#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

namespace strix::server {
namespace {

struct ScheduledRequest {
  std::uint64_t id{0};
  std::vector<TextRunnerToken> prompt;
  std::size_t token_limit{1};
  float temperature{0.0F};
  TextGenerationScheduler::CancellationCheck external_cancellation;
  bool publish_token_pieces{false};
  TextGenerationScheduler::Clock::time_point request_start;

  std::atomic<bool> cancellation_requested{false};
  std::atomic<TextRequestPhase> phase{TextRequestPhase::kQueued};

  TextRunnerPool::Request runner_request;
  TextGenerationScheduler::Result result;
  std::optional<TextGenerationScheduler::Clock::time_point> previous_token;
  std::chrono::duration<double, std::milli> inter_token_total{0};
  std::size_t inter_token_samples{0};

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::deque<std::string> output_pieces;
  std::exception_ptr failure;
  bool terminal{false};
};

bool IsTerminal(const std::shared_ptr<ScheduledRequest>& request) noexcept {
  return request->phase.load(std::memory_order_acquire) ==
         TextRequestPhase::kTerminal;
}

void PublishTerminal(const std::shared_ptr<ScheduledRequest>& request,
                     std::exception_ptr failure = {},
                     bool discard_pending_output = false) noexcept {
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (discard_pending_output) {
      request->output_pieces.clear();
    }
    request->failure = std::move(failure);
    request->terminal = true;
    request->phase.store(TextRequestPhase::kTerminal,
                         std::memory_order_release);
  }
  request->output_condition.notify_all();
}

void PublishPiece(const std::shared_ptr<ScheduledRequest>& request,
                  std::string piece) {
  if (!request->publish_token_pieces) {
    return;
  }
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    request->output_pieces.push_back(std::move(piece));
  }
  request->output_condition.notify_one();
}

bool CancellationRequested(const std::shared_ptr<ScheduledRequest>& request) {
  if (request->cancellation_requested.load(std::memory_order_acquire)) {
    return true;
  }
  return request->external_cancellation && request->external_cancellation();
}

}  // namespace

struct TextGenerationScheduler::Request::Impl {
  explicit Impl(std::shared_ptr<ScheduledRequest> scheduled_request)
      : request(std::move(scheduled_request)) {}

  std::shared_ptr<ScheduledRequest> request;
  bool waited{false};
};

struct TextGenerationScheduler::Impl {
  explicit Impl(std::shared_ptr<TextRunnerPool> model_runner_pool)
      : runner_pool(std::move(model_runner_pool)) {
    if (runner_pool == nullptr) {
      throw std::invalid_argument(
          "text generation scheduler runner pool must not be null");
    }
    worker = std::jthread(
        [this](const std::stop_token& stop_token) { Run(stop_token); });
  }

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    worker.request_stop();
    queue_condition.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  [[nodiscard]] std::shared_ptr<ScheduledRequest> PopQueued() {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (queued.empty()) {
      return {};
    }
    auto request = std::move(queued.front());
    queued.pop_front();
    return request;
  }

  [[nodiscard]] bool RemoveQueued(
      const std::shared_ptr<ScheduledRequest>& request) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    const auto iterator = std::find(queued.begin(), queued.end(), request);
    if (iterator == queued.end()) {
      return false;
    }
    queued.erase(iterator);
    return true;
  }

  [[nodiscard]] std::vector<std::shared_ptr<ScheduledRequest>> QueuedSnapshot()
      const {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    return {queued.begin(), queued.end()};
  }

  void FinalizeResult(const std::shared_ptr<ScheduledRequest>& request,
                      TextGenerationBackend::FinishReason finish_reason) {
    request->result.completion_tokens = request->result.tokens.size();
    request->result.finish_reason = finish_reason;
    if (request->inter_token_samples > 0) {
      request->result.mean_inter_token_ms =
          request->inter_token_total.count() /
          static_cast<double>(request->inter_token_samples);
    }
    request->result.text = runner_pool->runner().Decode(request->result.tokens);
  }

  void CompleteCancelled(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    try {
      if (request->runner_request) {
        request->runner_request.Invalidate();
      }
      request->result.cancelled = true;
      FinalizeResult(request, TextGenerationBackend::FinishReason::kCancelled);
      PublishTerminal(request, {}, true);
    } catch (...) {
      PublishTerminal(request, std::current_exception(), true);
    }
  }

  void CompleteFailure(const std::shared_ptr<ScheduledRequest>& request,
                       std::exception_ptr failure) noexcept {
    if (request->runner_request) {
      request->runner_request.Invalidate();
    }
    request->result.completion_tokens = request->result.tokens.size();
    PublishTerminal(request, std::move(failure), true);
  }

  void CompleteSuccess(const std::shared_ptr<ScheduledRequest>& request,
                       TextGenerationBackend::FinishReason finish_reason) {
    FinalizeResult(request, finish_reason);
    request->runner_request.Commit();
    PublishTerminal(request);
  }

  void ProcessQueuedCancellations() {
    for (const auto& request : QueuedSnapshot()) {
      try {
        if (CancellationRequested(request) && RemoveQueued(request)) {
          CompleteCancelled(request);
        }
      } catch (...) {
        if (RemoveQueued(request)) {
          CompleteFailure(request, std::current_exception());
        }
      }
    }
  }

  void Admit(std::deque<std::shared_ptr<ScheduledRequest>>& active) {
    while (active.size() < runner_pool->capacity()) {
      auto request = PopQueued();
      if (request == nullptr) {
        return;
      }
      try {
        if (CancellationRequested(request)) {
          CompleteCancelled(request);
          continue;
        }

        request->runner_request =
            runner_pool->Acquire(std::move(request->prompt), [request] {
              return request->cancellation_requested.load(
                  std::memory_order_acquire);
            });
        if (!request->runner_request) {
          CompleteCancelled(request);
          continue;
        }

        request->result.cache_hit = request->runner_request.cache_hit();
        request->result.cached_prompt_tokens =
            request->runner_request.cached_prompt_tokens();
        request->phase.store(TextRequestPhase::kAdmitted,
                             std::memory_order_release);
        request->phase.store(request->runner_request.prefill_complete()
                                 ? TextRequestPhase::kDecodeReady
                                 : TextRequestPhase::kPrefilling,
                             std::memory_order_release);
        active.push_back(std::move(request));
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
  }

  void Step(const std::shared_ptr<ScheduledRequest>& request) {
    try {
      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return;
      }

      if (!request->runner_request.prefill_complete()) {
        request->phase.store(TextRequestPhase::kPrefilling,
                             std::memory_order_release);
        (void)request->runner_request.Prefill(
            request->runner_request.prompt_tokens());
        request->phase.store(TextRequestPhase::kDecodeReady,
                             std::memory_order_release);
        return;
      }

      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      const auto selection =
          request->runner_request.SelectNext(request->temperature);
      if (selection.stop) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
        return;
      }

      const auto now = Clock::now();
      if (!request->previous_token.has_value()) {
        request->result.ttft_ms = std::chrono::duration<double, std::milli>(
                                      now - request->request_start)
                                      .count();
      } else {
        request->inter_token_total += now - *request->previous_token;
        ++request->inter_token_samples;
      }
      request->previous_token = now;
      request->result.tokens.push_back(selection.token);
      PublishPiece(request, selection.piece);

      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return;
      }
      request->runner_request.Advance();
      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return;
      }
      if (request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
      }
    } catch (...) {
      CompleteFailure(request, std::current_exception());
    }
  }

  void CancelRemaining(
      std::deque<std::shared_ptr<ScheduledRequest>>& active) noexcept {
    std::deque<std::shared_ptr<ScheduledRequest>> remaining_queued;
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      remaining_queued = std::move(queued);
    }
    for (const auto& request : remaining_queued) {
      CompleteCancelled(request);
    }
    for (const auto& request : active) {
      CompleteCancelled(request);
    }
    active.clear();
  }

  void Run(const std::stop_token& stop_token) noexcept {
    std::deque<std::shared_ptr<ScheduledRequest>> active;
    while (!stop_token.stop_requested()) {
      ProcessQueuedCancellations();
      Admit(active);

      if (active.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_condition.wait(lock, [&] {
          return stop_token.stop_requested() || stopping || !queued.empty();
        });
        continue;
      }

      auto request = std::move(active.front());
      active.pop_front();
      Step(request);
      if (!IsTerminal(request)) {
        active.push_back(std::move(request));
      }
    }
    CancelRemaining(active);
  }

  std::shared_ptr<TextRunnerPool> runner_pool;
  mutable std::mutex queue_mutex;
  std::condition_variable queue_condition;
  std::deque<std::shared_ptr<ScheduledRequest>> queued;
  bool stopping{false};
  std::atomic<std::uint64_t> next_request_id{1};
  std::jthread worker;
};

TextGenerationScheduler::Request::Request() = default;

TextGenerationScheduler::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TextGenerationScheduler::Request::~Request() {
  Cancel();
}

TextGenerationScheduler::Request::Request(Request&&) noexcept = default;

TextGenerationScheduler::Request& TextGenerationScheduler::Request::operator=(
    Request&& other) noexcept {
  if (this != &other) {
    Cancel();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

TextGenerationScheduler::Request::operator bool() const noexcept {
  return impl_ != nullptr && impl_->request != nullptr;
}

std::uint64_t TextGenerationScheduler::Request::id() const noexcept {
  return *this ? impl_->request->id : 0;
}

TextRequestPhase TextGenerationScheduler::Request::phase() const noexcept {
  return *this ? impl_->request->phase.load(std::memory_order_acquire)
               : TextRequestPhase::kTerminal;
}

TextGenerationScheduler::Result TextGenerationScheduler::Request::Wait(
    const TokenCallback& on_token) {
  if (!*this) {
    throw std::logic_error("text scheduler request is empty");
  }
  if (impl_->waited) {
    throw std::logic_error("text scheduler request was already consumed");
  }
  impl_->waited = true;

  bool deliver_pieces = true;
  bool consumer_cancelled = false;
  std::exception_ptr callback_failure;
  Result result;
  std::exception_ptr scheduler_failure;

  while (true) {
    std::string piece;
    bool terminal = false;
    {
      std::unique_lock<std::mutex> lock(impl_->request->output_mutex);
      impl_->request->output_condition.wait(lock, [&] {
        return impl_->request->terminal ||
               !impl_->request->output_pieces.empty();
      });
      if (!impl_->request->output_pieces.empty()) {
        piece = std::move(impl_->request->output_pieces.front());
        impl_->request->output_pieces.pop_front();
      } else if (impl_->request->terminal) {
        result = impl_->request->result;
        scheduler_failure = impl_->request->failure;
        terminal = true;
      }
    }

    if (!piece.empty() && deliver_pieces && on_token) {
      try {
        if (!on_token(piece)) {
          consumer_cancelled = true;
          deliver_pieces = false;
          Cancel();
        }
      } catch (...) {
        callback_failure = std::current_exception();
        deliver_pieces = false;
        Cancel();
      }
    }
    if (terminal) {
      break;
    }
  }

  if (callback_failure != nullptr) {
    std::rethrow_exception(callback_failure);
  }
  if (scheduler_failure != nullptr) {
    std::rethrow_exception(scheduler_failure);
  }
  if (consumer_cancelled) {
    result.cancelled = true;
    result.finish_reason = TextGenerationBackend::FinishReason::kCancelled;
  }
  return result;
}

void TextGenerationScheduler::Request::Cancel() noexcept {
  if (*this && !IsTerminal(impl_->request)) {
    impl_->request->cancellation_requested.store(true,
                                                 std::memory_order_release);
  }
}

TextGenerationScheduler::TextGenerationScheduler(
    std::shared_ptr<TextRunnerPool> runner_pool)
    : impl_(std::make_unique<Impl>(std::move(runner_pool))) {}

TextGenerationScheduler::~TextGenerationScheduler() = default;

const TextModelRunner& TextGenerationScheduler::runner() const noexcept {
  return impl_->runner_pool->runner();
}

std::size_t TextGenerationScheduler::capacity() const noexcept {
  return impl_->runner_pool->capacity();
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces, Clock::time_point request_start) {
  if (prompt.empty()) {
    throw std::invalid_argument("text scheduler prompt must not be empty");
  }

  auto request = std::make_shared<ScheduledRequest>();
  request->id = impl_->next_request_id.fetch_add(1, std::memory_order_relaxed);
  request->result.prompt_tokens = prompt.size();
  request->prompt = std::move(prompt);
  request->token_limit = max_tokens > 0 ? max_tokens : 1;
  request->temperature = temperature;
  request->external_cancellation = is_cancelled;
  request->publish_token_pieces = publish_token_pieces;
  request->request_start = request_start;

  {
    const std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    if (impl_->stopping) {
      throw std::runtime_error("text generation scheduler is stopping");
    }
    impl_->queued.push_back(request);
  }
  impl_->queue_condition.notify_one();
  return Request(std::make_unique<Request::Impl>(std::move(request)));
}

}  // namespace strix::server
