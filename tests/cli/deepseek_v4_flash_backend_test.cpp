#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

using Model = strix::models::deepseek_v4_flash::Model;
using ModelOptions = strix::models::deepseek_v4_flash::ModelOptions;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<strix::tokenization::TokenId> GenerateDirect(
    const std::shared_ptr<Model>& model, std::span<const int> prompt,
    std::size_t max_tokens) {
  std::string error;
  auto session = model->CreateSession(512, &error);
  Expect(session != nullptr, error);
  Expect(session->Sync(prompt, &error), error);

  std::vector<strix::tokenization::TokenId> result;
  result.reserve(max_tokens);
  for (std::size_t index = 0; index < max_tokens; ++index) {
    const int token = session->SelectNext(0.0F, nullptr);
    Expect(token >= 0, "direct token selection");
    if (model->IsStopToken(token)) {
      break;
    }
    result.push_back(static_cast<strix::tokenization::TokenId>(token));
    if (index + 1 < max_tokens) {
      Expect(session->Evaluate(token, &error), error);
    }
  }
  return result;
}

}  // namespace

int main() {
  try {
    const char* model_path = std::getenv("STRIX_DEEPSEEK_V4_FLASH_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
      std::cout << "SKIP: STRIX_DEEPSEEK_V4_FLASH_MODEL is not set\n";
      return 77;
    }

    std::string error;
    auto model = Model::Load(model_path,
                             ModelOptions{
                                 .max_context = 512,
                                 .prefill_chunk = 512,
                                 .power_percent = 100,
                             },
                             &error);
    Expect(model != nullptr, error);

    strix::server::InferenceBackend backend;
    Expect(backend.load(model, &error, 512, 2), error);
    Expect(backend.model_id() == model->ModelName(), "HTTP model identifier");

    const std::string raw_prompt = "The capital of France is";
    const auto direct_raw =
        GenerateDirect(model, model->Tokenize(raw_prompt), 2);
    const auto http_raw = backend.complete(raw_prompt, 2, 0.0F);
    Expect(http_raw.tokens == direct_raw, "raw direct/HTTP token parity");
    Expect(http_raw.ttft_ms > 0.0, "raw TTFT");

    const std::vector<strix::tokenization::ChatMessage> messages = {
        {strix::tokenization::ChatRole::kSystem,
         "Answer with one short sentence.", "", ""},
        {strix::tokenization::ChatRole::kUser, "Name one primary color.", "",
         ""},
    };
    const std::vector<strix::models::deepseek_v4_flash::ChatMessage>
        direct_messages = {
            {.role = "system", .content = "Answer with one short sentence."},
            {.role = "user", .content = "Name one primary color."},
        };
    const auto direct_chat =
        GenerateDirect(model, model->EncodeChat(direct_messages), 2);
    const auto http_chat = backend.chat(messages, 2, 0.0F);
    Expect(http_chat.tokens == direct_chat, "chat direct/HTTP token parity");
    Expect(!http_chat.cache_hit, "first chat request is a cache miss");

    auto continued_messages = messages;
    continued_messages.emplace_back(strix::tokenization::ChatRole::kAssistant,
                                    http_chat.text);
    continued_messages.emplace_back(strix::tokenization::ChatRole::kUser,
                                    "Name a different primary color.");
    auto continued_direct_messages = direct_messages;
    continued_direct_messages.push_back(
        {.role = "assistant", .content = http_chat.text});
    continued_direct_messages.push_back(
        {.role = "user", .content = "Name a different primary color."});
    const auto direct_continuation =
        GenerateDirect(model, model->EncodeChat(continued_direct_messages), 2);
    const auto http_continuation = backend.chat(continued_messages, 2, 0.0F);
    Expect(http_continuation.cache_hit,
           "continued chat reuses the retained DeepSeek state");
    Expect(http_continuation.cached_prompt_tokens > 0,
           "DeepSeek cache reports reused tokens");
    Expect(http_continuation.cached_prompt_tokens <
               http_continuation.prompt_tokens,
           "continued DeepSeek chat prefills only a suffix");
    Expect(http_continuation.tokens == direct_continuation,
           "cached DeepSeek continuation differs from cold full prefill");

    strix::server::ChatRequest concurrent_a({
        {strix::tokenization::ChatRole::kUser,
         "Continue this sequence with four short items: one, two, three,", "",
         ""},
    });
    concurrent_a.client_id = "deepseek-a";
    strix::server::ChatRequest concurrent_b({
        {strix::tokenization::ChatRole::kUser,
         "Continue this sequence with four short items: red, green, blue,", "",
         ""},
    });
    concurrent_b.client_id = "deepseek-b";
    const std::vector<strix::models::deepseek_v4_flash::ChatMessage>
        direct_a_messages = {
            {.role = "user",
             .content =
                 "Continue this sequence with four short items: one, two, "
                 "three,"},
        };
    const std::vector<strix::models::deepseek_v4_flash::ChatMessage>
        direct_b_messages = {
            {.role = "user",
             .content =
                 "Continue this sequence with four short items: red, green, "
                 "blue,"},
        };
    const auto direct_a =
        GenerateDirect(model, model->EncodeChat(direct_a_messages), 4);
    const auto direct_b =
        GenerateDirect(model, model->EncodeChat(direct_b_messages), 4);

    auto pending_a = backend.start_chat(concurrent_a, 4, 0.0F, {}, true);
    auto pending_b = backend.start_chat(concurrent_b, 4, 0.0F, {}, true);
    Expect(pending_a != nullptr && pending_b != nullptr,
           "concurrent DeepSeek requests are admitted");

    std::mutex event_mutex;
    std::vector<char> events;
    strix::server::InferenceBackend::Result concurrent_result_a;
    strix::server::InferenceBackend::Result concurrent_result_b;
    std::exception_ptr failure_a;
    std::exception_ptr failure_b;
    std::jthread waiter_a([&] {
      try {
        concurrent_result_a = pending_a->Wait([&](std::string_view) {
          const std::lock_guard<std::mutex> lock(event_mutex);
          events.push_back('a');
          return true;
        });
      } catch (...) {
        failure_a = std::current_exception();
      }
    });
    std::jthread waiter_b([&] {
      try {
        concurrent_result_b = pending_b->Wait([&](std::string_view) {
          const std::lock_guard<std::mutex> lock(event_mutex);
          events.push_back('b');
          return true;
        });
      } catch (...) {
        failure_b = std::current_exception();
      }
    });
    waiter_a.join();
    waiter_b.join();
    if (failure_a != nullptr) {
      std::rethrow_exception(failure_a);
    }
    if (failure_b != nullptr) {
      std::rethrow_exception(failure_b);
    }

    Expect(concurrent_result_a.tokens == direct_a,
           "concurrent DeepSeek request A differs from isolated execution");
    Expect(concurrent_result_b.tokens == direct_b,
           "concurrent DeepSeek request B differs from isolated execution");
    Expect(concurrent_result_a.requested_logical_concurrency == 2 &&
               concurrent_result_b.requested_logical_concurrency == 2,
           "DeepSeek scheduler reports two logical sessions");
    Expect(concurrent_result_a.physical_execution_width == 1 &&
               concurrent_result_b.physical_execution_width == 1 &&
               concurrent_result_a.execution_plan == "serial-fallback" &&
               concurrent_result_b.execution_plan == "serial-fallback",
           "DeepSeek reports exact serialized fallback");
    const auto first_a = std::find(events.begin(), events.end(), 'a');
    const auto first_b = std::find(events.begin(), events.end(), 'b');
    const auto last_a = std::find(events.rbegin(), events.rend(), 'a').base();
    const auto last_b = std::find(events.rbegin(), events.rend(), 'b').base();
    Expect(first_a != events.end() && first_b != events.end(),
           "both DeepSeek sessions stream output");
    Expect(first_b < last_a && first_a < last_b,
           "both DeepSeek sessions make progress before the other completes");

    std::size_t cancellation_checks = 0;
    const auto cancelled =
        backend.complete(raw_prompt, 4, 0.0F, [&cancellation_checks] {
          ++cancellation_checks;
          return cancellation_checks > 2;
        });
    Expect(cancelled.cancelled, "cancellation status");
    Expect(cancelled.completion_tokens <= 1, "cancellation token bound");

    const auto recovered = backend.complete(raw_prompt, 2, 0.0F);
    Expect(recovered.tokens == direct_raw, "session reuse after cancellation");
    Expect(!recovered.cache_hit,
           "cancelled DeepSeek state must not remain cached");

    std::cout << "DeepSeek V4 Flash HTTP parity test passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
