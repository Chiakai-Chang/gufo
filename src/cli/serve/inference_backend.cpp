#include "src/cli/serve/inference_backend.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/continuation_cache.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/generator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/hip/executor.hpp"
#endif

namespace strix::server {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)
void EmitRequestMetrics(const InferenceBackend::Result& result,
                        std::string_view status) {
  static std::mutex output_mutex;
  std::ostringstream line;
  line << std::fixed << std::setprecision(3)
       << "{\"event\":\"http_inference\",\"status\":\"" << status
       << "\",\"prompt_tokens\":" << result.prompt_tokens
       << ",\"cache_hit\":" << (result.cache_hit ? "true" : "false")
       << ",\"cached_prompt_tokens\":" << result.cached_prompt_tokens
       << ",\"uncached_prompt_tokens\":"
       << (result.prompt_tokens - result.cached_prompt_tokens)
       << ",\"completion_tokens\":" << result.completion_tokens
       << ",\"ttft_ms\":" << result.ttft_ms
       << ",\"mean_inter_token_ms\":" << result.mean_inter_token_ms
       << ",\"cancelled\":" << (result.cancelled ? "true" : "false") << "}";
  const std::lock_guard<std::mutex> lock(output_mutex);
  std::clog << line.str() << '\n';
}

class QwenContinuationState final : public ContinuationState {
public:
  QwenContinuationState(std::shared_ptr<const hip::QwenGpuModel> model,
                        std::uint32_t max_context) {
    std::string error;
    executor_ =
        hip::QwenGpuExecutor::Create(std::move(model), &error, max_context);
    if (executor_ == nullptr) {
      throw std::runtime_error("Failed to create GPU session: " + error);
    }
  }

  void Invalidate() noexcept override { executor_->Reset(); }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }

private:
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
};

class DeepSeekContinuationState final : public ContinuationState {
public:
  DeepSeekContinuationState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
};

QwenContinuationState& RequireQwenState(ContinuationState& state) {
  auto* qwen = dynamic_cast<QwenContinuationState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("continuation state is not Qwen");
  }
  return *qwen;
}

DeepSeekContinuationState& RequireDeepSeekState(ContinuationState& state) {
  auto* deepseek = dynamic_cast<DeepSeekContinuationState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("continuation state is not DeepSeek");
  }
  return *deepseek;
}

std::vector<ContinuationToken> DeepSeekContinuationTokens(
    std::span<const int> tokens) {
  std::vector<ContinuationToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<ContinuationToken>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

std::string DeepSeekToolsPrompt(const ChatRequest& request) {
  if (request.tools.empty() ||
      request.tool_choice == ChatRequest::ToolChoice::kNone) {
    return {};
  }

  std::ostringstream prompt;
  prompt << "\n\n## Tools\n\n"
         << "You have access to tools. Invoke them with this exact syntax:\n"
         << "<｜DSML｜tool_calls｜>\n"
         << "<｜DS｜invoke name=\"$TOOL_NAME\">\n"
         << "<｜DS｜parameter name=\"$PARAMETER_NAME\" "
            "string=\"true|false\">$PARAMETER_VALUE"
            "</｜DS｜parameter>\n"
         << "</｜DS｜invoke>\n"
         << "</｜DSML｜tool_calls｜>\n\n"
         << "Available tool schemas:\n";
  for (const auto& tool : request.tools) {
    prompt << "{\"type\":\"function\",\"function\":{\"name\":"
           << std::quoted(tool.name)
           << ",\"description\":" << std::quoted(tool.description)
           << ",\"parameters\":"
           << (tool.parameters_json.empty() ? "{}" : tool.parameters_json)
           << "}}\n";
  }
  if (request.tool_choice == ChatRequest::ToolChoice::kRequired) {
    prompt << "\nYou must call at least one available tool.";
  }
  return prompt.str();
}

void AppendDeepSeekToolCalls(
    std::string& content,
    std::span<const tokenization::ChatMessage::ToolCall> calls) {
  if (calls.empty()) {
    return;
  }
  content.append("<｜DSML｜tool_calls｜>\n");
  for (const auto& call : calls) {
    content.append("<｜DS｜invoke name=\"");
    content.append(call.name);
    content.append("\">\n");
    for (const auto& argument : call.arguments) {
      content.append("<｜DS｜parameter name=\"");
      content.append(argument.name);
      content.append("\" string=\"");
      content.append(argument.is_string ? "true" : "false");
      content.append("\">");
      content.append(argument.value);
      content.append("</｜DS｜parameter>\n");
    }
    content.append("</｜DS｜invoke>\n");
  }
  content.append("</｜DSML｜tool_calls｜>");
}

#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    enum class Kind : std::uint8_t {
      kQwen,
      kDeepSeekV4Flash,
    };

    Kind kind = Kind::kQwen;
    std::shared_ptr<const hip::QwenGpuModel> qwen_model;
    std::shared_ptr<ContinuationCache> qwen_continuations;
    std::shared_ptr<models::deepseek_v4_flash::Model> deepseek_model;
    std::shared_ptr<ContinuationCache> deepseek_continuations;
    std::string model_id;
    SamplingDefaults sampling_defaults;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateQwen(std::shared_ptr<const State> current,
                      std::span<const tokenization::TokenId> prompt_tokens,
                      Clock::time_point request_start, std::size_t max_tokens,
                      float temperature, const CancellationCheck& is_cancelled,
                      const TokenCallback& on_token) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    auto lease =
        current->qwen_continuations->Acquire(prompt_tokens, is_cancelled);
    if (!lease) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }
    result.cache_hit = lease.cache_hit();
    result.cached_prompt_tokens = lease.cached_tokens();
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    models::GenerationOptions options;
    options.max_new_tokens = max_tokens > 0 ? max_tokens : 1;
    options.temperature = temperature;

    std::optional<Clock::time_point> previous_token;
    std::chrono::duration<double, std::milli> inter_token_total{0};
    std::size_t inter_token_samples = 0;
    try {
      result.tokens =
          RequireQwenState(lease.state())
              .executor()
              .GenerateFromPrefix(
                  prompt_tokens, result.cached_prompt_tokens, options,
                  [&](tokenization::TokenId, std::string_view piece) {
                    const auto now = Clock::now();
                    if (!previous_token.has_value()) {
                      result.ttft_ms =
                          std::chrono::duration<double, std::milli>(
                              now - request_start)
                              .count();
                    } else {
                      inter_token_total += now - *previous_token;
                      ++inter_token_samples;
                    }
                    previous_token = now;
                    if (is_cancelled && is_cancelled()) {
                      result.cancelled = true;
                      return false;
                    }
                    if (on_token && !on_token(piece)) {
                      result.cancelled = true;
                      return false;
                    }
                    return true;
                  });
    } catch (...) {
      EmitRequestMetrics(result, "error");
      throw;
    }

    result.completion_tokens = result.tokens.size();
    result.text = current->qwen_model->GetTokenizer().Decode(result.tokens);
    result.finish_reason =
        result.cancelled
            ? FinishReason::kCancelled
            : (result.completion_tokens >= max_tokens ? FinishReason::kLength
                                                      : FinishReason::kStop);
    if (inter_token_samples > 0) {
      result.mean_inter_token_ms =
          inter_token_total.count() / static_cast<double>(inter_token_samples);
    }
    if (!result.cancelled) {
      std::vector<ContinuationToken> checkpoint(prompt_tokens.begin(),
                                                prompt_tokens.end());
      checkpoint.insert(checkpoint.end(), result.tokens.begin(),
                        result.tokens.end());
      lease.Commit(std::move(checkpoint));
    }
    EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
    return result;
  }

  Result GenerateDeepSeek(std::shared_ptr<const State> current,
                          std::span<const int> prompt_tokens,
                          Clock::time_point request_start,
                          std::size_t max_tokens, float temperature,
                          const CancellationCheck& is_cancelled,
                          const TokenCallback& on_token) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    auto continuation_prompt = DeepSeekContinuationTokens(prompt_tokens);
    auto lease = current->deepseek_continuations->Acquire(continuation_prompt,
                                                          is_cancelled);
    if (!lease) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }
    result.cache_hit = lease.cache_hit();
    result.cached_prompt_tokens = lease.cached_tokens();
    auto& session = RequireDeepSeekState(lease.state()).session();
    session.SetCancellationCheck(is_cancelled);

    std::string error;
    if (!session.Sync(prompt_tokens, &error)) {
      if (is_cancelled && is_cancelled()) {
        result.cancelled = true;
        EmitRequestMetrics(result, "cancelled");
        return result;
      }
      EmitRequestMetrics(result, "error");
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }

    std::optional<Clock::time_point> previous_token;
    std::chrono::duration<double, std::milli> inter_token_total{0};
    std::size_t inter_token_samples = 0;
    std::random_device random_device;
    std::uint64_t rng_state =
        (static_cast<std::uint64_t>(random_device()) << 32U) ^
        static_cast<std::uint64_t>(random_device());
    for (std::size_t index = 0; index < max_tokens; ++index) {
      if (is_cancelled && is_cancelled()) {
        result.cancelled = true;
        break;
      }
      const int token =
          session.SelectNext(temperature, &rng_state, 0, 1.0F, 0.05F);
      if (token < 0) {
        EmitRequestMetrics(result, "error");
        throw std::runtime_error("DeepSeek token selection failed");
      }
      if (current->deepseek_model->IsStopToken(token)) {
        break;
      }

      const auto now = Clock::now();
      if (!previous_token.has_value()) {
        result.ttft_ms =
            std::chrono::duration<double, std::milli>(now - request_start)
                .count();
      } else {
        inter_token_total += now - *previous_token;
        ++inter_token_samples;
      }
      previous_token = now;
      result.tokens.push_back(static_cast<tokenization::TokenId>(token));
      const std::string piece = current->deepseek_model->DecodeToken(token);
      result.text += piece;
      if (on_token && !on_token(piece)) {
        result.cancelled = true;
        break;
      }

      if (index + 1 < max_tokens && !session.Evaluate(token, &error)) {
        if (is_cancelled && is_cancelled()) {
          result.cancelled = true;
          break;
        }
        EmitRequestMetrics(result, "error");
        throw std::runtime_error("DeepSeek decode failed: " + error);
      }
    }

    result.completion_tokens = result.tokens.size();
    result.finish_reason =
        result.cancelled
            ? FinishReason::kCancelled
            : (result.completion_tokens >= max_tokens ? FinishReason::kLength
                                                      : FinishReason::kStop);
    if (inter_token_samples > 0) {
      result.mean_inter_token_ms =
          inter_token_total.count() / static_cast<double>(inter_token_samples);
    }
    session.SetCancellationCheck({});
    if (!result.cancelled) {
      continuation_prompt.insert(continuation_prompt.end(),
                                 result.tokens.begin(), result.tokens.end());
      const int checkpoint_position = session.Position();
      if (checkpoint_position < 0 ||
          static_cast<std::size_t>(checkpoint_position) >
              continuation_prompt.size()) {
        throw std::runtime_error(
            "DeepSeek checkpoint position exceeds generated token history");
      }
      continuation_prompt.resize(static_cast<std::size_t>(checkpoint_position));
      lease.Commit(std::move(continuation_prompt));
    }
    EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
    return result;
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count) {
#if defined(ENGINE_ENABLE_HIP)
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (session_count != 1) {
      SetError(error, "DeepSeek V4 Flash currently supports one HTTP session");
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .prefill_chunk = 2048,
            .power_percent = 100,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    return load(std::move(model), error, max_context, session_count);
  }
  auto model = hip::QwenGpuModel::CreateFromGguf(reader, &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  return load(std::move(model), error, max_context, session_count);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->kind = Impl::State::Kind::kQwen;
    new_state->qwen_model = std::move(model);
    new_state->qwen_continuations = std::make_shared<ContinuationCache>(
        session_count, [model = new_state->qwen_model, max_context] {
          return std::make_unique<QwenContinuationState>(model, max_context);
        });
    new_state->model_id = new_state->qwen_model->GetConfig().model_name;
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (session_count != 1) {
    SetError(error, "DeepSeek V4 Flash currently supports one HTTP session");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->kind = Impl::State::Kind::kDeepSeekV4Flash;
    new_state->deepseek_model = std::move(model);
    new_state->deepseek_continuations = std::make_shared<ContinuationCache>(
        session_count, [model = new_state->deepseek_model, max_context] {
          return std::make_unique<DeepSeekContinuationState>(model,
                                                             max_context);
        });
    new_state->model_id = new_state->deepseek_model->ModelName();
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

void InferenceBackend::set_model_id(const std::string& model_id) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_id.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = model_id;
  impl_->state = std::move(updated);
#else
  (void)model_id;
#endif
}

void InferenceBackend::set_sampling_defaults(std::size_t max_tokens,
                                             float temperature) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults = {
      .max_tokens = max_tokens,
      .temperature = temperature,
  };
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)temperature;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    const auto prompt_tokens = state->deepseek_model->Tokenize(prompt);
    return impl_->GenerateDeepSeek(state, prompt_tokens, request_start,
                                   max_tokens, temperature, is_cancelled,
                                   on_token);
  }
  const auto prompt_tokens = state->qwen_model->GetTokenizer().Encode(prompt);
  return impl_->GenerateQwen(state, prompt_tokens, request_start, max_tokens,
                             temperature, is_cancelled, on_token);
#else
  (void)prompt;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    std::vector<models::deepseek_v4_flash::ChatMessage> deepseek_messages;
    deepseek_messages.reserve(request.messages.size() + 1);
    const std::string tools_prompt = DeepSeekToolsPrompt(request);
    bool tools_rendered = tools_prompt.empty();
    if (!tools_rendered &&
        (request.messages.empty() ||
         (request.messages.front().role != tokenization::ChatRole::kSystem &&
          request.messages.front().role !=
              tokenization::ChatRole::kDeveloper))) {
      deepseek_messages.push_back({
          .role = "system",
          .content = tools_prompt,
      });
      tools_rendered = true;
    }
    for (const auto& message : request.messages) {
      std::string content = message.content;
      if (!tools_rendered &&
          (message.role == tokenization::ChatRole::kSystem ||
           message.role == tokenization::ChatRole::kDeveloper)) {
        content += tools_prompt;
        tools_rendered = true;
      }
      if (message.role == tokenization::ChatRole::kAssistant) {
        AppendDeepSeekToolCalls(content, message.tool_calls);
      }
      deepseek_messages.push_back({
          .role = std::string(ChatRoleName(message.role)),
          .content = std::move(content),
      });
    }
    const auto prompt_tokens =
        state->deepseek_model->EncodeChat(deepseek_messages);
    return impl_->GenerateDeepSeek(state, prompt_tokens, request_start,
                                   max_tokens, temperature, is_cancelled,
                                   on_token);
  }
  tokenization::ChatTemplateOptions template_options;
  template_options.require_tool_call =
      request.tool_choice == ChatRequest::ToolChoice::kRequired;
  const auto prompt_tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
      state->qwen_model->GetTokenizer(), request.messages,
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools},
      template_options);
  if (!prompt_tokens.has_value() || prompt_tokens->empty()) {
    return {};
  }
  return impl_->GenerateQwen(state, *prompt_tokens, request_start, max_tokens,
                             temperature, is_cancelled, on_token);
#else
  (void)request;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, temperature, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  if (state->kind == Impl::State::Kind::kDeepSeekV4Flash) {
    return state->deepseek_model->Tokenize(text).size();
  }
  return state->qwen_model->GetTokenizer().Encode(text).size();
#else
  (void)text;
  return 0;
#endif
}

}  // namespace strix::server
