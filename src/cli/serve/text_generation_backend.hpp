#ifndef STRIX_SERVER_TEXT_GENERATION_BACKEND_HPP_
#define STRIX_SERVER_TEXT_GENERATION_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace strix::server {

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<tokenization::ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<tokenization::ChatMessage> messages;
  std::vector<tokenization::ChatTool> tools;
  ToolChoice tool_choice{ToolChoice::kAuto};
};

/// Model-agnostic text generation boundary used by the HTTP transport.
///
/// Implementations retain ownership of tokenization, templates, complete
/// continuation state, and execution resources. The transport observes only
/// generated text pieces and aggregate request metrics.
class TextGenerationBackend {
public:
  using CancellationCheck = std::function<bool()>;
  using TokenCallback = std::function<bool(std::string_view)>;

  enum class FinishReason : std::uint8_t {
    kStop,
    kLength,
    kCancelled,
  };

  struct SamplingDefaults {
    std::size_t max_tokens{128};
    float temperature{0.0F};
  };

  struct Result {
    std::string text;
    std::vector<tokenization::TokenId> tokens;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    std::size_t completion_tokens{0};
    std::size_t prefill_tokens{0};
    std::size_t prefill_chunks{0};
    std::size_t active_decode_prefill_chunks{0};
    std::size_t max_prefill_chunk_tokens{0};
    std::size_t max_consecutive_active_prefill_chunks{0};
    std::size_t configured_active_prefill_tokens{0};
    double ttft_ms{0.0};
    double mean_inter_token_ms{0.0};
    double max_inter_token_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::string prefill_fallback_reason;
    FinishReason finish_reason{FinishReason::kStop};
    bool incremental_prefill_supported{false};
    bool cache_hit{false};
    bool cancelled{false};
  };

  TextGenerationBackend() = default;
  virtual ~TextGenerationBackend() = default;

  TextGenerationBackend(const TextGenerationBackend&) = delete;
  TextGenerationBackend& operator=(const TextGenerationBackend&) = delete;
  TextGenerationBackend(TextGenerationBackend&&) = delete;
  TextGenerationBackend& operator=(TextGenerationBackend&&) = delete;

  [[nodiscard]] virtual std::string model_id() const = 0;
  [[nodiscard]] virtual bool ready() const = 0;
  [[nodiscard]] virtual SamplingDefaults sampling_defaults() const {
    return {};
  }

  virtual Result complete(std::string_view prompt, std::size_t max_tokens,
                          float temperature,
                          const CancellationCheck& is_cancelled = {},
                          const TokenCallback& on_token = {}) = 0;

  virtual Result chat(const ChatRequest& request, std::size_t max_tokens,
                      float temperature,
                      const CancellationCheck& is_cancelled = {},
                      const TokenCallback& on_token = {}) = 0;

  [[nodiscard]] virtual std::size_t count_tokens(
      std::string_view text) const = 0;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_TEXT_GENERATION_BACKEND_HPP_
