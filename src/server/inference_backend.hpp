#ifndef STRIX_SERVER_INFERENCE_BACKEND_HPP_
#define STRIX_SERVER_INFERENCE_BACKEND_HPP_

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen_generator.hpp"
#include "src/tokenization/qwen_chat_template.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

namespace strix::server {

/// Thread-safe wrapper around a loaded QwenGenerator exposing the text
/// operations the HTTP server needs. Every public method that can touch the
/// generator serializes on an internal mutex.
class InferenceBackend {
public:
  struct Result {
    std::string text;
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
  };

  /// Loads weights from a GGUF file. Returns false and sets *error on failure.
  bool load(const std::string& model_path, std::string* error);

  /// Stable model identifier used in API responses.
  std::string model_id() const;

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature);

  /// Framed chat conversation; returns the assistant reply.
  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, float temperature);

  /// Token count of raw text (no generation).
  std::size_t count_tokens(std::string_view text);

private:
  mutable std::mutex mutex_;
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<models::QwenGenerator> generator_;
  std::string model_id_{"unknown"};
};

}  // namespace strix::server

#endif  // STRIX_SERVER_INFERENCE_BACKEND_HPP_
