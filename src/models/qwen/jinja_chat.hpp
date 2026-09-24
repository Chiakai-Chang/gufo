#ifndef GUFO_MODELS_QWEN_JINJA_CHAT_HPP_
#define GUFO_MODELS_QWEN_JINJA_CHAT_HPP_

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::tokenization {

/// A user-supplied Jinja chat template (--chat-template-file), rendered with
/// the llama.cpp Jinja engine instead of the built-in Qwen formatter.
/// Requests that leave thinking/effort/preserve unset get the template's own
/// defaults; request chat_template_kwargs are passed through unchanged.
class JinjaChatTemplate {
public:
  ~JinjaChatTemplate();
  JinjaChatTemplate(const JinjaChatTemplate&) = delete;
  JinjaChatTemplate& operator=(const JinjaChatTemplate&) = delete;

  [[nodiscard]] static std::unique_ptr<JinjaChatTemplate> Load(
      const std::filesystem::path& path, std::string* error_msg = nullptr);

  /// Renders messages; image_offsets receives the byte offset after each
  /// <|vision_start|> that opens an image placeholder.
  [[nodiscard]] std::optional<std::string> Render(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options, std::string* error_msg = nullptr,
      std::vector<std::size_t>* image_offsets = nullptr) const;

  /// Thinking state the template chooses when a request does not set it.
  [[nodiscard]] bool DefaultThinking() const noexcept { return default_thinking_; }
  /// Assistant generation suffix for the given thinking state.
  [[nodiscard]] std::string_view GenerationPrompt(bool thinking) const noexcept {
    return thinking ? generation_thinking_ : generation_plain_;
  }
  [[nodiscard]] const std::string& Path() const noexcept { return path_; }

private:
  struct Impl;
  JinjaChatTemplate();

  std::unique_ptr<Impl> impl_;
  std::string path_;
  bool default_thinking_{true};
  std::string generation_thinking_;
  std::string generation_plain_;
};

/// Process-wide template used by the Qwen formatter when set (server/CLI).
void SetActiveJinjaTemplate(std::shared_ptr<const JinjaChatTemplate> tmpl);
[[nodiscard]] const JinjaChatTemplate* ActiveJinjaTemplate() noexcept;

}  // namespace gufo::tokenization

#endif  // GUFO_MODELS_QWEN_JINJA_CHAT_HPP_
