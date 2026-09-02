#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

namespace ds4 = gufo::models::deepseek_v4_flash;
namespace json = gufo::server::json;
namespace qwen = gufo::tokenization;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

const json::Value& Required(const json::Value& object, std::string_view key) {
  const json::Value* value = object.find(std::string(key));
  Expect(value != nullptr, "Golden fixture is missing a required field");
  return *value;
}

const json::Value& GoldenCase(const json::Value& fixture,
                              std::string_view model,
                              std::string_view case_name) {
  return Required(Required(Required(fixture, model), "cases"), case_name);
}

std::string Sha256(std::string_view value) {
  const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({data, value.size()});
}

template<typename Token>
std::string TokenSha256(std::span<const Token> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const Token token : tokens) {
    const auto value = static_cast<std::uint32_t>(token);
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24U));
  }
  return gufo::crypto::Sha256Hex(bytes);
}

void CheckQwenCase(const json::Value& fixture, std::string_view case_name,
                   const qwen::QwenTokenizer& tokenizer,
                   std::span<const qwen::ChatMessage> messages,
                   std::span<const qwen::ChatTool> tools,
                   const qwen::ChatTemplateOptions& options) {
  std::string error;
  const auto rendered =
      qwen::QwenChatTemplate::Render(messages, tools, options, &error);
  Expect(rendered.has_value(), "Qwen golden prompt renders: " + error);
  const auto tokens = qwen::QwenChatTemplate::RenderAndTokenize(
      tokenizer, messages, tools, options, &error);
  Expect(tokens.has_value(), "Qwen golden prompt tokenizes: " + error);

  const json::Value& expected = GoldenCase(fixture, "qwen", case_name);
  Expect(Sha256(*rendered) == expected.member_str("rendered_sha256"),
         "Qwen rendered bytes match Hugging Face");
  if (tokens->size() != expected.member_size("token_count")) {
    std::cerr << "Qwen case " << case_name << ": expected "
              << expected.member_size("token_count") << " tokens, got "
              << tokens->size() << "; token SHA-256 "
              << TokenSha256<qwen::TokenId>(*tokens) << '\n';
  }
  Expect(tokens->size() == expected.member_size("token_count"),
         "Qwen token count matches Hugging Face");
  Expect(TokenSha256<qwen::TokenId>(*tokens) ==
             expected.member_str("token_ids_le_u32_sha256"),
         "Qwen token IDs match Hugging Face");
}

void TestQwenGoldens(const json::Value& fixture,
                     const std::string& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader != nullptr, "Qwen GGUF opens: " + error);
  Expect(reader->GetMetadataString("tokenizer.ggml.pre") ==
             std::optional<std::string_view>{"qwen35"},
         "Qwen GGUF declares the qwen35 pre-tokenizer");
  Expect(qwen::QwenChatTemplate::ValidateGgufTemplate(*reader, &error),
         "Qwen GGUF template is recognized: " + error);
  const auto tokenizer = qwen::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, "Qwen tokenizer loads: " + error);

  const std::vector<qwen::ChatMessage> base = {
      {qwen::ChatRole::kUser, "Name one color.", "", ""},
  };
  const std::vector<qwen::ChatMessage> history = {
      {qwen::ChatRole::kUser, "Name one color.", "", ""},
      {qwen::ChatRole::kAssistant, "Red", "",
       "I should answer with one color."},
      {qwen::ChatRole::kUser, "Name another.", "", ""},
  };
  const std::vector<qwen::ChatTool> tools = {{
      .name = "get_weather",
      .description = "Get weather",
      .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{"
                         "\"type\":\"string\"}},"
                         "\"required\":[\"city\"]}",
  }};

  qwen::ChatTemplateOptions options;
  options.enable_thinking = false;
  CheckQwenCase(fixture, "chat", *tokenizer, base, {}, options);

  options.enable_thinking = true;
  options.reasoning_effort = qwen::QwenReasoningEffort::kLow;
  CheckQwenCase(fixture, "thinking_low", *tokenizer, base, {}, options);
  options.reasoning_effort = qwen::QwenReasoningEffort::kMedium;
  CheckQwenCase(fixture, "thinking_medium", *tokenizer, base, {}, options);
  options.reasoning_effort = qwen::QwenReasoningEffort::kXHigh;
  CheckQwenCase(fixture, "thinking_xhigh", *tokenizer, base, {}, options);

  options.reasoning_effort = qwen::QwenReasoningEffort::kMedium;
  options.preserve_thinking = false;
  CheckQwenCase(fixture, "history_drop", *tokenizer, history, {}, options);
  options.enable_thinking = false;
  options.preserve_thinking = true;
  CheckQwenCase(fixture, "history_preserve", *tokenizer, history, {}, options);

  options = {};
  CheckQwenCase(fixture, "tools", *tokenizer, base, tools, options);
}

void CheckDeepSeekCase(const json::Value& fixture, std::string_view case_name,
                       const ds4::Model& model,
                       std::span<const ds4::ChatMessage> messages,
                       std::span<const ds4::ChatTool> tools,
                       const ds4::ChatTemplateOptions& options) {
  const std::string rendered = ds4::RenderChat(messages, tools, options);
  const std::vector<int> tokens = model.EncodeChat(messages, tools, options);

  const json::Value& expected = GoldenCase(fixture, "deepseek", case_name);
  Expect(Sha256(rendered) == expected.member_str("rendered_sha256"),
         "DeepSeek rendered bytes match Hugging Face");
  Expect(tokens.size() == expected.member_size("token_count"),
         "DeepSeek token count matches Hugging Face");
  Expect(TokenSha256<int>(tokens) ==
             expected.member_str("token_ids_le_u32_sha256"),
         "DeepSeek token IDs match Hugging Face");
}

void TestDeepSeekGoldens(const json::Value& fixture,
                         const std::string& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader != nullptr, "DeepSeek GGUF opens: " + error);
  Expect(ds4::ValidateGgufTemplate(*reader, &error),
         "DeepSeek GGUF template is recognized: " + error);

  const auto model = ds4::Model::Load(model_path,
                                      ds4::ModelOptions{
                                          .max_context = 4096,
                                          .prefill_chunk = 512,
                                          .power_percent = 100,
                                          .dspark_model_path = {},
                                      },
                                      &error);
  Expect(model != nullptr, "DeepSeek model loads: " + error);

  const std::vector<ds4::ChatMessage> base = {
      {.role = "system",
       .content = "Be concise.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Name one color.",
       .reasoning_content = {},
       .tool_calls = {}},
  };
  const std::vector<ds4::ChatMessage> history = {
      {.role = "user",
       .content = "Name one color.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "assistant",
       .content = "Red",
       .reasoning_content = "I should answer with one color.",
       .tool_calls = {}},
      {.role = "user",
       .content = "Name another.",
       .reasoning_content = {},
       .tool_calls = {}},
  };

  ds4::ChatTemplateOptions options;
  CheckDeepSeekCase(fixture, "chat", *model, base, {}, options);
  options.enable_thinking = true;
  options.reasoning_effort = gufo::ReasoningEffort::kLow;
  CheckDeepSeekCase(fixture, "thinking_low", *model, base, {}, options);
  options.reasoning_effort = gufo::ReasoningEffort::kHigh;
  CheckDeepSeekCase(fixture, "thinking_high", *model, base, {}, options);
  options.reasoning_effort = gufo::ReasoningEffort::kMax;
  CheckDeepSeekCase(fixture, "thinking_max", *model, base, {}, options);

  options.reasoning_effort = gufo::ReasoningEffort::kLow;
  options.preserve_thinking = false;
  CheckDeepSeekCase(fixture, "history_drop", *model, history, {}, options);
  options.preserve_thinking = true;
  CheckDeepSeekCase(fixture, "history_preserve", *model, history, {}, options);

  ds4::ChatMessage assistant{
      .role = "assistant",
      .content = "",
      .reasoning_content = "Use the tool.",
      .tool_calls = {},
  };
  assistant.tool_calls.push_back({
      .name = "get_weather",
      .arguments = {{.name = "city", .value = "Rome", .is_string = true}},
  });
  const std::vector<ds4::ChatMessage> tool_messages = {
      {.role = "system",
       .content = "Be concise.",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Weather in Rome?",
       .reasoning_content = {},
       .tool_calls = {}},
      assistant,
      {.role = "tool",
       .content = "{\"temperature\":21}",
       .reasoning_content = {},
       .tool_calls = {}},
      {.role = "user",
       .content = "Summarize.",
       .reasoning_content = {},
       .tool_calls = {}},
  };
  const std::vector<ds4::ChatTool> tools = {{
      .name = "get_weather",
      .description = "Get weather",
      .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{"
                         "\"type\":\"string\"}},"
                         "\"required\":[\"city\"]}",
  }};
  options = {};
  options.enable_thinking = true;
  CheckDeepSeekCase(fixture, "tools", *model, tool_messages, tools, options);
}

}  // namespace

int main() {
  const char* qwen_model = std::getenv("GUFO_QWEN_GGUF");
  const char* deepseek_model = std::getenv("GUFO_DEEPSEEK_GGUF");
  if (qwen_model == nullptr || qwen_model[0] == '\0' ||
      deepseek_model == nullptr || deepseek_model[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN_GGUF and GUFO_DEEPSEEK_GGUF\n";
    return 77;
  }

  std::ifstream input(GUFO_CHAT_TEMPLATE_HF_GOLDENS, std::ios::binary);
  Expect(input.good(), "Hugging Face golden fixture opens");
  const std::string fixture_text{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
  const json::Value fixture = json::parse(fixture_text);

  TestQwenGoldens(fixture, qwen_model);
  TestDeepSeekGoldens(fixture, deepseek_model);
  std::cout << "Hugging Face chat-template token goldens passed\n";
  return 0;
}
