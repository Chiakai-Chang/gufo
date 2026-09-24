#include "src/models/qwen/jinja_chat.hpp"

#include <atomic>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json.h"

namespace gufo::tokenization {
namespace {

constexpr std::string_view kImageMarker = "<|vision_start|><|image_pad|><|vision_end|>";
constexpr std::string_view kVisionStart = "<|vision_start|>";

std::shared_ptr<const JinjaChatTemplate>& ActiveSlot() {
  static std::shared_ptr<const JinjaChatTemplate> slot;
  return slot;
}

common_json ToolArgumentValue(const ChatMessage::ToolArgument& arg) {
  if (!arg.is_string) {
    auto parsed = common_json::parse_no_throw(arg.value);
    if (!parsed.is_null() || arg.value == "null")
      return parsed;
  }
  return common_json(arg.value);
}

common_json MessageContent(const ChatMessage& msg) {
  if (msg.images.empty())
    return common_json(msg.content);
  // Split the text at each image offset, keeping input order.
  auto parts = common_json::array();
  std::size_t cursor = 0;
  for (const auto& image : msg.images) {
    const std::size_t at = std::min(image.offset, msg.content.size());
    if (at > cursor) {
      auto text = common_json::object();
      text["type"] = "text";
      text["text"] = msg.content.substr(cursor, at - cursor);
      parts.push_back(text);
      cursor = at;
    }
    auto img = common_json::object();
    img["type"] = "image";
    parts.push_back(img);
  }
  if (cursor < msg.content.size()) {
    auto text = common_json::object();
    text["type"] = "text";
    text["text"] = msg.content.substr(cursor);
    parts.push_back(text);
  }
  return parts;
}

common_json MessageJson(const ChatMessage& msg) {
  auto m = common_json::object();
  m["role"] = std::string(ToString(msg.role));
  m["content"] = MessageContent(msg);
  if (!msg.thought.empty())
    m["reasoning_content"] = msg.thought;
  if (!msg.name.empty())
    m["name"] = msg.name;
  if (!msg.tool_call_id.empty())
    m["tool_call_id"] = msg.tool_call_id;
  if (!msg.tool_calls.empty()) {
    auto calls = common_json::array();
    for (const auto& call : msg.tool_calls) {
      auto args = common_json::object();
      for (const auto& arg : call.arguments)
        args[arg.name] = ToolArgumentValue(arg);
      auto fn = common_json::object();
      fn["name"] = call.name;
      fn["arguments"] = args;
      auto c = common_json::object();
      if (!call.id.empty())
        c["id"] = call.id;
      c["type"] = "function";
      c["function"] = fn;
      calls.push_back(c);
    }
    m["tool_calls"] = calls;
  }
  return m;
}

common_json ToolJson(const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    auto parsed = common_json::parse_no_throw(tool.definition_json);
    if (parsed.is_object())
      return parsed;
  }
  auto fn = common_json::object();
  fn["name"] = tool.name;
  fn["description"] = tool.description;
  fn["parameters"] = common_json::parse_no_throw(tool.parameters_json);
  auto t = common_json::object();
  t["type"] = "function";
  t["function"] = fn;
  return t;
}

}  // namespace

struct JinjaChatTemplate::Impl {
  std::string source;
  jinja::program program;

  std::string Run(const common_json& inputs) const {
    jinja::context ctx(source);
    jinja::global_from_json(ctx, inputs, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(program);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
  }
};

JinjaChatTemplate::JinjaChatTemplate() : impl_(std::make_unique<Impl>()) {}
JinjaChatTemplate::~JinjaChatTemplate() = default;

std::unique_ptr<JinjaChatTemplate> JinjaChatTemplate::Load(
    const std::filesystem::path& path, std::string* error_msg) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error_msg)
      *error_msg = "cannot read chat template file: " + path.string();
    return nullptr;
  }
  std::ostringstream text;
  text << in.rdbuf();
  std::unique_ptr<JinjaChatTemplate> t(new JinjaChatTemplate());
  t->path_ = path.string();
  try {
    jinja::lexer lexer;
    auto tokens = lexer.tokenize(text.str());
    t->impl_->program = jinja::parse_from_tokens(tokens);
    t->impl_->source = tokens.source;

    // Probe the template's defaults with a one-message conversation.
    const std::vector<ChatMessage> probe{ChatMessage(ChatRole::kUser, "hi")};
    ChatTemplateOptions options;
    const auto render = [&](bool generation, std::optional<bool> thinking) {
      options.add_generation_prompt = generation;
      options.requested_thinking = thinking;
      std::string err;
      auto out = t->Render(probe, {}, options, &err);
      if (!out)
        throw std::runtime_error(err);
      return *out;
    };
    const auto suffix = [&](std::optional<bool> thinking) {
      const std::string base = render(false, thinking);
      const std::string full = render(true, thinking);
      if (full.size() < base.size() || full.compare(0, base.size(), base) != 0)
        throw std::runtime_error("add_generation_prompt does not extend the conversation");
      return full.substr(base.size());
    };
    t->generation_thinking_ = suffix(true);
    t->generation_plain_ = suffix(false);
    const std::string default_suffix = suffix(std::nullopt);
    t->default_thinking_ = default_suffix == t->generation_thinking_;
  } catch (const std::exception& e) {
    if (error_msg)
      *error_msg = "chat template " + path.string() + ": " + e.what();
    return nullptr;
  }
  return t;
}

std::optional<std::string> JinjaChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets) const {
  if (image_offsets)
    image_offsets->clear();
  try {
    auto inputs = common_json::object();
    // Client kwargs first; explicit request controls below take precedence.
    if (!options.template_kwargs_json.empty()) {
      auto kwargs = common_json::parse_no_throw(options.template_kwargs_json);
      if (kwargs.is_object())
        inputs = kwargs;
    }
    auto list = common_json::array();
    for (const auto& msg : messages)
      list.push_back(MessageJson(msg));
    inputs["messages"] = list;
    if (!tools.empty()) {
      auto tl = common_json::array();
      for (const auto& tool : tools)
        tl.push_back(ToolJson(tool));
      inputs["tools"] = tl;
    }
    inputs["add_generation_prompt"] = options.add_generation_prompt;
    inputs["add_vision_id"] = options.add_vision_id;
    inputs["bos_token"] = "";
    inputs["eos_token"] = "<|im_end|>";
    if (options.requested_thinking.has_value())
      inputs["enable_thinking"] = *options.requested_thinking;
    if (!options.requested_effort.empty())
      inputs["reasoning_effort"] = options.requested_effort;
    if (options.requested_preserve.has_value())
      inputs["preserve_thinking"] = *options.requested_preserve;

    std::string out = impl_->Run(inputs);
    if (out.size() > options.max_output_bytes)
      throw std::length_error("rendered chat prompt exceeds the output limit");
    if (image_offsets) {
      for (std::size_t pos = out.find(kImageMarker); pos != std::string::npos;
           pos = out.find(kImageMarker, pos + kImageMarker.size()))
        image_offsets->push_back(pos + kVisionStart.size());
    }
    return out;
  } catch (const std::exception& e) {
    if (error_msg)
      *error_msg = std::string("chat template render failed: ") + e.what();
    return std::nullopt;
  }
}

void SetActiveJinjaTemplate(std::shared_ptr<const JinjaChatTemplate> tmpl) {
  std::atomic_store(&ActiveSlot(), std::move(tmpl));
}

const JinjaChatTemplate* ActiveJinjaTemplate() noexcept {
  return std::atomic_load(&ActiveSlot()).get();
}

}  // namespace gufo::tokenization
