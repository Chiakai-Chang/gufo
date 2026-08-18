#include "src/server/prompt_cli.hpp"

#include <charconv>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/tokenization/qwen_chat_template.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

namespace strix::server {
namespace {

void PrintPromptHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " prompt [OPTIONS] <PROMPT>\n\n"
      << "Execute one prompt request and exit.\n\n"
      << "Options:\n"
      << "  -m, --model <PATH>      Path to GGUF model file\n"
      << "  -n, --max-tokens <N>    Maximum tokens to generate (default: 128)\n"
      << "  -t, --temperature <T>   Sampling temperature (default: 0.0, "
         "greedy)\n"
      << "  --system <PROMPT>       Custom system prompt\n"
      << "  --raw                   Disable chat template framing\n"
      << "  -v, --verbose           Print detailed timing and token metrics\n"
      << "  -h, --help              Print help\n";
}

}  // namespace

std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg) {
  PromptOptions opt;
  bool skip_next = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (skip_next) {
      skip_next = false;
      continue;
    }

    const std::string_view arg = args[i];
    if (arg == "-h" || arg == "--help") {
      return std::nullopt;
    }

    if (arg == "-m" || arg == "--model") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.model_path = args[i + 1];
      skip_next = true;
      continue;
    }

    if (arg == "-n" || arg == "--max-tokens") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      const std::string_view val = args[i + 1];
      std::size_t num = 0;
      const auto res =
          std::from_chars(val.data(), val.data() + val.size(), num);
      if (res.ec != std::errc{}) {
        if (error_msg != nullptr) {
          *error_msg = "Invalid integer for max-tokens: " + std::string(val);
        }
        return std::nullopt;
      }
      opt.max_tokens = num;
      skip_next = true;
      continue;
    }

    if (arg == "-t" || arg == "--temperature") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      try {
        opt.temperature = std::stof(std::string(args[i + 1]));
      } catch (...) {
        if (error_msg != nullptr) {
          *error_msg =
              "Invalid float for temperature: " + std::string(args[i + 1]);
        }
        return std::nullopt;
      }
      skip_next = true;
      continue;
    }

    if (arg == "--system") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for --system";
        }
        return std::nullopt;
      }
      opt.system_prompt = args[i + 1];
      skip_next = true;
      continue;
    }

    if (arg == "--raw") {
      opt.use_chat_template = false;
      continue;
    }

    if (arg == "-v" || arg == "--verbose") {
      opt.verbose = true;
      continue;
    }

    if (!arg.empty() && arg[0] != '-') {
      if (!opt.prompt_text.empty()) {
        opt.prompt_text += " ";
      }
      opt.prompt_text += arg;
      continue;
    }

    if (error_msg != nullptr) {
      *error_msg = "Unknown option: " + std::string(arg);
    }
    return std::nullopt;
  }

  return opt;
}

int RunPrompt(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParsePromptOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintPromptHelp("strix-server");
      return 2;
    }
    PrintPromptHelp("strix-server");
    return 0;
  }

  const auto& opt = *opt_res;
  if (opt.prompt_text.empty() && opt.model_path.empty()) {
    PrintPromptHelp("strix-server");
    return 0;
  }

  if (opt.model_path.empty()) {
    std::cout << "strix-server prompt: prompt received: \"" << opt.prompt_text
              << "\"\n"
              << "(Specify --model <PATH.gguf> to execute model inference)\n";
    return 0;
  }

  std::string err;
  const auto reader = strix::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    return 1;
  }

  const auto config_opt = reader->ExtractModelConfig(&err);
  if (!config_opt.has_value()) {
    std::cerr << "Error extracting model configuration: " << err << "\n";
    return 1;
  }
  const auto& config = *config_opt;

  const auto tokenizer =
      strix::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
  if (!tokenizer) {
    std::cerr << "Error extracting tokenizer from GGUF: " << err << "\n";
    return 1;
  }

  std::string rendered_prompt = opt.prompt_text;
  if (opt.use_chat_template) {
    std::vector<strix::tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {strix::tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {strix::tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto rendered =
        strix::tokenization::QwenChatTemplate::Render(messages);
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    }
  }

  const auto prompt_tokens = tokenizer->Encode(rendered_prompt);

  if (opt.verbose) {
    std::cout << "Model: " << config.architecture << " (" << config.num_layers
              << " layers, hidden=" << config.hidden_size
              << ", heads=" << config.num_attention_heads << ")\n"
              << "Tensors: " << reader->GetTensorCount() << "\n"
              << "Prompt tokens: " << prompt_tokens.size() << "\n"
              << "Max tokens: " << opt.max_tokens << "\n"
              << "Temperature: " << opt.temperature << "\n";
  }

  std::cout << rendered_prompt << "\n";
  return 0;
}

}  // namespace strix::server
