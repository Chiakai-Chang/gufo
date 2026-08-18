#include "src/server/prompt_cli.hpp"

#include <charconv>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen_generator.hpp"
#include "src/tokenization/qwen_chat_template.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include "src/core/hip/qwen_gpu_executor.hpp"
#endif

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

void PrintChatHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " chat [OPTIONS]\n\n"
      << "Start an interactive conversation session in the terminal.\n\n"
      << "Options:\n"
      << "  -m, --model <PATH>      Path to GGUF model file\n"
      << "  -n, --max-tokens <N>    Maximum tokens per turn (default: 256)\n"
      << "  -t, --temperature <T>   Sampling temperature (default: 0.0, "
         "greedy)\n"
      << "  --system <PROMPT>       Custom system prompt\n"
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
    std::cout
        << "strix-server prompt: prompt received: \"" << opt.prompt_text
        << "\"\n"
        << "(Specify --model <PATH.gguf> to execute local model generation)\n";
    return 0;
  }

  std::string err;
  const auto reader = strix::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    return 1;
  }

  std::string rendered_prompt = opt.prompt_text;
  if (opt.use_chat_template) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto rendered = tokenization::QwenChatTemplate::Render(messages);
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    }
  }

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  if (hipGetDeviceCount(&dev_count) == hipSuccess && dev_count > 0) {
    auto gpu_exec = strix::hip::QwenGpuExecutor::CreateFromGguf(*reader, &err);
    if (gpu_exec) {
      const auto prompt_tokens =
          gpu_exec->GetTokenizer().Encode(rendered_prompt);

      if (opt.verbose) {
        const auto& config = gpu_exec->GetConfig();
        std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n"
                  << "Model: " << config.architecture << " (" << config.num_layers
                  << " layers, hidden=" << config.hidden_size
                  << ", heads=" << config.num_attention_heads << ")\n"
                  << "Prompt tokens: " << prompt_tokens.size() << "\n"
                  << "Max tokens: " << opt.max_tokens << "\n"
                  << "--- Generation Output ---\n";
      }

      models::GenerationOptions gen_opts;
      gen_opts.max_new_tokens = opt.max_tokens;
      gen_opts.temperature = opt.temperature;

      const auto start_time = std::chrono::steady_clock::now();
      std::size_t generated_count = 0;

      gpu_exec->Generate(
          prompt_tokens, gen_opts,
          [&](tokenization::TokenId, std::string_view piece) -> bool {
            std::cout << piece << std::flush;
            ++generated_count;
            return true;
          });

      std::cout << "\n";

      if (opt.verbose && generated_count > 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time);
        const double sec = static_cast<double>(elapsed.count()) / 1000.0;
        const double tok_per_sec =
            (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
        std::cout << "Generated " << generated_count << " tokens on GPU in "
                  << sec << "s (" << tok_per_sec << " tok/s)\n";
      }
      return 0;
    }
  }
#endif

  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    return 1;
  }

  const auto prompt_tokens = generator->GetTokenizer().Encode(rendered_prompt);

  if (opt.verbose) {
    const auto& config = generator->GetConfig();
    std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n"
              << "Model: " << config.architecture << " (" << config.num_layers
              << " layers, hidden=" << config.hidden_size
              << ", heads=" << config.num_attention_heads << ")\n"
              << "Prompt tokens: " << prompt_tokens.size() << "\n"
              << "Max tokens: " << opt.max_tokens << "\n"
              << "Temperature: " << opt.temperature << "\n"
              << "--- Generation Output ---\n";
  }

  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.temperature = opt.temperature;

  const auto start_time = std::chrono::steady_clock::now();
  std::size_t generated_count = 0;

  const auto tokens = generator->Generate(
      prompt_tokens, gen_opts,
      [&](tokenization::TokenId, std::string_view piece) -> bool {
        std::cout << piece << std::flush;
        ++generated_count;
        return true;
      });

  std::cout << "\n";

  if (opt.verbose && generated_count > 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    const double sec = static_cast<double>(elapsed.count()) / 1000.0;
    const double tok_per_sec =
        (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
    std::cout << "Generated " << generated_count << " tokens on CPU in " << sec
              << "s (" << tok_per_sec << " tok/s)\n";
  }

  return 0;
}

int RunChat(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParsePromptOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintChatHelp("strix-server");
      return 2;
    }
    PrintChatHelp("strix-server");
    return 0;
  }

  const auto& opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cout << "strix-server chat: interactive conversation mode\n"
              << "(Specify --model <PATH.gguf> to load model weights)\n";
    return 0;
  }

  std::string err;
  const auto reader = strix::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    return 1;
  }

  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    return 1;
  }

  std::cout << "=== Strix Halo Interactive Chat ("
            << generator->GetConfig().architecture << ") ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  std::string user_input;
  while (true) {
    std::cout << ">>> User: ";
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    const auto rendered_prompt =
        tokenization::QwenChatTemplate::Render(history);
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      continue;
    }

    const auto prompt_tokens =
        generator->GetTokenizer().Encode(*rendered_prompt);

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;

    models::GenerationOptions gen_opts;
    gen_opts.max_new_tokens = opt.max_tokens > 0 ? opt.max_tokens : 256;
    gen_opts.temperature = opt.temperature;

    const auto reply_tokens = generator->Generate(
        prompt_tokens, gen_opts,
        [&](tokenization::TokenId, std::string_view piece) -> bool {
          std::cout << piece << std::flush;
          assistant_reply += piece;
          return true;
        });

    std::cout << "\n\n";
    history.push_back(
        {tokenization::ChatRole::kAssistant, assistant_reply, "", ""});
  }

  return 0;
}

}  // namespace strix::server
