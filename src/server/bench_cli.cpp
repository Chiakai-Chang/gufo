#include "src/server/bench_cli.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_executor.hpp"
#endif

namespace strix::server {
namespace {

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

void PrintBenchHelp(std::string_view program_name) {
  std::cout << "Usage: " << program_name << " bench [options]\n\n"
            << "Benchmark prompt processing (pp) and token generation (tg) "
               "throughput.\n\n"
            << "Options:\n"
            << "  -h, --help                  Print help\n"
            << "  -m, --model <PATH>          Path to GGUF model file "
               "(default: models/Qwen3.5-4B-BF16.gguf)\n"
            << "  -p, --n-prompt <n,n,...>    Prompt token lengths to "
               "benchmark (default: 64,128,512)\n"
            << "  -n, --n-gen <n,n,...>       Number of text generation tokens "
               "(default: 128)\n"
            << "  -r, --repetitions <N>       Number of repetitions per test "
               "(default: 3)\n"
            << "  -ngl, --n-gpu-layers <N>    Number of layers offloaded to "
               "GPU (default: 99)\n"
            << "  -v, --verbose               Verbose progress output\n";
}

std::vector<std::size_t> ParseCommaSeparatedSizes(std::string_view str) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start < str.size()) {
    auto end = str.find(',', start);
    if (end == std::string_view::npos) {
      end = str.size();
    }
    const auto part = str.substr(start, end - start);
    if (!part.empty()) {
      std::size_t val = 0;
      const auto [ptr, ec] =
          std::from_chars(part.data(), part.data() + part.size(), val);
      if (ec == std::errc{} && val > 0) {
        result.push_back(val);
      }
    }
    start = end + 1;
  }
  return result;
}

struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {.mean = 0.0, .stddev = 0.0};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

}  // namespace

std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg) {
  BenchOptions opt;
  bool explicit_p = false;
  bool explicit_n = false;
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

    if (arg == "-p" || arg == "--n-prompt") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.n_prompts = ParseCommaSeparatedSizes(args[i + 1]);
      explicit_p = true;
      skip_next = true;
      continue;
    }

    if (arg == "-n" || arg == "--n-gen") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.n_gens = ParseCommaSeparatedSizes(args[i + 1]);
      explicit_n = true;
      skip_next = true;
      continue;
    }

    if (arg == "-r" || arg == "--repetitions") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      std::size_t r = 0;
      const std::string_view val = args[i + 1];
      std::from_chars(val.data(), val.data() + val.size(), r);
      if (r > 0) {
        opt.repetitions = r;
      }
      skip_next = true;
      continue;
    }

    if (arg == "-ngl" || arg == "--n-gpu-layers") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      int ngl = 0;
      const std::string_view val = args[i + 1];
      std::from_chars(val.data(), val.data() + val.size(), ngl);
      opt.n_gpu_layers = ngl;
      skip_next = true;
      continue;
    }

    if (arg == "-v" || arg == "--verbose") {
      opt.verbose = true;
      continue;
    }

    if (!arg.empty() && arg[0] != '-') {
      opt.model_path = std::string(arg);
      continue;
    }
  }

  if (explicit_p && !explicit_n) {
    opt.n_gens.clear();
  } else if (!explicit_p && explicit_n) {
    opt.n_prompts.clear();
  }

  return opt;
}

int RunBench(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseBenchOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintBenchHelp("strix-server");
      return 2;
    }
    PrintBenchHelp("strix-server");
    return 0;
  }

  const auto& opt = *opt_res;

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  const auto reader = strix::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

#if defined(ENGINE_ENABLE_HIP)
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: No HIP GPU devices available for benchmarking.\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const double total_vram_mib =
      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);

  std::cout << "ggml_cuda_init: found " << device_count
            << " ROCm devices (Total VRAM: "
            << static_cast<std::size_t>(total_vram_mib) << " MiB):\n"
            << "  Device 0: " << prop.name << ", " << prop.gcnArchName
            << ", Wave Size: " << prop.warpSize
            << ", VRAM: " << static_cast<std::size_t>(total_vram_mib)
            << " MiB\n";

  auto gpu_exec = hip::QwenGpuExecutor::CreateFromGguf(*reader, &err);
  if (!gpu_exec) {
    std::cerr << "Error creating Qwen GPU executor: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto& config = gpu_exec->GetConfig();
  const std::string model_name = config.model_name + " BF16";
  const double model_size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  std::uint64_t parameter_count = 0;
  for (const auto& tensor : reader->GetTensors()) {
    parameter_count += tensor.ElementCount();
  }
  const double model_params_b =
      static_cast<double>(parameter_count) / 1'000'000'000.0;

  std::cout << "| " << std::left << std::setw(30) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::right << std::setw(10) << "params"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(3) << "ngl"
            << " | " << std::right << std::setw(15) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n";
  std::cout << "| " << std::string(30, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(10, '-')
            << " | " << std::string(3, '-') << " | " << std::string(15, '-')
            << " | " << std::string(21, '-') << " |\n";

  // Warmup run
  {
    std::vector<tokenization::TokenId> warmup_tokens(32, 100);
    gpu_exec->Reset();
    (void)gpu_exec->ForwardPromptBatch(warmup_tokens);
    HIP_CHECK(hipDeviceSynchronize());
  }

  // 1. Benchmark Prompt Processing (pp<N>)
  for (const std::size_t p_len : opt.n_prompts) {
    std::vector<tokenization::TokenId> prompt_tokens(p_len);
    for (std::size_t i = 0; i < p_len; ++i) {
      prompt_tokens[i] = static_cast<tokenization::TokenId>((i % 1000) + 100);
    }

    std::vector<double> runs;
    runs.reserve(opt.repetitions);

    for (std::size_t r = 0; r < opt.repetitions; ++r) {
      gpu_exec->Reset();
      HIP_CHECK(hipDeviceSynchronize());

      const auto t0 = std::chrono::high_resolution_clock::now();
      (void)gpu_exec->ForwardPromptBatch(prompt_tokens);
      HIP_CHECK(hipDeviceSynchronize());
      const auto t1 = std::chrono::high_resolution_clock::now();

      const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
      if (elapsed_sec > 0.0) {
        runs.push_back(static_cast<double>(p_len) / elapsed_sec);
      }
    }

    const auto stats = ComputeStats(runs);
    const std::string test_name = "pp" + std::to_string(p_len);
    std::ostringstream ss_ts;
    ss_ts << std::fixed << std::setprecision(2) << stats.mean << " ± "
          << stats.stddev;

    std::ostringstream ss_size, ss_params;
    ss_size << std::fixed << std::setprecision(2) << model_size_gib << " GiB";
    ss_params << std::fixed << std::setprecision(2) << model_params_b << " B";

    std::cout << "| " << std::left << std::setw(30) << model_name << " | "
              << std::right << std::setw(10) << ss_size.str() << " | "
              << std::right << std::setw(10) << ss_params.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(3) << opt.n_gpu_layers
              << " | " << std::right << std::setw(15) << test_name << " | "
              << std::right << std::setw(21) << ss_ts.str() << " |\n"
              << std::flush;
  }

  // 2. Benchmark Text Generation (tg<N>)
  for (const std::size_t g_len : opt.n_gens) {
    std::vector<double> runs;
    runs.reserve(opt.repetitions);

    for (std::size_t r = 0; r < opt.repetitions; ++r) {
      gpu_exec->Reset();
      HIP_CHECK(hipDeviceSynchronize());

      tokenization::TokenId next_tok = gpu_exec->ForwardToken(100, 0, false);
      HIP_CHECK(hipDeviceSynchronize());

      const auto t0 = std::chrono::high_resolution_clock::now();
      for (std::size_t step = 1; step <= g_len; ++step) {
        next_tok = gpu_exec->ForwardToken(
            next_tok, static_cast<std::uint32_t>(step), step == g_len);
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto t1 = std::chrono::high_resolution_clock::now();

      const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
      if (elapsed_sec > 0.0) {
        runs.push_back(static_cast<double>(g_len) / elapsed_sec);
      }
    }

    const auto stats = ComputeStats(runs);
    const std::string test_name = "tg" + std::to_string(g_len);
    std::ostringstream ss_ts;
    ss_ts << std::fixed << std::setprecision(2) << stats.mean << " ± "
          << stats.stddev;

    std::ostringstream ss_size, ss_params;
    ss_size << std::fixed << std::setprecision(2) << model_size_gib << " GiB";
    ss_params << std::fixed << std::setprecision(2) << model_params_b << " B";

    std::cout << "| " << std::left << std::setw(30) << model_name << " | "
              << std::right << std::setw(10) << ss_size.str() << " | "
              << std::right << std::setw(10) << ss_params.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(3) << opt.n_gpu_layers
              << " | " << std::right << std::setw(15) << test_name << " | "
              << std::right << std::setw(21) << ss_ts.str() << " |\n"
              << std::flush;
  }

  std::cout << "\n";
  return 0;
#else
  std::cerr << "Error: Strix GPU benchmark requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(model_load_start, false);
  return 1;
#endif
}

}  // namespace strix::server
