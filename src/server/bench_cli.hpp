#ifndef STRIX_SERVER_BENCH_CLI_HPP_
#define STRIX_SERVER_BENCH_CLI_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix::server {

struct BenchOptions {
  std::string model_path{"models/Qwen3.5-4B-BF16.gguf"};
  std::vector<std::size_t> n_prompts{64, 128, 512};
  std::vector<std::size_t> n_gens{128};
  std::size_t repetitions{3};
  int n_gpu_layers{99};
  bool verbose{false};
};

std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);

int RunBench(std::span<const char* const> args);

}  // namespace strix::server

#endif  // STRIX_SERVER_BENCH_CLI_HPP_
