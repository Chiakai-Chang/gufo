#include "src/server/bench_cli.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

void TestDefaultOptions() {
  const std::array<const char*, 0> args{};
  const auto options = strix::server::ParseBenchOptions(args);
  Expect(options.has_value(), "default options parse");
  Expect(options->n_depths == std::vector<std::size_t>{0},
         "default depth is zero");
  Expect(options->repetitions == 1, "default is one repetition");
}

void TestDepthOptions() {
  const std::array<const char*, 8> args = {
      "--n-prompt",    "2048",      "--n-gen",
      "128",           "--n-depth", "4096,8192,12288,16384",
      "--repetitions", "1"};
  const auto options = strix::server::ParseBenchOptions(args);
  Expect(options.has_value(), "depth options parse");
  Expect(options->n_prompts == std::vector<std::size_t>{2048},
         "prompt length parsed");
  Expect(options->n_gens == std::vector<std::size_t>{128},
         "generation length parsed");
  Expect(
      options->n_depths == std::vector<std::size_t>{4096, 8192, 12288, 16384},
      "depth list parsed");
  Expect(options->repetitions == 1, "repetition count parsed");
}

void TestHybridMtpOptions() {
  const std::array<const char*, 8> args = {
      "--speculative",  "mtp-npu", "--mtp-model", "mtp.gguf",
      "--draft-tokens", "2",       "--n-gen",     "128"};
  const auto options = strix::server::ParseBenchOptions(args);
  Expect(options.has_value(), "hybrid MTP options parse");
  Expect(options->speculative_backend == "mtp-npu", "hybrid MTP mode parsed");
  Expect(options->mtp_model_path == "mtp.gguf", "MTP model path parsed");
  Expect(options->draft_tokens == 2, "draft token count parsed");
}

void TestInvalidDepth() {
  std::string error;
  const std::array<const char*, 2> args = {"--n-depth", "invalid"};
  Expect(!strix::server::ParseBenchOptions(args, &error).has_value(),
         "invalid depth rejected");
  Expect(!error.empty(), "invalid depth reports an error");
}

}  // namespace

int main() {
  TestDefaultOptions();
  TestDepthOptions();
  TestHybridMtpOptions();
  TestInvalidDepth();
  std::cout << "All benchmark CLI tests passed.\n";
  return 0;
}
