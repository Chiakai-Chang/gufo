#include <span>

#include "src/server/bench_cli.hpp"

int main(int argc, char* argv[]) {
  return strix::server::RunBench(std::span<const char* const>(
      argv + 1, static_cast<std::size_t>(argc - 1)));
}
