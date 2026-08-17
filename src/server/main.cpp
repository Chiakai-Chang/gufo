#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/diagnose.h"

namespace strix::server {
namespace {

constexpr std::string_view version_string = "strix-server 0.1.0";

void print_version() {
  std::cout << version_string << '\n';
}

void print_help(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS] [COMMAND]\n\n"
      << "Commands:\n"
      << "  serve     Start the OpenAI-compatible server\n"
      << "  chat      Maintain an interactive conversation\n"
      << "  prompt    Execute one request and exit\n"
      << "  diagnose  Run non-interactive system and hardware diagnostics\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n";
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    print_help("strix-server");
    return 2;
  }

  const std::string_view program_name = args[0];
  const auto options = args.subspan(1);

  if (options.empty()) {
    std::cerr << "Error: no command or option provided\n";
    print_help(program_name);
    return 2;
  }

  const std::string_view first_arg = options[0];
  if (first_arg == "--version" || first_arg == "-v") {
    print_version();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    print_help(program_name);
    return 0;
  }

  if (first_arg == "diagnose") {
    return strix::cli::RunDiagnose(options);
  }

  std::cerr << "Error: unknown command or option '" << first_arg << "'\n";
  print_help(program_name);
  return 2;
}

}  // namespace
}  // namespace strix::server

int main(int argc, char* argv[]) {
  return strix::server::run(
      std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
}
