#pragma once

#include <span>
#include <string_view>

namespace gufo::cli {

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand = {});
int RunServe(std::span<const char* const> args);

}  // namespace gufo::cli
