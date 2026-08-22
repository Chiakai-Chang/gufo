#ifndef STRIX_SERVER_VIDEO_CLI_HPP_
#define STRIX_SERVER_VIDEO_CLI_HPP_

#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "src/models/minimax_h3/generation.hpp"

namespace strix::server {

struct VideoCliOptions {
  minimax_h3::GenerationRequest request;
  std::filesystem::path parameters_path;
  std::filesystem::path telemetry_path;
  bool profile{false};
};

[[nodiscard]] std::filesystem::path DefaultH3SourceManifest();
[[nodiscard]] std::optional<VideoCliOptions> ParseVideoOptions(
    std::span<const char* const> args, std::string* error = nullptr);
[[nodiscard]] int RunVideo(std::span<const char* const> args);

}  // namespace strix::server

#endif  // STRIX_SERVER_VIDEO_CLI_HPP_
