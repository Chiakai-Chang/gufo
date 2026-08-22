#include "src/server/video_cli.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "src/models/minimax_h3/sha256.hpp"

#ifndef STRIX_H3_SOURCE_MANIFEST
#define STRIX_H3_SOURCE_MANIFEST \
  "share/strix/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
#endif
#ifndef STRIX_H3_INSTALLED_SOURCE_MANIFEST
#define STRIX_H3_INSTALLED_SOURCE_MANIFEST STRIX_H3_SOURCE_MANIFEST
#endif

namespace strix::server {
namespace {

volatile std::sig_atomic_t g_cancel_requested = 0;  // NOLINT

std::uint64_t NextReportNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

void HandleInterrupt(int) {
  g_cancel_requested = 1;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

void PrintVideoHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " video [OPTIONS] <PROMPT>\n\n"
      << "Generate MiniMax H3 text-to-video on a Strix Halo gfx1151 GPU.\n"
      << "There is no CPU inference fallback and weights are never "
         "downloaded.\n\n"
      << "Required:\n"
      << "  -m, --model <DIR>          Operator-supplied MiniMax H3 directory\n"
      << "  -o, --output <MP4>         Atomic MP4 output (non-dev presets)\n\n"
      << "Quality presets:\n"
      << "  --preset exact             512, 50 sigma points / 49 evaluations, "
         "50 blocks, reuse 1\n"
      << "  --preset fast              384->512, 20 sigma points / 19 "
         "evaluations, 45 blocks, reuse 2\n"
      << "  --preset aggressive        320->512, 20 sigma points / 19 "
         "evaluations, 40 blocks, reuse 3\n"
      << "  --preset dev               256, 4 steps, selected frames, no "
         "mux\n\n"
      << "Controls:\n"
      << "  --seed <N>                 Deterministic seed (default: 42)\n"
      << "  --width/--height <N>       Internal denoiser canvas\n"
      << "  --output-width/--output-height <N>  Encoded canvas\n"
      << "  --frames <N>               VisualVAE-decodable count: 22+17n\n"
      << "  --steps <N>                Denoising evaluations\n"
      << "  --blocks <N>               Active DiT prefix blocks\n"
      << "  --reuse <N>                Whole-denoiser reuse interval\n"
      << "  --attention-kernel <NAME>  row-parallel (default) or scalar\n"
      << "  --selected-frames <LIST>   e.g. first,middle,last or 0,11,21\n"
      << "  --frames-dir <DIR>         Atomic PPM frame outputs\n"
      << "  --latents-dir <DIR>        Diagnostic final F32 latent outputs\n"
      << "  --no-audio                 Skip AudioVAE (requires --no-mux)\n"
      << "  --no-mux                   Skip MP4 composition\n"
      << "  --manifest <PATH>          Pinned source manifest override\n"
      << "  --parameters <PATH>        Versioned parameter report path\n"
      << "  --report <PATH>            Timing/memory report path\n"
      << "  --profile                  Print phase progress and final report\n"
      << "  -h, --help                 Print this help\n\n"
      << "First/last-frame and ordered-reference inputs are future work and "
         "are rejected.\n";
}

template<typename Integer>
bool ParseInteger(std::string_view text, Integer* output) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *output);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

std::optional<std::vector<int>> ParseSelectedFrames(std::string_view text,
                                                    int frames,
                                                    std::string* error) {
  std::vector<int> result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    const std::string_view item = text.substr(begin, end - begin);
    int frame = -1;
    if (item == "first") {
      frame = 0;
    } else if (item == "middle") {
      frame = frames / 2;
    } else if (item == "last") {
      frame = frames - 1;
    } else if (!ParseInteger(item, &frame)) {
      SetError(error,
               "invalid MiniMax H3 selected frame: " + std::string(item));
      return std::nullopt;
    }
    result.push_back(frame);
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create H3 report directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial = path.string() + ".strix-partial-" +
                                        std::to_string(getpid()) + "-" +
                                        std::to_string(NextReportNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output << contents;
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      SetError(error, "cannot write H3 report");
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    SetError(error, "cannot publish H3 report: " + filesystem_error.message());
    return false;
  }
  return true;
}

std::string PromptSha256(std::string_view prompt) {
  return minimax_h3::Sha256(std::span(
      reinterpret_cast<const unsigned char*>(prompt.data()), prompt.size()));
}

struct CliProgress {
  minimax_h3::CancellationToken* cancellation{nullptr};
  bool profile{false};
};

void PrintProgress(std::string_view phase, int completed, int total,
                   void* opaque) {
  auto* state = static_cast<CliProgress*>(opaque);
  if (state == nullptr) {
    return;
  }
  if (g_cancel_requested != 0 && state->cancellation != nullptr) {
    state->cancellation->Cancel();
  }
  if (state->profile) {
    std::cerr << "h3: " << phase << ' ' << completed << '/' << total << '\n';
  }
}

}  // namespace

std::filesystem::path DefaultH3SourceManifest() {
  std::filesystem::path build_path = STRIX_H3_SOURCE_MANIFEST;
  std::error_code error;
  if (std::filesystem::is_regular_file(build_path, error)) {
    return build_path;
  }
  return STRIX_H3_INSTALLED_SOURCE_MANIFEST;
}

std::optional<VideoCliOptions> ParseVideoOptions(
    std::span<const char* const> args, std::string* error) {
  std::string preset_name = "exact";
  for (std::size_t index = 0; index < args.size(); ++index) {
    if (std::string_view(args[index]) == "--preset") {
      if (index + 1 >= args.size()) {
        SetError(error, "missing argument for --preset");
        return std::nullopt;
      }
      preset_name = args[++index];
    }
  }
  auto parameters = minimax_h3::ResolveGenerationPreset(preset_name, error);
  if (!parameters.has_value()) {
    return std::nullopt;
  }
  VideoCliOptions options;
  options.request.parameters = std::move(*parameters);
  options.request.seed = 42;
  options.request.source_manifest = DefaultH3SourceManifest();
  bool selected_frames_set = false;
  std::string selected_frames;

  const auto take_value =
      [&](std::size_t* index,
          std::string_view option) -> std::optional<std::string_view> {
    if (*index + 1 >= args.size()) {
      SetError(error, "missing argument for " + std::string(option));
      return std::nullopt;
    }
    return std::string_view(args[++*index]);
  };
  for (std::size_t index = 0; index < args.size(); ++index) {
    const std::string_view arg = args[index];
    if (arg == "-h" || arg == "--help") {
      return std::nullopt;
    }
    if (arg == "--preset") {
      ++index;
      continue;
    }
    if (arg == "--first-frame" || arg == "--last-frame" ||
        arg == "--reference" || arg == "--ordered-reference") {
      SetError(error, std::string(arg) +
                          " is not implemented for the text-only H3 milestone");
      return std::nullopt;
    }
    if (arg == "-m" || arg == "--model") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.request.model_root = *value;
    } else if (arg == "-o" || arg == "--output") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.request.output_path = *value;
    } else if (arg == "--frames-dir") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.request.frames_directory = *value;
    } else if (arg == "--latents-dir") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.request.latents_directory = *value;
    } else if (arg == "--manifest") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.request.source_manifest = *value;
    } else if (arg == "--parameters") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.parameters_path = *value;
    } else if (arg == "--report") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      options.telemetry_path = *value;
    } else if (arg == "--selected-frames") {
      const auto value = take_value(&index, arg);
      if (!value) {
        return std::nullopt;
      }
      selected_frames = *value;
      selected_frames_set = true;
    } else if (arg == "--profile") {
      options.profile = true;
    } else if (arg == "--attention-kernel") {
      const auto value = take_value(&index, arg);
      if (!value || (*value != "row-parallel" && *value != "scalar")) {
        SetError(error, "--attention-kernel must be row-parallel or scalar");
        return std::nullopt;
      }
      options.request.parameters.row_parallel_attention =
          *value == "row-parallel";
    } else if (arg == "--no-audio") {
      options.request.parameters.decode_audio = false;
    } else if (arg == "--no-mux") {
      options.request.parameters.mux = false;
    } else if (arg == "--seed") {
      const auto value = take_value(&index, arg);
      if (!value || !ParseInteger(*value, &options.request.seed)) {
        SetError(error, "invalid integer for --seed");
        return std::nullopt;
      }
    } else if (arg == "--width" || arg == "--height" ||
               arg == "--output-width" || arg == "--output-height" ||
               arg == "--frames" || arg == "--steps" || arg == "--blocks" ||
               arg == "--reuse") {
      const auto value = take_value(&index, arg);
      int parsed = 0;
      if (!value || !ParseInteger(*value, &parsed)) {
        SetError(error, "invalid integer for " + std::string(arg));
        return std::nullopt;
      }
      if (arg == "--width") {
        options.request.parameters.internal_width = parsed;
      } else if (arg == "--height") {
        options.request.parameters.internal_height = parsed;
      } else if (arg == "--output-width") {
        options.request.parameters.output_width = parsed;
      } else if (arg == "--output-height") {
        options.request.parameters.output_height = parsed;
      } else if (arg == "--frames") {
        options.request.parameters.frames = parsed;
      } else if (arg == "--steps") {
        options.request.parameters.evaluations = parsed;
      } else if (arg == "--blocks") {
        options.request.parameters.active_blocks = parsed;
      } else {
        options.request.parameters.reuse_interval = parsed;
      }
    } else if (!arg.empty() && arg.front() == '-') {
      SetError(error, "unknown MiniMax H3 video option: " + std::string(arg));
      return std::nullopt;
    } else if (options.request.prompt.empty()) {
      options.request.prompt = arg;
    } else {
      options.request.prompt += " ";
      options.request.prompt += arg;
    }
  }
  if (selected_frames_set) {
    auto frames = ParseSelectedFrames(selected_frames,
                                      options.request.parameters.frames, error);
    if (!frames.has_value()) {
      return std::nullopt;
    }
    options.request.parameters.selected_frames = std::move(*frames);
  }
  if (options.request.model_root.empty() || options.request.prompt.empty()) {
    SetError(error, "MiniMax H3 video requires --model and a prompt");
    return std::nullopt;
  }
  if (!options.request.parameters.mux &&
      options.request.frames_directory.empty()) {
    SetError(error, "--no-mux requires --frames-dir");
    return std::nullopt;
  }
  if (!options.request.parameters.decode_audio &&
      options.request.parameters.mux) {
    SetError(error, "--no-audio requires --no-mux");
    return std::nullopt;
  }
  if (options.request.parameters.mux && options.request.output_path.empty()) {
    SetError(error, "the selected H3 preset requires --output");
    return std::nullopt;
  }
  if (!minimax_h3::ValidateGenerationParameters(options.request.parameters,
                                                error)) {
    return std::nullopt;
  }
  const std::filesystem::path report_base =
      options.request.parameters.mux ? options.request.output_path
                                     : options.request.frames_directory;
  if (options.parameters_path.empty()) {
    options.parameters_path = report_base.string() + ".parameters.json";
  }
  if (options.telemetry_path.empty()) {
    options.telemetry_path = report_base.string() + ".telemetry.json";
  }
  return options;
}

int RunVideo(std::span<const char* const> args) {
  const bool help =
      std::find_if(args.begin(), args.end(), [](const char* value) {
        const std::string_view arg = value;
        return arg == "-h" || arg == "--help";
      }) != args.end();
  std::string error;
  auto options = ParseVideoOptions(args, &error);
  if (!options.has_value()) {
    if (help && error.empty()) {
      PrintVideoHelp("strix-server");
      return 0;
    }
    std::cerr << "Error: " << error << '\n';
    PrintVideoHelp("strix-server");
    return 2;
  }
  if (!WriteAtomic(options->parameters_path,
                   minimax_h3::GenerationParametersJson(
                       options->request.parameters, options->request.seed,
                       PromptSha256(options->request.prompt)),
                   &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }

  g_cancel_requested = 0;
  struct sigaction action{};
  action.sa_handler = HandleInterrupt;
  sigemptyset(&action.sa_mask);
  struct sigaction previous{};
  (void)sigaction(SIGINT, &action, &previous);
  minimax_h3::CancellationToken cancellation;
  std::jthread interrupt_watcher([&cancellation](const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      if (g_cancel_requested != 0) {
        cancellation.Cancel();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  CliProgress progress{&cancellation, options->profile};
  minimax_h3::GenerationTelemetry telemetry;
  const bool generated = minimax_h3::GenerateTextVideo(
      options->request, &cancellation, PrintProgress, &progress, &telemetry,
      &error);
  interrupt_watcher.request_stop();
  interrupt_watcher.join();
  (void)sigaction(SIGINT, &previous, nullptr);
  if (!generated) {
    std::cerr << "Error: " << error << '\n';
    return cancellation.IsCancelled() ? 130 : 1;
  }
  const std::string report = minimax_h3::GenerationTelemetryJson(telemetry);
  if (!WriteAtomic(options->telemetry_path, report, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  if (options->profile) {
    std::cerr << report;
  }
  return 0;
}

}  // namespace strix::server
