#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/minimax_h3/generation.hpp"
#include "src/models/minimax_h3/json.hpp"

namespace {

namespace h3 = strix::minimax_h3;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL minimax_h3_generation_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

bool HasPartial(const std::filesystem::path& target) {
  const std::filesystem::path parent = target.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : target.parent_path();
  const std::string prefix = target.filename().string() + ".strix-partial-";
  for (const auto& entry : std::filesystem::directory_iterator(parent)) {
    if (entry.path().filename().string().starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

h3::VideoFrames SyntheticFrames() {
  h3::VideoFrames frames;
  frames.width = 2;
  frames.height = 2;
  frames.frame_indices = {0, 11, 21};
  frames.rgb.resize(3U * 2U * 2U * 3U);
  for (std::size_t index = 0; index < frames.rgb.size(); ++index) {
    frames.rgb[index] =
        static_cast<float>(index % 13U) / static_cast<float>(12U);
  }
  return frames;
}

void TestPresetsAndReports() {
  std::string error;
  const auto exact = h3::ResolveGenerationPreset("exact", &error);
  const auto fast = h3::ResolveGenerationPreset("fast", &error);
  const auto aggressive = h3::ResolveGenerationPreset("aggressive", &error);
  const auto dev = h3::ResolveGenerationPreset("dev", &error);
  Check(exact && fast && aggressive && dev, "all frozen presets resolve");
  Check(exact->evaluations == 50 && exact->active_blocks == 50 &&
            exact->reuse_interval == 1,
        "exact preset contract");
  Check(fast->internal_width == 384 && fast->active_blocks == 45 &&
            fast->reuse_interval == 2,
        "fast preset contract");
  Check(aggressive->internal_width == 320 && aggressive->active_blocks == 40 &&
            aggressive->reuse_interval == 3,
        "aggressive preset contract");
  Check(!dev->mux && !dev->decode_audio &&
            dev->selected_frames == std::vector<int>({0, 11, 21}),
        "development preset contract");
  Check(h3::ValidateGenerationParameters(*dev, &error),
        "development preset is end-to-end decodable");
  auto too_short = *dev;
  too_short.frames = 5;
  too_short.selected_frames = {4};
  Check(!h3::ValidateGenerationParameters(too_short, &error) &&
            error.find("at least 22 aligned frames") != std::string::npos,
        "generation rejects geometry that the released VisualVAE cannot "
        "decode");
  const std::string report = h3::GenerationParametersJson(
      *fast, 42,
      "df0ff96bcdb3a350f115cc39daaf0d7523814258890aa90c06fa784e3ad06169");
  Check(report.find("strix.minimax-h3-text-generation.v1") != std::string::npos,
        "parameter schema is versioned");
  Check(report.find("\"token_reduction\": false") != std::string::npos,
        "token reduction is frozen off");
  Check(report.find("\"attention_kernel\": \"row_parallel\"") !=
            std::string::npos,
        "parameter report declares the attention kernel");
  Check(report.find("42ed227ee7df40d41602854ae760620d6eb651fe") !=
            std::string::npos,
        "parameter report pins model revision");
  Check(report.find("8974cc055ea9c02fcd14cc27dfda3e1027c05153") !=
            std::string::npos,
        "parameter report pins implementation reference");
  Check(report.find("/var/") == std::string::npos,
        "parameter report contains no private model path");
  Check(h3::json::Parse(report).IsObject(), "parameter report is valid JSON");

  h3::GenerationTelemetry telemetry;
  telemetry.prompt_tokens = 6;
  telemetry.inventory.json_bytes_read = 10;
  telemetry.inventory.safetensors_header_bytes_read = 20;
  telemetry.prompt_encoder.layers.push_back({.layer = 0,
                                             .weight_bytes = 30,
                                             .load_ms = 1.0,
                                             .prefetch_wait_ms = 2.0,
                                             .submit_ms = 3.0,
                                             .gpu_ms = 4.0,
                                             .dispatches = 17});
  telemetry.denoiser.adaln_precompute_ms = 5.0;
  telemetry.video_vae.tile_ms = {6.0};
  telemetry.audio_vae.stage_ms = {7.0};
  telemetry.first_preview_ms = 8.0;
  telemetry.denoiser.forward_ms = {12.5, 13.25};
  telemetry.denoiser.dispatches = 123;
  const std::string telemetry_json = h3::GenerationTelemetryJson(telemetry);
  Check(h3::json::Parse(telemetry_json).IsObject(),
        "telemetry report is valid JSON");
  Check(telemetry_json.find("\"denoiser_evaluations\": 2") != std::string::npos,
        "telemetry records fresh denoiser evaluations");
  Check(telemetry_json.find("\"prompt_weight_bytes\": 30") != std::string::npos,
        "telemetry records prompt weight traffic");
  Check(telemetry_json.find("\"prompt_layers\": [{\"layer\": 0") !=
            std::string::npos,
        "telemetry records per-layer prompt timings");
  Check(telemetry_json.find("\"denoiser_adaln_precompute_ms\": 5.000") !=
            std::string::npos,
        "telemetry records AdaLN precompute");
  Check(
      telemetry_json.find("\"denoiser_attention_kernel\": \"row_parallel\"") !=
          std::string::npos,
      "telemetry identifies the active attention kernel");
  Check(telemetry_json.find("\"first_preview_ms\": 8.000") != std::string::npos,
        "telemetry records preview latency");
}

void TestAtomicFrames() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("strix-h3-generation-test-" + std::to_string(getpid()));
  const std::filesystem::path target = root / "frames";
  std::error_code ignored;
  std::filesystem::remove_all(root, ignored);
  std::filesystem::create_directories(root);
  const h3::VideoFrames frames = SyntheticFrames();
  std::string error;

  h3::CancellationToken cancelled;
  cancelled.Cancel();
  Check(!h3::WriteDiagnosticFrames(target, frames, &cancelled, &error),
        "pre-cancelled frame publication fails");
  Check(!std::filesystem::exists(target) && !HasPartial(target),
        "cancelled frame publication leaves no output");

  Check(h3::WriteDiagnosticFrames(target, frames, nullptr, &error),
        "frame directory publishes");
  Check(std::filesystem::is_directory(target) && !HasPartial(target),
        "frame directory is complete and has no staging residue");
  for (const int frame : frames.frame_indices) {
    std::ostringstream name;
    name << "frame-" << std::setw(4) << std::setfill('0') << frame << ".ppm";
    std::ifstream input(target / name.str(), std::ios::binary);
    const std::string prefix(2, '\0');
    std::string magic = prefix;
    input.read(magic.data(), 2);
    Check(input.good() && magic == "P6", "PPM frame header");
  }

  Check(!h3::WriteDiagnosticFrames(target, frames, nullptr, &error),
        "existing frame directory is not replaced");
  Check(std::filesystem::is_directory(target) && !HasPartial(target),
        "existing complete output is preserved");
  std::filesystem::remove_all(root, ignored);
}

void TestAtomicLatents() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("strix-h3-latent-test-" + std::to_string(getpid()));
  const std::filesystem::path target = root / "latents";
  std::error_code ignored;
  std::filesystem::remove_all(root, ignored);
  std::filesystem::create_directories(root);
  std::string error;
  const auto geometry = h3::ResolveGenerationGeometry(32, 32, 22, &error);
  Check(geometry.has_value(), "latent test geometry resolves");
  if (!geometry.has_value()) {
    return;
  }
  std::vector<float> video(
      h3::kH3VideoLatentChannels *
          static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
          static_cast<std::size_t>(geometry->latent_height) *
          static_cast<std::size_t>(geometry->latent_width),
      0.25F);
  std::vector<float> audio(
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
          static_cast<std::size_t>(geometry->temporal.audio_latent_frames),
      -0.5F);
  h3::CancellationToken cancelled;
  cancelled.Cancel();
  Check(!h3::WriteDiagnosticLatents(target, *geometry, video, audio, &cancelled,
                                    &error),
        "pre-cancelled latent publication fails");
  Check(!std::filesystem::exists(target) && !HasPartial(target),
        "cancelled latent publication leaves no output");
  Check(h3::WriteDiagnosticLatents(target, *geometry, video, audio, nullptr,
                                   &error),
        "latent directory publishes");
  Check(std::filesystem::file_size(target / "video_final.f32") ==
            video.size() * sizeof(float),
        "video latent byte count");
  Check(std::filesystem::file_size(target / "audio_final.f32") ==
            audio.size() * sizeof(float),
        "audio latent byte count");
  std::ifstream manifest_stream(target / "manifest.json");
  const std::string manifest((std::istreambuf_iterator<char>(manifest_stream)),
                             std::istreambuf_iterator<char>());
  Check(h3::json::Parse(manifest).IsObject(), "latent manifest is valid JSON");
  Check(manifest.find("strix.minimax-h3-final-latents.v1") != std::string::npos,
        "latent manifest is versioned");
  Check(!h3::WriteDiagnosticLatents(target, *geometry, video, audio, nullptr,
                                    &error),
        "existing latent directory is not replaced");
  std::filesystem::remove_all(root, ignored);
}

}  // namespace

int main() {
  TestPresetsAndReports();
  TestAtomicFrames();
  TestAtomicLatents();
#if !defined(ENGINE_ENABLE_HIP)
  h3::GenerationTelemetry telemetry;
  std::string error;
  Check(
      !h3::GenerateTextVideo({}, nullptr, nullptr, nullptr, &telemetry, &error),
      "host-only build rejects inference");
  Check(error.find("no CPU inference fallback") != std::string::npos,
        "host-only rejection names the no-CPU contract");
#endif
  std::cout << "MiniMax H3 generation host contracts passed.\n";
  return 0;
}
