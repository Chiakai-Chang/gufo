#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace {

namespace h3 = strix::minimax_h3;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::vector<std::uint16_t> ReadBf16(const std::filesystem::path& path,
                                    std::size_t elements) {
  std::vector<std::uint16_t> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()),
                  static_cast<std::streamsize>(values.size() *
                                               sizeof(std::uint16_t))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return values;
}

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t elements) {
  std::vector<float> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(
          reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(float))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return values;
}

float ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool Finite(std::span<const float> values) {
  return std::ranges::all_of(values,
                             [](float value) { return std::isfinite(value); });
}

template<typename Left, typename Right, typename ConvertLeft,
         typename ConvertRight>
void CheckBoundary(std::string_view name, std::span<const Left> got,
                   std::span<const Right> expected, double bound,
                   ConvertLeft convert_left, ConvertRight convert_right) {
  CHECK(got.size() == expected.size());
  double error_square = 0.0;
  double value_square = 0.0;
  double maximum_error = 0.0;
  double maximum_value = 0.0;
  std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0; index < std::min(got.size(), expected.size());
       ++index) {
    const double actual = convert_left(got[index]);
    const double want = convert_right(expected[index]);
    const double delta = actual - want;
    error_square += delta * delta;
    value_square += want * want;
    maximum_error = std::max(maximum_error, std::abs(delta));
    maximum_value = std::max(maximum_value, std::abs(want));
    if (first_mismatch == std::numeric_limits<std::size_t>::max() &&
        actual != want) {
      first_mismatch = index;
    }
  }
  const double relative_l2 =
      std::sqrt(error_square / std::max(value_square, 1.0e-24));
  const double relative_max = maximum_error / std::max(maximum_value, 1.0e-12);
  std::cout << name << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max
            << " first_mismatch=" << first_mismatch << '\n';
  CHECK(relative_l2 < bound);
  CHECK(relative_max < bound);
}

void CheckBf16Boundary(std::string_view name,
                       std::span<const std::uint16_t> got,
                       std::span<const std::uint16_t> expected, double bound) {
  CheckBoundary(name, got, expected, bound, ToFloat, ToFloat);
}

void CheckF32Boundary(std::string_view name, std::span<const float> got,
                      std::span<const float> expected, double bound) {
  CheckBoundary(
      name, got, expected, bound, [](float value) { return value; },
      [](float value) { return value; });
}

struct CancelAfterStep {
  h3::CancellationToken* token{nullptr};
  int step{0};
};

void CancelProgress(int completed, int, void* opaque) {
  auto* state = static_cast<CancelAfterStep*>(opaque);
  if (state != nullptr && state->token != nullptr && completed == state->step) {
    state->token->Cancel();
  }
}

int CheckReal() {
  const char* root = std::getenv("STRIX_H3_MODEL_ROOT");
  const char* conditioning_path = std::getenv("STRIX_H3_LAYER50_CONDITIONING");
  if (root == nullptr || conditioning_path == nullptr) {
    std::cout << "SKIP: STRIX_H3_MODEL_ROOT or "
                 "STRIX_H3_LAYER50_CONDITIONING unset\n";
    return 77;
  }
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, STRIX_H3_SOURCE_MANIFEST, &error);
  CHECK(inventory.has_value());
  if (!inventory) {
    std::cerr << error << '\n';
    return 1;
  }
  constexpr std::size_t kTextRows = 6;
  auto conditioning =
      ReadBf16(conditioning_path, kTextRows * h3::kH3TextConditionWidth);
  CHECK(conditioning.size() == kTextRows * h3::kH3TextConditionWidth);
  auto geometry = h3::ResolveGenerationGeometry(32, 32, 5, &error);
  CHECK(geometry.has_value());
  if (!geometry || conditioning.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::CancellationToken cancelled_creation;
  cancelled_creation.Cancel();
  h3::DenoiserTelemetry rejected_telemetry;
  auto rejected = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 2, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, &cancelled_creation,
      &rejected_telemetry, &error);
  CHECK(rejected == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 2, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 24);
  CHECK(session->block0_modulation().size() ==
        3 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize);
  CHECK(session->refined_text().size() == kTextRows * h3::kDitHiddenSize);
  CHECK(creation.core_weight_bytes > 30ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.peak_live_bytes < 120ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.scratch_address != 0);
  CHECK(creation.swap_bytes == 0);

  const std::size_t video_elements =
      h3::kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  const std::size_t audio_elements =
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
      static_cast<std::size_t>(geometry->temporal.audio_latent_frames);
  std::vector<float> video(video_elements);
  std::vector<float> audio(audio_elements);
  for (std::size_t index = 0; index < video.size(); ++index) {
    video[index] = std::sin(static_cast<float>(index) * 0.031F) * 0.5F;
  }
  for (std::size_t index = 0; index < audio.size(); ++index) {
    audio[index] = std::cos(static_cast<float>(index) * 0.019F) * 0.5F;
  }
  h3::DenoiserVelocity first;
  h3::DenoiserVelocity repeated;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video, audio, nullptr, &first, &telemetry, &error));
  CHECK(Finite(first.video));
  CHECK(Finite(first.audio));
  CHECK(session->Forward(0, video, audio, nullptr, &repeated, &telemetry,
                         &error));
  CHECK(first.video == repeated.video);
  CHECK(first.audio == repeated.audio);

  h3::CancellationToken cancelled;
  cancelled.Cancel();
  h3::DenoiserVelocity ignored;
  CHECK(!session->Forward(0, video, audio, &cancelled, &ignored, &telemetry,
                          &error));
  CHECK(error.find("cancelled") != std::string::npos);
  error.clear();
  h3::CancellationToken step_cancellation;
  CancelAfterStep cancel_after = {&step_cancellation, 1};
  std::vector<float> cancelled_video = video;
  std::vector<float> cancelled_audio = audio;
  CHECK(!session->DenoiseWithProgress(cancelled_video, cancelled_audio,
                                      &step_cancellation, CancelProgress,
                                      &cancel_after, &telemetry, &error));
  CHECK(error.find("after step 1") != std::string::npos);
  error.clear();
  std::vector<float> denoised_video = video;
  std::vector<float> denoised_audio = audio;
  CHECK(session->Denoise(denoised_video, denoised_audio, nullptr, &telemetry,
                         &error));
  CHECK(Finite(denoised_video));
  CHECK(Finite(denoised_audio));
  CHECK(telemetry.forward_ms.size() == 5);
  CHECK(telemetry.dispatches == 5 * (2 + h3::kDitBlocks * 11 + 6) + 6);
  CHECK(telemetry.swap_bytes == 0);
  session.reset();
  h3::DenoiserTelemetry reload_creation;
  auto reloaded = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 2, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, nullptr, &reload_creation, &error);
  CHECK(reloaded != nullptr);
  if (reloaded == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DenoiserVelocity reloaded_velocity;
  h3::DenoiserTelemetry reload_forward;
  CHECK(reloaded->Forward(0, video, audio, nullptr, &reloaded_velocity,
                          &reload_forward, &error));
  CHECK(reloaded_velocity.video == first.video);
  CHECK(reloaded_velocity.audio == first.audio);
  CHECK(reload_forward.swap_bytes == 0);
  std::cout << "H3 denoiser smoke rows=" << reloaded->layout().rows()
            << " refiner_ms=" << telemetry.text_refiner_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms
            << " core_load_ms=" << telemetry.core_load_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " forward_ms=" << telemetry.forward_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckOracle() {
  const char* root = std::getenv("STRIX_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("STRIX_H3_DENOISER_GOLDEN");
  if (root == nullptr || golden_value == nullptr) {
    std::cout
        << "SKIP: STRIX_H3_MODEL_ROOT or STRIX_H3_DENOISER_GOLDEN unset\n";
    return 77;
  }
  const std::filesystem::path golden(golden_value);
  const std::array<std::pair<std::string_view, std::string_view>, 10> hashes = {
      {
          {"conditioning.bf16",
           "8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692"},
          {"refined_text.bf16",
           "be0d360ced628aa8778c95574a18f4593f938bdd0b72b292e1a8d452641e7aba"},
          {"block0_modulation.bf16",
           "8aeaefd51d570ed2a4e0527132a8eca5591fe6a392c3410f551ed0a0463c41cc"},
          {"row_map_step0.u32",
           "816629a33e18daef4f94301aba7b54ba832c50db86fd8ae16a66f86f8d03f5d0"},
          {"video_initial.f32",
           "0e81b17167a0f768cbfebd6d6dca241c6b876a7b93b2f4d601efb945ae8ac056"},
          {"audio_initial.f32",
           "6447c3a4ed7108e3994d753d08ec78180d9e41f80f2e16a8d5fedda050af55fc"},
          {"video_velocity_step0.f32",
           "06183875c33ab88046827226feac1a843baf4b0686d978c9034f9416c0fac7f5"},
          {"audio_velocity_step0.f32",
           "b659c241a6abbbb06c0ec255ed5e3185f70e28b787102799c9eb202dd33d6955"},
          {"video_final.f32",
           "23411f438f351f96632884476def1d1f17513a6d2105b42ebf698a6d69d79d52"},
          {"audio_final.f32",
           "f742e12e3f55d91886a1b4218e6e5a3e64efcd65bce63de6a6d42976f3b2e04c"},
      }};
  for (const auto& [path, hash] : hashes) {
    CHECK(h3::Sha256File(golden / path) == hash);
  }
  constexpr std::size_t kTextRows = 6;
  constexpr std::size_t kVideoElements = 24 * 7 * 16 * 16;
  constexpr std::size_t kAudioElements = 32 * 2 * 37;
  constexpr std::size_t kBlockModulationElements =
      7 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize;
  auto conditioning = ReadBf16(golden / "conditioning.bf16",
                               kTextRows * h3::kH3TextConditionWidth);
  auto expected_refined =
      ReadBf16(golden / "refined_text.bf16", kTextRows * h3::kDitHiddenSize);
  auto expected_modulation =
      ReadBf16(golden / "block0_modulation.bf16", kBlockModulationElements);
  auto video_initial = ReadF32(golden / "video_initial.f32", kVideoElements);
  auto audio_initial = ReadF32(golden / "audio_initial.f32", kAudioElements);
  auto expected_video_velocity =
      ReadF32(golden / "video_velocity_step0.f32", kVideoElements);
  auto expected_audio_velocity =
      ReadF32(golden / "audio_velocity_step0.f32", kAudioElements);
  auto expected_video_final =
      ReadF32(golden / "video_final.f32", kVideoElements);
  auto expected_audio_final =
      ReadF32(golden / "audio_final.f32", kAudioElements);
  CHECK(!conditioning.empty());
  CHECK(!expected_refined.empty());
  CHECK(!expected_modulation.empty());
  CHECK(!video_initial.empty());
  CHECK(!audio_initial.empty());
  if (failures != 0) {
    return 1;
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, STRIX_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(256, 256, 22, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  if (!inventory || !geometry) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 4, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 528);
  CheckBf16Boundary("refined_text", session->refined_text(), expected_refined,
                    0.02);
  CheckBf16Boundary("block0_modulation", session->block0_modulation(),
                    expected_modulation, 0.02);

  h3::DenoiserVelocity velocity;
  h3::DenoiserVelocity repeated;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video_initial, audio_initial, nullptr, &velocity,
                         &telemetry, &error));
  if (velocity.video.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("video_velocity", velocity.video, expected_video_velocity,
                   0.04);
  CheckF32Boundary("audio_velocity", velocity.audio, expected_audio_velocity,
                   0.05);
  CHECK(session->Forward(0, video_initial, audio_initial, nullptr, &repeated,
                         &telemetry, &error));
  CHECK(repeated.video == velocity.video);
  CHECK(repeated.audio == velocity.audio);

  std::vector<float> video_final = video_initial;
  std::vector<float> audio_final = audio_initial;
  CHECK(
      session->Denoise(video_final, audio_final, nullptr, &telemetry, &error));
  if (video_final.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("video_final_4step", video_final, expected_video_final,
                   0.08);
  CheckF32Boundary("audio_final_4step", audio_final, expected_audio_final,
                   0.08);
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.forward_ms.size() == 6);
  CHECK(telemetry.swap_bytes == 0);
  std::cout << "H3 denoiser oracle rows=" << session->layout().rows()
            << " load_ms=" << telemetry.core_load_ms
            << " adaln_ms=" << telemetry.adaln_precompute_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " first_forward_ms=" << telemetry.forward_ms.front()
            << " denoise_ms=" << telemetry.sampler_ms << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckProductionShapeOracle() {
  const char* root = std::getenv("STRIX_H3_MODEL_ROOT");
  const char* golden_value =
      std::getenv("STRIX_H3_DENOISER_GOLDEN_512");
  if (root == nullptr || golden_value == nullptr) {
    std::cout << "SKIP: STRIX_H3_MODEL_ROOT or "
                 "STRIX_H3_DENOISER_GOLDEN_512 unset\n";
    return 77;
  }
  const std::filesystem::path golden(golden_value);
  const std::array<std::pair<std::string_view, std::string_view>, 9> hashes = {
      {
          {"conditioning.bf16",
           "8015af1d2a551a0509bccc84372a30087f66721613f909c3f99e36ad7e269692"},
          {"refined_text.bf16",
           "be0d360ced628aa8778c95574a18f4593f938bdd0b72b292e1a8d452641e7aba"},
          {"block0_modulation.bf16",
           "f963df621ea2c2d5a447b474011b213885567b3de371e5a0636428c0946add14"},
          {"video_initial.f32",
           "6788a83c4b5aa596cde87c901156b6ad5982da2a6ce674d7067f16699924f293"},
          {"audio_initial.f32",
           "83489726502370f61a72a214e90c4bf909dca954bec260e83051b3e254798dbd"},
          {"video_velocity_step0.f32",
           "b0f403062d77ad29fd5f5961c2c9a47f4736d574b365d87b416f90f2736e2aa8"},
          {"audio_velocity_step0.f32",
           "43c5ab061c3b749882919a987e613f4cc23185f462d98069ee78b01d014304fc"},
          {"video_final.f32",
           "55d2460a8a6ef5bd30f2123145028979d6820063a1c9c569336b04d5f95c298e"},
          {"audio_final.f32",
           "20de4a1ca708fd32a2bd637fbd420a69faffbe56dd2ff048abf10b8d83709d56"},
      }};
  for (const auto& [path, hash] : hashes) {
    CHECK(h3::Sha256File(golden / path) == hash);
  }
  constexpr std::size_t kTextRows = 6;
  constexpr std::size_t kVideoElements = 24 * 7 * 32 * 32;
  constexpr std::size_t kAudioElements = 32 * 2 * 37;
  constexpr std::size_t kBlockModulationElements =
      3 * 3 * h3::kDitModulationSlots * h3::kDitHiddenSize;
  auto conditioning = ReadBf16(golden / "conditioning.bf16",
                               kTextRows * h3::kH3TextConditionWidth);
  auto expected_refined =
      ReadBf16(golden / "refined_text.bf16", kTextRows * h3::kDitHiddenSize);
  auto expected_modulation =
      ReadBf16(golden / "block0_modulation.bf16", kBlockModulationElements);
  auto video_initial = ReadF32(golden / "video_initial.f32", kVideoElements);
  auto audio_initial = ReadF32(golden / "audio_initial.f32", kAudioElements);
  auto expected_video_velocity =
      ReadF32(golden / "video_velocity_step0.f32", kVideoElements);
  auto expected_audio_velocity =
      ReadF32(golden / "audio_velocity_step0.f32", kAudioElements);
  auto expected_video_final =
      ReadF32(golden / "video_final.f32", kVideoElements);
  auto expected_audio_final =
      ReadF32(golden / "audio_final.f32", kAudioElements);
  CHECK(!conditioning.empty());
  CHECK(!expected_refined.empty());
  CHECK(!expected_modulation.empty());
  CHECK(!video_initial.empty());
  CHECK(!audio_initial.empty());
  if (failures != 0) {
    return 1;
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, STRIX_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(512, 512, 22, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  if (!inventory || !geometry) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry, .evaluations = 2, .text_rows = kTextRows},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->layout().rows() == 1872);
  CheckBf16Boundary("production_refined_text", session->refined_text(),
                    expected_refined, 0.02);
  CheckBf16Boundary("production_block0_modulation",
                    session->block0_modulation(), expected_modulation, 0.02);

  h3::DenoiserVelocity velocity;
  h3::DenoiserTelemetry telemetry;
  CHECK(session->Forward(0, video_initial, audio_initial, nullptr, &velocity,
                         &telemetry, &error));
  if (velocity.video.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("production_video_velocity", velocity.video,
                   expected_video_velocity, 0.04);
  CheckF32Boundary("production_audio_velocity", velocity.audio,
                   expected_audio_velocity, 0.05);

  std::vector<float> video_final = video_initial;
  std::vector<float> audio_final = audio_initial;
  CHECK(
      session->Denoise(video_final, audio_final, nullptr, &telemetry, &error));
  if (video_final.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CheckF32Boundary("production_video_final_2step", video_final,
                   expected_video_final, 0.08);
  CheckF32Boundary("production_audio_final_2step", audio_final,
                   expected_audio_final, 0.08);
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.forward_ms.size() == 3);
  CHECK(telemetry.swap_bytes == 0);
  std::cout << "H3 production-shape denoiser oracle rows="
            << session->layout().rows() << " load_ms="
            << telemetry.core_load_ms << " adaln_ms="
            << telemetry.adaln_precompute_ms << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " first_forward_ms=" << telemetry.forward_ms.front()
            << " denoise_ms=" << telemetry.sampler_ms << '\n';
  return failures == 0 ? 0 : 1;
}

int CheckReuseSmoke() {
  const char* root = std::getenv("STRIX_H3_MODEL_ROOT");
  const char* conditioning_path = std::getenv("STRIX_H3_LAYER50_CONDITIONING");
  if (root == nullptr || conditioning_path == nullptr) {
    std::cout << "SKIP: STRIX_H3_MODEL_ROOT or "
                 "STRIX_H3_LAYER50_CONDITIONING unset\n";
    return 77;
  }
  constexpr std::size_t kTextRows = 6;
  auto conditioning =
      ReadBf16(conditioning_path, kTextRows * h3::kH3TextConditionWidth);
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, STRIX_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(32, 32, 5, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  CHECK(!conditioning.empty());
  if (!inventory || !geometry || conditioning.empty()) {
    std::cerr << error << '\n';
    return 1;
  }

  h3::DenoiserTelemetry invalid_telemetry;
  auto invalid_blocks =
      h3::DenoiserSession::Create(*inventory,
                                  {.geometry = *geometry,
                                   .evaluations = 5,
                                   .text_rows = kTextRows,
                                   .active_blocks = 0,
                                   .reuse_interval = 3},
                                  {.layer50 = conditioning, .tags = {}},
                                  nullptr, &invalid_telemetry, &error);
  CHECK(invalid_blocks == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();
  auto invalid_reuse =
      h3::DenoiserSession::Create(*inventory,
                                  {.geometry = *geometry,
                                   .evaluations = 5,
                                   .text_rows = kTextRows,
                                   .active_blocks = 2,
                                   .reuse_interval = 33},
                                  {.layer50 = conditioning, .tags = {}},
                                  nullptr, &invalid_telemetry, &error);
  CHECK(invalid_reuse == nullptr);
  CHECK(error.find("invalid") != std::string::npos);
  error.clear();

  h3::DenoiserTelemetry creation;
  auto session = h3::DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = 5,
       .text_rows = kTextRows,
       .active_blocks = 2,
       .reuse_interval = 3},
      {.layer50 = conditioning, .tags = {}}, nullptr, &creation, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(creation.core_weight_bytes > 0);
  CHECK(creation.core_weight_bytes < 4ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.peak_live_bytes < 8ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.swap_bytes == 0);

  const std::size_t video_elements =
      h3::kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);
  const std::size_t audio_elements =
      h3::kH3AudioLatentChannels * h3::kH3AudioTracks *
      static_cast<std::size_t>(geometry->temporal.audio_latent_frames);
  std::vector<float> initial_video(video_elements);
  std::vector<float> initial_audio(audio_elements);
  for (std::size_t index = 0; index < initial_video.size(); ++index) {
    initial_video[index] = std::sin(static_cast<float>(index) * 0.031F) * 0.5F;
  }
  for (std::size_t index = 0; index < initial_audio.size(); ++index) {
    initial_audio[index] = std::cos(static_cast<float>(index) * 0.019F) * 0.5F;
  }
  std::vector<float> first_video = initial_video;
  std::vector<float> first_audio = initial_audio;
  h3::DenoiserTelemetry first;
  CHECK(session->Denoise(first_video, first_audio, nullptr, &first, &error));
  CHECK(Finite(first_video));
  CHECK(Finite(first_audio));
  CHECK(first.forward_ms.size() == 3);
  CHECK(first.dispatches == 100);
  CHECK(first.swap_bytes == 0);

  std::vector<float> repeated_video = initial_video;
  std::vector<float> repeated_audio = initial_audio;
  h3::DenoiserTelemetry repeated;
  CHECK(session->Denoise(repeated_video, repeated_audio, nullptr, &repeated,
                         &error));
  CHECK(repeated_video == first_video);
  CHECK(repeated_audio == first_audio);
  CHECK(repeated.forward_ms.size() == 6);
  CHECK(repeated.dispatches == 200);
  CHECK(repeated.swap_bytes == 0);
  std::cout << "H3 denoiser reuse smoke active_blocks=2 steps=5 "
               "reuse=3 fresh_evaluations=3 peak_gib="
            << static_cast<double>(repeated.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " sampler_ms=" << repeated.sampler_ms << '\n';
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view(argv[1]) != "--smoke" &&
                    std::string_view(argv[1]) != "--oracle" &&
                    std::string_view(argv[1]) != "--oracle-512" &&
                    std::string_view(argv[1]) != "--reuse-smoke")) {
    std::cerr << "usage: " << argv[0]
              << " --smoke|--oracle|--oracle-512|--reuse-smoke\n";
    return 2;
  }
  if (std::string_view(argv[1]) == "--oracle") {
    return CheckOracle();
  }
  if (std::string_view(argv[1]) == "--oracle-512") {
    return CheckProductionShapeOracle();
  }
  if (std::string_view(argv[1]) == "--reuse-smoke") {
    return CheckReuseSmoke();
  }
  return CheckReal();
}
