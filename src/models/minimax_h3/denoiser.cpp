#include "src/models/minimax_h3/denoiser.hpp"

#include <utility>

namespace strix::minimax_h3 {

#if !defined(ENGINE_ENABLE_HIP)

struct DenoiserSession::Impl {};

DenoiserSession::DenoiserSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenoiserSession::~DenoiserSession() = default;
DenoiserSession::DenoiserSession(DenoiserSession&&) noexcept = default;
DenoiserSession& DenoiserSession::operator=(DenoiserSession&&) noexcept =
    default;

std::unique_ptr<DenoiserSession> DenoiserSession::Create(
    const ModelInventory&, const DenoiserOptions&, const DenoiserConditioning&,
    const CancellationToken*, DenoiserTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return nullptr;
}

bool DenoiserSession::Forward(int, std::span<const float>,
                              std::span<const float>, const CancellationToken*,
                              DenoiserVelocity*, DenoiserTelemetry*,
                              std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

bool DenoiserSession::Denoise(std::span<float>, std::span<float>,
                              const CancellationToken*, DenoiserTelemetry*,
                              std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

bool DenoiserSession::DenoiseWithProgress(std::span<float>, std::span<float>,
                                          const CancellationToken*,
                                          DenoiserProgress, void*,
                                          DenoiserTelemetry*,
                                          std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

const GenerationGeometry& DenoiserSession::geometry() const noexcept {
  static const GenerationGeometry empty;
  return empty;
}

const PackedLayout& DenoiserSession::layout() const noexcept {
  static const PackedLayout empty;
  return empty;
}

const SigmaSchedule& DenoiserSession::schedule() const noexcept {
  static const SigmaSchedule empty;
  return empty;
}

std::span<const std::uint16_t> DenoiserSession::block0_modulation()
    const noexcept {
  return {};
}

std::span<const std::uint16_t> DenoiserSession::refined_text() const noexcept {
  return {};
}

#endif

}  // namespace strix::minimax_h3
