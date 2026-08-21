#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/models/minimax_h3/runtime.hpp"

namespace {

double Seconds(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main() {
  const char* model_root = std::getenv("STRIX_H3_MODEL_ROOT");
  const char* manifest = std::getenv("STRIX_H3_SOURCE_MANIFEST");
  if (model_root == nullptr || manifest == nullptr) {
    std::cerr << "SKIP: set STRIX_H3_MODEL_ROOT and "
                 "STRIX_H3_SOURCE_MANIFEST\n";
    return 77;
  }
  std::string error;
  const auto inspection_start = std::chrono::steady_clock::now();
  auto inventory =
      strix::minimax_h3::ModelInventory::Inspect(model_root, manifest, &error);
  if (!inventory.has_value()) {
    std::cerr << error << '\n';
    return 1;
  }
  const double inspection_seconds = Seconds(inspection_start);
  if (inventory->telemetry().payload_bytes_read != 0 ||
      inventory->telemetry().mapped_bytes != 0 ||
      inventory->telemetry().device_bytes != 0) {
    std::cerr << "metadata inspection touched weight payload residency\n";
    return 1;
  }
  auto backend = strix::minimax_h3::CreateHipResidencyBackend(&error);
  if (!backend) {
    std::cerr << error << '\n';
    return 1;
  }

  const auto load = [&](strix::minimax_h3::ResidencyMode mode) {
    const auto start = std::chrono::steady_clock::now();
    auto session = strix::minimax_h3::PhaseSession::Load(
        *inventory, strix::minimax_h3::Phase::kAudioVae,
        strix::minimax_h3::LoadOptions{mode, 16U << 20U, 16U << 20U, true},
        *backend, nullptr, nullptr, &error);
    const double elapsed = Seconds(start);
    if (!session.has_value()) {
      std::cerr << error << '\n';
      std::exit(1);
    }
    std::cout << strix::minimax_h3::ToString(mode) << ": " << elapsed
              << " s, file-backed " << session->telemetry().file_backed_bytes
              << ", registered " << session->telemetry().registered_host_bytes
              << ", device " << session->telemetry().device_weight_bytes
              << ", peak " << session->telemetry().peak_live_bytes << '\n';
    return elapsed;
  };

  const double mapped = load(strix::minimax_h3::ResidencyMode::kMappedReadOnly);
  const double copied = load(strix::minimax_h3::ResidencyMode::kDeviceCopy);
  std::cout << "inspection: " << inspection_seconds
            << " s; audio-VAE mapped/device-copy load ratio: "
            << mapped / copied << '\n';
  return 0;
}
