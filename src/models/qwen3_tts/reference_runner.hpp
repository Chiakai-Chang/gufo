#ifndef STRIX_MODELS_QWEN3_TTS_REFERENCE_RUNNER_HPP_
#define STRIX_MODELS_QWEN3_TTS_REFERENCE_RUNNER_HPP_

#include <cstddef>
#include <filesystem>
#include <string>

#include "src/models/qwen3_tts/synthesis.hpp"

namespace strix::models::qwen3_tts {
struct OfficialReferenceOptions {
  std::filesystem::path model_root;
  std::filesystem::path reference_root;
  std::filesystem::path python_executable;
  std::filesystem::path runner_script;
};

/// Runs the pinned official Qwen3-TTS implementation as an intentionally slow
/// process-isolated oracle. This is a validation backend, not the production
/// HIP implementation.
class OfficialReferenceRunner {
public:
  using CancellationCheck = ::strix::models::qwen3_tts::CancellationCheck;

  explicit OfficialReferenceRunner(OfficialReferenceOptions options);

  [[nodiscard]] bool Validate(std::string* error = nullptr) const;
  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const CancellationCheck& is_cancelled,
                              SynthesisResult* result,
                              std::string* error = nullptr) const;

  [[nodiscard]] const OfficialReferenceOptions& options() const noexcept {
    return options_;
  }

private:
  OfficialReferenceOptions options_;
};

}  // namespace strix::models::qwen3_tts

#endif  // STRIX_MODELS_QWEN3_TTS_REFERENCE_RUNNER_HPP_
