#ifndef STRIX_MODELS_QWEN3_TTS_LOADER_HPP_
#define STRIX_MODELS_QWEN3_TTS_LOADER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/tensor.hpp"

namespace strix::models::qwen3_tts {

struct MappedRegion {
  const std::byte* data = nullptr;
  std::size_t size = 0;
};

/// Result of loading the model directory. `tensors` payload pointers stay
/// valid for the lifetime of this struct (mappings are owned by it).
struct LoadResult {
  bool ok = false;
  std::string error;
  ModelConfig config;
  std::unique_ptr<TensorStore> store;

  // Owned file mappings; payload pointers in `store` point into these.
  std::vector<std::shared_ptr<void>> mappings;
  std::vector<MappedRegion> mapped_regions;
};

/// Loads a Qwen3-TTS 12Hz CustomVoice model directory.
///
/// Layout expected:
///   <dir>/config.json
///   <dir>/model.safetensors
///   <dir>/speech_tokenizer/model.safetensors
[[nodiscard]] LoadResult LoadModelDirectory(const std::string& model_dir);

/// True if `model_dir/config.json` has model_type "qwen3_tts".
[[nodiscard]] bool LooksLikeQwen3Tts(const std::string& model_dir);

}  // namespace strix::models::qwen3_tts

#endif  // STRIX_MODELS_QWEN3_TTS_LOADER_HPP_
