#ifndef STRIX_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
#define STRIX_MODELS_QWEN3_TTS_SYNTHESIS_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"

namespace strix::models::qwen3_tts {

struct SynthesisRequest {
  std::string text;
  std::string speaker{"vivian"};
  std::string language{"english"};
  std::string instruct;
  AudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only{false};
  std::size_t max_new_tokens{3000};
  std::uint32_t seed{42};
  bool greedy{false};
};

struct SynthesisResult {
  std::uint32_t sample_rate{0};
  std::uint32_t code_groups{0};
  std::vector<float> samples;
  // Frame-major [frames, code_groups].
  std::vector<std::int32_t> codes;
};

using CancellationCheck = std::function<bool()>;

}  // namespace strix::models::qwen3_tts

#endif  // STRIX_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
