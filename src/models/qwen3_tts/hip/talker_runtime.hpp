#ifndef STRIX_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_
#define STRIX_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::models::qwen3_tts::hip {

struct TalkerPrefillOutput {
  std::vector<float> logits;
  std::vector<float> last_hidden;
  std::vector<float> layer0_output;
};

struct CustomVoicePromptOutput {
  std::vector<float> embeddings;
  std::vector<float> trailing_text;
  std::vector<float> tts_pad;
  std::size_t tokens{0};
};

struct TalkerGenerationOutput {
  std::vector<std::uint32_t> codes;
  std::size_t frames{0};
  std::size_t code_groups{0};
};

struct TalkerSamplingOptions {
  bool sample{true};
  std::uint32_t seed{42};
  std::size_t top_k{50};
  float temperature{0.9F};
  std::size_t predictor_top_k{50};
  float predictor_temperature{0.9F};
  float repetition_penalty{1.05F};
};

struct CodePredictorOutput {
  std::vector<std::uint32_t> codes;
  std::vector<float> logits;
};

/// Native gfx1151 talker prefill used as the first exact HIP parity boundary.
/// Input embeddings must use row-major [tokens, hidden_size] layout.
class TalkerHipRuntime {
public:
  ~TalkerHipRuntime();

  TalkerHipRuntime(const TalkerHipRuntime&) = delete;
  TalkerHipRuntime& operator=(const TalkerHipRuntime&) = delete;
  TalkerHipRuntime(TalkerHipRuntime&&) = delete;
  TalkerHipRuntime& operator=(TalkerHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<TalkerHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_tokens,
      std::string* error = nullptr);

  [[nodiscard]] bool Prefill(std::span<const float> input_embeddings,
                             std::size_t tokens, TalkerPrefillOutput* output,
                             std::string* error = nullptr);

  /// Builds the official non-streaming CustomVoice prompt embeddings.
  [[nodiscard]] bool BuildCustomVoicePrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
      std::string_view language, CustomVoicePromptOutput* output,
      std::string* error = nullptr);

  /// Greedily expands one first-codebook token into all codec groups.
  [[nodiscard]] bool PredictCodeFrame(std::span<const float> talker_hidden,
                                      std::uint32_t first_code,
                                      std::vector<std::uint32_t>* codes,
                                      std::string* error = nullptr);

  /// Expands a frame and retains one [vocab] logit row for each sub-codebook.
  [[nodiscard]] bool PredictCodeFrameTrace(std::span<const float> talker_hidden,
                                           std::uint32_t first_code,
                                           CodePredictorOutput* output,
                                           std::string* error = nullptr);

  /// Feeds one complete codec frame back into the cached talker.
  [[nodiscard]] bool DecodeCodeFrame(std::span<const std::uint32_t> codes,
                                     std::span<const float> text_embedding,
                                     TalkerPrefillOutput* output,
                                     std::string* error = nullptr);

  [[nodiscard]] bool BuildCodeFrameEmbedding(
      std::span<const std::uint32_t> codes,
      std::span<const float> text_embedding, std::vector<float>* embedding,
      std::string* error = nullptr);

  [[nodiscard]] bool GenerateGreedy(const CustomVoicePromptOutput& prompt,
                                    std::size_t maximum_new_tokens,
                                    TalkerGenerationOutput* output,
                                    std::string* error = nullptr);

  /// Generates with the checkpoint's native top-k sampling policy by default.
  [[nodiscard]] bool Generate(const CustomVoicePromptOutput& prompt,
                              std::size_t maximum_new_tokens,
                              const TalkerSamplingOptions& sampling,
                              TalkerGenerationOutput* output,
                              std::string* error = nullptr);

private:
  struct Impl;
  explicit TalkerHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::models::qwen3_tts::hip

#endif  // STRIX_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_
