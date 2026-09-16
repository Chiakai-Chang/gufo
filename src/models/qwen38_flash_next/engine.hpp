#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

struct ModelWeights;
struct MtpWeights;
class NgramTable;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional MTP draft sidecar (`mtp-...-shared-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Prefill chunk; also bounds the draft block's batch.
  std::uint32_t max_batch = 512;
  /// Tokens the draft block proposes per speculative cycle.
  std::uint32_t max_draft_tokens = 3;
  /// Vocabulary prefix the draft block scores (0 = full). Token ids follow
  /// merge order, so a prefix holds the frequent tokens; verification
  /// always uses the full head, so only which draft is proposed changes.
  std::uint32_t draft_vocab = 0;
};

class Session;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions, executed one at a time.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<NgramTable> ngram_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Samples the target and verifies deterministic MTP proposals with the
  /// same sampler. Updates its history/RNG only for emitted tokens (and the
  /// stop draw), leaving the session at exactly the emitted prefix. A
  /// one-token budget or a model without MTP uses ordinary decoding.
  /// Benchmarks may continue past EOS by setting stop_at_eos to false.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return logits_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return tokens_;
  }
  void Reset();

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg);
  bool DraftCatchUp(std::int32_t next_token, std::string* error_msg);
  [[nodiscard]] std::int32_t Argmax(const float* row,
                                    std::size_t count = 0) const noexcept;

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::vector<float> draft_logits_;
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  SpeculativeStats stats_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
