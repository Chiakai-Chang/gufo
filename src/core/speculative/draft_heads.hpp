#ifndef GUFO_CORE_SPECULATIVE_DRAFT_HEADS_HPP_
#define GUFO_CORE_SPECULATIVE_DRAFT_HEADS_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::speculative {

/// Configuration for multi-token prediction (MTP) / Medusa draft heads
struct MtpDraftHeadConfig {
  std::uint32_t num_heads{3};
  std::uint32_t hidden_size{5120};
  std::uint32_t vocab_size{152064};
};

/// Multi-Token Prediction (MTP) / Medusa draft head backend
class MtpDraftBackend : public IDraftBackend {
public:
  explicit MtpDraftBackend(MtpDraftHeadConfig config);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "MtpDraftBackend";
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

  /// Sets synthetic projection weights for testing / evaluation
  void InitializeSyntheticWeights(float scale = 0.02F);

private:
  MtpDraftHeadConfig config_;
  std::vector<std::vector<float>> head_projections_;
  std::vector<tokenization::TokenId> last_drafted_;
};

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_DRAFT_HEADS_HPP_
