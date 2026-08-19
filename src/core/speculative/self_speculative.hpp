#ifndef STRIX_CORE_SPECULATIVE_SELF_SPECULATIVE_HPP_
#define STRIX_CORE_SPECULATIVE_SELF_SPECULATIVE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace strix::speculative {

/// Configuration for layer-skipping self-speculative draft proposal
struct SelfSpeculativeConfig {
  std::uint32_t total_layers{32};
  std::uint32_t exit_layer{
      8};  // Early exit at layer 8 out of 32 (4x speedup during draft)
  std::uint32_t draft_step_count{3};
};

/// Layer-skipping / early-exit self-speculative draft backend
class SelfSpeculativeBackend : public IDraftBackend {
public:
  explicit SelfSpeculativeBackend(SelfSpeculativeConfig config);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "SelfSpeculativeBackend";
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

  [[nodiscard]] const SelfSpeculativeConfig& GetConfig() const noexcept {
    return config_;
  }

private:
  SelfSpeculativeConfig config_;
  std::vector<tokenization::TokenId> history_;
};

}  // namespace strix::speculative

#endif  // STRIX_CORE_SPECULATIVE_SELF_SPECULATIVE_HPP_
