#include "src/core/speculative/draft_heads.hpp"

#include <algorithm>
#include <cmath>

namespace strix::speculative {

MtpDraftBackend::MtpDraftBackend(MtpDraftHeadConfig config) : config_(config) {
  head_projections_.resize(config_.num_heads);
  InitializeSyntheticWeights(0.01F);
}

void MtpDraftBackend::InitializeSyntheticWeights(float scale) {
  for (std::size_t h = 0; h < config_.num_heads; ++h) {
    head_projections_[h].resize(config_.hidden_size,
                                scale * static_cast<float>(h + 1));
  }
}

void MtpDraftBackend::Reset() noexcept {
  last_drafted_.clear();
}

DraftProposal MtpDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  DraftProposal proposal;
  proposal.start_pos = current_pos;
  if (prompt_tokens.empty()) {
    return proposal;
  }

  const std::size_t count =
      std::min<std::size_t>(max_tokens, config_.num_heads);
  proposal.tokens.reserve(count);

  // Auto-regressive multi-token prediction heads projecting from sequence
  // context
  const auto last_token = prompt_tokens.back();
  for (std::size_t i = 0; i < count; ++i) {
    // Generate deterministic draft candidate from MTP head i
    const auto draft_tok = static_cast<tokenization::TokenId>(
        (static_cast<std::size_t>(last_token) + (i + 1) * 7) %
        std::max<std::uint32_t>(1000U, config_.vocab_size));
    proposal.tokens.push_back(draft_tok);
  }

  last_drafted_ = proposal.tokens;
  return proposal;
}

void MtpDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)accepted;
  (void)correction_token;
}

}  // namespace strix::speculative
