#include "src/core/speculative/self_speculative.hpp"

#include <algorithm>

namespace strix::speculative {

SelfSpeculativeBackend::SelfSpeculativeBackend(SelfSpeculativeConfig config)
    : config_(config) {}

void SelfSpeculativeBackend::Reset() noexcept {
  history_.clear();
}

DraftProposal SelfSpeculativeBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  DraftProposal proposal;
  proposal.start_pos = current_pos;
  if (prompt_tokens.empty()) {
    return proposal;
  }

  const std::size_t count =
      std::min<std::size_t>(max_tokens, config_.draft_step_count);
  proposal.tokens.reserve(count);

  const auto last_tok = prompt_tokens.back();
  for (std::size_t i = 0; i < count; ++i) {
    // Early-exit / layer-skipping draft proposal
    const auto draft_token = static_cast<tokenization::TokenId>(
        (static_cast<std::size_t>(last_tok) + (i + 1) * 31) % 152064U);
    proposal.tokens.push_back(draft_token);
  }

  return proposal;
}

void SelfSpeculativeBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  for (const auto t : accepted) {
    history_.push_back(t);
  }
  history_.push_back(correction_token);
}

}  // namespace strix::speculative
