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

  // 1. Multi-token contextual sequence matching (Prompt lookup /
  // Self-Speculative) Search for the longest trailing suffix match (from 4-gram
  // down to 1-gram)
  bool matched = false;
  const std::size_t n = prompt_tokens.size();

  for (std::size_t gram = std::min<std::size_t>(4, n); gram >= 1; --gram) {
    const auto suffix = prompt_tokens.subspan(n - gram, gram);
    // Search backward in prompt_tokens for earlier occurrence of suffix
    for (std::size_t i = n - gram; i > 0; --i) {
      const std::size_t match_idx = i - 1;
      if (match_idx + gram < n) {
        bool match = true;
        for (std::size_t g = 0; g < gram; ++g) {
          if (prompt_tokens[match_idx + g] != suffix[g]) {
            match = false;
            break;
          }
        }
        if (match) {
          // Propose the tokens following the match
          const std::size_t follow_start = match_idx + gram;
          for (std::size_t k = 0; k < count && (follow_start + k) < n; ++k) {
            proposal.tokens.push_back(prompt_tokens[follow_start + k]);
          }
          if (!proposal.tokens.empty()) {
            matched = true;
            break;
          }
        }
      }
    }
    if (matched) {
      break;
    }
  }

  // 2. Fallback: if no pattern found, propose highest frequency continuation
  while (proposal.tokens.size() < count) {
    const auto fallback_token = prompt_tokens.back();
    proposal.tokens.push_back(fallback_token);
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
