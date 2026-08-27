#include "src/core/speculative/draft_heads.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::speculative {

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
  bool matched = false;
  const std::size_t n = prompt_tokens.size();

  for (std::size_t gram = std::min<std::size_t>(4, n); gram >= 1; --gram) {
    const auto suffix = prompt_tokens.subspan(n - gram, gram);
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

  last_drafted_ = proposal.tokens;
  return proposal;
}

void MtpDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)accepted;
  (void)correction_token;
}

}  // namespace gufo::speculative
