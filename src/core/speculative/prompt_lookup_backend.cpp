#include "src/core/speculative/prompt_lookup_backend.hpp"

#include <algorithm>

namespace strix::speculative {

void PromptLookupDraftBackend::Reset() noexcept {
  matches_found_ = 0;
  fallbacks_ = 0;
}

DraftProposal PromptLookupDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  DraftProposal proposal;
  proposal.start_pos = current_pos;

  const std::size_t total_tokens = prompt_tokens.size();
  if (total_tokens < config_.min_ngram_size + 1) {
    ++fallbacks_;
    return proposal;  // Propose 0 tokens -> pure 1-token decode, 0 rollback
                      // penalty
  }

  const std::size_t max_k =
      std::min<std::size_t>(max_tokens, config_.max_draft_tokens);

  // Search backward across descending n-gram sizes (e.g. 3-gram down to 2-gram)
  const std::size_t max_gram =
      std::min<std::size_t>(config_.max_ngram_size, total_tokens - 1);

  for (std::size_t gram = max_gram; gram >= config_.min_ngram_size; --gram) {
    const auto suffix = prompt_tokens.subspan(total_tokens - gram, gram);

    // Search earlier in sequence (prior to current trailing suffix)
    for (std::size_t i = total_tokens - gram; i > 0; --i) {
      const std::size_t candidate_idx = i - 1;
      if (candidate_idx + gram <= total_tokens - gram) {
        bool match = true;
        for (std::size_t g = 0; g < gram; ++g) {
          if (prompt_tokens[candidate_idx + g] != suffix[g]) {
            match = false;
            break;
          }
        }

        if (match) {
          // Found match at candidate_idx! Grab subsequent K tokens
          const std::size_t follow_start = candidate_idx + gram;
          for (std::size_t k = 0;
               k < max_k && (follow_start + k) < (total_tokens - gram); ++k) {
            proposal.tokens.push_back(prompt_tokens[follow_start + k]);
          }

          if (!proposal.tokens.empty()) {
            ++matches_found_;
            return proposal;
          }
        }
      }
    }
  }

  ++fallbacks_;
  // No pattern match found in context: return empty proposal to avoid rollback
  return proposal;
}

void PromptLookupDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)accepted;
  (void)correction_token;
}

}  // namespace strix::speculative
