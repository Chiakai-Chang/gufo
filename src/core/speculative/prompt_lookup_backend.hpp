#ifndef GUFO_CORE_SPECULATIVE_PROMPT_LOOKUP_BACKEND_HPP_
#define GUFO_CORE_SPECULATIVE_PROMPT_LOOKUP_BACKEND_HPP_

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::speculative {

struct PromptLookupConfig {
  std::size_t max_ngram_size{3};
  std::size_t min_ngram_size{2};
  std::size_t max_draft_tokens{4};
};

/// Industry-standard Prompt Lookup Decoding (PLD) provider
class PromptLookupDraftBackend : public IDraftBackend {
public:
  explicit PromptLookupDraftBackend(PromptLookupConfig config = {})
      : config_(config) {}

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "PromptLookupBackend";
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

private:
  PromptLookupConfig config_;
  std::size_t matches_found_{0};
  std::size_t fallbacks_{0};
};

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_PROMPT_LOOKUP_BACKEND_HPP_
