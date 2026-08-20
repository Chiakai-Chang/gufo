#include <cassert>
#include <iostream>
#include <vector>

#include "src/core/speculative/prompt_lookup_backend.hpp"

void TestPromptLookupExactMatch() {
  strix::speculative::PromptLookupConfig config;
  config.max_ngram_size = 3;
  config.min_ngram_size = 2;
  config.max_draft_tokens = 3;

  strix::speculative::PromptLookupDraftBackend backend(config);
  assert(backend.Name() == "PromptLookupBackend");

  // Context: [10, 20, 30, 40, 50, 99, 10, 20]
  // Suffix [10, 20] matches earlier [10, 20] followed by [30, 40, 50]
  std::vector<strix::tokenization::TokenId> seq = {10, 20, 30, 40,
                                                   50, 99, 10, 20};
  auto proposal = backend.Propose(seq, 8, 3);

  assert(proposal.tokens.size() == 3);
  assert(proposal.tokens[0] == 30);
  assert(proposal.tokens[1] == 40);
  assert(proposal.tokens[2] == 50);

  std::cout << "TestPromptLookupExactMatch passed.\n";
}

void TestPromptLookupNoMatchReturnsEmpty() {
  strix::speculative::PromptLookupConfig config;
  strix::speculative::PromptLookupDraftBackend backend(config);

  // Context: [1, 2, 3, 4, 5, 6, 7]
  // Trailing suffix [6, 7] does not appear earlier in context
  std::vector<strix::tokenization::TokenId> seq = {1, 2, 3, 4, 5, 6, 7};
  auto proposal = backend.Propose(seq, 7, 4);

  // Must return 0 tokens to avoid rollback penalty
  assert(proposal.tokens.empty());

  std::cout << "TestPromptLookupNoMatchReturnsEmpty passed.\n";
}

void TestPromptLookupShortSequence() {
  strix::speculative::PromptLookupConfig config;
  strix::speculative::PromptLookupDraftBackend backend(config);

  std::vector<strix::tokenization::TokenId> seq = {10};
  auto proposal = backend.Propose(seq, 1, 4);
  assert(proposal.tokens.empty());

  std::cout << "TestPromptLookupShortSequence passed.\n";
}

int main() {
  TestPromptLookupExactMatch();
  TestPromptLookupNoMatchReturnsEmpty();
  TestPromptLookupShortSequence();
  std::cout << "Prompt lookup test passed.\n";
  return 0;
}
