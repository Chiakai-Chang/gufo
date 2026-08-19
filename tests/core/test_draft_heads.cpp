#include <cassert>
#include <iostream>
#include <vector>

#include "src/core/speculative/draft_heads.hpp"
#include "src/core/speculative/self_speculative.hpp"

void TestMtpDraftBackend() {
  strix::speculative::MtpDraftHeadConfig config;
  config.num_heads = 4;
  config.hidden_size = 256;
  config.vocab_size = 1000;

  strix::speculative::MtpDraftBackend backend(config);
  assert(backend.Name() == "MtpDraftBackend");

  const std::vector<strix::tokenization::TokenId> prompt = {10, 20, 30, 40,
                                                            50, 60, 10, 20};
  const auto proposal = backend.Propose(prompt, 8, 3);
  assert(proposal.tokens ==
         std::vector<strix::tokenization::TokenId>({30, 40, 50}));
  assert(proposal.start_pos == 8);

  const auto proposal_max = backend.Propose(prompt, 8, 5);
  assert(proposal_max.tokens ==
         std::vector<strix::tokenization::TokenId>({30, 40, 50, 60}));

  const std::vector<strix::tokenization::TokenId> no_match = {42, 43, 44};
  const auto empty_proposal = backend.Propose(no_match, 3, 3);
  assert(empty_proposal.tokens.empty());

  std::cout << "TestMtpDraftBackend passed.\n";
}

void TestSelfSpeculativeBackend() {
  strix::speculative::SelfSpeculativeConfig config;
  config.total_layers = 32;
  config.exit_layer = 8;
  config.draft_step_count = 3;

  strix::speculative::SelfSpeculativeBackend backend(config);
  assert(backend.Name() == "SelfSpeculativeBackend");

  const std::vector<strix::tokenization::TokenId> prompt = {10, 20, 30};
  const auto proposal = backend.Propose(prompt, 3, 3);
  assert(proposal.tokens.size() == 3);
  assert(proposal.start_pos == 3);

  const std::vector<strix::tokenization::TokenId> accepted = {
      proposal.tokens[0]};
  backend.AcceptFeedback(accepted, 999);

  std::cout << "TestSelfSpeculativeBackend passed.\n";
}

int main() {
  TestMtpDraftBackend();
  TestSelfSpeculativeBackend();
  std::cout << "All MTP and self-speculative draft head tests passed.\n";
  return 0;
}
