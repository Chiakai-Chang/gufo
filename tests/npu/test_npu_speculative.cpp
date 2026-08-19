#include <cassert>
#include <iostream>
#include <vector>

#include "src/core/heterogeneous/npu_drafter.hpp"

void TestNpuDrafterInitialization() {
  strix::heterogeneous::NpuDrafterConfig config;
  config.enable_xrt = true;
  config.max_draft_tokens = 4;
  config.vocab_size = 152064;

  strix::heterogeneous::NpuDraftBackend drafter(config);
  assert(drafter.Name() == "NpuXdna2DraftBackend");

  std::cout << "NPU Drafter status: " << drafter.GetStatusMessage() << "\n";

  std::vector<strix::tokenization::TokenId> prompt = {100, 101, 102};
  auto proposal = drafter.Propose(prompt, 3, 4);
  assert(proposal.tokens.size() == 4);
  assert(proposal.start_pos == 3);

  std::vector<strix::tokenization::TokenId> accepted = {proposal.tokens[0],
                                                        proposal.tokens[1]};
  drafter.AcceptFeedback(accepted, 999);

  drafter.Reset();
  std::cout << "TestNpuDrafterInitialization passed.\n";
}

int main() {
  TestNpuDrafterInitialization();
  std::cout << "All NPU speculative tests passed.\n";
  return 0;
}
