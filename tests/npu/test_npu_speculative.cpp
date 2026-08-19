#include <cassert>
#include <iostream>
#include <vector>

#include "src/core/heterogeneous/npu_drafter.hpp"

void TestNpuDrafterInitialization() {
  strix::heterogeneous::NpuDrafterConfig config;
  config.mtp_model_path.clear();
  config.enable_xrt = false;
  config.max_draft_tokens = 4;
  config.vocab_size = 152064;

  strix::heterogeneous::NpuDraftBackend drafter(config);
  assert(drafter.Name() == "NpuXdna2DraftBackend");
  assert(!drafter.IsNpuActive());
  assert(!drafter.HasMtpModel());
  assert(!drafter.GetStatusMessage().empty());

  std::cout << "NPU Drafter status: " << drafter.GetStatusMessage() << "\n";

  const std::vector<strix::tokenization::TokenId> prompt = {10, 20, 30, 40,
                                                            50, 60, 10, 20};
  const auto proposal = drafter.Propose(prompt, 8, 4);
  assert(proposal.tokens ==
         std::vector<strix::tokenization::TokenId>({30, 40, 50, 60}));
  assert(proposal.start_pos == 8);

  const std::vector<strix::tokenization::TokenId> accepted = {
      proposal.tokens[0], proposal.tokens[1]};
  drafter.AcceptFeedback(accepted, 999);

  const std::vector<strix::tokenization::TokenId> no_match = {100, 101, 102};
  const auto empty_proposal = drafter.Propose(no_match, 3, 4);
  assert(empty_proposal.tokens.empty());

  drafter.Reset();
  std::cout << "TestNpuDrafterInitialization passed.\n";
}

int main() {
  TestNpuDrafterInitialization();
  std::cout << "All NPU speculative tests passed.\n";
  return 0;
}
