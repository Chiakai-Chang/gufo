#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/qwen_gpu_executor.hpp"

void TestSpeculativeDraftBackendInterface() {
  std::vector<strix::tokenization::TokenId> pool = {101, 102, 103, 104};
  strix::speculative::MockDraftBackend backend(pool);

  assert(backend.Name() == "MockDraftBackend");

  std::vector<strix::tokenization::TokenId> prefix = {1, 2, 3};
  auto proposal = backend.Propose(prefix, 3, 3);
  assert(proposal.tokens.size() == 3);
  assert(proposal.tokens[0] == 101);
  assert(proposal.tokens[1] == 102);
  assert(proposal.tokens[2] == 103);
  assert(proposal.start_pos == 3);

  std::vector<strix::tokenization::TokenId> accepted = {101, 102};
  backend.AcceptFeedback(accepted, 999);
  std::cout << "TestSpeculativeDraftBackendInterface passed.\n";
}

void TestSpeculativeVerificationMockExecution() {
  strix::speculative::SpeculativeStats stats;
  assert(stats.AcceptanceRate() == 0.0F);

  stats.total_draft_tokens = 10;
  stats.total_accepted_tokens = 8;
  assert(std::abs(stats.AcceptanceRate() - 0.80F) < 1e-5F);

  std::cout << "TestSpeculativeVerificationMockExecution passed.\n";
}

int main() {
  TestSpeculativeDraftBackendInterface();
  TestSpeculativeVerificationMockExecution();
  std::cout << "All speculative verification tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping speculative GPU tests.\n";
  return 0;
}
#endif
