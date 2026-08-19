#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using strix::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

class ScriptedTargetExecutor final
    : public strix::speculative::ISpeculativeTargetExecutor {
public:
  ScriptedTargetExecutor(std::vector<TokenId> generated_tokens, TokenId eos_id)
      : generated_tokens_(std::move(generated_tokens)), eos_id_(eos_id) {}

  void Reset() noexcept override {
    prompt_.clear();
    state_.clear();
    saved_state_.clear();
    restore_count_ = 0;
  }

  TokenId ForwardPromptBatch(std::span<const TokenId> prompt_tokens) override {
    prompt_.assign(prompt_tokens.begin(), prompt_tokens.end());
    state_ = prompt_;
    if (generated_tokens_.empty()) {
      return eos_id_;
    }
    return generated_tokens_.front();
  }

  TokenId ForwardToken(TokenId token_id, std::uint32_t pos,
                       bool compute_logits) override {
    (void)compute_logits;
    Expect(pos == state_.size(), "target position must match committed state");
    Expect(pos >= prompt_.size(), "target position must follow the prompt");
    const std::size_t generated_index = pos - prompt_.size();
    Expect(generated_index < generated_tokens_.size(),
           "scripted target exhausted");
    Expect(token_id == generated_tokens_[generated_index],
           "target input token must match the greedy sequence");
    state_.push_back(token_id);
    if (generated_index + 1 < generated_tokens_.size()) {
      return generated_tokens_[generated_index + 1];
    }
    return eos_id_;
  }

  void SaveState(std::uint32_t valid_context) override {
    Expect(valid_context == state_.size(),
           "snapshot length must match committed target state");
    saved_state_ = state_;
  }

  void RestoreState() override {
    Expect(!saved_state_.empty(), "target state must be saved before restore");
    state_ = saved_state_;
    ++restore_count_;
  }

  TokenId GetEosTokenId() const noexcept override { return eos_id_; }

  std::string_view DecodeToken(TokenId token_id) const noexcept override {
    (void)token_id;
    return "token";
  }

  [[nodiscard]] const std::vector<TokenId>& State() const noexcept {
    return state_;
  }

  [[nodiscard]] std::size_t RestoreCount() const noexcept {
    return restore_count_;
  }

private:
  std::vector<TokenId> generated_tokens_;
  TokenId eos_id_;
  std::vector<TokenId> prompt_;
  std::vector<TokenId> state_;
  std::vector<TokenId> saved_state_;
  std::size_t restore_count_{0};
};

strix::models::GenerationOptions GenerationOptions(std::size_t max_tokens,
                                                   TokenId eos_id) {
  strix::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  options.eos_token_id = eos_id;
  return options;
}

void TestSpeculativeDraftBackendInterface() {
  std::vector<TokenId> pool = {101, 102, 103, 104};
  strix::speculative::MockDraftBackend backend(pool);

  Expect(backend.Name() == "MockDraftBackend", "mock backend name");

  const std::vector<TokenId> prefix = {1, 2, 3};
  const auto proposal = backend.Propose(prefix, 3, 3);
  Expect(proposal.tokens == std::vector<TokenId>({101, 102, 103}),
         "mock proposal tokens");
  Expect(proposal.start_pos == 3, "mock proposal position");

  const std::vector<TokenId> accepted = {101, 102};
  backend.AcceptFeedback(accepted, 999);
}

void TestSpeculativeStats() {
  strix::speculative::SpeculativeStats stats;
  Expect(stats.AcceptanceRate() == 0.0F, "empty acceptance rate");

  stats.total_draft_tokens = 10;
  stats.total_accepted_tokens = 8;
  Expect(std::abs(stats.AcceptanceRate() - 0.80F) < 1e-5F,
         "non-empty acceptance rate");
}

void TestFullAcceptanceProducesTargetBonusToken() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12, 13}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 12});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(4, eos_id));

  Expect(output == std::vector<TokenId>({10, 11, 12, 13}),
         "full acceptance must emit the target bonus token");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11, 12}),
         "full acceptance target state");
  Expect(target.RestoreCount() == 0, "full acceptance must not roll back");
  Expect(verifier.GetStats().total_accepted_tokens == 2,
         "full acceptance statistics");
}

void TestPartialRejectionRestoresAndReplaysState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 99});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(3, eos_id));

  Expect(output == std::vector<TokenId>({10, 11, 12}),
         "partial rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11}),
         "partial rejection target state");
  Expect(target.RestoreCount() == 1, "partial rejection must restore once");
}

void TestImmediateRejectionRestoresGreedyState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{99, 98});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(2, eos_id));

  Expect(output == std::vector<TokenId>({10, 11}),
         "immediate rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10}),
         "immediate rejection target state");
  Expect(target.RestoreCount() == 1, "immediate rejection must restore once");
}

void TestFirstPrefillTokenHonorsBudgetAndCallback() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};
  std::vector<TokenId> callback_tokens;

  const auto output =
      verifier.Generate(prompt, GenerationOptions(1, eos_id),
                        [&](TokenId token, std::string_view piece) {
                          callback_tokens.push_back(token);
                          Expect(piece == "token", "callback decoded token");
                          return true;
                        });

  Expect(output == std::vector<TokenId>({10}), "one-token output budget");
  Expect(callback_tokens == output, "first token callback");
  Expect(target.State() == prompt,
         "first token remains uncommitted until the next decode step");
}

void TestFirstPrefillEosIsNotEmitted() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({eos_id}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{1});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};
  std::size_t callback_count = 0;

  const auto output = verifier.Generate(prompt, GenerationOptions(4, eos_id),
                                        [&](TokenId, std::string_view) {
                                          ++callback_count;
                                          return true;
                                        });

  Expect(output.empty(), "prefill EOS must not be emitted");
  Expect(callback_count == 0, "prefill EOS callback");
  Expect(target.State() == prompt, "prefill EOS target state");
}

}  // namespace

int main() {
  TestSpeculativeDraftBackendInterface();
  TestSpeculativeStats();
  TestFullAcceptanceProducesTargetBonusToken();
  TestPartialRejectionRestoresAndReplaysState();
  TestImmediateRejectionRestoresGreedyState();
  TestFirstPrefillTokenHonorsBudgetAndCallback();
  TestFirstPrefillEosIsNotEmitted();
  std::cout << "All speculative verification tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping speculative GPU tests.\n";
  return 0;
}
#endif
