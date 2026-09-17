#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

// Length depends on committed acceptance history, never request timings.
// Only the first rejected draft is an observed failure; its unverified tail
// and a fully accepted, budget-limited chain are censored observations.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)) {}

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0) {
      return 0;
    }
    const float probability = successes_ / (successes_ + failures_);
    float survival = 1.0F;
    float expected_tokens = 1.0F;  // The sampled target anchor.
    float best_score = 0.0F;
    std::uint32_t best = 1;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= probability;
      expected_tokens += survival;
      // C1 controls at widths 1/3/7 put each sequential draft near 0.35
      // of the fixed target verification cost. Maximize expected output
      // per unit of work instead of maximizing acceptance alone.
      const float cost = 1.0F + 0.35F * static_cast<float>(length);
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  void Observe(std::size_t accepted, std::size_t drafted) noexcept {
    if (drafted == 0) {
      return;
    }
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    successes_ = 0.75F * successes_ + static_cast<float>(accepted);
    failures_ = 0.75F * failures_ + (accepted < drafted ? 1.0F : 0.0F);
  }

  void Reset() noexcept {
    successes_ = 3.0F;
    failures_ = 1.0F;
  }

  [[nodiscard]] std::array<float, 2> State() const noexcept {
    return {successes_, failures_};
  }

  [[nodiscard]] bool Restore(std::array<float, 2> state) noexcept {
    if (!std::isfinite(state[0]) || !std::isfinite(state[1]) ||
        state[0] < 0.0F || state[1] < 0.0F ||
        !std::isfinite(state[0] + state[1]) || state[0] + state[1] <= 0.0F) {
      return false;
    }
    successes_ = state[0];
    failures_ = state[1];
    return true;
  }

private:
  std::uint32_t limit_;
  float successes_{3.0F};
  float failures_{1.0F};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
