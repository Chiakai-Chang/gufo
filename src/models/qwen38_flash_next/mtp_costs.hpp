#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, pinned Flash-Next Q4 target/Q8 MTP, 2026-09-19. Milliseconds per
// cohort, including catch-up, recursive proposals and target verification.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary decode).
// Reproduce with qwen38_flash_next_gpu_probe --cost-audit 0. Allocation and C1
// graph capture are warmed outside the measurement. See the benchmark README.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {38.7F, 48.4F, 57.5F, 68.4F, 78.3F, 88.0F, 97.4F, 106.6F},         // C1
        {51.3F, 66.1F, 81.9F, 100.2F, 130.7F, 146.7F, 163.5F, 175.4F},     // C2
        {72.4F, 100.0F, 141.7F, 172.0F, 215.2F, 246.4F, 293.3F, 320.6F},   // C4
        {93.7F, 149.8F, 202.8F, 246.7F, 300.0F, 366.5F, 426.0F, 466.7F},   // C6
        {114.5F, 182.3F, 248.3F, 320.2F, 392.8F, 463.2F, 538.5F, 609.2F},  // C8
    },
    {
        // Context 4096
        {40.0F, 50.9F, 60.5F, 71.6F, 80.6F, 90.6F, 101.4F, 112.1F},        // C1
        {54.0F, 70.5F, 87.6F, 105.0F, 136.6F, 153.3F, 171.8F, 190.4F},     // C2
        {77.1F, 107.8F, 154.0F, 183.6F, 226.5F, 254.8F, 303.4F, 344.3F},   // C4
        {100.8F, 161.8F, 222.2F, 265.6F, 321.4F, 386.2F, 449.8F, 501.0F},  // C6
        {124.2F, 197.9F, 272.0F, 348.0F, 418.8F, 493.5F, 574.6F, 657.8F},  // C8
    },
    {
        // Context 32768
        {42.8F, 53.0F, 62.2F, 72.6F, 84.4F, 95.3F, 107.0F, 117.8F},        // C1
        {59.8F, 75.6F, 92.0F, 110.1F, 144.2F, 161.5F, 180.8F, 198.4F},     // C2
        {88.5F, 116.8F, 160.6F, 192.6F, 244.2F, 275.5F, 325.1F, 359.0F},   // C4
        {117.8F, 176.7F, 233.5F, 279.6F, 342.3F, 411.0F, 474.6F, 525.9F},  // C6
        {147.1F, 216.0F, 285.2F, 365.1F, 446.4F, 525.6F, 608.0F, 686.7F},  // C8
    },
};

// Sampled policy uses fixed configured capacity for reproducibility; greedy
// batch policy uses these curves to bootstrap each physical occupancy.
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = kMtpCycleMilliseconds[interval][cohort][i];
    const auto high = kMtpCycleMilliseconds[interval + 1][cohort][i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal =
          high - kMtpCycleMilliseconds[interval + 1][cohort][i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
