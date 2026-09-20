#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, pinned Flash-Next Q4 target/Q8 MTP, 2026-09-20. Milliseconds per
// cohort, including catch-up, recursive proposals and target verification.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary decode).
// Reproduce with qwen38_flash_next_gpu_probe --cost-audit 0. Allocation and C1
// graph capture are warmed outside the measurement; each row is the median
// of three complete cycles. Use --depth N to audit one depth. See benchmarks.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {38.4F, 47.2F, 56.7F, 67.2F, 78.4F, 85.8F, 94.5F, 102.6F},  // C1
        {46.0F, 59.9F, 74.1F, 90.5F, 112.3F, 125.9F, 140.8F, 150.2F},  // C2
        {57.7F, 80.8F, 112.0F, 136.2F, 171.1F, 193.2F, 230.2F, 250.1F},  // C4
        {68.7F, 108.9F, 148.1F, 185.7F, 227.5F, 273.9F, 318.7F, 350.6F},  // C6
        {79.4F, 126.3F, 175.9F, 231.9F, 287.4F, 336.1F, 390.5F, 443.6F},  // C8
    },
    {
        // Context 4096
        {39.7F, 49.3F, 59.8F, 69.7F, 79.1F, 89.1F, 98.6F, 108.8F},  // C1
        {49.0F, 64.4F, 80.3F, 96.7F, 121.8F, 134.9F, 150.0F, 161.7F},  // C2
        {62.2F, 87.9F, 122.3F, 145.2F, 183.3F, 207.2F, 244.6F, 271.5F},  // C4
        {75.2F, 120.3F, 165.2F, 201.3F, 249.5F, 298.1F, 345.5F, 383.3F},  // C6
        {88.3F, 138.6F, 198.8F, 252.3F, 316.6F, 368.0F, 428.4F, 489.4F},  // C8
    },
    {
        // Context 32768
        {42.5F, 51.9F, 61.5F, 71.1F, 82.5F, 92.5F, 103.6F, 112.6F},  // C1
        {54.8F, 68.8F, 83.0F, 99.3F, 124.5F, 139.3F, 155.2F, 166.3F},  // C2
        {73.5F, 96.9F, 129.4F, 154.2F, 194.6F, 221.6F, 256.7F, 282.9F},  // C4
        {91.9F, 132.7F, 176.4F, 215.5F, 265.2F, 317.6F, 359.7F, 396.9F},  // C6
        {110.6F, 156.8F, 212.3F, 271.4F, 346.7F, 397.2F, 458.9F, 507.8F},  // C8
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
