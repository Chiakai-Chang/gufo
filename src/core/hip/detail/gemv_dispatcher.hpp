#ifndef STRIX_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_
#define STRIX_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_

#include <cstddef>
#include <cstdint>

namespace strix::hip::detail {

enum class GemvStrategy : std::uint8_t {
  kBaselineBlock = 0,
  kWave32SingleRow = 1,
  kWave32DualRow = 2,
  kWave32QuadRow = 3,
};

[[nodiscard]] constexpr GemvStrategy SelectGemvStrategy(std::size_t M,
                                                        std::size_t K,
                                                        bool is_bf16) noexcept {
  if (!is_bf16 || (K % 8 != 0)) {
    return GemvStrategy::kBaselineBlock;
  }
  // For very wide inputs (e.g. FFN down projection K >= 8192), block reduction
  // with 8-16 waves per row provides higher memory request concurrency.
  if (K >= 8192) {
    return GemvStrategy::kBaselineBlock;
  }
  // For large output dimensions (e.g. LM head M >= 16384), Wave32 single-row
  // eliminates 150K+ LDS allocations and block synchronization barriers.
  return GemvStrategy::kWave32SingleRow;
}

}  // namespace strix::hip::detail

#endif  // STRIX_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_
