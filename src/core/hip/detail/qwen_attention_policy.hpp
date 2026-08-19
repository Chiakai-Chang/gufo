#ifndef STRIX_CORE_HIP_DETAIL_QWEN_ATTENTION_POLICY_HPP_
#define STRIX_CORE_HIP_DETAIL_QWEN_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>

namespace strix::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionQueryHeads{24};
inline constexpr std::uint32_t kTiledAttentionKvHeads{4};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};
inline constexpr std::uint32_t kCkAttentionKvHeads{4};
inline constexpr std::uint32_t kCkAttentionHeadDim{256};
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{4096};
inline constexpr std::uint32_t kSplitKDecodeAttentionMaxSplits{32};

struct AttentionSupportParams {
  std::size_t batch_size{0};
  std::uint32_t start_pos{0};
  std::uint32_t max_context{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  bool has_k_cache_f16{false};
  bool has_v_cache_f16{false};
  bool has_scratch_f16{false};
};

[[nodiscard]] constexpr bool ShouldAttemptOptimizedAttention(
    std::size_t visible_context) noexcept {
  return visible_context >= kOptimizedAttentionMinBatch;
}

[[nodiscard]] constexpr std::uint32_t SelectDecodeAttentionSplitCount(
    std::size_t sequence_length) noexcept {
  if (sequence_length < kSplitKDecodeAttentionMinContext) {
    return 1;
  }
  return kSplitKDecodeAttentionMaxSplits;
}

[[nodiscard]] constexpr bool IsSplitKDecodeAttentionSupported(
    std::size_t sequence_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  return SelectDecodeAttentionSplitCount(sequence_length) > 1 &&
         num_heads != 0 && num_kv_heads != 0 &&
         (num_heads % num_kv_heads) == 0 && head_dim == 256;
}

[[nodiscard]] constexpr std::size_t DecodeAttentionScratchElements(
    std::uint32_t num_heads, std::uint32_t head_dim) noexcept {
  return static_cast<std::size_t>(num_heads) * kSplitKDecodeAttentionMaxSplits *
         (static_cast<std::size_t>(head_dim) + 2);
}

[[nodiscard]] constexpr bool IsTiledAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 &&
         params.num_heads == kTiledAttentionQueryHeads &&
         params.num_kv_heads == kTiledAttentionKvHeads &&
         params.head_dim == kTiledAttentionHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         params.has_k_cache_f16 && params.has_v_cache_f16;
}

[[nodiscard]] constexpr bool IsCkAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 && params.start_pos == 0 &&
         params.num_kv_heads == kCkAttentionKvHeads &&
         params.num_heads % kCkAttentionKvHeads == 0 &&
         params.head_dim == kCkAttentionHeadDim &&
         params.batch_size <= params.max_context && params.has_k_cache_f16 &&
         params.has_v_cache_f16 && params.has_scratch_f16;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then Composable Kernel, then the baseline implementation.
template<typename TiledLauncher, typename CkLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     CkLauncher&& launch_ck,
                                     BaselineLauncher&& launch_baseline) {
  if (ShouldAttemptOptimizedAttention(visible_context)) {
    if (std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
    if (std::forward<CkLauncher>(launch_ck)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace strix::hip::detail

#endif  // STRIX_CORE_HIP_DETAIL_QWEN_ATTENTION_POLICY_HPP_
