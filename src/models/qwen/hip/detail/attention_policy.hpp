#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionQueryHeads{24};
inline constexpr std::uint32_t kTiledAttentionKvHeads{4};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};
inline constexpr std::uint32_t kCkAttentionKvHeads{4};
inline constexpr std::uint32_t kCkAttentionHeadDim{256};
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{4096};
inline constexpr std::uint32_t kSplitKDecodeAttentionMaxSplits{32};
inline constexpr std::uint32_t kFusedQkNormMaxHeadDim{256};

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

// opt-c010-qk-rope-kv: fuse per-head Q/K RMSNorm, RoPE, and the KV-cache write
// into a single kernel per token (decode) / per token row (prefill). Flip to
// false to revert to the unfused chain (PerHeadRMSNorm x2 + RoPE +
// WriteKVCache*), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_qk_norm_rope_kv;
}

[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite() noexcept {
  return ShouldFuseQKNormRoPEKvWrite(QwenExecutionPolicy::Production());
}

// opt-c010-residual-rmsnorm: fuse the post-attention residual add with the
// subsequent FFN RMSNorm into a single kernel per token (decode) / per token
// row (prefill). Flip to false to revert to the unfused chain
// (ResidualAdd + RMSNorm), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseResidualAddRMSNorm(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_residual_rmsnorm;
}

[[nodiscard]] constexpr bool ShouldFuseResidualAddRMSNorm() noexcept {
  return ShouldFuseResidualAddRMSNorm(QwenExecutionPolicy::Production());
}

// opt-c010-ffn-swiglu: fuse the FFN gate/up projections with the SwiGLU
// activation into a single batched kernel for prefill. Rejected on gfx1151:
// the naive per-row fused kernel regresses prefill ~37x (pp1024 9.70 vs 362.87
// tok/s) against the hipBLASLt BF16 gate/up GEMMs plus SwiGLU activation, so
// the unfused chain is the production route. Kept at false for re-evaluation.
[[nodiscard]] constexpr bool ShouldFuseFFNSwiGLU(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_prefill_ffn_swiglu;
}

[[nodiscard]] constexpr bool ShouldFuseFFNSwiGLU() noexcept {
  return ShouldFuseFFNSwiGLU(QwenExecutionPolicy::Production());
}

// opt-c010-ssm-gate-residual: fuse the SSM per-head post-RMSNorm + SiLU gate
// into the DeltaNet recurrence epilogue (prefill, replacing
// BatchedSSMPostNormGateKernel) and fold the post-SSM residual add into the
// ssm_out GEMV (decode). Flip to false to revert to the unfused chain
// (recurrence + post-norm kernel; ssm_out GEMV + residual add), which stays
// wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseDecodeSSMOutputResidual(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_decode_ssm_output_residual;
}

[[nodiscard]] constexpr bool ShouldFusePrefillSSMPostNormGate(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_prefill_ssm_post_norm_gate;
}

[[nodiscard]] constexpr bool ShouldFuseSSMGateResidual() noexcept {
  const auto policy = QwenExecutionPolicy::Production();
  return ShouldFuseDecodeSSMOutputResidual(policy) &&
         ShouldFusePrefillSSMPostNormGate(policy);
}

// opt-c010-rmsnorm-projection: fuse the layer pre-RMSNorm into the fused
// decode projection GEMVs (QKV, SSM input, and FFN SwiGLU), so the projection
// kernel reads the raw hidden row and prepares its own normed input. Flip to
// false to revert to the unfused chain (RMSNormKernel + projection kernel),
// which stays wired as the independent reference. Prefill is unaffected: its
// batched norm kernel already fuses the FP32 + BF16 input preparation.
[[nodiscard]] constexpr bool ShouldFuseRMSNormProjection(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_decode_rmsnorm_projection;
}

[[nodiscard]] constexpr bool ShouldFuseRMSNormProjection() noexcept {
  return ShouldFuseRMSNormProjection(QwenExecutionPolicy::Production());
}

// opt-c014-layer-prefetch: issue an asynchronous GPU touch of the next layer's
// weight pages on a side stream while the current layer executes, so the next
// layer's projection kernels do not stall on first-touch page walks. Flip to
// false to revert to the no-prefetch route, which stays wired as the
// independent reference.
[[nodiscard]] constexpr bool ShouldPrefetchNextLayer(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.prefetch_next_layer;
}

[[nodiscard]] constexpr bool ShouldPrefetchNextLayer() noexcept {
  return ShouldPrefetchNextLayer(QwenExecutionPolicy::Production());
}

// opt-q4kxl-rows3: three rows amortizes the exact small-batch K-quant
// verifier's shared activation tile over 1.5x as many output rows and improves
// Q4 DFlash-2 end to end. Q6_K stays on two rows because its extra per-half
// scale state spills with three; four rows remains a rejected, measurable
// alternative. `2` pins the independent production reference.
enum class KQuantSmallBatchRows : std::uint8_t {
  kTwo = 2,
  kThree = 3,
  kFour = 4,
};

[[nodiscard]] inline KQuantSmallBatchRows ResolveKQuantSmallBatchRows(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantSmallBatchRows::kThree;
  }
  const std::string_view text{value};
  if (text == "3" || text == "three" || text == "rows3") {
    return KQuantSmallBatchRows::kThree;
  }
  return text == "4" || text == "four" || text == "rows4"
             ? KQuantSmallBatchRows::kFour
             : KQuantSmallBatchRows::kTwo;
}

[[nodiscard]] inline KQuantSmallBatchRows KQuantSmallBatchRowsFromEnv() {
  static const KQuantSmallBatchRows rows =
      ResolveKQuantSmallBatchRows(std::getenv("GUFO_KQUANT_SMALL_BATCH_ROWS"));
  return rows;
}

[[nodiscard]] constexpr std::uint32_t SelectDecodeAttentionSplitCount(
    std::size_t sequence_length) noexcept {
  if (sequence_length < kSplitKDecodeAttentionMinContext) {
    return 1;
  }
  return kSplitKDecodeAttentionMaxSplits;
}

[[nodiscard]] constexpr bool IsFusedQkNormSupported(
    std::uint32_t head_dim) noexcept {
  return head_dim != 0 && head_dim <= kFusedQkNormMaxHeadDim;
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

/// Prefill DeltaNet recurrence backend preference. The production route is the
/// row-split kernel (opt-c170-deltanet-rowsplit);
/// `GUFO_SSM_RECURRENCE=baseline` pins the previous single-block kernel so it
/// stays measurable in the same binary rather than only being reachable when a
/// shape is unsupported.
enum class SsmRecurrencePreference : std::uint8_t {
  kAuto,
  kRowSplit,
  kBaseline,
};

[[nodiscard]] inline SsmRecurrencePreference ResolveSsmRecurrencePreference(
    const char* value) noexcept {
  if (value == nullptr) {
    return SsmRecurrencePreference::kAuto;
  }
  const std::string_view text{value};
  if (text == "rowsplit" || text == "row_split") {
    return SsmRecurrencePreference::kRowSplit;
  }
  if (text == "baseline") {
    return SsmRecurrencePreference::kBaseline;
  }
  return SsmRecurrencePreference::kAuto;
}

[[nodiscard]] inline SsmRecurrencePreference SsmRecurrencePreferenceFromEnv() {
  static const SsmRecurrencePreference preference =
      ResolveSsmRecurrencePreference(std::getenv("GUFO_SSM_RECURRENCE"));
  return preference;
}

/// The row-split recurrence needs its two prologue scratch planes and the
/// 128 x 128 per-head state tile the kernel's register geometry is built for.
[[nodiscard]] inline bool ShouldUseSsmRowSplitRecurrence(bool shape_supported,
                                                         bool scratch_ready) {
  const SsmRecurrencePreference preference = SsmRecurrencePreferenceFromEnv();
  if (preference == SsmRecurrencePreference::kBaseline) {
    return false;
  }
  return shape_supported && scratch_ready;
}

/// opt-c177-attn-wmma: the masked WMMA kernel covers the whole visible range in
/// one pass and beats the tiled kernel by 3.5x at depth 0 and the tiled
/// diagonal plus AOTriton prefix by 1.3-1.4x at depth, so it is the production
/// route and the split path is no longer used. `GUFO_PREFILL_ATTENTION=split`
/// (or `tile`, `ck`, `baseline`) pins an alternative for comparison.
[[nodiscard]] inline bool ShouldUseWmmaPrefillAttention();

/// Prefill attention backend preference. The fallback order behind the WMMA
/// kernel is tiled, then Composable Kernel, then baseline;
/// `GUFO_PREFILL_ATTENTION` pins one backend so the alternatives stay
/// measurable in the same binary instead of only being reachable when the
/// preferred one rejects a shape.
enum class PrefillAttentionPreference : std::uint8_t {
  kAuto,
  kWmma,
  kTiled,
  kComposableKernel,
  kBaseline,
  kSplit,
};

[[nodiscard]] inline PrefillAttentionPreference
ResolvePrefillAttentionPreference(const char* value) noexcept {
  if (value == nullptr) {
    return PrefillAttentionPreference::kAuto;
  }
  const std::string_view text{value};
  if (text == "wmma") {
    return PrefillAttentionPreference::kWmma;
  }
  if (text == "tile" || text == "tiled") {
    return PrefillAttentionPreference::kTiled;
  }
  if (text == "ck" || text == "composable_kernel") {
    return PrefillAttentionPreference::kComposableKernel;
  }
  if (text == "baseline") {
    return PrefillAttentionPreference::kBaseline;
  }
  if (text == "split") {
    return PrefillAttentionPreference::kSplit;
  }
  return PrefillAttentionPreference::kAuto;
}

[[nodiscard]] inline PrefillAttentionPreference
PrefillAttentionPreferenceFromEnv() {
  static const PrefillAttentionPreference preference =
      ResolvePrefillAttentionPreference(std::getenv("GUFO_PREFILL_ATTENTION"));
  return preference;
}

[[nodiscard]] inline bool ShouldUseWmmaPrefillAttention() {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  return preference == PrefillAttentionPreference::kAuto ||
         preference == PrefillAttentionPreference::kWmma;
}

inline constexpr std::size_t kPrefillAttentionSplitMinPrefix{1024};

/// The split prefill attention (AOTriton non-causal prefix plus the tiled
/// causal diagonal) only pays off once the prefix is a meaningful share of the
/// work: it costs one extra kernel plus an FP16 query conversion and a merge
/// pass. Below the threshold, or with no prefix at all, the unsplit chain wins.
[[nodiscard]] constexpr bool IsPrefillAttentionSplitProfitable(
    std::uint32_t start_pos, std::size_t batch_size) noexcept {
  return start_pos >= kPrefillAttentionSplitMinPrefix && batch_size >= 64;
}

[[nodiscard]] inline bool ShouldUsePrefillAttentionSplit(
    std::uint32_t start_pos, std::size_t batch_size) {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  if (preference == PrefillAttentionPreference::kSplit) {
    return start_pos > 0 && batch_size > 0;
  }
  // Not auto-selected any more: the WMMA kernel covers the same work in one
  // pass and is faster at every depth.
  return false;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then Composable Kernel, then the baseline implementation.
template<typename TiledLauncher, typename CkLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     CkLauncher&& launch_ck,
                                     BaselineLauncher&& launch_baseline) {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  if (preference != PrefillAttentionPreference::kBaseline &&
      ShouldAttemptOptimizedAttention(visible_context)) {
    if (preference != PrefillAttentionPreference::kComposableKernel &&
        std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
    if (preference != PrefillAttentionPreference::kTiled &&
        std::forward<CkLauncher>(launch_ck)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
