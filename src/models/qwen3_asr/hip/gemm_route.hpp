#ifndef GUFO_MODELS_QWEN3_ASR_HIP_GEMM_ROUTE_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_GEMM_ROUTE_HPP_

#include <cstdlib>
#include <string_view>

namespace gufo::models::qwen3_asr::hip {

/// Whether the multi-token projections may use hipBLASLt.
///
/// Both the audio tower and the text prefill multiply a transposed BF16 weight
/// by BF16 activations into a float32 result. rocBLAS has no WMMA kernel for
/// that combination, so hipBLAS resolves these to a scalar MT64x32x8 Tensile
/// kernel at roughly 4 TFLOPS against a 55 TFLOPS WMMA ceiling. hipBLASLt
/// reaches 13-19 TFLOPS on the same shapes -- see
/// tools/bench/bf16_gemm_bench.hip at batch 211 -- so hipBLASLt is tried first
/// and hipBLAS is retained as the fallback whenever no plan is available.
///
/// `GUFO_QWEN3_ASR_PREFILL_GEMM=hipblas` (or `0`) forces the fallback, which is
/// how the two routes are A/B compared.
[[nodiscard]] inline bool UsePrefillHipblasLt() {
  const char* value = std::getenv("GUFO_QWEN3_ASR_PREFILL_GEMM");
  if (value == nullptr) {
    return true;
  }
  const std::string_view mode(value);
  return mode != "0" && mode != "hipblas";
}

}  // namespace gufo::models::qwen3_asr::hip

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_GEMM_ROUTE_HPP_
