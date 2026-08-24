#ifndef STRIX_MODELS_QWEN3_TTS_HIP_TALKER_OPS_HPP_
#define STRIX_MODELS_QWEN3_TTS_HIP_TALKER_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::models::qwen3_tts::hip {

void LaunchBfloat16PerHeadRMSNorm(float* values, const float* weight,
                                  std::size_t batch_size,
                                  std::uint32_t num_heads,
                                  std::uint32_t head_dim, float epsilon,
                                  hipStream_t stream);

void LaunchBfloat16RoPEAndRoundV(float* query, float* key, float* value,
                                 std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t num_key_value_heads,
                                 std::uint32_t head_dim,
                                 std::uint32_t start_position, float rope_theta,
                                 hipStream_t stream);

/// Fuses the q/k per-head RMSNorm, the q/k rotation and the V rounding into one
/// launch. Equivalent to the standalone per-head RMSNorm and RoPE kernels,
/// including every intermediate BF16 rounding.
///
/// Returns false without launching when `head_dim` exceeds one workgroup, so
/// the caller can fall back to the separate launches.
[[nodiscard]] bool LaunchBfloat16QkNormRoPE(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    hipStream_t stream);

/// Adds `update` into `hidden` and writes the RMS-normalized BF16 result, in
/// one launch. Matches the standalone residual-add and RMSNorm kernels bit for
/// bit.
void LaunchBfloat16ResidualAddRMSNorm(float* hidden, const float* update,
                                      const float* weight,
                                      void* output_bfloat16,
                                      std::size_t batch_size,
                                      std::size_t dimension, float epsilon,
                                      hipStream_t stream);

void LaunchRoundBfloat16InPlace(float* values, std::size_t count,
                                hipStream_t stream);

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t count, hipStream_t stream);

void LaunchBfloat16ResidualAdd(const float* residual, const float* update,
                               float* output, std::size_t count,
                               hipStream_t stream);

void LaunchBfloat16SwiGlu(const float* gate, const float* up, float* output,
                          void* output_bfloat16, std::size_t count,
                          hipStream_t stream);

void LaunchBfloat16BiasSilu(const float* input, const void* bias_bfloat16,
                            float* output, void* output_bfloat16,
                            std::size_t rows, std::size_t columns,
                            hipStream_t stream);

void LaunchBfloat16Bias(const float* input, const void* bias_bfloat16,
                        float* output, std::size_t rows, std::size_t columns,
                        hipStream_t stream);

void LaunchSumCodecEmbeddings(const float* embeddings,
                              const float* text_embedding, float* output,
                              std::size_t groups, std::size_t hidden_size,
                              hipStream_t stream);

/// Computes `output[(b * rows) + m] = sum_k weights[(m * columns) + k] *
/// inputs[(b * columns) + k]` for the decode-time projections, accumulating in
/// float32. Matches the row-major weight layout and column-major output of the
/// batched BF16 GEMM.
///
/// One workgroup reduces one output row, so every row issues eight
/// independently scheduled coalesced streams and the projection reaches the
/// gfx1151 DRAM roofline instead of the tile-shaped batched path. All `batch`
/// columns share a single pass over the weights. The reduction order is fixed,
/// which keeps the result reproducible.
///
/// Returns false without launching when the batch or layout is not supported,
/// so the caller can fall back to the batched GEMM.
[[nodiscard]] bool LaunchBfloat16Gemv(const void* weights_bfloat16,
                                      const void* inputs_bfloat16,
                                      float* output, std::size_t batch,
                                      std::size_t rows, std::size_t columns,
                                      hipStream_t stream);

}  // namespace strix::models::qwen3_tts::hip
#endif

#endif  // STRIX_MODELS_QWEN3_TTS_HIP_TALKER_OPS_HPP_
