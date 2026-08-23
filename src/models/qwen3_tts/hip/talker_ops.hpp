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

}  // namespace strix::models::qwen3_tts::hip
#endif

#endif  // STRIX_MODELS_QWEN3_TTS_HIP_TALKER_OPS_HPP_
