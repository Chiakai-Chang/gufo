#ifndef STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_
#define STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

/// Asynchronously copies embedding row for token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, bool is_bf16,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream = nullptr);

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) on Q and K heads
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream = nullptr);

/// Computes SwiGLU: out = SiLU(gate) * up
void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream = nullptr);

/// Computes Matrix-Vector Multiplication: y = A * x
/// Supports both F32 and BF16 weights A
void LaunchGEMV(const void* A, bool is_bf16, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention with KV-cache and optional gating on GPU
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     float* out_context, std::uint32_t layer_idx,
                     std::uint32_t pos, std::uint32_t max_context,
                     std::uint32_t num_heads, std::uint32_t num_kv_heads,
                     std::uint32_t head_dim, hipStream_t stream = nullptr);

/// Computes Gated DeltaNet 1D Conv and Recurrence on GPU
void LaunchSSMConvRecurrence(const float* qkv_in, const float* conv_weights,
                             float* conv_state, float* deltanet_state,
                             const float* alpha_buf, const float* beta_buf,
                             const float* ssm_a, const float* ssm_dt,
                             const float* ssm_norm, const float* gate,
                             float* out_buf, std::uint32_t num_heads,
                             std::uint32_t key_dim, std::uint32_t val_dim,
                             hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_
