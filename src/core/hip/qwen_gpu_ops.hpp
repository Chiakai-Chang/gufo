#ifndef STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_
#define STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace strix::hip {

struct HipblasLtDispatchInfo {
  int algorithm_id{-1};
  std::string solution_name;
  std::string kernel_name;
  bool plan_cache_hit{false};
};

/// Cached hipBLASLt BF16 GEMM plans for prompt-processing projections.
class HipblasLtGemm {
public:
  HipblasLtGemm();
  ~HipblasLtGemm();

  HipblasLtGemm(const HipblasLtGemm&) = delete;
  HipblasLtGemm& operator=(const HipblasLtGemm&) = delete;
  HipblasLtGemm(HipblasLtGemm&&) noexcept;
  HipblasLtGemm& operator=(HipblasLtGemm&&) noexcept;

  /// Computes Y[B, M] = X_bf16[B, K] * A_bf16[M, K]^T.
  /// Returns false when hipBLASLt cannot provide a supported plan.
  [[nodiscard]] bool RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                             std::size_t batch_size, std::size_t m,
                             std::size_t k, hipStream_t stream = nullptr,
                             HipblasLtDispatchInfo* dispatch_info = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Asynchronously copies embedding row for token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, bool is_bf16,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Asynchronously copies embedding row for *d_token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, bool is_bf16,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes per-head RMSNorm across num_heads
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps = 1e-6F, hipStream_t stream = nullptr);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// De-interleaves [Q0 (head_dim), Gate0 (head_dim), Q1, Gate1, ...] into
/// separate Q and Gate buffers
void LaunchUnpackQG(const float* qg_interleaved, float* q_out, float* gate_out,
                    std::uint32_t num_heads, std::uint32_t head_dim,
                    hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) on Q and K heads
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) reading position from device
/// memory
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                float rope_theta, hipStream_t stream = nullptr);

/// Computes SwiGLU: out = SiLU(gate) * up
void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream = nullptr);

/// Computes Matrix-Vector Multiplication: y = A * x
/// Supports both F32 and BF16 weights A
void LaunchGEMV(const void* A, bool is_bf16, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream = nullptr);

/// Computes Fused SSM Input Projections (QKV, Gate, Alpha, Beta) in a single
/// kernel
void LaunchFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* x, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t hidden_size,
    std::size_t qkv_size, std::size_t inner_size, std::size_t time_step_rank,
    hipStream_t stream = nullptr);

/// Computes Fused QKV Projections for Full Attention layers in a single kernel
void LaunchFusedQKVProjections(const void* q_w, bool q_is_bf16, const void* k_w,
                               bool k_is_bf16, const void* v_w, bool v_is_bf16,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream = nullptr);

/// Computes Fused SwiGLU GEMV: out = SiLU(W_gate * x) * (W_up * x)
void LaunchFusedSwiGLUGEMV(const void* gate_w, bool gate_is_bf16,
                           const void* up_w, bool up_is_bf16, const float* x,
                           float* out, std::size_t intermediate_size,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention with KV-cache and optional gating
/// on GPU (maintaining both FP32 and FP16 cache representations)
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr,
                     float* split_k_scratch = nullptr);

/// Computes Grouped-Query Softmax Attention reading position from device memory
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr);

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream = nullptr);

/// Computes parallel GPU argmax reduction over logits
void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream = nullptr);

// =========================================================================
// Batched GPU Operators for High-Throughput Prompt Processing (Prefill)
// =========================================================================

/// Batched Embedding lookup for B tokens
void LaunchBatchedEmbeddingLookup(const void* table, bool is_bf16,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
                          hipStream_t stream = nullptr);

/// Batched Per-Head RMSNorm across B tokens
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Batched Residual Add across B tokens
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream = nullptr);

/// Batched Unpack Q and Gate across B tokens
void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream = nullptr);

/// Batched RoPE across B tokens starting at pos
void LaunchBatchedRoPE(float* q, float* k, std::size_t batch_size,
                       std::uint32_t num_heads, std::uint32_t num_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rotary_dim,
                       std::uint32_t start_pos, float rope_theta,
                       hipStream_t stream = nullptr);

/// Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T
void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream = nullptr);

/// Converts float buffer to bfloat16 buffer on GPU
void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream = nullptr);

/// Hardware-accelerated Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T using
/// hipBLAS
void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream = nullptr);

/// Direct BF16 GEMM without input conversion: Y[B, M] = X_bf16[B, K] *
/// A_bf16[M, K]^T
void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream = nullptr);

/// Batched SwiGLU activation: out = (gate * sigmoid(gate)) * up (optional BF16
/// output)
void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream = nullptr);

/// Batched Fused SSM Input Projections across B tokens
void LaunchBatchedFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* X, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Batched Fused QKV Projections across B tokens
void LaunchBatchedFusedQKVProjections(
    const void* q_w, bool q_is_bf16, const void* k_w, bool k_is_bf16,
    const void* v_w, bool v_is_bf16, const float* X, float* q_out, float* k_out,
    float* v_out, std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size, hipStream_t stream = nullptr);

/// Batched Fused SwiGLU GEMM: Out[B, intermediate] = SiLU(X[B, K] * W_gate^T) *
/// (X[B, K] * W_up^T)
void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched Causal Attention for B tokens with KV Cache
void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            void* k_cache_f16, void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream = nullptr);

/// Qwen3.8-specific causal GQA tile for gfx1151. The kernel processes 16 query
/// positions and two query heads per block while reusing one FP16 K/V tile.
/// Returns false for unsupported model shapes.
[[nodiscard]] bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Causal GQA through ROCm Composable Kernel. Inputs and outputs remain FP32
/// at the executor boundary; the fused attention operator uses FP16 tiles with
/// FP32 accumulation and online softmax. Returns false for unsupported shapes.
[[nodiscard]] bool LaunchBatchedAttentionCk(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    void* scratch_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Large-batch causal attention using float32 QK/PV GEMMs and one reusable
/// [batch, context] score buffer.
void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* scores, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Batched Causal SSM Conv1D + DeltaNet Recurrence for B tokens
void LaunchBatchedSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_CORE_HIP_QWEN_GPU_OPS_HPP_
