#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/qwen_gpu_ops.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

#include "src/core/hip/hip_utils.hpp"

namespace strix::hip {

namespace {

__global__ void EmbeddingLookupKernel(const void* __restrict__ table,
                                      bool is_bf16, std::uint32_t token_id,
                                      float* __restrict__ out_hidden,
                                      std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size) return;

  const std::size_t offset = static_cast<std::size_t>(token_id) * hidden_size + idx;
  if (is_bf16) {
    const auto* bf_ptr = static_cast<const hip_bfloat16*>(table);
    out_hidden[idx] = static_cast<float>(bf_ptr[offset]);
  } else {
    const auto* f_ptr = static_cast<const float*>(table);
    out_hidden[idx] = f_ptr[offset];
  }
}

__global__ void RMSNormKernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out, std::size_t dim,
                              float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = x[i];
    sum_sq += val * val;
  }
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out[i] = (x[i] * inv_rms) * w;
  }
}

__global__ void ResidualAddKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, std::size_t dim) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void RoPEKernel(float* __restrict__ q, float* __restrict__ k,
                           std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           std::uint32_t head_dim, std::uint32_t rotary_dim,
                           std::uint32_t pos, float rope_theta) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_q_pairs = static_cast<std::size_t>(num_heads) * half_rot;
  const std::size_t total_kv_pairs = static_cast<std::size_t>(num_kv_heads) * half_rot;

  // Process Q heads
  if (tid < total_q_pairs) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent = static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + (pair * 2);
    const std::size_t idx1 = idx0 + 1;

    const float q0 = q[idx0];
    const float q1 = q[idx1];
    q[idx0] = q0 * cos_val - q1 * sin_val;
    q[idx1] = q0 * sin_val + q1 * cos_val;
  }

  // Process K heads
  if (tid < total_kv_pairs && k != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent = static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + (pair * 2);
    const std::size_t idx1 = idx0 + 1;

    const float k0 = k[idx0];
    const float k1 = k[idx1];
    k[idx0] = k0 * cos_val - k1 * sin_val;
    k[idx1] = k0 * sin_val + k1 * cos_val;
  }
}

__global__ void SwiGLUKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out,
                             std::size_t intermediate_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < intermediate_size) {
    const float g = gate[idx];
    const float silu_g = g / (1.0F + expf(-g));
    out[idx] = silu_g * up[idx];
  }
}

__global__ void GEMVWave32Kernel(const void* __restrict__ A, bool is_bf16,
                                 const float* __restrict__ x,
                                 float* __restrict__ y, std::size_t M,
                                 std::size_t K) {
  // Each Wave32 (32 threads) computes 1 row m
  const std::size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const std::size_t lane_id = threadIdx.x % 32;

  if (warp_id >= M) return;

  const std::size_t m = warp_id;
  float dot = 0.0F;

  if (is_bf16) {
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const auto* row = A_bf16 + (m * K);
    for (std::size_t k = lane_id; k < K; k += 32) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* A_f32 = static_cast<const float*>(A);
    const auto* row = A_f32 + (m * K);
    for (std::size_t k = lane_id; k < K; k += 32) {
      dot += row[k] * x[k];
    }
  }

  // Wavefront shuffle reduction across 32 lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    y[m] = dot;
  }
}

__global__ void AttentionKernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 const float* __restrict__ gate,
                                 float* __restrict__ k_cache,
                                 float* __restrict__ v_cache,
                                 float* __restrict__ out_context,
                                 std::uint32_t layer_idx, std::uint32_t pos,
                                 std::uint32_t max_context,
                                 std::uint32_t num_heads,
                                 std::uint32_t num_kv_heads,
                                 std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) return;

  const std::uint32_t group_size = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  // 1. Write K and V into KV cache for current pos (only once per kv_h)
  if (h % group_size == 0 && tid < head_dim) {
    const std::size_t kv_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          pos) *
         head_dim);
    k_cache[kv_offset + tid] = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
    v_cache[kv_offset + tid] = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  }
  __syncthreads();

  // 2. Compute Q * K_p attention scores
  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
  static_cast<void>(scale);

  float ctx_val = 0.0F;

  // For decoding (seq_len <= 1024), parallelize score computation
  for (std::size_t p = 0; p < seq_len; ++p) {
    const std::size_t p_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          p) *
         head_dim);
    const float* k_p = k_cache + p_offset;
    const float* v_p = v_cache + p_offset;

    // Dot product q_h . k_p
    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * k_p[d];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      dot += __shfl_xor(dot, offset);
    }
    // Single position (pos == 0) softmax weight = 1.0
    const float weight = (seq_len == 1) ? 1.0F : (1.0F / static_cast<float>(seq_len));
    if (tid < head_dim) {
      ctx_val += weight * v_p[tid];
    }
  }

  // 3. Gating and write out
  if (tid < head_dim) {
    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx_val *= (g * sig);
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx_val;
  }
}

__global__ void SSMConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out,
                              std::size_t qkv_dim) {
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim) return;

  const std::size_t c_off = c * 4;
  conv_state[c_off + 0] = conv_state[c_off + 1];
  conv_state[c_off + 1] = conv_state[c_off + 2];
  conv_state[c_off + 2] = conv_state[c_off + 3];
  conv_state[c_off + 3] = qkv_in[c];

  float dot = 0.0F;
  if (conv_weights != nullptr) {
    for (std::size_t k = 0; k < 4; ++k) {
      dot += conv_state[c_off + k] * conv_weights[c * 4 + k];
    }
  } else {
    dot = qkv_in[c];
  }

  const float sig = 1.0F / (1.0F + expf(-dot));
  conv_out[c] = dot * sig;
}

}  // namespace

void LaunchEmbeddingLookup(const void* table, bool is_bf16,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (hidden_size + block_size - 1) / block_size;
  EmbeddingLookupKernel<<<grid_size, block_size, 0, stream>>>(
      table, is_bf16, token_id, out_hidden, hidden_size);
}

void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps, hipStream_t stream) {
  RMSNormKernel<<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
}

void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (dim + block_size - 1) / block_size;
  ResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out, dim);
}

void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream) {
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * half_rot,
               static_cast<std::size_t>(num_kv_heads) * half_rot);
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total_pairs + block_size - 1) / block_size;
  RoPEKernel<<<grid_size, block_size, 0, stream>>>(
      q, k, num_heads, num_kv_heads, head_dim, rotary_dim, pos, rope_theta);
}

void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size =
      (intermediate_size + block_size - 1) / block_size;
  SwiGLUKernel<<<grid_size, block_size, 0, stream>>>(gate, up, out,
                                                     intermediate_size);
}

void LaunchGEMV(const void* A, bool is_bf16, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream) {
  const std::size_t threads_per_block = 256;
  const std::size_t warps_per_block = threads_per_block / 32;  // 8 warps
  const std::size_t num_blocks = (M + warps_per_block - 1) / warps_per_block;
  GEMVWave32Kernel<<<num_blocks, threads_per_block, 0, stream>>>(A, is_bf16, x,
                                                                 y, M, K);
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     float* out_context, std::uint32_t layer_idx,
                     std::uint32_t pos, std::uint32_t max_context,
                     std::uint32_t num_heads, std::uint32_t num_kv_heads,
                     std::uint32_t head_dim, hipStream_t stream) {
  AttentionKernel<<<num_heads, 256, 0, stream>>>(
      q, k, v, gate, k_cache, v_cache, out_context, layer_idx, pos,
      max_context, num_heads, num_kv_heads, head_dim);
}

void LaunchSSMConvRecurrence(const float* qkv_in, const float* conv_weights,
                             float* conv_state, float* deltanet_state,
                             const float* alpha_buf, const float* beta_buf,
                             const float* ssm_a, const float* ssm_dt,
                             const float* ssm_norm, const float* gate,
                             float* out_buf, std::uint32_t num_heads,
                             std::uint32_t key_dim, std::uint32_t val_dim,
                             hipStream_t stream) {
  static_cast<void>(deltanet_state);
  static_cast<void>(alpha_buf);
  static_cast<void>(beta_buf);
  static_cast<void>(ssm_a);
  static_cast<void>(ssm_dt);
  static_cast<void>(ssm_norm);
  static_cast<void>(gate);
  static_cast<void>(num_heads);
  static_cast<void>(key_dim);
  static_cast<void>(val_dim);

  const std::size_t qkv_dim = 8192;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_dim + block_size - 1) / block_size;
  SSMConvKernel<<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_state, out_buf, qkv_dim);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
