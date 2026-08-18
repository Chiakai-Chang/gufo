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

__global__ void UnpackQGKernel(const float* __restrict__ qg_interleaved,
                               float* __restrict__ q_out,
                               float* __restrict__ gate_out,
                               std::uint32_t num_heads,
                               std::uint32_t head_dim) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(num_heads) * head_dim;
  if (tid >= total) return;

  const std::uint32_t h = static_cast<std::uint32_t>(tid / head_dim);
  const std::uint32_t d = static_cast<std::uint32_t>(tid % head_dim);

  const std::size_t src_base = static_cast<std::size_t>(h) * head_dim * 2;
  q_out[tid] = qg_interleaved[src_base + d];
  gate_out[tid] = qg_interleaved[src_base + head_dim + d];
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

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

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

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

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
  extern __shared__ float shx[];
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = (blockIdx.x * blockDim.x + tid) >> 5u;

  for (std::size_t i = tid; i < K; i += blockDim.x) {
    shx[i] = x[i];
  }
  __syncthreads();

  if (warp_id >= M) return;

  const std::size_t m = warp_id;
  float dot = 0.0F;

  if (is_bf16) {
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const auto* row = A_bf16 + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      dot += static_cast<float>(vals[0]) * shx[k + 0];
      dot += static_cast<float>(vals[1]) * shx[k + 1];
      dot += static_cast<float>(vals[2]) * shx[k + 2];
      dot += static_cast<float>(vals[3]) * shx[k + 3];
      dot += static_cast<float>(vals[4]) * shx[k + 4];
      dot += static_cast<float>(vals[5]) * shx[k + 5];
      dot += static_cast<float>(vals[6]) * shx[k + 6];
      dot += static_cast<float>(vals[7]) * shx[k + 7];
    }
    for (std::size_t k = K_vec * 8 + lane_id; k < K; k += 32) {
      dot += static_cast<float>(row[k]) * shx[k];
    }
  } else {
    const auto* A_f32 = static_cast<const float*>(A);
    const auto* row = A_f32 + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      dot += v.x * shx[k + 0];
      dot += v.y * shx[k + 1];
      dot += v.z * shx[k + 2];
      dot += v.w * shx[k + 3];
    }
    for (std::size_t k = K_vec * 4 + lane_id; k < K; k += 32) {
      dot += row[k] * shx[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    y[m] = dot;
  }
}

__global__ void FusedSwiGLUGEMVWave32Kernel(
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ up_w, bool up_is_bf16,
    const float* __restrict__ x, float* __restrict__ out,
    std::size_t M, std::size_t K) {
  extern __shared__ float shx[];
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = (blockIdx.x * blockDim.x + tid) >> 5u;

  for (std::size_t i = tid; i < K; i += blockDim.x) {
    shx[i] = x[i];
  }
  __syncthreads();

  if (warp_id >= M) return;

  const std::size_t m = warp_id;
  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_is_bf16 && up_is_bf16) {
    const auto* g_bf16 = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
    const auto* u_bf16 = static_cast<const hip_bfloat16*>(up_w) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_bf16);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_bf16);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;

      dot_gate += static_cast<float>(g_vals[0]) * shx[k + 0];
      dot_up   += static_cast<float>(u_vals[0]) * shx[k + 0];

      dot_gate += static_cast<float>(g_vals[1]) * shx[k + 1];
      dot_up   += static_cast<float>(u_vals[1]) * shx[k + 1];

      dot_gate += static_cast<float>(g_vals[2]) * shx[k + 2];
      dot_up   += static_cast<float>(u_vals[2]) * shx[k + 2];

      dot_gate += static_cast<float>(g_vals[3]) * shx[k + 3];
      dot_up   += static_cast<float>(u_vals[3]) * shx[k + 3];

      dot_gate += static_cast<float>(g_vals[4]) * shx[k + 4];
      dot_up   += static_cast<float>(u_vals[4]) * shx[k + 4];

      dot_gate += static_cast<float>(g_vals[5]) * shx[k + 5];
      dot_up   += static_cast<float>(u_vals[5]) * shx[k + 5];

      dot_gate += static_cast<float>(g_vals[6]) * shx[k + 6];
      dot_up   += static_cast<float>(u_vals[6]) * shx[k + 6];

      dot_gate += static_cast<float>(g_vals[7]) * shx[k + 7];
      dot_up   += static_cast<float>(u_vals[7]) * shx[k + 7];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up   += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

__global__ void PerHeadRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     std::uint32_t num_heads,
                                     std::uint32_t head_dim,
                                     float eps) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) return;

  const float* x_head = x + (static_cast<std::size_t>(h) * head_dim);
  float* out_head = out + (static_cast<std::size_t>(h) * head_dim);

  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
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

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}

__global__ void WriteKVCacheKernel(const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   float* __restrict__ k_cache,
                                   float* __restrict__ v_cache,
                                   std::uint32_t layer_idx,
                                   std::uint32_t pos,
                                   std::uint32_t max_context,
                                   std::uint32_t num_kv_heads,
                                   std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads) return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim) return;

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  k_cache[kv_offset + tid] = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  v_cache[kv_offset + tid] = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
}

__global__ void AttentionKernel(const float* __restrict__ q,
                                 const float* __restrict__ gate,
                                 const float* __restrict__ k_cache,
                                 const float* __restrict__ v_cache,
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

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  extern __shared__ float s_scores[];

  // Step 1: Compute scores for each position p = 0..seq_len-1
  for (std::size_t p = 0; p < seq_len; ++p) {
    const std::size_t p_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          p) *
         head_dim);
    const float* k_p = k_cache + p_offset;

    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * k_p[d];
    }
    __shared__ float s_dot[256];
    s_dot[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_dot[tid] += s_dot[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p] = s_dot[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..seq_len-1]
  __shared__ float s_max[256];
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_max[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_max[0];
  __syncthreads();

  __shared__ float s_sum_exp[256];
  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_sum_exp[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum_exp[tid] += s_sum_exp[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_sum_exp[0] + 1e-9F);
  __syncthreads();

  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    s_scores[p] *= inv_sum_exp;
  }
  __syncthreads();

  // Step 3: Weighted sum of V
  if (tid < head_dim) {
    float ctx = 0.0F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const float* v_p = v_cache + p_offset;
      ctx += s_scores[p] * v_p[tid];
    }

    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx;
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

__global__ void DeltaNetRecurrenceKernel(
    const float* __restrict__ conv_out,
    float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf,
    const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a,
    const float* __restrict__ ssm_dt,
    const float* __restrict__ ssm_norm,
    const float* __restrict__ gate,
    float* __restrict__ out_buf,
    std::uint32_t layer_idx,
    std::uint32_t num_heads,
    std::uint32_t key_dim,
    std::uint32_t val_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) return;

  const std::size_t tid = threadIdx.x;

  const float* q_ptr = conv_out;
  const float* k_ptr = conv_out + (16 * key_dim);
  const float* v_ptr = conv_out + (16 * key_dim * 2);

  const std::uint32_t kh_idx = h / 2;
  const float* q_h = q_ptr + (kh_idx * key_dim);
  const float* k_h = k_ptr + (kh_idx * key_dim);
  const float* v_h = v_ptr + (h * val_dim);
  float* o_h = out_buf + (h * val_dim);

  float* s_matrix = deltanet_state +
      ((static_cast<std::size_t>(layer_idx) * num_heads + h) * key_dim * val_dim);

  // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta = sigmoid(beta)
  const float dt = (ssm_dt != nullptr) ? ssm_dt[h % 32] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h % 32] : -0.05F;
  const float alpha_in = (alpha_buf != nullptr) ? alpha_buf[h] : 0.0F;
  const float beta_in = (beta_buf != nullptr) ? beta_buf[h] : 0.0F;

  const float alpha_biased = alpha_in + dt;
  const float alpha_softplus = (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
  const float alpha_h = expf(alpha_softplus * a_val);
  const float beta_h = 1.0F / (1.0F + expf(-beta_in));

  // 2. Normalize key ||k_h||_2 and query ||q_h||_2
  __shared__ float s_norm[2];
  if (tid == 0) {
    float k_sq = 0.0F;
    float q_sq = 0.0F;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      k_sq += k_h[i] * k_h[i];
      q_sq += q_h[i] * q_h[i];
    }
    s_norm[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    s_norm[1] = 1.0F / sqrtf(q_sq + 1e-6F);
  }
  __syncthreads();
  const float q_scale =
      (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1];
  const float inv_k_norm = s_norm[0];

  // 3. Associative retrieval: u_j = sum_i S[j, i] * (k[i] * inv_k_norm)
  if (tid < val_dim) {
    const std::size_t j = tid;
    float* s_row = s_matrix + (j * key_dim);

    // Decay state
    for (std::size_t i = 0; i < key_dim; ++i) {
      s_row[i] *= alpha_h;
    }

    float u_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      u_j += s_row[i] * (k_h[i] * inv_k_norm);
    }
    const float d_j = (v_h[j] - u_j) * beta_h;

    // Delta update
    for (std::size_t i = 0; i < key_dim; ++i) {
      s_row[i] += d_j * (k_h[i] * inv_k_norm);
    }

    // Query readout
    float o_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      o_j += s_row[i] * (q_h[i] * q_scale);
    }
    o_h[j] = o_j;
  }
  __syncthreads();

  // 4. Per-head RMSNorm on o_h (val_dim = 256 or 128 elements)
  __shared__ float s_sum[256];
  if (tid < val_dim) {
    s_sum[tid] = o_h[tid] * o_h[tid];
  } else {
    s_sum[tid] = 0.0F;
  }
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) s_sum[tid] += s_sum[tid + s];
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(val_dim);
  const float rms = rsqrtf(mean_sq + 1e-6F);
  __syncthreads();

  if (tid < val_dim) {
    const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
    float val = o_h[tid] * rms * w;
    if (gate != nullptr) {
      const float g = gate[h * val_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      val *= (g * sig); // SiLU gating
    }
    o_h[tid] = val;
  }
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

void LaunchUnpackQG(const float* qg_interleaved, float* q_out, float* gate_out,
                    std::uint32_t num_heads, std::uint32_t head_dim,
                    hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total + block_size - 1) / block_size;
  UnpackQGKernel<<<grid_size, block_size, 0, stream>>>(
      qg_interleaved, q_out, gate_out, num_heads, head_dim);
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
  const std::size_t warps_per_block = (M >= 32) ? 32 : M;
  const std::size_t threads_per_block = warps_per_block * 32;
  const std::size_t num_blocks = (M + warps_per_block - 1) / warps_per_block;
  const std::size_t shared_mem_bytes = K * sizeof(float);
  GEMVWave32Kernel<<<num_blocks, threads_per_block, shared_mem_bytes, stream>>>(
      A, is_bf16, x, y, M, K);
}

void LaunchFusedSwiGLUGEMV(const void* gate_w, bool gate_is_bf16,
                           const void* up_w, bool up_is_bf16,
                           const float* x, float* out,
                           std::size_t intermediate_size, std::size_t hidden_size,
                           hipStream_t stream) {
  const std::size_t warps_per_block = (intermediate_size >= 32) ? 32 : intermediate_size;
  const std::size_t threads_per_block = warps_per_block * 32;
  const std::size_t num_blocks =
      (intermediate_size + warps_per_block - 1) / warps_per_block;
  const std::size_t shared_mem_bytes = hidden_size * sizeof(float);
  FusedSwiGLUGEMVWave32Kernel<<<num_blocks, threads_per_block,
                                shared_mem_bytes, stream>>>(
      gate_w, gate_is_bf16, up_w, up_is_bf16, x, out, intermediate_size,
      hidden_size);
}

void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(
      x, weight, out, num_heads, head_dim, eps);
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     float* out_context, std::uint32_t layer_idx,
                     std::uint32_t pos, std::uint32_t max_context,
                     std::uint32_t num_heads, std::uint32_t num_kv_heads,
                     std::uint32_t head_dim, hipStream_t stream) {
  // 1. Write K and V into KV cache across all num_kv_heads
  WriteKVCacheKernel<<<num_kv_heads, 256, 0, stream>>>(
      k, v, k_cache, v_cache, layer_idx, pos, max_context, num_kv_heads,
      head_dim);

  // 2. Launch Attention with exact Softmax and dynamic score buffer
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
  const std::size_t shared_scores_bytes = seq_len * sizeof(float);
  AttentionKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
      q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
      num_heads, num_kv_heads, head_dim);
}

void LaunchSSMConvRecurrence(const float* qkv_in, const float* conv_weights,
                             float* conv_state, float* deltanet_state,
                             const float* alpha_buf, const float* beta_buf,
                             const float* ssm_a, const float* ssm_dt,
                             const float* ssm_norm, const float* gate,
                             float* out_buf, std::uint32_t layer_idx,
                             std::uint32_t num_heads, std::uint32_t key_dim,
                             std::uint32_t val_dim, hipStream_t stream) {
  const std::size_t qkv_dim = 8192;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_dim + block_size - 1) / block_size;
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * 8192 * 4);

  SSMConvKernel<<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_layer_state, out_buf, qkv_dim);

  DeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      out_buf, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm,
      gate, out_buf, layer_idx, num_heads, key_dim, val_dim);
}

__global__ void ArgmaxKernel(const float* __restrict__ logits,
                             std::uint32_t* __restrict__ out_token,
                             std::size_t vocab_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;

  float max_val = -1e30F;
  std::uint32_t max_idx = 0;

  for (std::size_t i = tid; i < vocab_size; i += block_size) {
    const float val = logits[i];
    if (val > max_val) {
      max_val = val;
      max_idx = static_cast<std::uint32_t>(i);
    }
  }

  __shared__ float s_max[256];
  __shared__ std::uint32_t s_idx[256];
  s_max[tid] = max_val;
  s_idx[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      if (s_max[tid + stride] > s_max[tid]) {
        s_max[tid] = s_max[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out_token = s_idx[0];
  }
}

void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream) {
  ArgmaxKernel<<<1, 256, 0, stream>>>(logits, out_token, vocab_size);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
