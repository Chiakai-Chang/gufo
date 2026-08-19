#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_query_attention_forward_wmma.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

namespace {

constexpr std::size_t kHipblasLtMaxAlgorithms = 16;
constexpr std::size_t kHipblasLtWideFfnThreshold = 3;
constexpr std::size_t kHipblasLtLargeBatch = 256;
constexpr std::size_t kHipblasLtLargeWideRank = 4;
constexpr std::size_t kHipblasLtTallRank = 5;
constexpr std::size_t kLargeGemvInput = 16384;
constexpr ck::index_t kCkNumKvHeads = 4;

template<ck::index_t... Values>
using CkSequence = ck::Sequence<Values...>;

using CkPassThrough = ck::tensor_operation::element_wise::PassThrough;
using CkScale = ck::tensor_operation::element_wise::Scale;
using CkGqaOperation =
    ck::tensor_operation::device::DeviceGroupedQueryAttentionForward_Wmma<
        2, 1, 1, 1, 1, ck::half_t, ck::half_t, ck::half_t, float, ck::Tuple<>,
        float, ck::Tuple<>, float, float, CkPassThrough, CkPassThrough, CkScale,
        CkPassThrough, CkPassThrough,
        ck::tensor_operation::device::GemmSpecialization::MNKOPadding,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default, 1,
        kCkNumKvHeads, 256,
        // QK: 128x128 output tile, K tile 64.
        128, 128, 64, 8, 8,
        // PV: output width 64, reduction tile 64.
        64, 64, 8, 16, 16, 16, 1, 8, 4,
        // Q and K transfers.
        CkSequence<2, 128, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>, 2, 8,
        8, true, CkSequence<8, 32, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>,
        2, 8, 8, true,
        // V and output transfers.
        CkSequence<2, 16, 8>, CkSequence<0, 2, 1>, CkSequence<0, 2, 1>, 1, 1, 1,
        false, 1, 1, CkSequence<1, 128, 1, 2>, 8,
        ck::tensor_operation::device::MaskingSpecialization::
            MaskOutUpperTriangle>;

struct HipblasLtPlan {
  std::size_t batch_size{0};
  std::size_t m{0};
  std::size_t k{0};
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};

  HipblasLtPlan() = default;
  HipblasLtPlan(const HipblasLtPlan&) = delete;
  HipblasLtPlan& operator=(const HipblasLtPlan&) = delete;

  ~HipblasLtPlan() {
    if (a_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    }
    if (x_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
    }
    if (y_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
    }
    if (operation != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation);
    }
  }
};

[[nodiscard]] std::size_t SelectHipblasLtHeuristicRank(
    std::size_t batch_size, std::size_t m, std::size_t k,
    std::size_t result_count) noexcept {
  if (batch_size >= kHipblasLtLargeBatch) {
    if (m <= 1024) {
      return kHipblasLtLargeWideRank;
    }
    return k >= kHipblasLtWideFfnThreshold * m ? kHipblasLtTallRank
                                               : kHipblasLtLargeWideRank;
  }
  const std::size_t tall_m_threshold = batch_size >= 128 ? k : 2 * k;
  if (m > tall_m_threshold || k >= kHipblasLtWideFfnThreshold * m) {
    return kHipblasLtTallRank;
  }
  return result_count - 1;
}

[[nodiscard]] bool IsUsable(
    const hipblasLtMatmulHeuristicResult_t& result) noexcept {
  return result.state == HIPBLAS_STATUS_SUCCESS && result.workspaceSize == 0;
}

[[nodiscard]] std::unique_ptr<HipblasLtPlan> CreateHipblasLtPlan(
    hipblasLtHandle_t handle, std::size_t batch_size, std::size_t m,
    std::size_t k) {
  auto plan = std::make_unique<HipblasLtPlan>();
  plan->batch_size = batch_size;
  plan->m = m;
  plan->k = k;

  if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  if (hipblasLtMatmulDescSetAttribute(
          plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  if (hipblasLtMatrixLayoutCreate(&plan->a_layout, HIP_R_16BF, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->x_layout, HIP_R_16BF, k, batch_size,
                                  k) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->y_layout, HIP_R_32F, m, batch_size,
                                  m) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }

  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  constexpr std::size_t max_workspace_bytes = 0;
  const auto preference_status = hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_bytes, sizeof(max_workspace_bytes));

  std::vector<hipblasLtMatmulHeuristicResult_t> results(
      kHipblasLtMaxAlgorithms);
  int result_count = 0;
  const auto heuristic_status =
      preference_status == HIPBLAS_STATUS_SUCCESS
          ? hipblasLtMatmulAlgoGetHeuristic(
                handle, plan->operation, plan->a_layout, plan->x_layout,
                plan->y_layout, plan->y_layout, preference,
                static_cast<int>(results.size()), results.data(), &result_count)
          : preference_status;
  (void)hipblasLtMatmulPreferenceDestroy(preference);

  if (heuristic_status != HIPBLAS_STATUS_SUCCESS || result_count <= 0) {
    return nullptr;
  }

  const auto result_span =
      std::span(results.data(), static_cast<std::size_t>(result_count));
  const auto preferred_rank =
      SelectHipblasLtHeuristicRank(batch_size, m, k, result_span.size());
  if (preferred_rank < result_span.size() &&
      IsUsable(result_span[preferred_rank])) {
    plan->algorithm = result_span[preferred_rank].algo;
    return plan;
  }

  const auto fallback = std::ranges::find_if(
      result_span, [](const auto& result) { return IsUsable(result); });
  if (fallback == result_span.end()) {
    return nullptr;
  }
  plan->algorithm = fallback->algo;
  return plan;
}

__global__ void EmbeddingLookupKernel(const void* __restrict__ table,
                                      bool is_bf16, std::uint32_t token_id,
                                      float* __restrict__ out_hidden,
                                      std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;

  const std::size_t offset =
      static_cast<std::size_t>(token_id) * hidden_size + idx;
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
  if (tid >= total)
    return;

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
  const std::size_t total_q_pairs =
      static_cast<std::size_t>(num_heads) * half_rot;
  const std::size_t total_kv_pairs =
      static_cast<std::size_t>(num_kv_heads) * half_rot;

  // Process Q heads
  if (tid < total_q_pairs) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
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

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
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

__global__ void FastGEMVBlockKernel(const void* __restrict__ A, bool is_bf16,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, std::size_t M,
                                    std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[16];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      y[m] = block_sum;
    }
  }
}

__global__ void FastFusedSwiGLUGEMVBlockKernel(
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ up_w, bool up_is_bf16, const float* __restrict__ x,
    float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_is_bf16 && up_is_bf16) {
    const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
    const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[16];
  __shared__ float s_up_sums[16];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      out[m] = (block_gate * sig_g) * block_up;
    }
  }
}

__global__ void FusedSSMInputProjectionsKernel(
    const void* __restrict__ qkv_w, bool qkv_is_bf16,
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ alpha_w, bool alpha_is_bf16,
    const void* __restrict__ beta_w, bool beta_is_bf16,
    const float* __restrict__ x, float* __restrict__ qkv_out,
    float* __restrict__ gate_out, float* __restrict__ alpha_out,
    float* __restrict__ beta_out, std::size_t hidden_size, std::size_t qkv_size,
    std::size_t inner_size, std::size_t time_step_rank) {
  const std::size_t m = blockIdx.x;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    is_bf16 = qkv_is_bf16;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    is_bf16 = gate_is_bf16;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    is_bf16 = alpha_is_bf16;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    is_bf16 = beta_is_bf16;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

__global__ void FusedQKVProjectionsKernel(
    const void* __restrict__ q_w, bool q_is_bf16, const void* __restrict__ k_w,
    bool k_is_bf16, const void* __restrict__ v_w, bool v_is_bf16,
    const float* __restrict__ x, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    is_bf16 = q_is_bf16;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    is_bf16 = k_is_bf16;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    is_bf16 = v_is_bf16;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

__global__ void PerHeadRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     std::uint32_t num_heads,
                                     std::uint32_t head_dim, float eps) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

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

__global__ void WriteKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  k_cache[kv_offset + tid] = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  v_cache[kv_offset + tid] = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
}

__global__ void AttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];

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
    s_scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_scratch[tid] += s_scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p] = s_scratch[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..seq_len-1]
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_scratch[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
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

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__global__ void SSMConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out,
                              std::size_t qkv_dim) {
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim)
    return;

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
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    const float* __restrict__ ssm_norm, const float* __restrict__ gate,
    float* __restrict__ out_buf, std::uint32_t layer_idx,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;

  const float* q_ptr = conv_out;
  const float* k_ptr = conv_out + (num_key_heads * key_dim);
  const float* v_ptr = conv_out + (num_key_heads * key_dim * 2);

  const std::uint32_t kh_idx = h % num_key_heads;
  const float* q_h = q_ptr + (kh_idx * key_dim);
  const float* k_h = k_ptr + (kh_idx * key_dim);
  const float* v_h = v_ptr + (h * val_dim);
  float* o_h = out_buf + (h * val_dim);

  float* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta =
  // sigmoid(beta)
  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
  const float alpha_in = (alpha_buf != nullptr) ? alpha_buf[h] : 0.0F;
  const float beta_in = (beta_buf != nullptr) ? beta_buf[h] : 0.0F;

  const float alpha_biased = alpha_in + dt;
  const float alpha_softplus =
      (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
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
  const float q_scale = (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1];
  const float inv_k_norm = s_norm[0];

  // 3. Associative retrieval: u_j = sum_i S[j, i] * (k[i] * inv_k_norm)
  if (tid < val_dim) {
    const std::size_t j = tid;
    float* s_row = s_matrix + (j * key_dim);

    // Decay state while computing the retrieval. Keeping the decayed value
    // live avoids a second full read of the state row.
    float u_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] * alpha_h;
      s_row[i] = state;
      u_j += state * (k_h[i] * inv_k_norm);
    }
    const float d_j = (v_h[j] - u_j) * beta_h;

    // Apply the delta and compute the query readout from the same updated
    // value, avoiding another full read of the state row.
    float o_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] + d_j * (k_h[i] * inv_k_norm);
      s_row[i] = state;
      o_j += state * (q_h[i] * q_scale);
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
    if (tid < s)
      s_sum[tid] += s_sum[tid + s];
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
      val *= (g * sig);  // SiLU gating
    }
    o_h[tid] = val;
  }
}

}  // namespace

struct HipblasLtGemm::Impl {
  hipblasLtHandle_t handle{nullptr};
  std::vector<std::unique_ptr<HipblasLtPlan>> plans;

  Impl() {
    if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      handle = nullptr;
    }
  }

  ~Impl() {
    plans.clear();
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
  }

  [[nodiscard]] HipblasLtPlan* GetPlan(std::size_t batch_size, std::size_t m,
                                       std::size_t k) {
    const auto existing = std::ranges::find_if(plans, [&](const auto& plan) {
      return plan->batch_size == batch_size && plan->m == m && plan->k == k;
    });
    if (existing != plans.end()) {
      return existing->get();
    }
    auto plan = CreateHipblasLtPlan(handle, batch_size, m, k);
    if (plan == nullptr) {
      return nullptr;
    }
    auto* result = plan.get();
    plans.push_back(std::move(plan));
    return result;
  }
};

HipblasLtGemm::HipblasLtGemm() : impl_(std::make_unique<Impl>()) {}

HipblasLtGemm::~HipblasLtGemm() = default;

HipblasLtGemm::HipblasLtGemm(HipblasLtGemm&&) noexcept = default;

HipblasLtGemm& HipblasLtGemm::operator=(HipblasLtGemm&&) noexcept = default;

bool HipblasLtGemm::RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                            std::size_t batch_size, std::size_t m,
                            std::size_t k, hipStream_t stream) {
  if (impl_ == nullptr || impl_->handle == nullptr || batch_size == 0 ||
      m == 0 || k == 0) {
    return false;
  }
  auto* plan = impl_->GetPlan(batch_size, m, k);
  if (plan == nullptr) {
    return false;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  return hipblasLtMatmul(impl_->handle, plan->operation, &alpha, a_bf16,
                         plan->a_layout, x_bf16, plan->x_layout, &beta, y,
                         plan->y_layout, y, plan->y_layout, &plan->algorithm,
                         nullptr, 0, stream) == HIPBLAS_STATUS_SUCCESS;
}

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
  const std::size_t block_size = K >= kLargeGemvInput ? 512 : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, is_bf16, x, y, M, K);
}

void LaunchFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* x, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t hidden_size,
    std::size_t qkv_size, std::size_t inner_size, std::size_t time_step_rank,
    hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  FusedSSMInputProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      qkv_w, qkv_is_bf16, gate_w, gate_is_bf16, alpha_w, alpha_is_bf16, beta_w,
      beta_is_bf16, x, qkv_out, gate_out, alpha_out, beta_out, hidden_size,
      qkv_size, inner_size, time_step_rank);
}

void LaunchFusedQKVProjections(const void* q_w, bool q_is_bf16, const void* k_w,
                               bool k_is_bf16, const void* v_w, bool v_is_bf16,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  FusedQKVProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      q_w, q_is_bf16, k_w, k_is_bf16, v_w, v_is_bf16, x, q_out, k_out, v_out,
      q_dim, kv_dim, hidden_size);
}

void LaunchFusedSwiGLUGEMV(const void* gate_w, bool gate_is_bf16,
                           const void* up_w, bool up_is_bf16, const float* x,
                           float* out, std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream) {
  FastFusedSwiGLUGEMVBlockKernel<<<intermediate_size, 512, 0, stream>>>(
      gate_w, gate_is_bf16, up_w, up_is_bf16, x, out, intermediate_size,
      hidden_size);
}

void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(x, weight, out, num_heads,
                                                      head_dim, eps);
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

  // 2. Qwen3.8 uses D=256. A one-wave online softmax avoids the context-sized
  // LDS score buffer, which exceeds the gfx1151 block limit beyond 16K tokens.
  if (head_dim == 256 && num_kv_heads > 0 && (num_heads % num_kv_heads) == 0) {
    QwenDecodeOnlineAttentionKernel<<<num_heads, 32, 0, stream>>>(
        q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
        num_heads, num_kv_heads);
  } else {
    const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
    const std::size_t shared_scores_bytes = seq_len * sizeof(float);
    AttentionKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
        q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
        num_heads, num_kv_heads, head_dim);
  }
}

void LaunchSSMConvRecurrence(const float* qkv_in, const float* conv_weights,
                             float* conv_state, float* conv_out,
                             float* deltanet_state, const float* alpha_buf,
                             const float* beta_buf, const float* ssm_a,
                             const float* ssm_dt, const float* ssm_norm,
                             const float* gate, float* out_buf,
                             std::uint32_t layer_idx, std::size_t qkv_size,
                             std::uint32_t num_key_heads,
                             std::uint32_t num_heads, std::uint32_t key_dim,
                             std::uint32_t val_dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_size + block_size - 1) / block_size;
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  SSMConvKernel<<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_layer_state, conv_out, qkv_size);

  DeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      conv_out, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm,
      gate, out_buf, layer_idx, num_key_heads, num_heads, key_dim, val_dim);
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

// =========================================================================
// Batched GPU Operators for High-Throughput Prompt Processing (Prefill)
// =========================================================================

__global__ void BatchedEmbeddingLookupKernel(
    const void* __restrict__ table, bool is_bf16,
    const std::uint32_t* __restrict__ token_ids, float* __restrict__ out_hidden,
    std::size_t batch_size, std::size_t hidden_size) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;
  const std::size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= hidden_size)
    return;

  const std::uint32_t tok = token_ids[b];
  float* out_row = out_hidden + (b * hidden_size);

  if (is_bf16) {
    const auto* table_bf16 = static_cast<const hip_bfloat16*>(table);
    out_row[tid] = static_cast<float>(
        table_bf16[(static_cast<std::size_t>(tok) * hidden_size) + tid]);
  } else {
    const auto* table_f32 = static_cast<const float*>(table);
    out_row[tid] =
        table_f32[(static_cast<std::size_t>(tok) * hidden_size) + tid];
  }
}

void LaunchBatchedEmbeddingLookup(const void* table, bool is_bf16,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_x = (hidden_size + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedEmbeddingLookupKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                                 stream>>>(table, is_bf16, token_ids,
                                           out_hidden, batch_size, hidden_size);
}

__global__ void BatchedRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     hip_bfloat16* __restrict__ out_bf16,
                                     std::size_t batch_size, std::size_t dim,
                                     float eps) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  float* out_row = (out != nullptr) ? out + (b * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * dim) : nullptr;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += block_size) {
    const float val = in_row[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += block_size) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    const float normed = (in_row[i] * inv_rms) * w;
    if (out_row != nullptr) {
      out_row[i] = normed;
    }
    if (out_bf16_row != nullptr) {
      out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps, hipStream_t stream) {
  BatchedRMSNormKernel<<<batch_size, 256, 0, stream>>>(
      x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
      eps);
}

__global__ void BatchedPerHeadRMSNormKernel(const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            float* __restrict__ out,
                                            std::size_t batch_size,
                                            std::uint32_t num_heads,
                                            std::uint32_t head_dim, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  if (b >= batch_size || h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t head_total = static_cast<std::size_t>(num_heads) * head_dim;
  const float* x_head =
      x + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);
  float* out_head =
      out + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
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

void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps,
                                 hipStream_t stream) {
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedPerHeadRMSNormKernel<<<grid, 256, 0, stream>>>(
      x, weight, out, batch_size, num_heads, head_dim, eps);
}

__global__ void BatchedResidualAddKernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         std::size_t total_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < total_elements) {
    out[idx] = a[idx] + b[idx];
  }
}

void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream) {
  const std::size_t total = batch_size * dim;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total + block_size - 1) / block_size;
  BatchedResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out,
                                                                 total);
}

__global__ void BatchedUnpackQGKernel(const float* __restrict__ qg,
                                      float* __restrict__ q_out,
                                      float* __restrict__ gate_out,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;

  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  if (idx >= total_h)
    return;

  const std::size_t head_idx = idx / head_dim;
  const std::size_t dim_idx = idx % head_dim;
  const std::size_t src_idx =
      (b * total_h * 2) + (head_idx * 2 * head_dim) + dim_idx;

  q_out[(b * total_h) + idx] = qg[src_idx];
  gate_out[(b * total_h) + idx] = qg[src_idx + head_dim];
}

void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream) {
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_h + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedUnpackQGKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                          stream>>>(qg_interleaved, q_out, gate_out, batch_size,
                                    num_heads, head_dim);
}

__global__ void BatchedRoPEKernel(
    float* __restrict__ q, float* __restrict__ k, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t start_pos, float rope_theta) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;

  const std::size_t pair_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t half_rot = rotary_dim / 2;
  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(b);

  if (pair_idx < static_cast<std::size_t>(num_heads) * half_rot) {
    const std::size_t h = pair_idx / half_rot;
    const std::size_t i = pair_idx % half_rot;
    const float freq =
        1.0F / powf(rope_theta,
                    static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(cur_pos) * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t q_stride = static_cast<std::size_t>(num_heads) * head_dim;
    float* q_head = q + (b * q_stride) + (h * head_dim);
    const float q0 = q_head[i];
    const float q1 = q_head[i + half_rot];
    q_head[i] = (q0 * cos_val) - (q1 * sin_val);
    q_head[i + half_rot] = (q0 * sin_val) + (q1 * cos_val);
  }

  if (pair_idx < static_cast<std::size_t>(num_kv_heads) * half_rot) {
    const std::size_t h = pair_idx / half_rot;
    const std::size_t i = pair_idx % half_rot;
    const float freq =
        1.0F / powf(rope_theta,
                    static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(cur_pos) * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t k_stride =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    float* k_head = k + (b * k_stride) + (h * head_dim);
    const float k0 = k_head[i];
    const float k1 = k_head[i + half_rot];
    k_head[i] = (k0 * cos_val) - (k1 * sin_val);
    k_head[i + half_rot] = (k0 * sin_val) + (k1 * cos_val);
  }
}

void LaunchBatchedRoPE(float* q, float* k, std::size_t batch_size,
                       std::uint32_t num_heads, std::uint32_t num_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rotary_dim,
                       std::uint32_t start_pos, float rope_theta,
                       hipStream_t stream) {
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * half_rot,
               static_cast<std::size_t>(num_kv_heads) * half_rot);
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_pairs + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedRoPEKernel<<<grid, static_cast<unsigned int>(block_size), 0, stream>>>(
      q, k, batch_size, num_heads, num_kv_heads, head_dim, rotary_dim,
      start_pos, rope_theta);
}

__global__ void BatchedGEMMKernel(const void* __restrict__ A, bool is_bf16,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y, std::size_t batch_size,
                                  std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= M || b >= batch_size)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t block_threads = blockDim.x;
  const float* x_row = X + (b * K);
  float* y_row = Y + (b * M);

  float dot = 0.0F;
  if (is_bf16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      y_row[m] = block_sum;
    }
  }
}

void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(M),
                  static_cast<unsigned int>(batch_size));
  BatchedGEMMKernel<<<grid, 128, 0, stream>>>(A, is_bf16, X, Y, batch_size, M,
                                              K);
}

__global__ void FloatToBfloat16Kernel(const float* __restrict__ in,
                                      hip_bfloat16* __restrict__ out,
                                      std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    out[idx] = hip_bfloat16(in[idx]);
  }
}

void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  FloatToBfloat16Kernel<<<num_blocks, block_size, 0, stream>>>(
      in, static_cast<hip_bfloat16*>(out), num_elements);
}

__global__ void BatchedSwiGLUActivationKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    const float g = gate[idx];
    const float u = up[idx];
    const float sig_g = 1.0F / (1.0F + expf(-g));
    const float val = (g * sig_g) * u;
    if (out != nullptr) {
      out[idx] = val;
    }
    if (out_bf16 != nullptr) {
      out_bf16[idx] = hip_bfloat16(val);
    }
  }
}

void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  BatchedSwiGLUActivationKernel<<<num_blocks, block_size, 0, stream>>>(
      gate, up, out, static_cast<hip_bfloat16*>(out_bf16), num_elements);
}

void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  if (is_bf16) {
    LaunchFloatToBfloat16(X, d_x_bf16_buf, batch_size * K, stream);
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A,
        HIP_R_16BF, static_cast<int>(K), d_x_bf16_buf, HIP_R_16BF,
        static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  } else {
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A, HIP_R_32F,
        static_cast<int>(K), X, HIP_R_32F, static_cast<int>(K), &beta, Y,
        HIP_R_32F, static_cast<int>(M), HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT));
  }
}

void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  HIPBLAS_CHECK(hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(batch_size), static_cast<int>(K), &alpha, A_bf16,
      HIP_R_16BF, static_cast<int>(K), d_x_bf16, HIP_R_16BF,
      static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

__global__ void BatchedFusedSSMInputProjectionsKernel(
    const void* __restrict__ qkv_w, bool qkv_is_bf16,
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ alpha_w, bool alpha_is_bf16,
    const void* __restrict__ beta_w, bool beta_is_bf16,
    const float* __restrict__ X, float* __restrict__ qkv_out,
    float* __restrict__ gate_out, float* __restrict__ alpha_out,
    float* __restrict__ beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows || b >= batch_size)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    is_bf16 = qkv_is_bf16;
    out_target = qkv_out + (b * qkv_size);
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    is_bf16 = gate_is_bf16;
    out_target = gate_out + (b * inner_size);
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    is_bf16 = alpha_is_bf16;
    out_target = alpha_out + (b * time_step_rank);
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    is_bf16 = beta_is_bf16;
    out_target = beta_out + (b * time_step_rank);
    row_idx = m - alpha_end;
  }

  const float* x_row = X + (b * hidden_size);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

void LaunchBatchedFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* X, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  const dim3 grid(static_cast<unsigned int>(total_rows),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedSSMInputProjectionsKernel<<<grid, 128, 0, stream>>>(
      qkv_w, qkv_is_bf16, gate_w, gate_is_bf16, alpha_w, alpha_is_bf16, beta_w,
      beta_is_bf16, X, qkv_out, gate_out, alpha_out, beta_out, batch_size,
      hidden_size, qkv_size, inner_size, time_step_rank);
}

__global__ void BatchedFusedQKVProjectionsKernel(
    const void* __restrict__ q_w, bool q_is_bf16, const void* __restrict__ k_w,
    bool k_is_bf16, const void* __restrict__ v_w, bool v_is_bf16,
    const float* __restrict__ X, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out,
    std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows || b >= batch_size)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    is_bf16 = q_is_bf16;
    out_target = q_out + (b * q_dim);
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    is_bf16 = k_is_bf16;
    out_target = k_out + (b * kv_dim);
    row_idx = m - q_dim;
  } else {
    A = v_w;
    is_bf16 = v_is_bf16;
    out_target = v_out + (b * kv_dim);
    row_idx = m - q_dim - kv_dim;
  }

  const float* x_row = X + (b * hidden_size);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

void LaunchBatchedFusedQKVProjections(
    const void* q_w, bool q_is_bf16, const void* k_w, bool k_is_bf16,
    const void* v_w, bool v_is_bf16, const float* X, float* q_out, float* k_out,
    float* v_out, std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size, hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  const dim3 grid(static_cast<unsigned int>(total_rows),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedQKVProjectionsKernel<<<grid, 128, 0, stream>>>(
      q_w, q_is_bf16, k_w, k_is_bf16, v_w, v_is_bf16, X, q_out, k_out, v_out,
      batch_size, q_dim, kv_dim, hidden_size);
}

__global__ void BatchedFusedSwiGLUGEMMKernel(
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ up_w, bool up_is_bf16, const float* __restrict__ X,
    float* __restrict__ out, std::size_t batch_size,
    std::size_t intermediate_size, std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= intermediate_size || b >= batch_size)
    return;

  const float* x_row = X + (b * hidden_size);
  float* out_row = out + (b * intermediate_size);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_is_bf16 && up_is_bf16) {
    const auto* g_row =
        static_cast<const hip_bfloat16*>(gate_w) + (m * hidden_size);
    const auto* u_row =
        static_cast<const hip_bfloat16*>(up_w) + (m * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[8];
  __shared__ float s_up_sums[8];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      out_row[m] = (block_gate * sig_g) * block_up;
    }
  }
}

void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(intermediate_size),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedSwiGLUGEMMKernel<<<grid, 128, 0, stream>>>(
      gate_w, gate_is_bf16, up_w, up_is_bf16, X, out, batch_size,
      intermediate_size, hidden_size);
}

__global__ void WriteBatchedKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t kv_head = blockIdx.x;
  if (b >= batch_size || kv_head >= num_kv_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(b);
  const std::size_t k_src_offset =
      (b * num_kv_heads * head_dim) +
      (static_cast<std::size_t>(kv_head) * head_dim);

  const std::size_t cache_offset =
      (static_cast<std::size_t>(layer_idx) * num_kv_heads * max_context *
       head_dim) +
      (static_cast<std::size_t>(kv_head) * max_context * head_dim) +
      (static_cast<std::size_t>(pos) * head_dim);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    k_cache[cache_offset + i] = k[k_src_offset + i];
    v_cache[cache_offset + i] = v[k_src_offset + i];
  }
}

__global__ void BatchedAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (h >= num_heads || t >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::uint32_t kv_h = h / (num_heads / num_kv_heads);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  __shared__ float s_scratch[256];
  extern __shared__ float s_scores[];

  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(t);
  const std::size_t seq_len = static_cast<std::size_t>(cur_pos) + 1;
  const float* q_h =
      q + (t * num_heads * head_dim) + (static_cast<std::size_t>(h) * head_dim);

  const auto* q_h_vec = reinterpret_cast<const float4*>(q_h);
  const std::size_t num_vec = head_dim / 4;
  constexpr std::size_t wave_dot_min_sequence = 64;
  if (seq_len >= wave_dot_min_sequence) {
    const std::size_t lane_id = tid & 31U;
    const std::size_t wave_id = tid >> 5U;
    const std::size_t waves_per_block = blockDim.x >> 5U;
    for (std::size_t p = wave_id; p < seq_len; p += waves_per_block) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const auto* k_p_vec = reinterpret_cast<const float4*>(k_cache + p_offset);

      float dot = 0.0F;
      for (std::size_t d = lane_id; d < num_vec; d += 32) {
        const float4 qv = q_h_vec[d];
        const float4 kv = k_p_vec[d];
        dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor(dot, offset);
      }
      if (lane_id == 0) {
        s_scores[p] = dot * scale;
      }
    }
  } else {
    for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const auto* k_p_vec = reinterpret_cast<const float4*>(k_cache + p_offset);

      float dot = 0.0F;
      for (std::size_t d = 0; d < num_vec; ++d) {
        const float4 qv = q_h_vec[d];
        const float4 kv = k_p_vec[d];
        dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
      s_scores[p] = dot * scale;
    }
  }
  __syncthreads();

  // Step 2: Softmax
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_scratch[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
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
      const float g =
          (gate + (t * num_heads *
                   head_dim))[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[(t * num_heads * head_dim) +
                (static_cast<std::size_t>(h) * head_dim) + tid] = ctx;
  }
}

void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream) {
  const dim3 kv_grid(num_kv_heads, static_cast<unsigned int>(batch_size));
  WriteBatchedKVCacheKernel<<<kv_grid, 256, 0, stream>>>(
      k, v, k_cache, v_cache, layer_idx, start_pos, batch_size, max_context,
      num_kv_heads, head_dim);

  const dim3 attn_grid(num_heads, static_cast<unsigned int>(batch_size));
  const std::size_t max_seq_len =
      static_cast<std::size_t>(start_pos) + batch_size;
  const std::size_t shared_scores_bytes = max_seq_len * sizeof(float);
  BatchedAttentionKernel<<<attn_grid, 256, shared_scores_bytes, stream>>>(
      q, gate, k_cache, v_cache, out_context, layer_idx, start_pos, batch_size,
      max_context, num_heads, num_kv_heads, head_dim);
}

__global__ void PackCkAttentionInputsKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, __half* __restrict__ q_f16,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (token >= batch_size) {
    return;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  if (index < attention_width) {
    const std::size_t offset = (token * attention_width) + index;
    q_f16[offset] = __float2half_rn(q[offset]);
  }

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (index < kv_width) {
    const std::size_t source_offset = (token * kv_width) + index;
    const std::size_t position = start_pos + token;
    const std::size_t f16_cache_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + position) *
         kv_width) +
        index;
    const std::size_t kv_head = index / head_dim;
    const std::size_t dimension = index % head_dim;
    const std::size_t f32_cache_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
              max_context +
          position) *
         head_dim) +
        dimension;
    const float key = k[source_offset];
    const float value = v[source_offset];
    k_cache_f16[f16_cache_offset] = __float2half_rn(key);
    v_cache_f16[f16_cache_offset] = __float2half_rn(value);
    k_cache[f32_cache_offset] = key;
    v_cache[f32_cache_offset] = value;
  }
}

// This Qwen-specific tile follows the causal online-softmax structure used by
// llama.cpp's MIT-licensed ggml-cuda/fattn-tile.cuh (commit e9fa078). The data
// movement is native HIP and specialized for gfx1151, D=256 and GQA=6.
constexpr std::uint32_t kTileAttentionHeadDim = 256;
constexpr std::uint32_t kTileAttentionQueryHeads = 24;
constexpr std::uint32_t kTileAttentionKvHeads = 4;
constexpr std::uint32_t kTileAttentionGqa = 6;
constexpr std::uint32_t kTileAttentionQueryRows = 16;
constexpr std::uint32_t kTileAttentionHeadsPerBlock = 2;
constexpr std::uint32_t kTileAttentionColumns =
    kTileAttentionQueryRows * kTileAttentionHeadsPerBlock;
constexpr std::uint32_t kTileAttentionKeyRows = 64;
constexpr std::uint32_t kTileAttentionHalf2Dim = kTileAttentionHeadDim / 2;
constexpr std::uint32_t kTileAttentionKeyHalf2Dim = kTileAttentionHalf2Dim / 2;
constexpr std::uint32_t kTileAttentionValueRows = 32;
constexpr std::uint32_t kTileAttentionHalf2Padding = 4;

__device__ __forceinline__ float TileWaveReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float TileWaveReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ void TileHalf2Dot(float& accumulator, __half2 lhs,
                                             __half2 rhs) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
               : "+v"(accumulator)
               : "v"(lhs), "v"(rhs));
}

__global__ void PackTiledAttentionKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  if (token >= batch_size || index >= kv_width) {
    return;
  }

  const std::size_t source_offset = (token * kv_width) + index;
  const std::size_t position = start_pos + token;
  const std::size_t kv_head = index / kTileAttentionHeadDim;
  const std::size_t dimension = index % kTileAttentionHeadDim;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + position) *
       kv_width) +
      index;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * kTileAttentionKvHeads +
         kv_head) *
            max_context +
        position) *
       kTileAttentionHeadDim) +
      dimension;
  const float key = k[source_offset];
  const float value = v[source_offset];
  k_cache_f16[f16_cache_offset] = __float2half_rn(key);
  v_cache_f16[f16_cache_offset] = __float2half_rn(value);
  k_cache[f32_cache_offset] = key;
  v_cache[f32_cache_offset] = value;
}

__launch_bounds__(256, 3) __global__ void QwenTiledAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache_f16,
    const __half* __restrict__ v_cache_f16, float* __restrict__ out_context,
    std::uint32_t start_pos, std::size_t batch_size) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t wave = threadIdx.y;
  const std::uint32_t query_start = blockIdx.x * kTileAttentionQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kTileAttentionGqa / 2);
  const std::uint32_t pair_in_group = head_pair % (kTileAttentionGqa / 2);
  const std::uint32_t first_query_head =
      (kv_head * kTileAttentionGqa) +
      (pair_in_group * kTileAttentionHeadsPerBlock);
  constexpr std::size_t attention_width =
      kTileAttentionQueryHeads * kTileAttentionHeadDim;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  constexpr std::size_t kv_half2_stride = kv_width / 2;
  constexpr float attention_scale = 1.0F / 16.0F;

  __shared__ __half2 q_tile[kTileAttentionColumns][kTileAttentionHalf2Dim];
  __shared__ __half2
      kv_tile[kTileAttentionKeyRows *
              (kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding)];
  __shared__ __half weights[kTileAttentionColumns][kTileAttentionKeyRows];

  constexpr std::uint32_t columns_per_wave = kTileAttentionColumns / 8;
  float running_max[columns_per_wave];
  float running_sum[columns_per_wave];
  float2 output[columns_per_wave][kTileAttentionHalf2Dim / 32];
#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    running_max[column] = -INFINITY;
    running_sum[column] = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      output[column][slot] = make_float2(0.0F, 0.0F);
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
#pragma unroll
    for (std::uint32_t half2_index = lane; half2_index < kTileAttentionHalf2Dim;
         half2_index += 32) {
      __half2 packed =
          __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      if (local_query_position < batch_size) {
        const std::size_t source =
            (static_cast<std::size_t>(local_query_position) * attention_width) +
            (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
            (2 * half2_index);
        packed =
            __halves2half2(__float2half_rn(q[source] * attention_scale),
                           __float2half_rn(q[source + 1] * attention_scale));
      }
      q_tile[tiled_column][half2_index] = packed;
    }
  }
  __syncthreads();

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kTileAttentionQueryRows);
  for (std::uint32_t key_start = 0; key_start < max_visible;
       key_start += kTileAttentionKeyRows) {
    float scores[columns_per_wave][2] = {};
#pragma unroll
    for (std::uint32_t key_dimension = 0; key_dimension < 2; ++key_dimension) {
      for (std::uint32_t key_row = wave; key_row < kTileAttentionKeyRows;
           key_row += 8) {
        const std::uint32_t key_position = key_start + key_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionKeyHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_key =
                reinterpret_cast<const __half2*>(k_cache_f16);
            packed = layer_key
                [(static_cast<std::size_t>(key_position) * kv_half2_stride) +
                 (static_cast<std::size_t>(kv_head) * kTileAttentionHalf2Dim) +
                 (key_dimension * kTileAttentionKeyHalf2Dim) + half2_index];
          }
          kv_tile[(key_row *
                   (kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding)) +
                  half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll 4
      for (std::uint32_t half2_index = 0;
           half2_index < kTileAttentionKeyHalf2Dim; ++half2_index) {
        const __half2 key_values[2] = {
            kv_tile[(lane *
                     (kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding)) +
                    half2_index],
            kv_tile[((lane + 32) *
                     (kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding)) +
                    half2_index]};
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          const __half2 query_value =
              q_tile[tiled_column]
                    [(key_dimension * kTileAttentionKeyHalf2Dim) + half2_index];
          TileHalf2Dot(scores[column][0], key_values[0], query_value);
          TileHalf2Dot(scores[column][1], key_values[1], query_value);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
      const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
      const std::uint32_t query_offset =
          tiled_column / kTileAttentionHeadsPerBlock;
      const std::uint32_t local_query_position = query_start + query_offset;
      const std::uint32_t absolute_query_position =
          start_pos + local_query_position;
      const std::uint32_t key_positions[2] = {key_start + lane,
                                              key_start + lane + 32};
      const bool valid[2] = {local_query_position < batch_size &&
                                 key_positions[0] <= absolute_query_position &&
                                 key_positions[0] < context_end,
                             local_query_position < batch_size &&
                                 key_positions[1] <= absolute_query_position &&
                                 key_positions[1] < context_end};
      const float score_values[2] = {valid[0] ? scores[column][0] : -INFINITY,
                                     valid[1] ? scores[column][1] : -INFINITY};
      const float tile_max =
          TileWaveReduceMax(fmaxf(score_values[0], score_values[1]));
      const float next_max = fmaxf(running_max[column], tile_max);
      const float prior_scale = isfinite(running_max[column])
                                    ? expf(running_max[column] - next_max)
                                    : 0.0F;
      const float weight_values[2] = {
          valid[0] ? expf(score_values[0] - next_max) : 0.0F,
          valid[1] ? expf(score_values[1] - next_max) : 0.0F};
      running_max[column] = next_max;
      running_sum[column] = (running_sum[column] * prior_scale) +
                            weight_values[0] + weight_values[1];
#pragma unroll
      for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
        output[column][slot].x *= prior_scale;
        output[column][slot].y *= prior_scale;
      }
      weights[tiled_column][lane] = __float2half_rn(weight_values[0]);
      weights[tiled_column][lane + 32] = __float2half_rn(weight_values[1]);
    }
    __syncthreads();

    for (std::uint32_t value_group = 0; value_group < 2; ++value_group) {
      for (std::uint32_t value_row = wave; value_row < kTileAttentionValueRows;
           value_row += 8) {
        const std::uint32_t key_position =
            key_start + (value_group * kTileAttentionValueRows) + value_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_value =
                reinterpret_cast<const __half2*>(v_cache_f16);
            packed = layer_value
                [(static_cast<std::size_t>(key_position) * kv_half2_stride) +
                 (static_cast<std::size_t>(kv_head) * kTileAttentionHalf2Dim) +
                 half2_index];
          }
          kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll
      for (std::uint32_t value_row = 0; value_row < kTileAttentionValueRows;
           value_row += 2) {
        __half2 weight_values[columns_per_wave];
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          weight_values[column] = __halves2half2(
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row],
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row + 1]);
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32;
             ++slot) {
          const std::uint32_t half2_index = lane + (slot * 32);
          const __half2 value0 =
              kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index];
          const __half2 value1 =
              kv_tile[((value_row + 1) * kTileAttentionHalf2Dim) + half2_index];
          const __half2 values_x =
              __halves2half2(__low2half(value0), __low2half(value1));
          const __half2 values_y =
              __halves2half2(__high2half(value0), __high2half(value1));
#pragma unroll
          for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
            TileHalf2Dot(output[column][slot].x, values_x,
                         weight_values[column]);
            TileHalf2Dot(output[column][slot].y, values_y,
                         weight_values[column]);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
    const float denominator = TileWaveReduceSum(running_sum[column]);
    if (local_query_position >= batch_size || denominator == 0.0F) {
      continue;
    }
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      const std::uint32_t half2_index = lane + (slot * 32);
      const std::uint32_t dimension = 2 * half2_index;
      const std::size_t output_offset =
          (static_cast<std::size_t>(local_query_position) * attention_width) +
          (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
          dimension;
      const float2 result = output[column][slot];
      const float gate0 =
          gate == nullptr ? 1.0F : 1.0F / (1.0F + expf(-gate[output_offset]));
      const float gate1 = gate == nullptr
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset + 1]));
      out_context[output_offset] = (result.x / denominator) * gate0;
      out_context[output_offset + 1] = (result.y / denominator) * gate1;
    }
  }
}

bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream) {
  const bool supported =
      batch_size != 0 && num_heads == kTileAttentionQueryHeads &&
      num_kv_heads == kTileAttentionKvHeads &&
      head_dim == kTileAttentionHeadDim &&
      static_cast<std::size_t>(start_pos) + batch_size <= max_context &&
      k_cache_f16 != nullptr && v_cache_f16 != nullptr;
  if (!supported) {
    return false;
  }

  constexpr std::size_t block_size = 256;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  auto* layer_k = static_cast<__half*>(k_cache_f16) + layer_cache_offset;
  auto* layer_v = static_cast<__half*>(v_cache_f16) + layer_cache_offset;
  const dim3 pack_grid(
      static_cast<unsigned int>((kv_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  PackTiledAttentionKvKernel<<<pack_grid, block_size, 0, stream>>>(
      k, v, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx, start_pos,
      batch_size, max_context);

  const dim3 attention_grid(
      static_cast<unsigned int>((batch_size + kTileAttentionQueryRows - 1) /
                                kTileAttentionQueryRows),
      kTileAttentionKvHeads * (kTileAttentionGqa / 2));
  const dim3 attention_block(32, 8);
  QwenTiledAttentionKernel<<<attention_grid, attention_block, 0, stream>>>(
      q, gate, layer_k, layer_v, out_context, start_pos, batch_size);
  return true;
}

__global__ void ApplyCkAttentionGateKernel(const float* __restrict__ gate,
                                           float* __restrict__ output,
                                           std::size_t num_elements) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float gate_value = gate[index];
  output[index] *= 1.0F / (1.0F + expf(-gate_value));
}

bool LaunchBatchedAttentionCk(const float* q, const float* k, const float* v,
                              const float* gate, float* k_cache, float* v_cache,
                              void* k_cache_f16, void* v_cache_f16,
                              void* scratch_f16, float* out_context,
                              std::uint32_t layer_idx, std::uint32_t start_pos,
                              std::size_t batch_size, std::uint32_t max_context,
                              std::uint32_t num_heads,
                              std::uint32_t num_kv_heads,
                              std::uint32_t head_dim, hipStream_t stream) {
  const bool supported = batch_size != 0 && start_pos == 0 &&
                         num_kv_heads == kCkNumKvHeads &&
                         num_heads % num_kv_heads == 0 && head_dim == 256 &&
                         batch_size <= max_context && k_cache_f16 != nullptr &&
                         v_cache_f16 != nullptr && scratch_f16 != nullptr;
  if (!supported) {
    return false;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  auto* q_f16 = static_cast<__half*>(scratch_f16);
  constexpr std::size_t block_size = 256;
  const std::size_t pack_width = std::max(attention_width, kv_width);
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  const auto* layer_k =
      static_cast<const ck::half_t*>(k_cache_f16) + layer_cache_offset;
  const auto* layer_v =
      static_cast<const ck::half_t*>(v_cache_f16) + layer_cache_offset;
  CkGqaOperation operation;
  const auto argument = operation.MakeArgument(
      reinterpret_cast<const ck::half_t*>(q_f16), layer_k, layer_v, out_context,
      static_cast<ck::index_t>(batch_size),
      static_cast<ck::index_t>(batch_size), static_cast<ck::index_t>(head_dim),
      static_cast<ck::index_t>(head_dim), 1,
      static_cast<ck::index_t>(num_heads),
      1.0F / sqrtf(static_cast<float>(head_dim)), true, true);
  if (!operation.IsSupportedArgument(argument)) {
    return false;
  }

  const dim3 pack_grid(
      static_cast<unsigned int>((pack_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  PackCkAttentionInputsKernel<<<pack_grid, block_size, 0, stream>>>(
      q, k, v, q_f16, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim);

  auto invoker = operation.MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});

  const std::size_t num_elements = batch_size * attention_width;
  if (gate != nullptr) {
    const std::size_t output_grid =
        (num_elements + block_size - 1) / block_size;
    ApplyCkAttentionGateKernel<<<output_grid, block_size, 0, stream>>>(
        gate, out_context, num_elements);
  }
  return true;
}

__global__ void CausalSoftmaxKernel(float* __restrict__ scores,
                                    std::uint32_t start_pos,
                                    std::size_t batch_size,
                                    std::size_t sequence_length,
                                    std::size_t matrix_stride) {
  const std::size_t query = blockIdx.x;
  const std::size_t matrix = blockIdx.y;
  const std::size_t tid = threadIdx.x;
  if (query >= batch_size) {
    return;
  }

  float* column = scores + (matrix * matrix_stride) + (query * sequence_length);
  const std::size_t visible = static_cast<std::size_t>(start_pos) + query + 1;
  __shared__ float scratch[256];

  float local_max = -INFINITY;
  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    local_max = fmaxf(local_max, column[key]);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];

  float local_sum = 0.0F;
  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    const float value = expf(column[key] - max_score);
    column[key] = value;
    local_sum += value;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / (scratch[0] + 1e-9F);

  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    column[key] *= inverse_sum;
  }
  for (std::size_t key = visible + tid; key < sequence_length;
       key += blockDim.x) {
    column[key] = 0.0F;
  }
}

__global__ void ApplyAttentionGateKernel(const float* __restrict__ gate,
                                         float* __restrict__ out,
                                         std::size_t num_elements) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float value = gate[index];
  out[index] *= 1.0F / (1.0F + expf(-value));
}

void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, float* scores,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream) {
  if (batch_size == 0) {
    return;
  }
  const std::size_t sequence_length =
      static_cast<std::size_t>(start_pos) + batch_size;
  if (sequence_length > max_context) {
    return;
  }
  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const dim3 kv_grid(num_kv_heads, static_cast<unsigned int>(batch_size));
  WriteBatchedKVCacheKernel<<<kv_grid, 256, 0, stream>>>(
      k, v, k_cache, v_cache, layer_idx, start_pos, batch_size, max_context,
      num_kv_heads, head_dim);

  const float attention_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  constexpr float one = 1.0F;
  constexpr float zero = 0.0F;
  const std::size_t head_stride =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t cache_head_stride =
      static_cast<std::size_t>(max_context) * head_dim;
  const std::uint32_t heads_per_kv = num_heads / num_kv_heads;
  const std::size_t score_matrix_stride = sequence_length * batch_size;
  constexpr std::size_t grouped_attention_min_batch = 2048;

  if (batch_size < grouped_attention_min_batch) {
    for (std::uint32_t head = 0; head < num_heads; ++head) {
      const std::uint32_t kv_head = head / heads_per_kv;
      const std::size_t cache_head =
          (static_cast<std::size_t>(layer_idx) * num_kv_heads) + kv_head;
      const float* key_head = k_cache + (cache_head * cache_head_stride);
      const float* value_head = v_cache + (cache_head * cache_head_stride);
      const float* query_head = q + (static_cast<std::size_t>(head) * head_dim);
      float* output_head =
          out_context + (static_cast<std::size_t>(head) * head_dim);

      HIPBLAS_CHECK(hipblasGemmEx(
          handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(sequence_length),
          static_cast<int>(batch_size), static_cast<int>(head_dim),
          &attention_scale, key_head, HIP_R_32F, static_cast<int>(head_dim),
          query_head, HIP_R_32F, static_cast<int>(head_stride), &zero, scores,
          HIP_R_32F, static_cast<int>(sequence_length), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));

      const dim3 softmax_grid(static_cast<unsigned int>(batch_size), 1);
      CausalSoftmaxKernel<<<softmax_grid, 256, 0, stream>>>(
          scores, start_pos, batch_size, sequence_length, score_matrix_stride);

      HIPBLAS_CHECK(hipblasGemmEx(
          handle, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
          static_cast<int>(batch_size), static_cast<int>(sequence_length), &one,
          value_head, HIP_R_32F, static_cast<int>(head_dim), scores, HIP_R_32F,
          static_cast<int>(sequence_length), &zero, output_head, HIP_R_32F,
          static_cast<int>(head_stride), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));
    }
  } else {
    for (std::uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
      const std::uint32_t first_head = kv_head * heads_per_kv;
      const std::size_t cache_head =
          (static_cast<std::size_t>(layer_idx) * num_kv_heads) + kv_head;
      const float* key_head = k_cache + (cache_head * cache_head_stride);
      const float* value_head = v_cache + (cache_head * cache_head_stride);
      const float* query_group =
          q + (static_cast<std::size_t>(first_head) * head_dim);
      float* output_group =
          out_context + (static_cast<std::size_t>(first_head) * head_dim);

      HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
          handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(sequence_length),
          static_cast<int>(batch_size), static_cast<int>(head_dim),
          &attention_scale, key_head, HIP_R_32F, static_cast<int>(head_dim), 0,
          query_group, HIP_R_32F, static_cast<int>(head_stride),
          static_cast<hipblasStride>(head_dim), &zero, scores, HIP_R_32F,
          static_cast<int>(sequence_length),
          static_cast<hipblasStride>(score_matrix_stride),
          static_cast<int>(heads_per_kv), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));

      const dim3 softmax_grid(static_cast<unsigned int>(batch_size),
                              heads_per_kv);
      CausalSoftmaxKernel<<<softmax_grid, 256, 0, stream>>>(
          scores, start_pos, batch_size, sequence_length, score_matrix_stride);

      HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
          handle, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
          static_cast<int>(batch_size), static_cast<int>(sequence_length), &one,
          value_head, HIP_R_32F, static_cast<int>(head_dim), 0, scores,
          HIP_R_32F, static_cast<int>(sequence_length),
          static_cast<hipblasStride>(score_matrix_stride), &zero, output_group,
          HIP_R_32F, static_cast<int>(head_stride),
          static_cast<hipblasStride>(head_dim), static_cast<int>(heads_per_kv),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
  }

  if (gate != nullptr) {
    const std::size_t num_elements = batch_size * head_stride;
    constexpr std::size_t block_size = 256;
    const std::size_t grid_size = (num_elements + block_size - 1) / block_size;
    ApplyAttentionGateKernel<<<grid_size, block_size, 0, stream>>>(
        gate, out_context, num_elements);
  }
}

__global__ void BatchedSSMConvKernel(const float* __restrict__ qkv_in,
                                     const float* __restrict__ conv_weights,
                                     float* __restrict__ conv_state,
                                     float* __restrict__ conv_out,
                                     std::size_t batch_size,
                                     std::size_t qkv_dim) {
  const std::size_t c = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t t = blockIdx.y;
  if (c >= qkv_dim || t >= batch_size) {
    return;
  }

  const std::size_t c_off = c * 4;
  const float w0 = (conv_weights != nullptr) ? conv_weights[c * 4 + 0] : 0.0F;
  const float w1 = (conv_weights != nullptr) ? conv_weights[c * 4 + 1] : 0.0F;
  const float w2 = (conv_weights != nullptr) ? conv_weights[c * 4 + 2] : 0.0F;
  const float w3 = (conv_weights != nullptr) ? conv_weights[c * 4 + 3] : 1.0F;

  const float s3 = qkv_in[(t * qkv_dim) + c];
  const float s2 =
      (t >= 1) ? qkv_in[((t - 1) * qkv_dim) + c] : conv_state[c_off + 3];
  const float s1 = (t >= 2)   ? qkv_in[((t - 2) * qkv_dim) + c]
                   : (t == 1) ? conv_state[c_off + 3]
                              : conv_state[c_off + 2];
  const float s0 = (t >= 3)   ? qkv_in[((t - 3) * qkv_dim) + c]
                   : (t == 2) ? conv_state[c_off + 3]
                   : (t == 1) ? conv_state[c_off + 2]
                              : conv_state[c_off + 1];

  float dot = 0.0F;
  if (conv_weights != nullptr) {
    dot = (s0 * w0) + (s1 * w1) + (s2 * w2) + (s3 * w3);
  } else {
    dot = s3;
  }

  const float sig = 1.0F / (1.0F + expf(-dot));
  conv_out[(t * qkv_dim) + c] = dot * sig;

  // Final token updates state for future decode steps
  if (t == batch_size - 1) {
    conv_state[c_off + 0] = s0;
    conv_state[c_off + 1] = s1;
    conv_state[c_off + 2] = s2;
    conv_state[c_off + 3] = s3;
  }
}

__global__ void BatchedDeltaNetRecurrenceKernel(
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ raw_out, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::uint32_t kh_idx = h % num_key_heads;

  float* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;

  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];
  __shared__ float s_warp_kq[8];
  __shared__ float s_scales[3];  // inv_k_norm, q_scale, kq_dot

  // Two adjacent lanes own one state row, cutting the live register tile in
  // half while preserving the token-serial recurrence.
  constexpr std::size_t row_vectors_per_thread = 16;
  constexpr std::size_t row_values_per_thread =
      row_vectors_per_thread * sizeof(float4) / sizeof(float);
  const std::size_t j = tid >> 1U;
  const std::size_t row_segment = tid & 1U;
  const std::size_t row_offset = row_segment * row_values_per_thread;

  float4 s_reg[row_vectors_per_thread];
  if (j < val_dim) {
    const float* s_row = s_matrix + (j * key_dim) + row_offset;
    const auto* s_row_vec = reinterpret_cast<const float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_reg[vi] = s_row_vec[vi];
    }
  }

  for (std::size_t t = 0; t < batch_size; ++t) {
    const float* conv_t = conv_out + (t * qkv_size);
    const float* q_h = conv_t + (kh_idx * key_dim);
    const float* k_h = conv_t + (num_key_heads * key_dim) + (kh_idx * key_dim);
    const float* v_h = conv_t + (2 * num_key_heads * key_dim) + (h * val_dim);
    const std::size_t inner_size =
        static_cast<std::size_t>(num_heads) * val_dim;
    float* o_h = raw_out + (t * inner_size) + (h * val_dim);

    const float alpha_in =
        (alpha_buf != nullptr) ? (alpha_buf + (t * num_heads))[h] : 0.0F;
    const float beta_in =
        (beta_buf != nullptr) ? (beta_buf + (t * num_heads))[h] : 0.0F;

    const float alpha_biased = alpha_in + dt;
    const float alpha_softplus =
        (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
    const float alpha_h = expf(alpha_softplus * a_val);
    const float beta_h = 1.0F / (1.0F + expf(-beta_in));

    // 1. Normalize k and q. The upper four waves contribute zeros.
    const float raw_k = (tid < key_dim) ? k_h[tid] : 0.0F;
    const float raw_q = (tid < key_dim) ? q_h[tid] : 0.0F;
    float k_sq = raw_k * raw_k;
    float q_sq = raw_q * raw_q;

    for (int offset = 16; offset > 0; offset >>= 1) {
      k_sq += __shfl_xor(k_sq, offset);
      q_sq += __shfl_xor(q_sq, offset);
    }
    if (lane_id == 0) {
      s_warp_k[warp_id] = k_sq;
      s_warp_q[warp_id] = q_sq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_k_sq = 0.0F;
      float total_q_sq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_k_sq += s_warp_k[warp];
        total_q_sq += s_warp_q[warp];
      }
      s_scales[0] = 1.0F / sqrtf(total_k_sq + 1e-6F);
      s_scales[1] = (1.0F / sqrtf(static_cast<float>(key_dim))) *
                    (1.0F / sqrtf(total_q_sq + 1e-6F));
    }
    __syncthreads();

    const float inv_k_norm = s_scales[0];
    const float q_scale = s_scales[1];
    const float norm_k = raw_k * inv_k_norm;
    const float norm_q = raw_q * q_scale;

    if (tid < 128) {
      s_k[tid] = norm_k;
      s_q[tid] = norm_q;
    }

    // 2. Parallel reduction for kq_dot
    float local_kq = norm_k * norm_q;
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_kq += __shfl_xor(local_kq, offset);
    }
    if (lane_id == 0) {
      s_warp_kq[warp_id] = local_kq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_kq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_kq += s_warp_kq[warp];
      }
      s_scales[2] = total_kq;
    }
    __syncthreads();
    const float kq_dot = s_scales[2];

    // 3. Each lane processes half a row, then exchanges its two partial dots
    // with the adjacent lane that owns the other half.
    if (j < val_dim) {
      const auto* k_vec = reinterpret_cast<const float4*>(s_k) +
                          (row_segment * row_vectors_per_thread);
      const auto* q_vec = reinterpret_cast<const float4*>(s_q) +
                          (row_segment * row_vectors_per_thread);

      float u_j = 0.0F;
      float p_j = 0.0F;

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[vi] = s4;

        const float4 kv = k_vec[vi];
        const float4 qv = q_vec[vi];

        u_j += (s4.x * kv.x) + (s4.y * kv.y) + (s4.z * kv.z) + (s4.w * kv.w);
        p_j += (s4.x * qv.x) + (s4.y * qv.y) + (s4.z * qv.z) + (s4.w * qv.w);
      }

      u_j += __shfl_xor(u_j, 1);
      p_j += __shfl_xor(p_j, 1);
      const float d_j = (v_h[j] - u_j) * beta_h;
      if (row_segment == 0) {
        o_h[j] = p_j + (d_j * kq_dot);
      }

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        const float4 kv = k_vec[vi];
        s4.x += (d_j * kv.x);
        s4.y += (d_j * kv.y);
        s4.z += (d_j * kv.z);
        s4.w += (d_j * kv.w);
        s_reg[vi] = s4;
      }
    }
  }

  // Write final row state back to global memory ONCE
  if (j < val_dim) {
    float* s_row = s_matrix + (j * key_dim) + row_offset;
    auto* s_row_vec = reinterpret_cast<float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_row_vec[vi] = s_reg[vi];
    }
  }
}

__global__ void BatchedSSMPostNormGateKernel(
    const float* __restrict__ raw_out, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, float* __restrict__ out_buf,
    std::size_t batch_size, std::size_t inner_size, std::uint32_t num_heads,
    std::uint32_t val_dim) {
  const std::size_t h = blockIdx.x;
  const std::size_t t = blockIdx.y;
  if (h >= num_heads || t >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const float* in_h = raw_out + (t * inner_size) + (h * val_dim);
  float* o_h = out_buf + (t * inner_size) + (h * val_dim);

  const float in_val = (tid < val_dim) ? in_h[tid] : 0.0F;
  float o_sq = in_val * in_val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    o_sq += __shfl_xor(o_sq, offset);
  }

  __shared__ float s_warp_rms[4];
  if (lane_id == 0) {
    s_warp_rms[warp_id] = o_sq;
  }
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    const float total_o_sq =
        s_warp_rms[0] + s_warp_rms[1] + s_warp_rms[2] + s_warp_rms[3];
    const float mean_sq = total_o_sq / static_cast<float>(val_dim);
    s_inv_rms = rsqrtf(mean_sq + 1e-6F);
  }
  __syncthreads();

  if (tid < val_dim) {
    const float normed = in_val * s_inv_rms;
    const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
    float val = normed * w;
    if (gate != nullptr) {
      const float g = (gate + (t * inner_size))[h * val_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      val *= (g * sig);
    }
    o_h[tid] = val;
  }
}

void LaunchBatchedSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_x = (qkv_size + block_size - 1) / block_size;
  const dim3 conv_grid(static_cast<unsigned int>(grid_x),
                       static_cast<unsigned int>(batch_size));
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  BatchedSSMConvKernel<<<conv_grid, static_cast<unsigned int>(block_size), 0,
                         stream>>>(qkv_in, conv_weights, conv_layer_state,
                                   conv_out, batch_size, qkv_size);

  BatchedDeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      conv_out, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, out_buf,
      layer_idx, batch_size, qkv_size, num_key_heads, num_heads, key_dim,
      val_dim);

  const dim3 post_grid(num_heads, static_cast<unsigned int>(batch_size));
  const std::size_t inner_size = static_cast<std::size_t>(num_heads) * val_dim;
  BatchedSSMPostNormGateKernel<<<post_grid, 128, 0, stream>>>(
      out_buf, ssm_norm, gate, out_buf, batch_size, inner_size, num_heads,
      val_dim);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
