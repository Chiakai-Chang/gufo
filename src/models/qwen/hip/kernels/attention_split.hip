// Split prefill attention: AOTriton non-causal prefix plus the existing causal
// diagonal, merged by log-sum-exp (opt-c165-attn-split).
//
// At context depth the prefill attention cost is dominated by the fully visible
// prefix: for a chunk of N new tokens at depth D, N*D of the N*(D + N/2) query
// key pairs need no causal mask at all. The production tiled kernel computes
// QK^T and PV with `v_dot2_f32_f16` and reaches 4.36 TFLOPS, which is 15% of
// even its own instruction ceiling and 8% of the WMMA ceiling. AOTriton's
// pretuned gfx11xx `attn_fwd` reaches 15.6 TFLOPS on the same shapes, but its
// causal variants are not usable here: only `causal_type` None and
// WindowedAttention are compiled into the shipped images, and every
// WindowedAttention encoding returns success while writing nothing.
//
// So the prefix goes through AOTriton with no mask, the N x N diagonal keeps the
// tiled kernel, and the two partial softmaxes are combined exactly:
//
//   m  = max(lse_a, lse_b)
//   wa = exp(lse_a - m), wb = exp(lse_b - m)
//   O  = (O_a * wa + O_b * wb) / (wa + wb)
//
// which is the standard flash-attention partition merge. Both halves return
// normalized outputs and their log-sum-exp, so this is algebraically identical
// to one softmax over the union of the key ranges. The SiLU gate is applied once
// on the merged result, matching the unsplit kernel's epilogue.
#if defined(ENGINE_ENABLE_HIP)
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace strix::hip {
namespace {

// One thread per (token, query head, dim) element.
__global__ void ConvertQueriesToHalfKernel(const float* __restrict__ q,
                                           __half* __restrict__ out,
                                           std::size_t n) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = __float2half_rn(q[i]);
  }
}

// Merges the prefix and diagonal partial attentions and applies the gate.
// `lse_prefix` / `lse_diag` are [num_heads, batch] as AOTriton lays them out;
// the tiled kernel writes the same shape.
__global__ void MergeSplitAttentionKernel(
    const __half* __restrict__ out_prefix,
    const float* __restrict__ lse_prefix, const float* __restrict__ out_diag,
    const float* __restrict__ lse_diag, const float* __restrict__ gate,
    float* __restrict__ out, std::size_t batch, std::uint32_t num_heads,
    std::uint32_t head_dim) {
  const std::uint32_t head = blockIdx.y;
  const std::size_t token = blockIdx.x;
  if (token >= batch || head >= num_heads) {
    return;
  }
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t row = (token * attention_width) +
                          (static_cast<std::size_t>(head) * head_dim);
  const std::size_t stat = (static_cast<std::size_t>(head) * batch) + token;

  const float la = lse_prefix[stat];
  const float lb = lse_diag[stat];
  // A partial with no visible keys reports -inf; guard so it contributes zero.
  const float m = fmaxf(la, lb);
  const float wa = isfinite(la) ? __expf(la - m) : 0.0F;
  const float wb = isfinite(lb) ? __expf(lb - m) : 0.0F;
  const float denominator = wa + wb;
  const float inv = (denominator > 0.0F) ? (1.0F / denominator) : 0.0F;

  for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float a = __half2float(out_prefix[row + d]);
    const float b = out_diag[row + d];
    float value = ((a * wa) + (b * wb)) * inv;
    if (gate != nullptr) {
      value *= 1.0F / (1.0F + __expf(-gate[row + d]));
    }
    out[row + d] = value;
  }
}

std::uint32_t* AtomicCounter() {
  // AOTriton's dispatch expects a real persistent-counter tensor rather than a
  // null one. One zeroed word is enough and it is never read back on the host.
  static std::uint32_t* counter = [] {
    std::uint32_t* p = nullptr;
    HIP_CHECK(hipMalloc(&p, sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(p, 0, sizeof(std::uint32_t)));
    return p;
  }();
  return counter;
}

}  // namespace

bool LaunchQwenAotritonPrefixAttention(
    const __half* q_half, const void* k_cache_f16, const void* v_cache_f16,
    __half* out_prefix, float* lse_prefix, std::size_t batch_size,
    std::uint32_t prefix_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream) {
  using aotriton::DType;
  using aotriton::TensorView;
  if (prefix_length == 0 || batch_size == 0) {
    return false;
  }

  const std::uint64_t heads_q = num_heads;
  const std::uint64_t heads_kv = num_kv_heads;
  const std::uint64_t dim = head_dim;
  const std::uint64_t seq_q = batch_size;
  const std::uint64_t seq_k = prefix_length;
  const std::uint64_t q_width = heads_q * dim;
  const std::uint64_t kv_width = heads_kv * dim;

  // Q and the output are [token][head][dim]; the FP16 KV cache is
  // [position][kv_head][dim]. Both are expressed to AOTriton as strides over its
  // canonical {batch, head, seqlen, dim} view.
  const std::array<std::uint64_t, 4> q_sizes{1, heads_q, seq_q, dim};
  const std::array<std::uint64_t, 4> q_strides{seq_q * q_width, dim, q_width, 1};
  const std::array<std::uint64_t, 4> kv_sizes{1, heads_kv, seq_k, dim};
  const std::array<std::uint64_t, 4> kv_strides{seq_k * kv_width, dim, kv_width,
                                                1};

  aotriton::v3::flash::attn_fwd_params params;
  params.Q = TensorView<4>(reinterpret_cast<intptr_t>(q_half), q_sizes,
                           q_strides, DType::kFloat16);
  params.K = TensorView<4>(reinterpret_cast<intptr_t>(k_cache_f16), kv_sizes,
                           kv_strides, DType::kFloat16);
  params.V = TensorView<4>(reinterpret_cast<intptr_t>(v_cache_f16), kv_sizes,
                           kv_strides, DType::kFloat16);
  params.B = TensorView<4>::get_null_tensor(DType::kFloat16);
  params.A = TensorView<2>::get_null_tensor(DType::kFloat32);
  params.Sm_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  params.L = TensorView<2>(reinterpret_cast<intptr_t>(lse_prefix),
                           {heads_q, seq_q}, {seq_q, 1}, DType::kFloat32);
  params.Out = TensorView<4>(reinterpret_cast<intptr_t>(out_prefix), q_sizes,
                             q_strides, DType::kFloat16);
  params.Max_seqlen_q = static_cast<std::int32_t>(seq_q);
  params.Max_seqlen_k = static_cast<std::int32_t>(seq_k);
  params.dropout_p = 0.0F;
  params.encoded_softmax = TensorView<4>::get_null_tensor(DType::kFloat32);
  params.persistent_atomic_counter =
      TensorView<0>(reinterpret_cast<intptr_t>(AtomicCounter()),
                    DType::kUInt32);
  // The whole prefix is visible to every query in the chunk, so no mask.
  params.causal_type = aotriton::v3::flash::CausalType::None;
  params.varlen_type = aotriton::v3::flash::VarlenType::None;

  const hipError_t status = aotriton::v3::flash::attn_fwd(
      params, aotriton::v3::flash::attn_fwd_params::kVersion,
      aotriton::Stream(stream));
  return status == hipSuccess;
}

void LaunchConvertQueriesToHalf(const float* q, void* q_half,
                                std::size_t num_elements, hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((num_elements + threads - 1) / threads);
  ConvertQueriesToHalfKernel<<<blocks, threads, 0, stream>>>(
      q, static_cast<__half*>(q_half), num_elements);
}

void LaunchMergeSplitAttention(const void* out_prefix, const float* lse_prefix,
                               const float* out_diag, const float* lse_diag,
                               const float* gate, float* out,
                               std::size_t batch_size, std::uint32_t num_heads,
                               std::uint32_t head_dim, hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || head_dim == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(batch_size), num_heads);
  const unsigned int threads = (head_dim < 256) ? head_dim : 256U;
  MergeSplitAttentionKernel<<<grid, threads, 0, stream>>>(
      static_cast<const __half*>(out_prefix), lse_prefix, out_diag, lse_diag,
      gate, out, batch_size, num_heads, head_dim);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
