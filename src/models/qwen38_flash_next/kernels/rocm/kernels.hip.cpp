#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>

// First-light kernels: one thread or one block per output, no tiling beyond
// what correctness needs. Every kernel is written against the float32
// reference (reference.cpp); the layouts are the reference's layouts.

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

__global__ void EmbedKernel(const void* table, WeightType type,
                            const std::int32_t* tokens, float* res,
                            std::uint32_t hidden, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const auto* row = static_cast<const std::uint8_t*>(table) +
                    RowBytes(type, hidden) * tokens[t];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = RowElement(row, type, i);
    for (std::uint32_t s = 0; s < streams; ++s) {
      res[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
    }
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    ss += src[i] * src[i];
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
  // A grouped norm shares one gamma row across the groups of a token.
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0f);
  }
}

/// grid (tokens, hidden chunks of kThreads): the stream loop stays inside
/// the thread. With `inject_w` every block also contributes its slice of the
/// inject dot products, written as partial sums [t][stream][chunk] that
/// HcCombine totals (a fixed-order reduction, so the result is stable).
__global__ void HcMixEpilogueKernel(const float* xn, const float* gate,
                                    const float* inject_w, float* mixed,
                                    float* inject, std::uint32_t hidden,
                                    std::uint32_t streams) {
  constexpr std::uint32_t kMaxStreams = 8;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(streams) * hidden;
  const float* x = xn + static_cast<std::size_t>(t) * hc_dim;
  float v[kMaxStreams];
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < streams; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    v[s] = i < hidden ? x[idx] : 0.0f;
    if (i < hidden) {
      acc += v[s] * SigmoidF(gate[static_cast<std::size_t>(t) * hc_dim + idx]);
    }
  }
  if (i < hidden) {
    mixed[static_cast<std::size_t>(t) * hidden + i] =
        acc / static_cast<float>(streams);
  }
  if (inject_w == nullptr) {
    return;
  }
  // All inject dots reduce through one shared pass: wave sums first, then
  // one thread per logit totals the waves.
  __shared__ float partial[kMaxStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  for (std::uint32_t o = 0; o < streams; ++o) {
    const float* w = inject_w + o * hc_dim;
    float dot = 0.0f;
    if (i < hidden) {
      for (std::uint32_t s = 0; s < streams; ++s) {
        dot += w[static_cast<std::size_t>(s) * hidden + i] * v[s];
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < streams) {
    float total = 0.0f;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * streams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// Four adjacent hidden lanes per thread. This keeps the stream arithmetic in
/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}

template<typename XnT>
__global__ void HcMixEpilogueVec4Kernel(const XnT* xn, const float* gate,
                                        const float* inject_w, float* mixed,
                                        float* inject, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const XnT* x = xn + static_cast<std::size_t>(t) * hc_dim;
  float4 v[kStreams];
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    v[s] = i < hidden ? Load4(x + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (i < hidden) {
      const float4 g = *reinterpret_cast<const float4*>(
          gate + static_cast<std::size_t>(t) * hc_dim + idx);
      acc.x += v[s].x * SigmoidF(g.x);
      acc.y += v[s].y * SigmoidF(g.y);
      acc.z += v[s].z * SigmoidF(g.z);
      acc.w += v[s].w * SigmoidF(g.w);
    }
  }
  if (i < hidden) {
    constexpr float kInvStreams = 1.0F / static_cast<float>(kStreams);
    acc.x *= kInvStreams;
    acc.y *= kInvStreams;
    acc.z *= kInvStreams;
    acc.w *= kInvStreams;
    *reinterpret_cast<float4*>(mixed + static_cast<std::size_t>(t) * hidden +
                               i) = acc;
  }
  if (inject_w == nullptr) {
    return;
  }
  __shared__ float partial[kStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    const float* w = inject_w + static_cast<std::size_t>(o) * hc_dim;
    float dot = 0.0F;
    if (i < hidden) {
#pragma unroll
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        const float4 q = *reinterpret_cast<const float4*>(
            w + static_cast<std::size_t>(s) * hidden + i);
        dot += q.x * v[s].x;
        dot += q.y * v[s].y;
        dot += q.z * v[s].z;
        dot += q.w * v[s].w;
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// grid (tokens, streams): one block owns one residual stream, so the
/// grouped norm of the next mixer reduces over exactly its own elements.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// `XnT` is float for the reference route and __half for the F16 mixer
/// input route; a non-null `xn_q8` also receives the norm quantized into
/// the tiled Q8 layout (hidden % 32 == 0) for the W8A8 down projection.
template<typename XnT>
__global__ void HcCombineKernel(float* res, const float* block_out,
                                const float* inject, std::uint32_t inject_parts,
                                const float* gamma, XnT* xn, void* xn_q8,
                                std::uint32_t hidden, std::uint32_t streams,
                                float eps) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  float logit = 0.0f;
  for (std::uint32_t p = 0; p < inject_parts; ++p) {
    logit +=
        inject[(static_cast<std::size_t>(t) * streams + s) * inject_parts + p];
  }
  const float w = 2.0f * SigmoidF(logit / static_cast<float>(streams));
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float* dst = res + base;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] + src[i] * w;
    dst[i] = v;
    ss += v * v;
  }
  if (gamma == nullptr) {
    return;
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(hidden) + eps);
  const float* g = gamma + static_cast<std::size_t>(s) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] * scale * g[i];
    xn[base + i] = static_cast<XnT>(v);
    if (xn_q8 != nullptr) {
      // A wave holds one 32-wide block of the row per iteration: quantize
      // it for the next mixer's W8A8 down projection, K = streams * hidden.
      float max_abs = fabsf(v);
      for (int off = 16; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q = static_cast<std::int8_t>(roundf(v * id));
      const std::size_t num_blocks =
          (static_cast<std::size_t>(streams) * hidden) / 32;
      const std::size_t kb = (static_cast<std::size_t>(s) * hidden + i) / 32;
      const std::size_t lane = threadIdx.x & 31u;
      std::int8_t* tile =
          Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
      if (lane == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
}

__global__ void SiluScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] = SiluF(x[i] * scale);
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

__global__ void AddKernel(float* dst, const float* src, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] += src[i];
  }
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

/// One wave per weight row, kSmallGemmRows rows per block: the row is read
/// once (four consecutive elements per lane per step) while up to eight
/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmTokens = 8;
constexpr unsigned kSmallGemmRows = kThreads / 32;
__global__ void SmallGemmKernel(const void* w, WeightType type, const float* x,
                                float* out, std::uint32_t n_tokens,
                                std::uint32_t m, std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  float acc[kSmallGemmTokens] = {};
  for (std::uint32_t i0 = lane * 4; i0 < k; i0 += warpSize * 4) {
    float wv[4];
#pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
      wv[r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j < kSmallGemmTokens; ++j) {
      if (j < n_tokens) {
        const float* xr = x + static_cast<std::size_t>(j) * k + i0;
        float dot = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          dot += wv[r] * (i0 + r < k ? xr[r] : 0.0f);
        }
        acc[j] += dot;
      }
    }
  }
#pragma unroll
  for (unsigned j = 0; j < kSmallGemmTokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0 && j < n_tokens) {
      out[static_cast<std::size_t>(j) * m + row] = total;
    }
  }
}

__global__ void PleGateKernel(const float* key_n, const float* query_n,
                              const float* value, float* gated,
                              std::uint32_t hidden, std::uint32_t streams) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float dot = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dot += key_n[base + i] * query_n[base + i];
  }
  dot = BlockSum(dot, shared);
  const float sc = dot * rsqrtf(static_cast<float>(hidden));
  const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
  const float sign = sc < 0.0f ? -1.0f : (sc > 0.0f ? 1.0f : 0.0f);
  const float gate = SigmoidF(sign * mag);
  const float* v = value + static_cast<std::size_t>(t) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    gated[base + i] = v[i] * gate;
  }
}

__global__ void PleConvKernel(const float* in, const float* w,
                              const float* history, float* out,
                              std::uint32_t n_tokens, std::uint32_t channels,
                              std::uint32_t kernel, std::uint32_t dilation,
                              std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t) -
        static_cast<std::int32_t>((kernel - 1 - k) * dilation);
    const float v =
        src_t >= 0
            ? in[static_cast<std::size_t>(src_t) * channels + c]
            : history[static_cast<std::size_t>(hist + src_t) * channels + c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// New history row j is row (n_tokens + j) of [history ; in].
__global__ void HistoryShiftKernel(const float* in, std::uint32_t in_stride,
                                   const float* history, float* scratch,
                                   std::uint32_t n_tokens,
                                   std::uint32_t channels, std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(hist) * channels) {
    return;
  }
  const std::uint32_t j = idx / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = n_tokens + j;
  scratch[idx] = src < hist
                     ? history[static_cast<std::size_t>(src) * channels + c]
                     : in[static_cast<std::size_t>(src - hist) * in_stride + c];
}

__global__ void PleInjectKernel(float* res, const float* gated,
                                const float* conv, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    res[i] += gated[i] + conv[i];
  }
}

/// L2-normalizes the convolved q and k of one (token, key head) into the
/// packed [t][kh][d] buffers the recurrence streams from, so the serial loop
/// carries no block-wide reductions. One wave per (token, head) row.
constexpr unsigned kGdnDim = 128;
__global__ void GdnPrepKernel(const float* conv_out, float* qn, float* kn,
                              std::uint32_t n_rows, std::uint32_t k_heads,
                              std::uint32_t channels, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  const std::uint32_t t = row / k_heads;
  const std::uint32_t kh = row % k_heads;
  const float* conv = conv_out + static_cast<std::size_t>(t) * channels;
  const std::uint32_t per_lane = d / warpSize;
  float qv[d / 32];
  float kv[d / 32];
  float qs = 0.0f;
  float ks = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    qv[r] = conv[kh * d + i];
    kv[r] = conv[k_heads * d + kh * d + i];
    qs += qv[r] * qv[r];
    ks += kv[r] * kv[r];
  }
  qs = rsqrtf(WaveSum(qs) + eps);
  ks = rsqrtf(WaveSum(ks) + eps);
  float* q_out = qn + static_cast<std::size_t>(row) * d;
  float* k_out = kn + static_cast<std::size_t>(row) * d;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    q_out[i] = qv[r] * qs;
    k_out[i] = kv[r] * ks;
  }
}

__global__ void GdnPrepKqKernel(const float* conv_out, float* scales,
                                std::uint32_t n_tokens, std::uint32_t k_heads,
                                std::uint32_t channels, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t kh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (t >= n_tokens || kh >= k_heads) {
    return;
  }
  const float* row = conv_out + static_cast<std::size_t>(t) * channels;
  const auto* q = reinterpret_cast<const float4*>(row + kh * d);
  const auto* k = reinterpret_cast<const float4*>(row + (k_heads + kh) * d);
  const float4 q4 = q[lane];
  const float4 k4 = k[lane];
  float qs = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
  float ks = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
  float kq = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset >>= 1) {
    qs += __shfl_xor(qs, offset);
    ks += __shfl_xor(ks, offset);
    kq += __shfl_xor(kq, offset);
  }
  if (lane == 0) {
    float* dst = scales + (static_cast<std::size_t>(t) * k_heads + kh) * 3;
    dst[0] = rsqrtf(ks + eps);
    dst[1] = rsqrtf(qs + eps) * rsqrtf(static_cast<float>(d));
    dst[2] = kq;
  }
}

__global__ void GdnPrepAbKernel(const float* alpha_beta, const float* a,
                                const float* dt, float* ab, std::size_t count,
                                std::uint32_t v_heads) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = i % v_heads;
  const std::size_t t = i / v_heads;
  const float alpha = alpha_beta[t * 2 * v_heads + h];
  const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
  ab[i * 2] = __expf(a[h] * log1pf(__expf(alpha + dt[h])));
  ab[i * 2 + 1] = SigmoidF(beta);
}

template<int kMask>
__device__ __forceinline__ float GdnXorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

__device__ __forceinline__ void GdnRowReduce(float& u, float& p) {
  u = GdnXorAddDpp<1>(u);
  p = GdnXorAddDpp<1>(p);
  u = GdnXorAddDpp<2>(u);
  p = GdnXorAddDpp<2>(p);
}

/// Qwen row-split recurrence specialized to the model's 128x128 state. Four
/// lanes own one row, eight rows share a wave, and two blocks cover a head.
__launch_bounds__(256) __global__
    void GdnRowSplitKernel(const float* conv_out, const float* scales,
                           const float* ab, float* state, float* raw,
                           std::uint32_t n_tokens, std::uint32_t k_heads,
                           std::uint32_t v_heads) {
  constexpr int d = kGdnDim;
  constexpr int kKeysPerLane = 32;
  constexpr int kVec = kKeysPerLane / 4;
  constexpr int kLanesPerRow = d / kKeysPerLane;
  constexpr int kRowsPerWave = 32 / kLanesPerRow;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int segment = lane % kLanesPerRow;
  const int row_group = lane / kLanesPerRow;
  const int row =
      blockIdx.x * kRowsPerBlock + (tid >> 5) * kRowsPerWave + row_group;
  const int kh = h % k_heads;
  const int vec0 = segment * kVec;
  const int channels = 2 * k_heads * d + v_heads * d;

  float* state_row = state + (static_cast<std::size_t>(h) * d + row) * d;
  auto* state4 = reinterpret_cast<float4*>(state_row);
  float4 s[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    s[i] = state4[vec0 + i];
  }

  const float* q_base = conv_out + kh * d;
  const float* k_base = conv_out + (k_heads + kh) * d;
  const float* v_base = conv_out + 2 * k_heads * d + h * d;
  const float* scale_base = scales + kh * 3;
  const float* ab_base = ab + h * 2;
  float* out_base = raw + h * d;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const auto* q4 = reinterpret_cast<const float4*>(q_base);
    const auto* k4 = reinterpret_cast<const float4*>(k_base);
    float u = 0.0F;
    float p = 0.0F;
    const float decay = ab_base[0];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      s[i].x *= decay;
      s[i].y *= decay;
      s[i].z *= decay;
      s[i].w *= decay;
      const float4 kv = k4[vec0 + i];
      const float4 qv = q4[vec0 + i];
      u += s[i].x * kv.x + s[i].y * kv.y + s[i].z * kv.z + s[i].w * kv.w;
      p += s[i].x * qv.x + s[i].y * qv.y + s[i].z * qv.z + s[i].w * qv.w;
    }
    GdnRowReduce(u, p);
    const float inv_k = scale_base[0];
    const float q_scale = scale_base[1];
    const float delta = (v_base[row] - u * inv_k) * ab_base[1];
    if (segment == 0) {
      out_base[row] = p * q_scale + delta * inv_k * q_scale * scale_base[2];
    }
    const float update = delta * inv_k;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float4 kv = k4[vec0 + i];
      s[i].x += update * kv.x;
      s[i].y += update * kv.y;
      s[i].z += update * kv.z;
      s[i].w += update * kv.w;
    }
    q_base += channels;
    k_base += channels;
    v_base += channels;
    scale_base += k_heads * 3;
    ab_base += v_heads * 2;
    out_base += v_heads * d;
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    state4[vec0 + i] = s[i];
  }
}

/// grid (value heads, row groups): each block owns kGdnRowsPerBlock state
/// rows of one head, kGdnLanes threads per row, each lane a contiguous
/// slice of the key dimension. Rows of the delta rule are independent, so
/// splitting a head over blocks only re-reads its q/k. The token loop is
/// serial; every reduction stays inside a lane group, so the loop runs
/// barrier-free. The raw attention rows go out unnormalized;
/// GdnEpilogueKernel finishes them.
constexpr unsigned kGdnLanes = 4;
constexpr unsigned kGdnRowsPerBlock = 32;
__global__ void GdnKernel(const float* conv_out, const float* qn,
                          const float* kn, const float* alpha_beta,
                          const float* a, const float* dt, float* state,
                          float* raw, float* snapshots, std::uint32_t n_tokens,
                          std::uint32_t k_heads, std::uint32_t v_heads) {
  constexpr std::uint32_t d = kGdnDim;
  constexpr std::uint32_t slice = d / kGdnLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j =
      blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnLanes;
  const std::uint32_t lane = threadIdx.x % kGdnLanes;
  const std::uint32_t i0 = lane * slice;
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const float a_h = a[h];
  const float dt_h = dt[h];
  float* S = state + static_cast<std::size_t>(h) * d * d + j * d + i0;
  float row[slice];
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    row[i] = S[i];
  }
  const float q_scale = rsqrtf(static_cast<float>(d));
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* q = qn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float* k = kn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float vv = conv_out[static_cast<std::size_t>(t) * channels +
                              2 * k_heads * d + h * d + j];
    const float alpha = alpha_beta[t * 2 * v_heads + h];
    const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
    const float decay = __expf(a_h * log1pf(__expf(alpha + dt_h)));
    const float b = SigmoidF(beta);
    float kr[slice];
    float u = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      kr[i] = k[i];
      row[i] *= decay;
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      u += __shfl_xor(u, off, kGdnLanes);
    }
    const float delta = (vv - u) * b;
    float acc = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      row[i] += delta * kr[i];
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      acc += __shfl_xor(acc, off, kGdnLanes);
    }
    if (lane == 0) {
      raw[static_cast<std::size_t>(t) * v_heads * d + h * d + j] =
          acc * q_scale;
    }
    if (snapshots != nullptr) {
      float* snap = snapshots +
                    (static_cast<std::size_t>(t) * v_heads + h) * d * d +
                    j * d + i0;
#pragma unroll
      for (std::uint32_t i = 0; i < slice; ++i) {
        snap[i] = row[i];
      }
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    S[i] = row[i];
  }
}

/// Per-head RMSNorm of the raw attention rows and the sigmoid output gate,
/// one wave per (token, head) row.
__global__ void GdnEpilogueKernel(const float* raw, const float* z,
                                  std::uint32_t z_stride, const float* norm_w,
                                  float* out, std::uint32_t n_rows,
                                  std::uint32_t v_heads, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::size_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  // z rows are [t][v_heads*d] with a caller-side row stride.
  const float* zrow = z + (row / v_heads) * z_stride + (row % v_heads) * d;
  const std::uint32_t per_lane = d / warpSize;
  const float* src = raw + row * d;
  float v[d / 32];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    v[r] = src[r * warpSize + lane];
    ss += v[r] * v[r];
  }
  const float scale = rsqrtf(WaveSum(ss) / static_cast<float>(d) + eps);
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    out[row * d + i] = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
  }
}

/// Row j of the rolling state after token t is row (t + 1 + j) of the
/// concatenation [history ; rows], for any history depth `hist`.
__global__ void RollingSnapshotKernel(const float* rows,
                                      std::uint32_t row_stride,
                                      const float* history, float* snapshots,
                                      std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t per_token = static_cast<std::size_t>(hist) * channels;
  if (idx >= per_token * n_tokens) {
    return;
  }
  const std::uint32_t t = idx / per_token;
  const std::uint32_t j = (idx % per_token) / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = t + 1 + j;
  snapshots[idx] =
      src < hist ? history[static_cast<std::size_t>(src) * channels + c]
                 : rows[static_cast<std::size_t>(src - hist) * row_stride + c];
}

/// Causal conv over the chunk with the rolling state, SiLU applied.
__global__ void SsmConvKernel(const float* qkv, std::uint32_t qkv_stride,
                              const float* w, const float* conv_state,
                              float* out, std::uint32_t n_tokens,
                              std::uint32_t channels, std::uint32_t kernel) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>(kernel - 1 - k);
    const float v =
        src_t >= 0 ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
                   : conv_state[static_cast<std::size_t>(kernel - 1 + src_t) *
                                    channels +
                                c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

__global__ void UnpackQGateKernel(const float* qg, std::uint32_t qg_stride,
                                  float* q, float* gate, float* k, float* v,
                                  std::uint32_t heads, std::uint32_t d,
                                  std::uint32_t kv_width) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t width = static_cast<std::size_t>(heads) * d;
  const float* row = qg + static_cast<std::size_t>(t) * qg_stride;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const std::uint32_t h = i / d;
    const std::uint32_t j = i % d;
    q[t * width + i] = row[h * 2 * d + j];
    gate[t * width + i] = row[h * 2 * d + d + j];
  }
  // A stacked [q|gate ; k ; v] projection carries k and v after the heads.
  if (k != nullptr) {
    for (std::size_t i = threadIdx.x; i < kv_width; i += blockDim.x) {
      k[t * kv_width + i] = row[2 * width + i];
      v[t * kv_width + i] = row[2 * width + kv_width + i];
    }
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim,
                           const std::uint32_t* start_pos, float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  const float pos = static_cast<float>(*start_pos + t);
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(pos * freq, &s, &c);
    const float a = v[i];
    const float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::uint32_t row_dim,
                              const std::uint32_t* start_pos) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(*start_pos + t) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
}

__global__ void StoreRowsKernel(const float* src, float* dst,
                                std::uint32_t row_dim,
                                const std::uint32_t* start_pos) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    dst[static_cast<std::size_t>(*start_pos + t) * row_dim + i] =
        src[static_cast<std::size_t>(t) * row_dim + i];
  }
}

/// grid: the most blocks a batch can complete. Block b pools raw keys
/// [b*ratio, (b+1)*ratio) once every one of them is stored, i.e. for
/// b in [*first_block, (*start_pos + n_tokens) / ratio).
__global__ void PoolBlocksKernel(const float* raw, const float* gamma,
                                 float* blocks,
                                 const std::uint32_t* first_block,
                                 const std::uint32_t* start_pos,
                                 std::uint32_t n_tokens, std::uint32_t ratio,
                                 std::uint32_t dim, std::uint32_t rotary_dim,
                                 float theta, float eps) {
  __shared__ float v[256];
  __shared__ float shared[32];
  const std::uint32_t b = *first_block + blockIdx.x;
  if (b >= (*start_pos + n_tokens) / ratio) {
    return;
  }
  const std::uint32_t i = threadIdx.x;
  float mean = 0.0f;
  if (i < dim) {
    for (std::uint32_t r = 0; r < ratio; ++r) {
      mean += raw[(static_cast<std::size_t>(b) * ratio + r) * dim + i];
    }
    mean /= static_cast<float>(ratio);
  }
  const float ss = BlockSum(i < dim ? mean * mean : 0.0f, shared);
  const float scale = rsqrtf(ss / static_cast<float>(dim) + eps);
  if (i < dim) {
    v[i] = mean * scale * gamma[i];
  }
  __syncthreads();
  const std::uint32_t half = rotary_dim / 2;
  float outv = i < dim ? v[i] : 0.0f;
  if (i < rotary_dim) {
    const std::uint32_t p = i < half ? i : i - half;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(p) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(static_cast<float>(b * ratio) * freq, &s, &c);
    const float a = v[p];
    const float bb = v[p + half];
    outv = i < half ? a * c - bb * s : a * s + bb * c;
  }
  if (i < dim) {
    blocks[static_cast<std::size_t>(b) * dim + i] = outv;
  }
}

/// One block per query. Scores every complete block, then finds the
/// budget-th largest score by a 32-step bit search over the float ordering
/// and marks the blocks at or above it (ties resolved by lowest index).
__global__ void SelectBlocksKernel(
    const float* q, const float* blocks, std::uint32_t* mask, float* scores,
    const std::uint32_t* start_pos, std::uint32_t first_token,
    std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
    std::uint32_t budget, std::uint32_t mask_words, std::uint32_t max_blocks) {
  __shared__ float shared[32];
  __shared__ std::uint32_t counts[32];
  __shared__ float qs[4 * 128];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t pos = *start_pos + first_token + t;
  const std::uint32_t complete = (pos + 1) / ratio;
  std::uint32_t* words = mask + static_cast<std::size_t>(t) * mask_words;
  for (std::uint32_t w = threadIdx.x; w < mask_words; w += blockDim.x) {
    words[w] = complete <= budget ? 0xFFFFFFFFu : 0u;
  }
  if (complete <= budget) {
    return;
  }
  for (std::uint32_t i = threadIdx.x; i < heads * dim; i += blockDim.x) {
    qs[i] = q[static_cast<std::size_t>(t) * heads * dim + i];
  }
  __syncthreads();
  float* sc = scores + static_cast<std::size_t>(blockIdx.x) * max_blocks;
  for (std::uint32_t b = threadIdx.x; b < complete; b += blockDim.x) {
    const float* kb = blocks + static_cast<std::size_t>(b) * dim;
    float total = 0.0f;
    for (std::uint32_t h = 0; h < heads; ++h) {
      float dot = 0.0f;
      for (std::uint32_t i = 0; i < dim; ++i) {
        dot += qs[h * dim + i] * kb[i];
      }
      total += fmaxf(dot, 0.0f);
    }
    sc[b] = total;
  }
  __syncthreads();
  // Scores are non-negative, so their bit patterns order like unsigned ints.
  std::uint32_t threshold = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const std::uint32_t candidate = threshold | (1u << bit);
    std::uint32_t count = 0;
    for (std::uint32_t b = threadIdx.x; b < complete; b += blockDim.x) {
      count += __float_as_uint(sc[b]) >= candidate ? 1u : 0u;
    }
    const float total = BlockSum(static_cast<float>(count), shared);
    if (static_cast<std::uint32_t>(total + 0.5f) >= budget) {
      threshold = candidate;
    }
  }
  // Blocks strictly above the threshold are in; ties fill the remainder in
  // index order.
  std::uint32_t above = 0;
  for (std::uint32_t b = threadIdx.x; b < complete; b += blockDim.x) {
    above += __float_as_uint(sc[b]) > threshold ? 1u : 0u;
  }
  const std::uint32_t n_above = static_cast<std::uint32_t>(
      BlockSum(static_cast<float>(above), shared) + 0.5f);
  __syncthreads();
  if (threadIdx.x == 0) {
    counts[0] = budget - n_above;
  }
  __syncthreads();
  for (std::uint32_t b = threadIdx.x; b < complete; b += blockDim.x) {
    if (__float_as_uint(sc[b]) > threshold) {
      atomicOr(&words[b / 32], 1u << (b % 32));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint32_t remaining = counts[0];
    for (std::uint32_t b = 0; b < complete && remaining > 0; ++b) {
      if (__float_as_uint(sc[b]) == threshold) {
        words[b / 32] |= 1u << (b % 32);
        --remaining;
      }
    }
  }
}

/// grid (heads, queries), block 256 = head dim. Keys stream in tiles of 256
/// positions: lane j scores key j of the tile, then lane d accumulates
/// value column d with online softmax rescaling.
__global__ void AttentionKernel(const float* q, const __half* k_cache,
                                const __half* v_cache,
                                const std::uint32_t* mask,
                                std::uint32_t mask_words, float* out,
                                const std::uint32_t* start_pos,
                                std::uint32_t heads, std::uint32_t kv_heads,
                                std::uint32_t d, std::uint32_t ratio) {
  constexpr std::uint32_t kTile = 256;
  __shared__ float qs[256];
  __shared__ float p[kTile];
  __shared__ float shared[32];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t tail_start = (n_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr
                      : mask + static_cast<std::size_t>(t) * mask_words;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t i = threadIdx.x;
  qs[i] = i < d ? q[(static_cast<std::size_t>(t) * heads + h) * d + i] : 0.0f;
  __syncthreads();
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * d;
  for (std::uint32_t tile = 0; tile < n_kv; tile += kTile) {
    const std::uint32_t j = tile + i;
    float s = -INFINITY;
    if (j < n_kv) {
      bool visible = j >= tail_start || words == nullptr;
      if (!visible) {
        const std::uint32_t b = j / ratio;
        visible = (words[b / 32] >> (b % 32)) & 1u;
      }
      if (visible) {
        const __half* krow = k_cache + j * kv_stride + kvh * d;
        float dot = 0.0f;
        for (std::uint32_t x = 0; x < d; ++x) {
          dot += qs[x] * __half2float(krow[x]);
        }
        s = dot * scale;
      }
    }
    const float tile_max = BlockMax(s, shared);
    const float m_new = fmaxf(m, tile_max);
    const float rescale = m == -INFINITY ? 0.0f : __expf(m - m_new);
    const float pj = s == -INFINITY ? 0.0f : __expf(s - m_new);
    p[i] = pj;
    const float tile_sum = BlockSum(pj, shared);
    l = l * rescale + tile_sum;
    acc *= rescale;
    m = m_new;
    __syncthreads();
    if (i < d) {
      const std::uint32_t limit = min(kTile, n_kv - tile);
      for (std::uint32_t jj = 0; jj < limit; ++jj) {
        const float w = p[jj];
        if (w != 0.0f) {
          acc +=
              w * __half2float(v_cache[(tile + jj) * kv_stride + kvh * d + i]);
        }
      }
    }
    __syncthreads();
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  __shared__ float probs[1024];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = BlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    // Block argmax: value first, lowest index on ties.
    float best = -1.0f;
    std::uint32_t best_e = 0;
    for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
      if (probs[e] > best) {
        best = probs[e];
        best_e = e;
      }
    }
    const float block_best = BlockMax(best, shared);
    __syncthreads();
    if (threadIdx.x == 0) {
      chosen[slot] = 0xFFFFFFFFu;
    }
    __syncthreads();
    if (best == block_best) {
      atomicMin(&chosen[slot], best_e);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      chosen_p[slot] = probs[chosen[slot]];
      probs[chosen[slot]] = -1.0f;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}

/// grid (tokens, dim chunks of kThreads).
__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}

/// Four adjacent output lanes per thread: the same reduction over the top-k
/// slots with a quarter of the waves, so the per-wave issue overhead no
/// longer hides the streaming reads.
__global__ void MoeEpilogueVec4Kernel(const float* expert_out,
                                      const float* weights,
                                      const float* shared_out,
                                      const float* gate,
                                      std::uint32_t gate_stride, float* out,
                                      std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const float4 v = *reinterpret_cast<const float4*>(
        rows + static_cast<std::size_t>(s) * dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  acc.x += g * sh.x;
  acc.y += g * sh.y;
  acc.z += g * sh.z;
  acc.w += g * sh.w;
  *reinterpret_cast<float4*>(out + idx) = acc;
}

/// dst[t] = row < 0 ? alt[t] : base[(row + t)]: the draft block's hidden
/// input, a kept trunk row or its own carried residual.
__global__ void MtpHiddenKernel(const float* base, const float* alt,
                                const std::int32_t* row, float* dst,
                                std::uint32_t width) {
  const std::uint32_t t = blockIdx.x;
  const float* src = *row < 0
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(*row) + t) * width;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i];
  }
}

__global__ void MtpConcatKernel(const float* embd_n, const float* h_n,
                                float* concat, std::uint32_t hidden,
                                std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t s = 0; s < streams; ++s) {
    float* dst =
        concat + (static_cast<std::size_t>(t) * streams + s) * 2 * hidden;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      dst[i] = embd_n[static_cast<std::size_t>(t) * hidden + i];
      dst[hidden + i] =
          h_n[(static_cast<std::size_t>(t) * streams + s) * hidden + i];
    }
  }
}

__global__ void ArgmaxKernel(const float* logits, std::int32_t* out,
                             std::uint32_t vocab) {
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen;
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  float best = -INFINITY;
  std::uint32_t best_i = 0;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    if (src[i] > best) {
      best = src[i];
      best_i = i;
    }
  }
  const float block_best = BlockMax(best, shared);
  if (threadIdx.x == 0) {
    chosen = 0xFFFFFFFFu;
  }
  __syncthreads();
  if (best == block_best) {
    atomicMin(&chosen, best_i);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    out[t] = static_cast<std::int32_t>(chosen);
  }
}

__global__ void CopyKernel(const float* src, float* dst, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

__global__ void ChecksumKernel(const float* x, std::size_t count, float* out) {
  __shared__ float shared[32];
  float sum = 0.0f;
  float abs_sum = 0.0f;
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
    sum += x[i];
    abs_sum += fabsf(x[i]);
  }
  sum = BlockSum(sum, shared);
  abs_sum = BlockSum(abs_sum, shared);
  if (threadIdx.x == 0) {
    out[0] = sum;
    out[1] = abs_sum;
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// Masked prefill attention on the WMMA matrix cores, ported from the Qwen
// 27B route (src/models/qwen/hip/kernels/attention_wmma.hip) to this
// model's 24 x 256 query heads over two KV heads. Wave32 fragment layout: A
// holds row L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous
// k, and C element i is row 2i + L/16, column L%16. Every block owns 32
// queries x 2 heads = 64 rows against one 16-key tile at a time; eight waves
// split as four S tiles x two halves of the head dimension, and the next
// tile's K/V is prefetched into registers behind the current tile's S,
// softmax and PV phases. V reaches LDS transposed with one key per lane so
// the writes stay conflict-free. The optional block mask follows the model's
// sparse selection: keys at or past the query's incomplete tail block are
// always visible, earlier blocks only when their bit is set.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 24;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes: two 16-byte loads and a
// bitcast. Every fragment base here is 16-byte aligned by construction.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

template<std::uint32_t kQueryRows, std::uint32_t kKeys>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t ratio) {
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kWmmaGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][16];
  __shared__ __half p_lds[kRows][kKeys];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < n_tokens;
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][2] = {};
  for (std::uint32_t r = tid; r < kRows; r += 256) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * kHeadDim);

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    const std::uint32_t key_position = key_start + v_key;
    const auto* src =
        v_base + (static_cast<std::size_t>(key_position) * kWmmaKvWidth);
    const bool live = key_position < context_end;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? *reinterpret_cast<const uint4*>(
                    k_base +
                    (static_cast<std::size_t>(key_position) * kWmmaKvWidth) +
                    d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs];
  uint4 v_pre[kVRegs];
  load_k(0, k_cur);
  load_v(0, v_cur);

  for (std::uint32_t key_start = 0; key_start < max_visible;
       key_start += kKeys) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile. Everything below covers its latency.
    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      load_v(next_start, v_pre);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot; the reader sums them.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      // Keys in the query's own incomplete block are always visible; earlier
      // blocks follow the selection mask.
      const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
      const std::uint32_t* words =
          mask == nullptr || local_query >= n_tokens
              ? nullptr
              : mask + static_cast<std::size_t>(local_query) * mask_words;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        bool valid = local_query < n_tokens && key_position <= absolute_query &&
                     key_position < context_end;
        if (valid && words != nullptr && key_position < tail_start) {
          const std::uint32_t b = key_position / ratio;
          valid = ((words[b / 32] >> (b % 32)) & 1u) != 0u;
        }
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      if (seg == 0) {
        row_max[rg] = next_max;
        row_sum[rg] = (row_sum[rg] * prior_scale) + part_sum;
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

    // --- rescale the running O by the new maximum. A lane touches only rows
    // 2i + half_id, so the factors are read once per row block.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- stage V transposed. The barrier after the softmax already
    // separates every read of K from these writes to the same buffer.
    {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
    }
    __syncthreads();

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          o_acc[rb][t] = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
  }
  __syncthreads();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= n_tokens) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= SigmoidF(gate[offset]);
        }
        out[offset] = value;
      }
    }
  }
}

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

/// One wave per (token, 32-wide K block): per-block absmax scale, codes
/// written straight into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledKernel(const float* __restrict__ x,
                                      void* __restrict__ y, std::size_t batch,
                                      std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float val = x[(tok * k) + (blk * 32) + lane_id];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(val * id));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// Two-dimensionally blocked W8A8 WMMA GEMM: block = BM rows x BN tokens, BK
/// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
/// Out-of-range rows and K blocks are clamped and their scale zeroed, so
/// they contribute exactly zero without divergence. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
    void W8A8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                   const void* __restrict__ x_blocks,
                                   float* __restrict__ y, std::size_t batch,
                                   std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  // Transpose each 16x16 result tile through LDS so the global stores cover
  // 16 consecutive rows of one token instead of 16 separate cache lines.
  __syncthreads();
  float* tile_scratch =
      reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
            static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
          const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
          if (tok < batch && r < m) {
            y[(tok * m) + r] = tile_scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// Routed int8 WMMA expert GEMM (Q4_K gate/up, Q5_1 down) over the sorted
// assignment rows. The vendored MMQ tier issues `v_dot4` tiles per 32-column
// bucket slice and re-reads an expert's weight panel once per slice; here
// the 27B blocked W8A8 structure is kept (LDS-staged 16x16 int8 fragments,
// one K stage prefetched into registers) with the expert as the grid's z
// axis, the token macro tile sized to the mean bucket, and the weight fetch
// decoding the K-quant block to codes plus a per-32 (scale, offset) pair:
//
//     w = scale * q - offset,   sum_j w_j x_j = scale * sum q_j x_j - offset *
//     sum x_j
//
// with the per-token activation sum recovered in-register from the staged
// codes (`sudot4` against a ones vector). Assignment rows are compacted by
// expert with every bucket padded to a 16-row tile (`pad_bounds`), so a
// token tile never straddles experts; `rows_out` maps a compact row to its
// (token, slot) output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

__device__ __forceinline__ void GetQKScaleMin(std::size_t index,
                                              const std::uint8_t* packed,
                                              std::uint8_t& sc,
                                              std::uint8_t& m) {
  if (index < 4) {
    sc = packed[index] & 0x3FU;
    m = packed[index + 4] & 0x3FU;
    return;
  }
  sc = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                 ((packed[index - 4] >> 6U) << 4U));
  m = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                ((packed[index] >> 6U) << 4U));
}

/// One 32-element K block of a weight row as unsigned codes plus the affine
/// pair: w = scale * q - offset.
struct RoutedBlock {
  int32x4_t q0;
  int32x4_t q1;
  float scale;
  float offset;
};

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

/// `header` caches the 16-byte header of the 256-element block a thread's
/// row is currently in: every wave-wide load of it touches sixteen cache
/// lines for sixteen useful bytes, so it is fetched once per eight K blocks
/// rather than per stage.
template<WeightType kType>
__device__ __forceinline__ RoutedBlock
DecodeRoutedBlock(const std::uint8_t* __restrict__ row, std::size_t kb,
                  uint4& header, int& header_block) {
  RoutedBlock out;
  if constexpr (kType == WeightType::kQ4_K) {
    const int block = static_cast<int>(kb / 8);
    const auto* blk = reinterpret_cast<const uint4*>(row) + (block * 9);
    if (block != header_block) {
      header = blk[0];
      header_block = block;
    }
    const std::uint32_t sb32 = static_cast<std::uint32_t>(kb % 8);
    const unsigned shift = 4U * (sb32 & 1U);
    // block_q4_K is 144 bytes with qs at +16, so every 32-byte code group
    // of an expert table sits on a 16-byte boundary: two vector loads.
    const uint4 p0 = blk[1 + (sb32 / 2) * 2];
    const uint4 p1 = blk[2 + (sb32 / 2) * 2];
    const std::uint32_t packed[8] = {p0.x, p0.y, p0.z, p0.w,
                                     p1.x, p1.y, p1.z, p1.w};
    std::uint32_t lo[4];
    std::uint32_t hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      lo[i] = (packed[i] >> shift) & 0x0F0F0F0FU;
      hi[i] = (packed[4 + i] >> shift) & 0x0F0F0F0FU;
    }
    __builtin_memcpy(&out.q0, lo, 16);
    __builtin_memcpy(&out.q1, hi, 16);
    // The 6-bit scale/min pairs, packed as llama.cpp's get_scale_min_k4.
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    if (sb32 < 4) {
      sc = HeaderByte(header, 4 + sb32) & 0x3FU;
      mn = HeaderByte(header, 8 + sb32) & 0x3FU;
    } else {
      sc = (HeaderByte(header, 8 + sb32) & 0x0FU) |
           ((HeaderByte(header, sb32) >> 6U) << 4U);
      mn = (HeaderByte(header, 8 + sb32) >> 4U) |
           ((HeaderByte(header, 4 + sb32) >> 6U) << 4U);
    }
    const __half2 dm = __builtin_bit_cast(__half2, header.x);
    out.scale = __low2float(dm) * static_cast<float>(sc);
    out.offset = __high2float(dm) * static_cast<float>(mn);
  } else {
    static_assert(kType == WeightType::kQ5_1, "routed weight type");
    // block_q5_1 is 24 bytes: three 8-byte words, d|m, qh, then the codes.
    const auto* words = reinterpret_cast<const uint2*>(row) + (kb * 3);
    const uint2 w0 = words[0];
    const uint2 w1 = words[1];
    const uint2 w2 = words[2];
    const std::uint32_t packed[4] = {w1.x, w1.y, w2.x, w2.y};
    const std::uint32_t qh = w0.y;
    const __half2 dm = __builtin_bit_cast(__half2, w0.x);
    std::uint32_t lo[4];
    std::uint32_t hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Element j of the block: low nibble of qs[j % 16] for j < 16, high
      // nibble for j >= 16; bit j of qh is its fifth bit.
      std::uint32_t low_bits = 0;
      std::uint32_t high_bits = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int j = (i * 4) + b;
        low_bits |= ((qh >> j) & 1U) << (8 * b + 4);
        high_bits |= ((qh >> (j + 16)) & 1U) << (8 * b + 4);
      }
      lo[i] = (packed[i] & 0x0F0F0F0FU) | low_bits;
      hi[i] = ((packed[i] >> 4U) & 0x0F0F0F0FU) | high_bits;
    }
    __builtin_memcpy(&out.q0, lo, 16);
    __builtin_memcpy(&out.q1, hi, 16);
    out.scale = __low2float(dm);
    out.offset = -__high2float(dm);
  }
  return out;
}

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedRowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K ? (k / 256) * sizeof(Q4KBlock)
                                    : (k / 32) * sizeof(Q5_1Block);
}

/// grid (token macro tiles up to the widest padded bucket, m / BM, experts).
/// Block (j, y, e) computes rows y*BM.. of expert e against its compact rows
/// [pad_bounds[e] + j*BN, +BN) and scatters them to `out[rows_out[c]][row]`.
template<WeightType kType, int BM, int BN, int BK, int WM, int WN, int kDepth>
__launch_bounds__(256) __global__
    void RoutedWmmaGEMMKernel(const void* __restrict__ w,
                              const void* __restrict__ x_blocks,
                              const std::int32_t* __restrict__ pad_bounds,
                              const std::int32_t* __restrict__ rows_in,
                              const std::int32_t* __restrict__ rows_out,
                              float* __restrict__ out, std::size_t m,
                              std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8, "one wave per staged token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kRowTiles][16];
  __shared__ float s_off[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const int expert = static_cast<int>(blockIdx.z);
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int t_local = static_cast<int>(blockIdx.x) * BN;
  if (t_local >= bucket_rows) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t row_bytes = RoutedRowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  constexpr int kPrefetch = (BM * BK) / 256;
  // Two stages in flight: with only twelve WMMAs of work per stage, one
  // stage of prefetch leaves most of the scattered K-quant block reads'
  // latency exposed.
  struct Stage {
    int32x4_t q0[kPrefetch];
    int32x4_t q1[kPrefetch];
    float dw[kPrefetch];
    float off[kPrefetch];
    int32x4_t b[BK];
    float dx[BK];
  };
  static_assert(kDepth == 1 || kDepth == 2, "one or two stages in flight");
  Stage stage0;
  Stage stage1;
  const int b_tile = wave_id;

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const std::uint8_t* w_row[kPrefetch];
  float row_live[kPrefetch];
  uint4 header[kPrefetch];
  int header_block[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_expert + (static_cast<std::size_t>(r_clamped) * row_bytes);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
    header[p] = make_uint4(0u, 0u, 0u, 0u);
    header_block[p] = -1;
  }
  // The activations are quantized once per source row (token, or slot row
  // for the down projection) in the tiled layout; each lane of a token
  // subtile gathers its own row's fragment slice through `rows_in`. A row
  // past the bucket (padding, or the next expert's) reads row 0 with a zero
  // scale.
  const bool b_live = b_tile < kTokTiles;
  const int c_lane = t_local + (b_tile * 16) + sub_lane;
  const std::int32_t src_row =
      (b_live && c_lane < bucket_rows)
          ? rows_in[static_cast<std::size_t>(bucket_begin) + c_lane]
          : -1;
  const float tile_scale = src_row >= 0 ? 1.0F : 0.0F;
  const std::size_t src = src_row >= 0 ? static_cast<std::size_t>(src_row) : 0;
  const auto* b_base =
      static_cast<const std::int8_t*>(x_blocks) +
      ((src / kQ8ActTileTokens) * num_blocks * kQ8ActTileBytes);
  const std::size_t b_lane_offset = (static_cast<std::size_t>(half_id) * 256) +
                                    ((src % kQ8ActTileTokens) * 16);
  const std::size_t b_scale_offset =
      kQ8ActScaleOffset + ((src % kQ8ActTileTokens) * sizeof(float));

  const auto fetch_stage = [&](int kb0, Stage& st) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const float live = (kb < num_kb) ? row_live[p] : 0.0F;
      const RoutedBlock blk = DecodeRoutedBlock<kType>(
          w_row[p], static_cast<std::size_t>(kb_clamped), header[p],
          header_block[p]);
      st.q0[p] = blk.q0;
      st.q1[p] = blk.q1;
      st.dw[p] = blk.scale * live;
      st.off[p] = blk.offset * live;
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        st.b[i] = *reinterpret_cast<const int32x4_t*>(tile + b_lane_offset);
        st.dx[i] = ((kb < num_kb) ? tile_scale : 0.0F) *
                   *reinterpret_cast<const float*>(tile + b_scale_offset);
      }
    }
  };

  const auto commit_stage = [&](const Stage& st) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      const int slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
      s_a[kk][rs][rl] = st.q0[p];
      s_a[kk][rs][16 + rl] = st.q1[p];
      s_dw[kk][rs][slot] = st.dw[p];
      s_off[kk][rs][slot] = st.off[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = st.b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = st.dx[i];
        }
      }
    }
  };

  const auto compute_stage = [&]() {
  // Not unrolled: with four K blocks per stage the unrolled body hoists
  // every block's fragment and scale reads into registers and spills.
#pragma unroll 1
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
      float off[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
        const float4 olo =
            *reinterpret_cast<const float4*>(&s_off[kb][rs][half_id * 8]);
        const float4 oup =
            *reinterpret_cast<const float4*>(&s_off[kb][rs][(half_id * 8) + 4]);
        off[i][0] = olo.x;
        off[i][1] = olo.y;
        off[i][2] = olo.z;
        off[i][3] = olo.w;
        off[i][4] = oup.x;
        off[i][5] = oup.y;
        off[i][6] = oup.z;
        off[i][7] = oup.w;
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
        // Token sub_lane's activation sum over this K block, from its own
        // staged codes: signed codes against an unsigned ones vector.
        int qsum = 0;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          qsum = __builtin_amdgcn_sudot4(true, b0[v], false, 0x01010101, qsum,
                                         false);
          qsum = __builtin_amdgcn_sudot4(true, b1[v], false, 0x01010101, qsum,
                                         false);
        }
        const float sx = dx * static_cast<float>(qsum);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] +=
                ((dw[i][l] * dx) * static_cast<float>(c[l])) - (off[i][l] * sx);
          }
        }
      }
    }
  };

  if constexpr (kDepth == 2) {
    fetch_stage(0, stage0);
    fetch_stage(BK, stage1);
    for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * BK) {
      commit_stage(stage0);
      __syncthreads();
      if (kb0 + (2 * BK) < num_kb) {
        fetch_stage(kb0 + (2 * BK), stage0);
      }
      compute_stage();
      __syncthreads();
      if (kb0 + BK < num_kb) {
        commit_stage(stage1);
        __syncthreads();
        if (kb0 + (3 * BK) < num_kb) {
          fetch_stage(kb0 + (3 * BK), stage1);
        }
        compute_stage();
        __syncthreads();
      }
    }
  } else {
    fetch_stage(0, stage0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      commit_stage(stage0);
      __syncthreads();
      if (kb0 + BK < num_kb) {
        fetch_stage(kb0 + BK, stage0);
      }
      compute_stage();
      __syncthreads();
    }
  }

  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  __syncthreads();
  float* tile_scratch =
      reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const int t0 = t_local + (((wave_tok * kWaveTokTiles) + j) * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
        if (t < bucket_rows && r < m) {
          const std::int32_t dst =
              rows_out[static_cast<std::size_t>(bucket_begin) + t];
          if (dst >= 0) {
            out[(static_cast<std::size_t>(dst) * m) + r] = tile_scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

}  // namespace

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     table, type, tokens, res, hidden, streams);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  // Every (row, group) pair is one block; the kernel recovers the group
  // from the block index to pick its gamma slice.
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows * groups), dim3(kThreads), 0,
                     stream, x, gamma, out, dim / groups, groups, eps);
}

std::uint32_t HcInjectParts(std::uint32_t hidden) {
  return Blocks(hidden);
}

std::uint32_t HcInjectPartsVec4(std::uint32_t hidden) {
  return Blocks((hidden + 3) / 4);
}

void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream) {
  hipLaunchKernelGGL(HcMixEpilogueKernel, dim3(n_tokens, Blocks(hidden)),
                     dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                     inject, hidden, streams);
}

void HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream) {
  if (streams != 4 || hidden % 4 != 0) {
    HcMixEpilogue(xn, gate, inject_w, mixed, inject, n_tokens, hidden, streams,
                  stream);
    return;
  }
  hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<float>,
                     dim3(n_tokens, HcInjectPartsVec4(hidden)), dim3(kThreads),
                     0, stream, xn, gate, inject_w, mixed, inject, hidden);
}

void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream) {
  hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, HcInjectPartsVec4(hidden)), dim3(kThreads),
                     0, stream, xn, gate, inject_w, mixed, inject, hidden);
}

void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(HcCombineKernel<float>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, nullptr, hidden, streams, eps);
}

void HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(HcCombineKernel<__half>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, xn_q8, hidden, streams, eps);
}

void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SiluScaleKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, scale, count);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, count);
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(SigmoidMulKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count);
}

void AddInPlace(float* dst, const float* src, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(count)), dim3(kThreads), 0, stream,
                     dst, src, count);
}

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

std::size_t Q8TiledBytes(std::size_t batch, std::size_t k) {
  // The GEMM stages whole 128-token macro tiles, so the buffer is sized to
  // the batch rounded up to one (the padding tiles are never read as
  // results, only staged).
  constexpr std::size_t kMacroTokens = 128;
  const std::size_t padded =
      (batch + kMacroTokens - 1) / kMacroTokens * kMacroTokens;
  return (padded / kQ8ActTileTokens) * (k / 32) * kQ8ActTileBytes;
}

void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream) {
  const std::size_t blocks = batch * (k / 32);
  const std::size_t waves = kThreads / 32;
  hipLaunchKernelGGL(QuantizeQ8TiledKernel, dim3((blocks + waves - 1) / waves),
                     dim3(kThreads), 0, stream, x, out, batch, k);
}

bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // A 128-token macro tile is the throughput configuration; short chunks
  // would leave most of it idle and take the 64-token variant. A narrow
  // projection (the 320-row mixer down over K = 10240) gets 32-row tiles so
  // it still fills the device.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kNarrowBM, kBN, 4, 4, 2>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, batch,
                       m, k);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  }
  return true;
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  const std::size_t rows = RoutedCompactRows(slots, n_experts);
  (void)hipMemsetAsync(rows_token, 0xFF, rows * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(rows_slot, 0xFF, rows * sizeof(std::int32_t), stream);
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

bool RoutedWmmaGemm(const void* w, WeightType type, const void* x_tiled,
                    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                    const std::int32_t* rows_out, float* out, std::size_t m,
                    std::size_t k, std::uint32_t n_experts,
                    std::uint32_t max_bucket_rows, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBN = 48;
  // Four K blocks per stage: four lanes then cover one row's 128 contiguous
  // bytes, so a wave-wide weight load touches eight cache lines, not
  // sixteen.
  constexpr int kBK = 4;
  constexpr int kDepth = 1;
  const std::size_t block_elems = type == WeightType::kQ4_K ? 256 : 32;
  if (m == 0 || k == 0 || k % block_elems != 0 || max_bucket_rows == 0) {
    return false;
  }
  const dim3 grid((max_bucket_rows + kBN - 1) / kBN,
                  static_cast<unsigned int>((m + kBM - 1) / kBM), n_experts);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL((RoutedWmmaGEMMKernel<WeightType::kQ4_K, kBM, kBN, kBK,
                                               8, 1, kDepth>),
                         grid, dim3(kThreads), 0, stream, w, x_tiled,
                         pad_bounds, rows_in, rows_out, out, m, k);
      return true;
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedWmmaGEMMKernel<WeightType::kQ5_1, kBM, kBN, kBK,
                                               8, 1, kDepth>),
                         grid, dim3(kThreads), 0, stream, w, x_tiled,
                         pad_bounds, rows_in, rows_out, out, m, k);
      return true;
    default:
      return false;
  }
}

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += kSmallGemmTokens) {
    const std::uint32_t n = std::min(kSmallGemmTokens, n_tokens - t0);
    hipLaunchKernelGGL(SmallGemmKernel,
                       dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                       dim3(kThreads), 0, stream, w, type,
                       x + static_cast<std::size_t>(t0) * k,
                       out + static_cast<std::size_t>(t0) * m, n, m, k);
  }
}

void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(PleGateKernel, dim3(n_tokens, streams), dim3(kThreads), 0,
                     stream, key_n, query_n, value, gated, hidden, streams);
}

void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, float* snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream) {
  const std::uint32_t hist = (kernel - 1) * dilation;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  hipLaunchKernelGGL(PleConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, in, w, history, out, n_tokens, channels, kernel,
                     dilation, hist);
  if (snapshots != nullptr) {
    hipLaunchKernelGGL(RollingSnapshotKernel, dim3(Blocks(count * hist)),
                       dim3(kThreads), 0, stream, in, channels, history,
                       snapshots, n_tokens, channels, hist);
  }
  // Device copies stay kernels: a copy engine transfer is not reliably
  // ordered behind the kernels on this stream.
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, in, channels, history,
                     history_scratch, n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, history_scratch, history, hist_count);
}

void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(PleInjectKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, res, gated, conv, count);
}

void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   float* state_snapshots, float* conv_snapshots,
                   std::uint32_t n_tokens, std::uint32_t k_heads,
                   std::uint32_t v_heads, std::uint32_t d, std::uint32_t kernel,
                   bool row_split, float eps, hipStream_t stream) {
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  hipLaunchKernelGGL(SsmConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, qkv, qkv_stride, conv_w, conv_state, conv_scratch,
                     n_tokens, channels, kernel);
  if (conv_snapshots != nullptr) {
    hipLaunchKernelGGL(RollingSnapshotKernel,
                       dim3(Blocks(count * (kernel - 1))), dim3(kThreads), 0,
                       stream, qkv, qkv_stride, conv_state, conv_snapshots,
                       n_tokens, channels, kernel - 1);
  }
  // The rolling state is the last kernel-1 projections: [history ; qkv].
  const std::uint32_t hist = kernel - 1;
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, qkv, qkv_stride, conv_state,
                     conv_scratch + count, n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, conv_scratch + count, conv_state, hist_count);
  const unsigned waves = kThreads / 32;
  if (row_split && d == kGdnDim && state_snapshots == nullptr) {
    hipLaunchKernelGGL(GdnPrepKqKernel, dim3(k_heads, n_tokens), dim3(32), 0,
                       stream, conv_scratch, qn, n_tokens, k_heads, channels,
                       eps);
    hipLaunchKernelGGL(
        GdnPrepAbKernel,
        dim3(Blocks(static_cast<std::size_t>(n_tokens) * v_heads)),
        dim3(kThreads), 0, stream, alpha_beta, a, dt, kn,
        static_cast<std::size_t>(n_tokens) * v_heads, v_heads);
    hipLaunchKernelGGL(GdnRowSplitKernel, dim3(kGdnDim / 64, v_heads),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn, state,
                       raw, n_tokens, k_heads, v_heads);
  } else {
    hipLaunchKernelGGL(GdnPrepKernel,
                       dim3((n_tokens * k_heads + waves - 1) / waves),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn,
                       n_tokens * k_heads, k_heads, channels, eps);
    hipLaunchKernelGGL(GdnKernel, dim3(v_heads, kGdnDim / kGdnRowsPerBlock),
                       dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream,
                       conv_scratch, qn, kn, alpha_beta, a, dt, state, raw,
                       state_snapshots, n_tokens, k_heads, v_heads);
  }
  hipLaunchKernelGGL(GdnEpilogueKernel,
                     dim3((n_tokens * v_heads + waves - 1) / waves),
                     dim3(kThreads), 0, stream, raw, z, z_stride, norm_w, out,
                     n_tokens * v_heads, v_heads, eps);
}

void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream) {
  hipLaunchKernelGGL(UnpackQGateKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, qg, qg_stride, q, gate, k, v, heads, d, kv_width);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos);
}

void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               hipStream_t stream) {
  hipLaunchKernelGGL(StoreRowsKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, dst, row_dim, start_pos);
}

void PoolIndexerBlocks(const float* raw_keys, const float* gamma, float* blocks,
                       const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, hipStream_t stream) {
  if (grid_blocks == 0) {
    return;
  }
  hipLaunchKernelGGL(PoolBlocksKernel, dim3(grid_blocks), dim3(kThreads), 0,
                     stream, raw_keys, gamma, blocks, first_block, start_pos,
                     n_tokens, ratio, dim, rotary_dim, theta, eps);
}

void SelectBlocks(const float* q, const float* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream) {
  hipLaunchKernelGGL(SelectBlocksKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, q, blocks, mask, scores, start_pos, first_token,
                     heads, dim, ratio, budget, mask_words, max_blocks);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               std::uint32_t n_tokens, const std::uint32_t* start_pos,
               std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t ratio, hipStream_t stream) {
  hipLaunchKernelGGL(AttentionKernel, dim3(heads, n_tokens), dim3(kThreads), 0,
                     stream, q, k_cache, v_cache, mask, mask_words, out,
                     start_pos, heads, kv_heads, d, ratio);
}

bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream) {
  if (heads != kWmmaQueryHeads || kv_heads != kWmmaKvHeads ||
      d != kWmmaHeadDim || ratio == 0 || n_tokens == 0) {
    return false;
  }
  const dim3 grid((n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows,
                  kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
  hipLaunchKernelGGL((WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys>),
                     grid, dim3(kThreads), 0, stream, q, gate, k_cache, v_cache,
                     mask, mask_words, out, start_pos, n_tokens, ratio);
  return true;
}

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(RouterTopKKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, stride, ids, weights, n_experts, k);
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream) {
  if (dim % 4 != 0) {
    MoeEpilogue(expert_out, weights, shared, gate, gate_stride, out, n_tokens,
                k, dim, stream);
    return;
  }
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel, dim3(n_tokens, Blocks(dim / 4)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream) {
  hipLaunchKernelGGL(MtpHiddenKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     base, alt, row, dst, width);
}

void MtpConcat(const float* embd_n, const float* h_n, float* concat,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(MtpConcatKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     embd_n, h_n, concat, hidden, streams);
}

void Checksum(const float* x, std::size_t count, float* out,
              hipStream_t stream) {
  hipLaunchKernelGGL(ChecksumKernel, dim3(1), dim3(kThreads), 0, stream, x,
                     count, out);
}

void Argmax(const float* logits, std::int32_t* out, std::uint32_t n_tokens,
            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     logits, out, vocab);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
