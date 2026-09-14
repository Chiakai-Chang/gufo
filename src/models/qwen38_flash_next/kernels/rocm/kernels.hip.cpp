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
__device__ __forceinline__ float SiluF(float x) { return x * SigmoidF(x); }
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
  const float* g = gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
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
  __shared__ float shared[32];
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
  for (std::uint32_t o = 0; o < streams; ++o) {
    const float* w = inject_w + o * hc_dim;
    float dot = 0.0f;
    if (i < hidden) {
      for (std::uint32_t s = 0; s < streams; ++s) {
        dot += w[static_cast<std::size_t>(s) * hidden + i] * v[s];
      }
    }
    dot = BlockSum(dot, shared);
    if (threadIdx.x == 0) {
      inject[(static_cast<std::size_t>(t) * streams + o) * gridDim.y +
             blockIdx.y] = dot;
    }
  }
}

/// grid (tokens, streams): one block owns one residual stream, so the
/// grouped norm of the next mixer reduces over exactly its own elements.
__global__ void HcCombineKernel(float* res, const float* block_out,
                                const float* inject, std::uint32_t inject_parts,
                                const float* gamma, float* xn,
                                std::uint32_t hidden, std::uint32_t streams,
                                float eps) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  float logit = 0.0f;
  for (std::uint32_t p = 0; p < inject_parts; ++p) {
    logit += inject[(static_cast<std::size_t>(t) * streams + s) * inject_parts + p];
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
    xn[base + i] = dst[i] * scale * g[i];
  }
}

__global__ void SiluScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    x[i] = SiluF(x[i] * scale);
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

__global__ void AddKernel(float* dst, const float* src, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    dst[i] += src[i];
  }
}

template <typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

/// grid (m, token tiles of 8): the weight row is read once per block while
/// eight tokens accumulate; kThreads lanes stride over k.
constexpr unsigned kSmallGemmTokens = 8;
__global__ void SmallGemmKernel(const void* w, WeightType type, const float* x,
                                float* out, std::uint32_t n_tokens,
                                std::uint32_t m, std::uint32_t k) {
  __shared__ float shared[kSmallGemmTokens][32];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t t0 = blockIdx.y * kSmallGemmTokens;
  const auto* wrow = static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  float acc[kSmallGemmTokens] = {};
  for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const float wv = RowElement(wrow, type, i);
    for (unsigned j = 0; j < kSmallGemmTokens; ++j) {
      const std::uint32_t t = t0 + j;
      if (t < n_tokens) {
        acc[j] += wv * x[static_cast<std::size_t>(t) * k + i];
      }
    }
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (unsigned j = 0; j < kSmallGemmTokens; ++j) {
    float v = acc[j];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      v += __shfl_xor(v, offset);
    }
    if (lane == 0) {
      shared[j][warp] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x < kSmallGemmTokens) {
    const std::uint32_t t = t0 + threadIdx.x;
    if (t < n_tokens) {
      float total = 0.0f;
      for (int wv = 0; wv < static_cast<int>(blockDim.x / warpSize); ++wv) {
        total += shared[threadIdx.x][wv];
      }
      out[static_cast<std::size_t>(t) * m + row] = total;
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
  const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>((kernel - 1 - k) * dilation);
    const float v = src_t >= 0
                        ? in[static_cast<std::size_t>(src_t) * channels + c]
                        : history[static_cast<std::size_t>(hist + src_t) * channels + c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// New history row j is row (n_tokens + j) of [history ; in].
__global__ void HistoryShiftKernel(const float* in, const float* history,
                                   float* scratch, std::uint32_t n_tokens,
                                   std::uint32_t channels, std::uint32_t hist) {
  const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
  if (idx >= static_cast<std::size_t>(hist) * channels) {
    return;
  }
  const std::uint32_t j = idx / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = n_tokens + j;
  scratch[idx] = src < hist ? history[static_cast<std::size_t>(src) * channels + c]
                            : in[static_cast<std::size_t>(src - hist) * channels + c];
}

__global__ void PleInjectKernel(float* res, const float* gated,
                                const float* conv, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
  if (i < count) {
    res[i] += gated[i] + conv[i];
  }
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels below split their 128-wide rows as d / 32 elements per lane
/// (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
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
  const std::uint32_t row = blockIdx.x * (blockDim.x / warpSize) +
                            threadIdx.x / warpSize;
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

/// One block per value head, kGdnLanes threads per state row j, each owning
/// a contiguous slice of the key dimension. The token loop is serial; every
/// reduction stays inside a lane group, so the loop runs barrier-free. The
/// raw attention rows go out unnormalized; GdnEpilogueKernel finishes them.
constexpr unsigned kGdnLanes = 4;
__global__ void GdnKernel(const float* conv_out, const float* qn,
                          const float* kn, const float* alpha_beta,
                          const float* a, const float* dt, float* state,
                          float* raw, float* snapshots, std::uint32_t n_tokens,
                          std::uint32_t k_heads, std::uint32_t v_heads) {
  constexpr std::uint32_t d = kGdnDim;
  constexpr std::uint32_t slice = d / kGdnLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j = threadIdx.x / kGdnLanes;
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
      raw[static_cast<std::size_t>(t) * v_heads * d + h * d + j] = acc * q_scale;
    }
    if (snapshots != nullptr) {
      float* snap = snapshots +
                    (static_cast<std::size_t>(t) * v_heads + h) * d * d + j * d + i0;
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
                                  const float* norm_w, float* out,
                                  std::uint32_t n_rows, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::size_t row = blockIdx.x * (blockDim.x / warpSize) +
                          threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
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
    out[row * d + i] = v[r] * scale * norm_w[i] * SigmoidF(z[row * d + i]);
  }
}

/// Row j of the rolling state after token t is row (t + 1 + j) of the
/// concatenation [history ; rows], for any history depth `hist`.
__global__ void RollingSnapshotKernel(const float* rows, const float* history,
                                      float* snapshots, std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t hist) {
  const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
  const std::size_t per_token = static_cast<std::size_t>(hist) * channels;
  if (idx >= per_token * n_tokens) {
    return;
  }
  const std::uint32_t t = idx / per_token;
  const std::uint32_t j = (idx % per_token) / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = t + 1 + j;
  snapshots[idx] = src < hist
                       ? history[static_cast<std::size_t>(src) * channels + c]
                       : rows[static_cast<std::size_t>(src - hist) * channels + c];
}

/// Causal conv over the chunk with the rolling state, SiLU applied.
__global__ void SsmConvKernel(const float* qkv, const float* w,
                              const float* conv_state, float* out,
                              std::uint32_t n_tokens, std::uint32_t channels,
                              std::uint32_t kernel) {
  const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>(kernel - 1 - k);
    const float v = src_t >= 0
                        ? qkv[static_cast<std::size_t>(src_t) * channels + c]
                        : conv_state[static_cast<std::size_t>(kernel - 1 + src_t) * channels + c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

__global__ void UnpackQGateKernel(const float* qg, float* q, float* gate,
                                  std::uint32_t heads, std::uint32_t d) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t width = static_cast<std::size_t>(heads) * d;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const std::uint32_t h = i / d;
    const std::uint32_t j = i % d;
    q[t * width + i] = qg[t * 2 * width + h * 2 * d + j];
    gate[t * width + i] = qg[t * 2 * width + h * 2 * d + d + j];
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim, std::uint32_t start_pos,
                           float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  const float pos = static_cast<float>(start_pos + t);
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    const float freq = powf(theta, -2.0f * static_cast<float>(i) /
                                       static_cast<float>(rotary_dim));
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
                              std::uint32_t row_dim, std::uint32_t start_pos) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(start_pos + t) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
}

__global__ void PoolBlocksKernel(const float* raw, const float* gamma,
                                 float* blocks, std::uint32_t first_block,
                                 std::uint32_t ratio, std::uint32_t dim,
                                 std::uint32_t rotary_dim, float theta,
                                 float eps) {
  __shared__ float v[256];
  __shared__ float shared[32];
  const std::uint32_t b = first_block + blockIdx.x;
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
    const float freq = powf(theta, -2.0f * static_cast<float>(p) /
                                       static_cast<float>(rotary_dim));
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
__global__ void SelectBlocksKernel(const float* q, const float* blocks,
                                   std::uint32_t* mask, float* scores,
                                   std::uint32_t start_pos, std::uint32_t heads,
                                   std::uint32_t dim, std::uint32_t ratio,
                                   std::uint32_t budget,
                                   std::uint32_t mask_words,
                                   std::uint32_t max_blocks) {
  __shared__ float shared[32];
  __shared__ std::uint32_t counts[32];
  __shared__ float qs[4 * 128];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t pos = start_pos + t;
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
  const std::uint32_t n_above =
      static_cast<std::uint32_t>(BlockSum(static_cast<float>(above), shared) + 0.5f);
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
                                const __half* v_cache, const std::uint32_t* mask,
                                std::uint32_t mask_words, float* out,
                                std::uint32_t start_pos, std::uint32_t heads,
                                std::uint32_t kv_heads, std::uint32_t d,
                                std::uint32_t ratio) {
  constexpr std::uint32_t kTile = 256;
  __shared__ float qs[256];
  __shared__ float p[kTile];
  __shared__ float shared[32];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t tail_start = (n_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr : mask + static_cast<std::size_t>(t) * mask_words;
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
          acc += w * __half2float(v_cache[(tile + jj) * kv_stride + kvh * d + i]);
        }
      }
    }
    __syncthreads();
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

__global__ void AttentionSoftmaxKernel(const float* scores,
                                       const std::uint32_t* mask,
                                       std::uint32_t mask_words, __half* probs,
                                       std::uint32_t n_kv,
                                       std::uint32_t start_pos,
                                       std::uint32_t ratio, float scale) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t visible_kv = pos + 1;
  const std::uint32_t tail_start = (visible_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr : mask + static_cast<std::size_t>(t) * mask_words;
  const std::size_t row = (static_cast<std::size_t>(h) * gridDim.x + t) * n_kv;
  const float* src = scores + row;
  __half* dst = probs + row;
  float local_max = -INFINITY;
  for (std::uint32_t j = threadIdx.x; j < visible_kv; j += blockDim.x) {
    bool visible = j >= tail_start || words == nullptr;
    if (!visible) {
      const std::uint32_t b = j / ratio;
      visible = (words[b / 32] >> (b % 32)) & 1u;
    }
    if (visible) {
      local_max = fmaxf(local_max, src[j] * scale);
    }
  }
  const float m = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t j = threadIdx.x; j < visible_kv; j += blockDim.x) {
    bool visible = j >= tail_start || words == nullptr;
    if (!visible) {
      const std::uint32_t b = j / ratio;
      visible = (words[b / 32] >> (b % 32)) & 1u;
    }
    local_sum += visible ? __expf(src[j] * scale - m) : 0.0f;
  }
  const float inv = 1.0f / BlockSum(local_sum, shared);
  for (std::uint32_t j = threadIdx.x; j < n_kv; j += blockDim.x) {
    bool visible = j < visible_kv && (j >= tail_start || words == nullptr);
    if (!visible && j < visible_kv) {
      const std::uint32_t b = j / ratio;
      visible = (words[b / 32] >> (b % 32)) & 1u;
    }
    dst[j] = __float2half(visible ? __expf(src[j] * scale - m) * inv : 0.0f);
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

__global__ void MtpConcatKernel(const float* embd_n, const float* h_n,
                                float* concat, std::uint32_t hidden,
                                std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t s = 0; s < streams; ++s) {
    float* dst = concat + (static_cast<std::size_t>(t) * streams + s) * 2 * hidden;
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
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
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

std::uint32_t HcInjectParts(std::uint32_t hidden) { return Blocks(hidden); }

void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream) {
  hipLaunchKernelGGL(HcMixEpilogueKernel, dim3(n_tokens, Blocks(hidden)),
                     dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                     inject, hidden, streams);
}

void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(HcCombineKernel, dim3(n_tokens, streams), dim3(kThreads),
                     0, stream, res, block_out, inject, inject_parts, gamma, xn,
                     hidden, streams, eps);
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

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  const unsigned tiles = (n_tokens + kSmallGemmTokens - 1) / kSmallGemmTokens;
  hipLaunchKernelGGL(SmallGemmKernel, dim3(m, tiles), dim3(kThreads), 0,
                     stream, w, type, x, out, n_tokens, m, k);
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
    hipLaunchKernelGGL(RollingSnapshotKernel,
                       dim3(Blocks(count * hist)), dim3(kThreads), 0, stream,
                       in, history, snapshots, n_tokens, channels, hist);
  }
  // Device copies stay kernels: a copy engine transfer is not reliably
  // ordered behind the kernels on this stream.
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, in, history, history_scratch,
                     n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, history_scratch, history, hist_count);
}

void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(PleInjectKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, res, gated, conv, count);
}

void GatedDeltaNet(const float* qkv, const float* z, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   float* state_snapshots, float* conv_snapshots,
                   std::uint32_t n_tokens, std::uint32_t k_heads,
                   std::uint32_t v_heads, std::uint32_t d, std::uint32_t kernel,
                   float eps, hipStream_t stream) {
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  hipLaunchKernelGGL(SsmConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, qkv, conv_w, conv_state, conv_scratch, n_tokens,
                     channels, kernel);
  if (conv_snapshots != nullptr) {
    hipLaunchKernelGGL(RollingSnapshotKernel,
                       dim3(Blocks(count * (kernel - 1))), dim3(kThreads), 0,
                       stream, qkv, conv_state, conv_snapshots, n_tokens,
                       channels, kernel - 1);
  }
  // The rolling state is the last kernel-1 projections: [history ; qkv].
  const std::uint32_t hist = kernel - 1;
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, qkv, conv_state,
                     conv_scratch + count, n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, conv_scratch + count, conv_state, hist_count);
  const unsigned waves = kThreads / 32;
  hipLaunchKernelGGL(GdnPrepKernel,
                     dim3((n_tokens * k_heads + waves - 1) / waves),
                     dim3(kThreads), 0, stream, conv_scratch, qn, kn,
                     n_tokens * k_heads, k_heads, channels, eps);
  hipLaunchKernelGGL(GdnKernel, dim3(v_heads), dim3(kGdnDim * kGdnLanes), 0,
                     stream, conv_scratch, qn, kn, alpha_beta, a, dt, state,
                     raw, state_snapshots, n_tokens, k_heads, v_heads);
  hipLaunchKernelGGL(GdnEpilogueKernel,
                     dim3((n_tokens * v_heads + waves - 1) / waves),
                     dim3(kThreads), 0, stream, raw, z, norm_w, out,
                     n_tokens * v_heads, eps);
}

void UnpackQGate(const float* qg, float* q, float* gate, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, hipStream_t stream) {
  hipLaunchKernelGGL(UnpackQGateKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, qg, q, gate, heads, d);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim, std::uint32_t start_pos,
          float theta, hipStream_t stream) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t start_pos,
             hipStream_t stream) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos);
}

void PoolIndexerBlocks(const float* raw_keys, const float* gamma, float* blocks,
                       std::uint32_t first_block, std::uint32_t n_blocks,
                       std::uint32_t ratio, std::uint32_t dim,
                       std::uint32_t rotary_dim, float theta, float eps,
                       hipStream_t stream) {
  if (n_blocks == 0) {
    return;
  }
  hipLaunchKernelGGL(PoolBlocksKernel, dim3(n_blocks), dim3(kThreads), 0,
                     stream, raw_keys, gamma, blocks, first_block, ratio, dim,
                     rotary_dim, theta, eps);
}

void SelectBlocks(const float* q, const float* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens, std::uint32_t start_pos,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream) {
  hipLaunchKernelGGL(SelectBlocksKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, q, blocks, mask, scores, start_pos, heads, dim,
                     ratio, budget, mask_words, max_blocks);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               std::uint32_t n_tokens, std::uint32_t start_pos,
               std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t ratio, hipStream_t stream) {
  hipLaunchKernelGGL(AttentionKernel, dim3(heads, n_tokens), dim3(kThreads), 0,
                     stream, q, k_cache, v_cache, mask, mask_words, out,
                     start_pos, heads, kv_heads, d, ratio);
}

void AttentionSoftmax(const float* scores, const std::uint32_t* mask,
                      std::uint32_t mask_words, __half* probs,
                      std::uint32_t n_tokens, std::uint32_t n_kv,
                      std::uint32_t start_pos, std::uint32_t heads,
                      std::uint32_t d, std::uint32_t ratio, hipStream_t stream) {
  hipLaunchKernelGGL(AttentionSoftmaxKernel, dim3(n_tokens, heads),
                     dim3(kThreads), 0, stream, scores, mask, mask_words, probs,
                     n_kv, start_pos, ratio, rsqrtf(static_cast<float>(d)));
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(RouterTopKKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, stride, ids, weights, n_experts, k);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MtpConcat(const float* embd_n, const float* h_n, float* concat,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(MtpConcatKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     embd_n, h_n, concat, hidden, streams);
}

void Checksum(const float* x, std::size_t count, float* out, hipStream_t stream) {
  hipLaunchKernelGGL(ChecksumKernel, dim3(1), dim3(kThreads), 0, stream, x, count,
                     out);
}

void Argmax(const float* logits, std::int32_t* out, std::uint32_t n_tokens,
            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     logits, out, vocab);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
