// Masked prefill attention on the WMMA matrix cores (opt-c177-attn-wmma).
//
// Replaces both halves of the previous prefill attention: the tiled
// `v_dot2_f32_f16` kernel, which reached 4.58 TFLOPS -- 15% of its own
// instruction ceiling and 8% of the 55.05 TFLOPS WMMA ceiling -- and the
// AOTriton non-causal prefix plus log-sum-exp merge that `opt-c165-attn-split`
// bolted on to work around AOTriton's missing causal support on gfx11xx. One
// masked kernel over the whole visible range is faster than that pair at every
// depth, so the FP16 query conversion, the AOTriton call and the merge pass all
// go away.
//
// Measured against the tiled kernel plus AOTriton at batch 2048
// (tools/bench/attn_causal_bench.hip), one layer call:
//
//   depth      before    after   speedup
//       0      11.40     3.24     3.52x     (tiled only)
//    8192      37.6     27.26     1.38x     (tiled diagonal + AOTriton prefix)
//   16384      64.1     50.48     1.27x
//
// 15.9 to 17.4 TFLOPS, or 29-32% of the WMMA ceiling, against AOTriton's 15.6
// on the easier unmasked shapes.
//
// Design. Wave32 fragment layout, verified against a CPU double-precision
// reference: A holds row L%16 and 16 contiguous k, B holds column L%16 and 16
// contiguous k (so B is fed a transposed tile), and C element i is row
// 2i + L/16, column L%16.
//
// Three things set the shape of this kernel, each found by ablation:
//
//   * Global request count dominates, not bytes. Every block re-reads K and V
//     for its key range, so the cost per query row falls as the block covers
//     more rows. 32 rows per block spends 2.43 ms of a 5.03 ms call on global
//     memory; 64 rows halves that to 0.94 ms of 3.24 ms. 64 is the most the
//     eight-wave split supports without spilling.
//   * V has to reach LDS transposed, because both WMMA operands want 16
//     contiguous values of the contracted dimension and V is stored
//     [position][head][dim]. Writing it with each lane owning 8 contiguous dims
//     of one key -- what the coalesced global read wants -- puts all 32 lanes
//     of a wave on LDS rows kVtStride halves apart, which is a single bank and
//     a 32-way conflict. Giving each lane one key instead costs a strided
//     global read but makes the LDS writes conflict-free, and was worth 1.8x.
//   * V's global load is issued at the top of the key-tile iteration rather
//   than
//     at its point of use. Behind the barriers it needs, its latency is
//     otherwise fully exposed.
//
// Barriers are not a cost here: removing all four per key tile changes nothing
// measurable.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace strix::hip {

// Defined in kernels/attention_tile.hip; the WMMA route reuses the same FP16
// cache and the same pack/sync pair unchanged.
__global__ void PackTiledAttentionKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context);

__global__ void SyncTiledAttentionKvPrefixKernel(
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t prefix_length,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim);

namespace {

constexpr std::uint32_t kHeadDim = detail::kTiledAttentionHeadDim;
constexpr std::uint32_t kQueryHeads = detail::kTiledAttentionQueryHeads;
constexpr std::uint32_t kKvHeads = detail::kTiledAttentionKvHeads;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes. Reading it element by element
// costs 16 ds_read_u16 plus 16 __half -> _Float16 conversions; the two formats
// are bit-identical, so two 16-byte loads and a bitcast do the same work in two
// instructions. Every fragment base here is 16-byte aligned by construction.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads =
    2;  // query heads per block, must divide GQA
// Measured tile: 32 queries x 2 heads = 64 rows per block against 16 keys. See
// the header note on why more rows per block is the lever that matters.
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;

// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

// kQueryRows queries x kWmmaHeads heads form (kQueryRows / 16) * kWmmaHeads
// 16-row blocks, and kKeys keys form kKeys / 16 key blocks. Their product is
// pinned to 4 so the eight waves split as four S tiles x two halves of the head
// dimension, which is what keeps Q in registers at 64 VGPRs. Growing kQueryRows
// and shrinking kKeys in step therefore trades nothing in the tiling but halves
// the K/V bytes each block pulls per query row -- the dominant cost.
template<std::uint32_t kQueryRows, std::uint32_t kKeys>
__launch_bounds__(256, 2) __global__
    void WmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const __half* __restrict__ k_cache_f16,
                             const __half* __restrict__ v_cache_f16,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             float* __restrict__ lse_out,
                             std::uint32_t key_begin) {
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
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
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
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
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

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][2] = {};
  for (std::uint32_t r = tid; r < kRows; r += 256) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // V's global load is issued before the K staging and the whole S phase, and
    // parked in registers until the transpose. Issued at its point of use it
    // sits behind two barriers with nothing able to cover its latency.
    constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
    uint4 v_raw[kVRegs];
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_cache_f16 + (static_cast<std::size_t>(key_position) * kKvWidth) +
          (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        v_raw[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                        : make_uint4(0u, 0u, 0u, 0u);
      }
    }

    // --- stage K (coalesced: a wave reads one key row's 512 contiguous bytes)
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKeys * (kHeadDim / 8); idx += 256) {
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const std::uint32_t key_position = key_start + key_row;
      __half packed[8] = {};
      if (key_position < context_end) {
        const auto* src = k_cache_f16 +
                          (static_cast<std::size_t>(key_position) * kKvWidth) +
                          (static_cast<std::size_t>(kv_head) * kHeadDim) + d8;
        *reinterpret_cast<uint4*>(packed) =
            *reinterpret_cast<const uint4*>(src);
      }
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          *reinterpret_cast<const uint4*>(packed);
    }
    __syncthreads();

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
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
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
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
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
    // 2i + half_id, so the factors are read once and reused across its dim
    // tiles.
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

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks. The mapping the coalesced global read would
    // prefer -- each lane taking 8 contiguous dims of one key -- puts all 32
    // lanes on rows kVtStride halves apart, which is one bank, and costs a
    // 32-way conflict on every write.
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_raw[j]);
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
  }
  __syncthreads();

  // --- epilogue ---
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
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

}  // namespace

bool LaunchQwenWmmaAttention(const float* q, const float* k, const float* v,
                             const float* gate, float* k_cache, float* v_cache,
                             void* k_cache_f16, void* v_cache_f16,
                             float* out_context, std::uint32_t layer_idx,
                             std::uint32_t start_pos, std::size_t batch_size,
                             std::uint32_t max_context, std::uint32_t num_heads,
                             std::uint32_t num_kv_heads, std::uint32_t head_dim,
                             hipStream_t stream, float* lse_out,
                             std::uint32_t key_begin, bool skip_kv_write) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = k_cache_f16 != nullptr,
      .has_v_cache_f16 = v_cache_f16 != nullptr,
      .has_scratch_f16 = false,
  };
  if (!detail::IsTiledAttentionSupported(support_params)) {
    return false;
  }

  constexpr std::size_t block_size = 256;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kKvWidth;
  auto* layer_k = static_cast<__half*>(k_cache_f16) + layer_cache_offset;
  auto* layer_v = static_cast<__half*>(v_cache_f16) + layer_cache_offset;

  if (!skip_kv_write) {
    if (start_pos > 0 && k_cache != nullptr && v_cache != nullptr) {
      const dim3 sync_grid(
          static_cast<unsigned int>((kKvWidth + block_size - 1) / block_size),
          static_cast<unsigned int>(start_pos));
      SyncTiledAttentionKvPrefixKernel<<<sync_grid, block_size, 0, stream>>>(
          k_cache, v_cache, static_cast<__half*>(k_cache_f16),
          static_cast<__half*>(v_cache_f16), layer_idx, start_pos, max_context,
          num_kv_heads, head_dim);
    }
    const dim3 pack_grid(
        static_cast<unsigned int>((kKvWidth + block_size - 1) / block_size),
        static_cast<unsigned int>(batch_size));
    PackTiledAttentionKvKernel<<<pack_grid, block_size, 0, stream>>>(
        k, v, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx,
        start_pos, batch_size, max_context);
  }

  const dim3 grid(static_cast<unsigned int>((batch_size + kWmmaQueryRows - 1) /
                                            kWmmaQueryRows),
                  kKvHeads * (kGqa / kWmmaHeads));
  WmmaCausalAttention<kWmmaQueryRows, kWmmaKeys>
      <<<grid, block_size, 0, stream>>>(q, gate, layer_k, layer_v, out_context,
                                        start_pos, batch_size, lse_out,
                                        key_begin);
  return true;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
