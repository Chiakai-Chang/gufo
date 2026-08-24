#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

struct Q8_1ActivationBlock {
  float d;
  std::int8_t qs[32];
};

template<typename T>
__global__ void QuantizeActivationToQ8_1Kernel(
    const T* __restrict__ x, Q8_1ActivationBlock* __restrict__ y,
    std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float val = static_cast<float>(x[elem_idx]);
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  if (lane_id == 0) {
    y[b_idx].d = d;
  }
  y[b_idx].qs[lane_id] = q;
}

__global__ void BatchedFusedSwiGLUQuantizeQ8_1Kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    Q8_1ActivationBlock* __restrict__ y, std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float g = gate[elem_idx];
  const float u = up[elem_idx];
  const float silu_g = g / (1.0F + expf(-g));
  const float val = silu_g * u;

  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  if (lane_id == 0) {
    y[b_idx].d = d;
  }
  y[b_idx].qs[lane_id] = q;
}

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

template<bool SubOpsel>
__device__ __forceinline__ int32x8_t Wmma_16x16x16_iu8(int32x4_t a, int32x4_t b,
                                                       int32x8_t c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                    SubOpsel);
#else
  return c;
#endif
}

struct LdsWeight8Blocks {
  int32x4_t a0[8][16];
  int32x4_t a1[8][16];
  float dw[8][16];
};

__launch_bounds__(256, 4) __global__ void W8A8WmmaLdsBatchedGEMMKernel(
    const void* __restrict__ w,
    const Q8_1ActivationBlock* __restrict__ x_blocks, float* __restrict__ y,
    std::size_t batch, std::size_t m, std::size_t k) {
  __shared__ LdsWeight8Blocks s_w;

  const std::size_t num_blocks = k / 32;
  const auto* w_blocks = static_cast<const Q8_0Block*>(w);

  const std::size_t tid = threadIdx.x;
  const std::size_t wave_id =
      tid >> 5u;  // 0..7 (8 waves compute 128 tokens total)
  const std::size_t lane_id = tid & 31u;  // 0..31

  const std::size_t r_base = blockIdx.x * 16;
  const std::size_t t_base = (blockIdx.y * 128) + (wave_id * 16);

  const std::size_t sub_lane = lane_id % 16u;
  const std::size_t half_id = lane_id / 16u;

  const std::size_t tok = t_base + sub_lane;

  float acc[8] = {0.0F};

  const auto* tok_x = (tok < batch) ? (x_blocks + (tok * num_blocks)) : nullptr;

  const std::size_t block_sub =
      tid / 32;  // 0..7 (all 256 threads participate in 1 pass)
  const std::size_t lane_sub = tid % 32;  // 0..31

  for (std::size_t b = 0; b < num_blocks; b += 8) {
    const std::size_t b_curr = b + block_sub;
    if (lane_sub < 16) {
      const std::size_t r = r_base + lane_sub;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk = w_blocks[(r * num_blocks) + b_curr];
        s_w.dw[block_sub][lane_sub] = __half2float(blk.d);
        __builtin_memcpy(&s_w.a0[block_sub][lane_sub], blk.qs + 0, 16);
      } else {
        s_w.dw[block_sub][lane_sub] = 0.0F;
        s_w.a0[block_sub][lane_sub] = {0, 0, 0, 0};
      }
    } else {
      const std::size_t lane_idx = lane_sub - 16;
      const std::size_t r = r_base + lane_idx;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk = w_blocks[(r * num_blocks) + b_curr];
        __builtin_memcpy(&s_w.a1[block_sub][lane_idx], blk.qs + 16, 16);
      } else {
        s_w.a1[block_sub][lane_idx] = {0, 0, 0, 0};
      }
    }
    __syncthreads();

    int32x4_t act_b0[8] = {};
    int32x4_t act_b1[8] = {};
    float act_dx[8] = {0.0F};

    if (tok_x != nullptr) {
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const std::size_t b_step = b + static_cast<std::size_t>(step);
        if (b_step < num_blocks) {
          const auto& blk_x = tok_x[b_step];
          act_dx[step] = blk_x.d;
          act_b0[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 0);
          act_b1[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 16);
        }
      }
    }

#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const std::size_t b_step = b + static_cast<std::size_t>(step);
      if (b_step < num_blocks) {
        const int32x4_t a0 = s_w.a0[step][sub_lane];
        const int32x4_t a1 = s_w.a1[step][sub_lane];

        int32x8_t c_frag = {0, 0, 0, 0, 0, 0, 0, 0};
        c_frag = Wmma_16x16x16_iu8<true>(a0, act_b0[step], c_frag);
        c_frag = Wmma_16x16x16_iu8<true>(a1, act_b1[step], c_frag);

        const float dx = act_dx[step];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const std::size_t row_idx =
              (2u * static_cast<std::size_t>(l)) + half_id;
          const float dw = s_w.dw[step][row_idx];
          acc[l] += (dw * dx) * static_cast<float>(c_frag[l]);
        }
      }
    }

    __syncthreads();
  }

  if (tok < batch) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const std::size_t r =
          r_base + (2u * static_cast<std::size_t>(l)) + half_id;
      if (r < m) {
        y[(tok * m) + r] = acc[l];
      }
    }
  }
}

__launch_bounds__(256, 4) __global__ void W8A8DualWmmaLdsBatchedGEMMKernel(
    const void* __restrict__ w_gate, const void* __restrict__ w_up,
    const Q8_1ActivationBlock* __restrict__ x_blocks,
    float* __restrict__ y_gate, float* __restrict__ y_up, std::size_t batch,
    std::size_t m, std::size_t k) {
  __shared__ LdsWeight8Blocks s_w_gate;
  __shared__ LdsWeight8Blocks s_w_up;

  const std::size_t num_blocks = k / 32;
  const auto* w_blocks_gate = static_cast<const Q8_0Block*>(w_gate);
  const auto* w_blocks_up = static_cast<const Q8_0Block*>(w_up);

  const std::size_t tid = threadIdx.x;
  const std::size_t wave_id =
      tid >> 5u;  // 0..7 (8 waves compute 128 tokens total)
  const std::size_t lane_id = tid & 31u;  // 0..31

  const std::size_t r_base = blockIdx.x * 16;
  const std::size_t t_base = (blockIdx.y * 128) + (wave_id * 16);

  const std::size_t sub_lane = lane_id % 16u;
  const std::size_t half_id = lane_id / 16u;

  const std::size_t tok = t_base + sub_lane;

  float acc_gate[8] = {0.0F};
  float acc_up[8] = {0.0F};

  const auto* tok_x = (tok < batch) ? (x_blocks + (tok * num_blocks)) : nullptr;

  const std::size_t block_sub = tid / 32;  // 0..7
  const std::size_t lane_sub = tid % 32;   // 0..31

  for (std::size_t b = 0; b < num_blocks; b += 8) {
    const std::size_t b_curr = b + block_sub;
    if (lane_sub < 16) {
      const std::size_t r = r_base + lane_sub;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk_g = w_blocks_gate[(r * num_blocks) + b_curr];
        const Q8_0Block& blk_u = w_blocks_up[(r * num_blocks) + b_curr];
        s_w_gate.dw[block_sub][lane_sub] = __half2float(blk_g.d);
        s_w_up.dw[block_sub][lane_sub] = __half2float(blk_u.d);
        __builtin_memcpy(&s_w_gate.a0[block_sub][lane_sub], blk_g.qs + 0, 16);
        __builtin_memcpy(&s_w_up.a0[block_sub][lane_sub], blk_u.qs + 0, 16);
      } else {
        s_w_gate.dw[block_sub][lane_sub] = 0.0F;
        s_w_up.dw[block_sub][lane_sub] = 0.0F;
        s_w_gate.a0[block_sub][lane_sub] = {0, 0, 0, 0};
        s_w_up.a0[block_sub][lane_sub] = {0, 0, 0, 0};
      }
    } else {
      const std::size_t lane_idx = lane_sub - 16;
      const std::size_t r = r_base + lane_idx;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk_g = w_blocks_gate[(r * num_blocks) + b_curr];
        const Q8_0Block& blk_u = w_blocks_up[(r * num_blocks) + b_curr];
        __builtin_memcpy(&s_w_gate.a1[block_sub][lane_idx], blk_g.qs + 16, 16);
        __builtin_memcpy(&s_w_up.a1[block_sub][lane_idx], blk_u.qs + 16, 16);
      } else {
        s_w_gate.a1[block_sub][lane_idx] = {0, 0, 0, 0};
        s_w_up.a1[block_sub][lane_idx] = {0, 0, 0, 0};
      }
    }
    __syncthreads();

    int32x4_t act_b0[8] = {};
    int32x4_t act_b1[8] = {};
    float act_dx[8] = {0.0F};

    if (tok_x != nullptr) {
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const std::size_t b_step = b + static_cast<std::size_t>(step);
        if (b_step < num_blocks) {
          const auto& blk_x = tok_x[b_step];
          act_dx[step] = blk_x.d;
          act_b0[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 0);
          act_b1[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 16);
        }
      }
    }

#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const std::size_t b_step = b + static_cast<std::size_t>(step);
      if (b_step < num_blocks) {
        const int32x4_t a0_g = s_w_gate.a0[step][sub_lane];
        const int32x4_t a1_g = s_w_gate.a1[step][sub_lane];
        const int32x4_t a0_u = s_w_up.a0[step][sub_lane];
        const int32x4_t a1_u = s_w_up.a1[step][sub_lane];

        int32x8_t c_frag_g = {0, 0, 0, 0, 0, 0, 0, 0};
        c_frag_g = Wmma_16x16x16_iu8<true>(a0_g, act_b0[step], c_frag_g);
        c_frag_g = Wmma_16x16x16_iu8<true>(a1_g, act_b1[step], c_frag_g);

        int32x8_t c_frag_u = {0, 0, 0, 0, 0, 0, 0, 0};
        c_frag_u = Wmma_16x16x16_iu8<true>(a0_u, act_b0[step], c_frag_u);
        c_frag_u = Wmma_16x16x16_iu8<true>(a1_u, act_b1[step], c_frag_u);

        const float dx = act_dx[step];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const std::size_t row_idx =
              (2u * static_cast<std::size_t>(l)) + half_id;
          const float dw_g = s_w_gate.dw[step][row_idx];
          const float dw_u = s_w_up.dw[step][row_idx];
          acc_gate[l] += (dw_g * dx) * static_cast<float>(c_frag_g[l]);
          acc_up[l] += (dw_u * dx) * static_cast<float>(c_frag_u[l]);
        }
      }
    }

    __syncthreads();
  }

  if (tok < batch) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const std::size_t r =
          r_base + (2u * static_cast<std::size_t>(l)) + half_id;
      if (r < m) {
        y_gate[(tok * m) + r] = acc_gate[l];
        y_up[(tok * m) + r] = acc_up[l];
      }
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void TiledBatchedQ8_0GEMMKernel(const void* __restrict__ w,
                                    const hip_bfloat16* __restrict__ x,
                                    float* __restrict__ y, std::size_t batch,
                                    std::size_t m, std::size_t k) {
  constexpr std::size_t kRowsPerWave = 4;
  constexpr std::size_t kTokensPerWave = 8;
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const std::size_t row_base =
      (blockIdx.x * WavesPerBlock + warp_id) * kRowsPerWave;
  const std::size_t tok_base = blockIdx.y * kTokensPerWave;
  if (row_base >= m || tok_base >= batch) {
    return;
  }

  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* w_blocks = static_cast<const Q8_0Block*>(w);

  const auto* row0 = w_blocks + ((row_base + 0) * num_blocks);
  const auto* row1 =
      (row_base + 1 < m) ? (w_blocks + ((row_base + 1) * num_blocks)) : nullptr;
  const auto* row2 =
      (row_base + 2 < m) ? (w_blocks + ((row_base + 2) * num_blocks)) : nullptr;
  const auto* row3 =
      (row_base + 3 < m) ? (w_blocks + ((row_base + 3) * num_blocks)) : nullptr;

  const std::size_t group_id = lane_id >> 3u;       // 0..3
  const std::size_t lane_in_group = lane_id & 7u;   // 0..7
  const std::size_t elem_off = lane_in_group * 4u;  // 0, 4, 8..28

  float acc[4][8] = {{0.0F}};

  const std::size_t num_steps = (num_blocks + 3) / 4;
  for (std::size_t step = 0; step < num_steps; ++step) {
    const std::size_t b = (step * 4) + group_id;

    if (b < num_blocks) {
      // Load 4 bfloat16 activation values for each of the 8 tokens
      hip_bfloat16 xv[8][4];
#pragma unroll
      for (std::size_t t = 0; t < 8; ++t) {
        if (tok_base + t < batch) {
          *reinterpret_cast<uint64_t*>(&xv[t][0]) =
              *reinterpret_cast<const uint64_t*>(
                  &x[((tok_base + t) * k) + (b * kQ8_0BlockSize) + elem_off]);
        }
      }

      // Process row 0
      {
        const Q8_0Block& b0 = row0[b];
        const float d0 = __half2float(b0.d);
        const auto* qs0 = b0.qs + elem_off;
        float dot0[8] = {0.0F};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d0 * static_cast<float>(qs0[i]);
#pragma unroll
          for (std::size_t t = 0; t < 8; ++t) {
            dot0[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < 8; ++t) {
          acc[0][t] += dot0[t];
        }
      }

      // Process row 1
      if (row1 != nullptr) {
        const Q8_0Block& b1 = row1[b];
        const float d1 = __half2float(b1.d);
        const auto* qs1 = b1.qs + elem_off;
        float dot1[8] = {0.0F};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d1 * static_cast<float>(qs1[i]);
#pragma unroll
          for (std::size_t t = 0; t < 8; ++t) {
            dot1[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < 8; ++t) {
          acc[1][t] += dot1[t];
        }
      }

      // Process row 2
      if (row2 != nullptr) {
        const Q8_0Block& b2 = row2[b];
        const float d2 = __half2float(b2.d);
        const auto* qs2 = b2.qs + elem_off;
        float dot2[8] = {0.0F};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d2 * static_cast<float>(qs2[i]);
#pragma unroll
          for (std::size_t t = 0; t < 8; ++t) {
            dot2[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < 8; ++t) {
          acc[2][t] += dot2[t];
        }
      }

      // Process row 3
      if (row3 != nullptr) {
        const Q8_0Block& b3 = row3[b];
        const float d3 = __half2float(b3.d);
        const auto* qs3 = b3.qs + elem_off;
        float dot3[8] = {0.0F};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d3 * static_cast<float>(qs3[i]);
#pragma unroll
          for (std::size_t t = 0; t < 8; ++t) {
            dot3[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < 8; ++t) {
          acc[3][t] += dot3[t];
        }
      }
    }
  }

  // Reduce within group (8 lanes per block) and across groups (4 groups in
  // wave)
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
#pragma unroll
      for (std::size_t t = 0; t < 8; ++t) {
        acc[r][t] += __shfl_xor(acc[r][t], off);
      }
    }
  }

  if (lane_id == 0) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
      const std::size_t row = row_base + r;
      if (row < m) {
#pragma unroll
        for (std::size_t t = 0; t < 8; ++t) {
          const std::size_t tok = tok_base + t;
          if (tok < batch) {
            y[(tok * m) + row] = acc[r][t];
          }
        }
      }
    }
  }
}

__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const hip_bfloat16* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot =
      QuantWarpBlockDot<hip_bfloat16>(type, w, row, x + (b * k), k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

void LaunchQuantizeActivationQ8_1(const void* bf16_x, void* q8_1_out,
                                  std::size_t batch, std::size_t k,
                                  hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                   quant_threads, 0, stream>>>(
      reinterpret_cast<const hip_bfloat16*>(bf16_x),
      reinterpret_cast<Q8_1ActivationBlock*>(q8_1_out), batch, k);
}

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                   quant_threads, 0, stream>>>(
      fp32_x, reinterpret_cast<Q8_1ActivationBlock*>(q8_1_out), batch, k);
}

void LaunchBatchedFusedSwiGLUQuantizeQ8_1(const float* gate, const float* up,
                                          void* q8_1_out, std::size_t batch,
                                          std::size_t k, hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  BatchedFusedSwiGLUQuantizeQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                         quant_threads, 0, stream>>>(
      gate, up, reinterpret_cast<Q8_1ActivationBlock*>(q8_1_out), batch, k);
}

void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    const dim3 grid(static_cast<unsigned int>((m + 15) / 16),
                    static_cast<unsigned int>((batch + 127) / 128));
    W8A8WmmaLdsBatchedGEMMKernel<<<grid, 256, 0, stream>>>(
        w, reinterpret_cast<const Q8_1ActivationBlock*>(q8_1_x), y, batch, m,
        k);
  }
}

void LaunchBatchedDualQuantGEMMPreQuantized(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, float* y_up, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    const dim3 grid(static_cast<unsigned int>((m + 15) / 16),
                    static_cast<unsigned int>((batch + 127) / 128));
    W8A8DualWmmaLdsBatchedGEMMKernel<<<grid, 256, 0, stream>>>(
        w_gate, w_up, reinterpret_cast<const Q8_1ActivationBlock*>(q8_1_x),
        y_gate, y_up, batch, m, k);
  }
}

void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    static thread_local void* s_act_q8_scratch = nullptr;
    static thread_local std::size_t s_act_q8_scratch_bytes = 0;

    const std::size_t num_blocks = k / 32;
    const std::size_t total_x_blocks = batch * num_blocks;
    const std::size_t req_bytes = total_x_blocks * sizeof(Q8_1ActivationBlock);
    if (s_act_q8_scratch_bytes < req_bytes) {
      if (s_act_q8_scratch != nullptr) {
        (void)hipFree(s_act_q8_scratch);
      }
      HIP_CHECK(hipMalloc(&s_act_q8_scratch, req_bytes));
      s_act_q8_scratch_bytes = req_bytes;
    }

    LaunchQuantizeActivationQ8_1(bf16_x, s_act_q8_scratch, batch, k, stream);
    LaunchBatchedQuantGEMMPreQuantized(type, w, s_act_q8_scratch, y, batch, m,
                                       k, stream);
    return;
  }

  constexpr std::size_t kWarp = 32;
  const std::size_t warps_per_block = 8;
  const std::size_t rows_per_block = warps_per_block;
  const std::size_t num_blocks_x = (m + rows_per_block - 1) / rows_per_block;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * warps_per_block));
  BatchedQuantGEMVKernel<<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
