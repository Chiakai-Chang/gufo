#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel(const void* __restrict__ A, core::GgmlType type,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const float sumf = QuantWarpBlockDot(type, A, m, x, K, lane_id, 0, 1);
  if (lane_id == 0) {
    y[m] = sumf;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel_2Rows(const void* __restrict__ A,
                                  core::GgmlType type,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, std::size_t M,
                                  std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t pair_idx = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  if (type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = K / kQ8_0BlockSize;
    const auto* row0_blocks =
        static_cast<const Q8_0Block*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const Q8_0Block*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const float xv = x[(b * kQ8_0BlockSize) + lane_id];
      const Q8_0Block& wblk0 = row0_blocks[b];
      float dot0 = static_cast<float>(wblk0.qs[lane_id]) * xv;
      for (int off = 16; off > 0; off >>= 1) {
        dot0 += __shfl_xor(dot0, off);
      }
      sum0 += __half2float(wblk0.d) * dot0;

      if (row1_blocks != nullptr) {
        const Q8_0Block& wblk1 = row1_blocks[b];
        float dot1 = static_cast<float>(wblk1.qs[lane_id]) * xv;
        for (int off = 16; off > 0; off >>= 1) {
          dot1 += __shfl_xor(dot1, off);
        }
        sum1 += __half2float(wblk1.d) * dot1;
      }
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  if (type == core::GgmlType::kQ8_K) {
    const std::size_t num_blocks = K / kQ8KBlockSize;
    const auto* row0_blocks =
        static_cast<const Q8KBlock*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const Q8KBlock*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlock& wblk0 = row0_blocks[b];
      const float d_w0 = wblk0.d;
      const Q8KBlock* wblk1 =
          (row1_blocks != nullptr) ? &row1_blocks[b] : nullptr;
      const float d_w1 = (wblk1 != nullptr) ? wblk1->d : 0.0F;
      const float* xb = x + (b * kQ8KBlockSize);
      float dot0 = 0.0F;
      float dot1 = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32 * k);
        const float xv = xb[idx];
        dot0 += static_cast<float>(wblk0.qs[idx]) * xv;
        if (wblk1 != nullptr) {
          dot1 += static_cast<float>(wblk1->qs[idx]) * xv;
        }
      }
      for (int off = 16; off > 0; off >>= 1) {
        dot0 += __shfl_xor(dot0, off);
        dot1 += __shfl_xor(dot1, off);
      }
      sum0 += d_w0 * dot0;
      sum1 += d_w1 * dot1;
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  const float s0 = QuantWarpBlockDot(type, A, m0, x, K, lane_id, 0, 1);
  const float s1 =
      (m1 < M) ? QuantWarpBlockDot(type, A, m1, x, K, lane_id, 0, 1) : 0.0F;
  if (lane_id == 0) {
    y[m0] = s0;
    if (m1 < M) {
      y[m1] = s1;
    }
  }
}

// opt-c1xx-q8k-gemv: launches the Q8_K / Q8_0 block GEMV.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  if (M >= 2 && (M % 2 == 0)) {
    const std::size_t num_pairs = M / 2;
    const std::size_t grid = (num_pairs + kWavesPerBlock - 1U) / kWavesPerBlock;
    Q8KBlockGEMVKernel_2Rows<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
    return;
  }

  const std::size_t grid = (M + kWavesPerBlock - 1U) / kWavesPerBlock;
  Q8KBlockGEMVKernel<kWavesPerBlock>
      <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
