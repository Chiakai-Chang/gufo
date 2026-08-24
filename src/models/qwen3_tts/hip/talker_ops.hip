#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/models/qwen3_tts/hip/talker_ops.hpp"

namespace strix::models::qwen3_tts::hip {
namespace {

template<std::uint32_t ThreadsPerBlock>
__launch_bounds__(ThreadsPerBlock, 4) __global__
    void Bfloat16PerHeadRMSNormKernel(float* values, const float* weight,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim, float epsilon) {
  const std::size_t batch = blockIdx.y;
  const std::uint32_t head = blockIdx.x;
  if (batch >= batch_size || head >= num_heads) {
    return;
  }

  const std::size_t thread = threadIdx.x;
  const std::size_t head_stride =
      static_cast<std::size_t>(num_heads) * head_dim;
  float* head_values = values + (batch * head_stride) +
                       (static_cast<std::size_t>(head) * head_dim);

  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    sum_squared += value * value;
  }

  __shared__ float shared_sum[ThreadsPerBlock];
  shared_sum[thread] = sum_squared;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }

  const float mean_squared = shared_sum[0] / static_cast<float>(head_dim);
  const float inverse_rms = rsqrtf(mean_squared + epsilon);
  for (std::size_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    head_values[index] =
        static_cast<float>(hip_bfloat16((value * inverse_rms) * scale));
  }
}

__launch_bounds__(256, 4) __global__ void Bfloat16RoPEAndRoundVKernel(
    float* query, float* key, float* value, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta) {
  const std::size_t batch = blockIdx.y;
  if (batch >= batch_size) {
    return;
  }

  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t half_rotary = head_dim / 2;
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(batch);
  if (index < static_cast<std::size_t>(num_heads) * half_rotary) {
    const std::size_t head = index / half_rotary;
    const std::size_t dimension = index % half_rotary;
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const std::size_t stride = static_cast<std::size_t>(num_heads) * head_dim;
    float* head_values = query + (batch * stride) + (head * head_dim);
    const float first = head_values[dimension];
    const float second = head_values[dimension + half_rotary];
    head_values[dimension] =
        static_cast<float>(hip_bfloat16((first * cosine) - (second * sine)));
    head_values[dimension + half_rotary] =
        static_cast<float>(hip_bfloat16((first * sine) + (second * cosine)));
  }

  if (index < static_cast<std::size_t>(num_key_value_heads) * half_rotary) {
    const std::size_t head = index / half_rotary;
    const std::size_t dimension = index % half_rotary;
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const std::size_t stride =
        static_cast<std::size_t>(num_key_value_heads) * head_dim;
    float* head_values = key + (batch * stride) + (head * head_dim);
    const float first = head_values[dimension];
    const float second = head_values[dimension + half_rotary];
    head_values[dimension] =
        static_cast<float>(hip_bfloat16((first * cosine) - (second * sine)));
    head_values[dimension + half_rotary] =
        static_cast<float>(hip_bfloat16((first * sine) + (second * cosine)));
  }

  const std::size_t value_elements =
      static_cast<std::size_t>(num_key_value_heads) * head_dim;
  if (index < value_elements) {
    float* batch_values = value + (batch * value_elements);
    batch_values[index] = static_cast<float>(hip_bfloat16(batch_values[index]));
  }
}

/// Normalizes one attention head and rotates it without a round trip through
/// memory between the two steps. Reproduces the standalone per-head RMSNorm and
/// RoPE kernels exactly, including every intermediate BF16 rounding.
__device__ void NormalizeAndRotateHead(float* head_values, const float* weight,
                                       std::uint32_t head_dim, float epsilon,
                                       std::uint32_t position, float rope_theta,
                                       float* shared_sum, float* normalized) {
  const std::uint32_t thread = threadIdx.x;
  float sum_squared = 0.0F;
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    sum_squared += value * value;
  }
  shared_sum[thread] = sum_squared;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }
  const float mean_squared = shared_sum[0] / static_cast<float>(head_dim);
  const float inverse_rms = rsqrtf(mean_squared + epsilon);
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    normalized[index] =
        static_cast<float>(hip_bfloat16((value * inverse_rms) * scale));
  }
  __syncthreads();

  const std::uint32_t half_rotary = head_dim / 2;
  for (std::uint32_t dimension = thread; dimension < half_rotary;
       dimension += blockDim.x) {
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = normalized[dimension];
    const float second = normalized[dimension + half_rotary];
    head_values[dimension] =
        static_cast<float>(hip_bfloat16((first * cosine) - (second * sine)));
    head_values[dimension + half_rotary] =
        static_cast<float>(hip_bfloat16((first * sine) + (second * cosine)));
  }
  __syncthreads();
}

template<std::uint32_t ThreadsPerBlock>
__launch_bounds__(ThreadsPerBlock) __global__ void Bfloat16QkNormRoPEKernel(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon) {
  const std::size_t batch = blockIdx.y;
  const std::uint32_t head = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  __shared__ float shared_sum[ThreadsPerBlock];
  __shared__ float normalized[ThreadsPerBlock];
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(batch);

  // Both branches are uniform across the workgroup, so the barriers inside the
  // helper stay collective.
  if (head < num_heads) {
    const std::size_t stride = static_cast<std::size_t>(num_heads) * head_dim;
    NormalizeAndRotateHead(
        query + (batch * stride) + (static_cast<std::size_t>(head) * head_dim),
        query_weight, head_dim, epsilon, position, rope_theta, shared_sum,
        normalized);
  }
  if (head < num_key_value_heads) {
    const std::size_t stride =
        static_cast<std::size_t>(num_key_value_heads) * head_dim;
    const std::size_t offset =
        (batch * stride) + (static_cast<std::size_t>(head) * head_dim);
    NormalizeAndRotateHead(key + offset, key_weight, head_dim, epsilon,
                           position, rope_theta, shared_sum, normalized);
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x) {
      value[offset + index] =
          static_cast<float>(hip_bfloat16(value[offset + index]));
    }
  }
}

/// Adds the residual and normalizes in one pass. The elementwise sum is order
/// independent and the reduction keeps the 256-thread tree of the standalone
/// RMSNorm, so the BF16 result is unchanged.
__launch_bounds__(256) __global__ void Bfloat16ResidualAddRMSNormKernel(
    float* __restrict__ hidden, const float* __restrict__ update,
    const float* __restrict__ weight, hip_bfloat16* __restrict__ output,
    std::size_t batch_size, std::size_t dimension, float epsilon) {
  const std::size_t batch = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  const std::size_t thread = threadIdx.x;
  float* row = hidden + (batch * dimension);
  const float* update_row = update + (batch * dimension);
  hip_bfloat16* output_row = output + (batch * dimension);

  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float left = static_cast<float>(hip_bfloat16(row[index]));
    const float right = static_cast<float>(hip_bfloat16(update_row[index]));
    const float sum = static_cast<float>(hip_bfloat16(left + right));
    row[index] = sum;
    sum_squared += sum * sum;
  }

  __shared__ float shared_sum[256];
  shared_sum[thread] = sum_squared;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }
  const float mean_squared = shared_sum[0] / static_cast<float>(dimension);
  const float inverse_rms = rsqrtf(mean_squared + epsilon);
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    output_row[index] = hip_bfloat16((row[index] * inverse_rms) * scale);
  }
}

__global__ void RoundBfloat16InPlaceKernel(float* values, std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    values[index] = static_cast<float>(hip_bfloat16(values[index]));
  }
}

__global__ void Bfloat16ToFloatKernel(const hip_bfloat16* input, float* output,
                                      std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(input[index]);
  }
}

__global__ void Bfloat16ResidualAddKernel(const float* residual,
                                          const float* update, float* output,
                                          std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float left = static_cast<float>(hip_bfloat16(residual[index]));
    const float right = static_cast<float>(hip_bfloat16(update[index]));
    output[index] = static_cast<float>(hip_bfloat16(left + right));
  }
}

__global__ void Bfloat16SwiGluKernel(const float* gate, const float* up,
                                     float* output,
                                     hip_bfloat16* output_bfloat16,
                                     std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float gate_bfloat16 = static_cast<float>(hip_bfloat16(gate[index]));
    const float up_bfloat16 = static_cast<float>(hip_bfloat16(up[index]));
    const float activated = static_cast<float>(
        hip_bfloat16(gate_bfloat16 / (1.0F + expf(-gate_bfloat16))));
    const hip_bfloat16 result(activated * up_bfloat16);
    if (output != nullptr) {
      output[index] = static_cast<float>(result);
    }
    if (output_bfloat16 != nullptr) {
      output_bfloat16[index] = result;
    }
  }
}

__global__ void Bfloat16BiasSiluKernel(const float* input,
                                       const hip_bfloat16* bias, float* output,
                                       hip_bfloat16* output_bfloat16,
                                       std::size_t count, std::size_t columns) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float biased = static_cast<float>(
        hip_bfloat16(input[index] + static_cast<float>(bias[index % columns])));
    const hip_bfloat16 result(biased / (1.0F + expf(-biased)));
    if (output != nullptr) {
      output[index] = static_cast<float>(result);
    }
    if (output_bfloat16 != nullptr) {
      output_bfloat16[index] = result;
    }
  }
}

__global__ void Bfloat16BiasKernel(const float* input, const hip_bfloat16* bias,
                                   float* output, std::size_t count,
                                   std::size_t columns) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(
        hip_bfloat16(input[index] + static_cast<float>(bias[index % columns])));
  }
}

__global__ void SumCodecEmbeddingsKernel(const float* embeddings,
                                         const float* text_embedding,
                                         float* output, std::size_t groups,
                                         std::size_t hidden_size) {
  const std::size_t dimension = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (dimension < hidden_size) {
    float sum = 0.0F;
    for (std::size_t group = 0; group < groups; ++group) {
      sum += embeddings[(group * hidden_size) + dimension];
    }
    output[dimension] =
        static_cast<float>(hip_bfloat16(sum + text_embedding[dimension]));
  }
}

__device__ __forceinline__ float LowBfloat16(std::uint32_t word) {
  return __uint_as_float(word << 16U);
}

__device__ __forceinline__ float HighBfloat16(std::uint32_t word) {
  return __uint_as_float(word & 0xFFFF0000U);
}

/// Dot product of the BF16 lanes packed into one vectorized load.
__device__ __forceinline__ float PackedDot(uint4 left, uint4 right) {
  return (LowBfloat16(left.x) * LowBfloat16(right.x)) +
         (HighBfloat16(left.x) * HighBfloat16(right.x)) +
         (LowBfloat16(left.y) * LowBfloat16(right.y)) +
         (HighBfloat16(left.y) * HighBfloat16(right.y)) +
         (LowBfloat16(left.z) * LowBfloat16(right.z)) +
         (HighBfloat16(left.z) * HighBfloat16(right.z)) +
         (LowBfloat16(left.w) * LowBfloat16(right.w)) +
         (HighBfloat16(left.w) * HighBfloat16(right.w));
}

__device__ __forceinline__ float PackedDot(uint2 left, uint2 right) {
  return (LowBfloat16(left.x) * LowBfloat16(right.x)) +
         (HighBfloat16(left.x) * HighBfloat16(right.x)) +
         (LowBfloat16(left.y) * LowBfloat16(right.y)) +
         (HighBfloat16(left.y) * HighBfloat16(right.y));
}

constexpr unsigned int kGemvWavefront = 32;
constexpr unsigned int kGemvWaves = 8;
constexpr std::size_t kGemvMaxBatch = 4;

/// Reduces one output row per workgroup. Eight wavefronts each walk the row
/// with their own coalesced stream, which is what lets a single-token
/// projection saturate DRAM. The reduction order is fixed by wavefront index,
/// so the result is reproducible.
///
/// `kVectors` input columns share one pass over the weight row, so the few
/// two-token predictor steps cost the same memory traffic as a single token.
template<typename Packed, unsigned int kVectors>
__launch_bounds__(kGemvWaves* kGemvWavefront) __global__
    void Bfloat16GemvKernel(const Packed* __restrict__ weights,
                            const Packed* __restrict__ inputs,
                            float* __restrict__ output,
                            std::uint32_t packed_columns, std::uint32_t rows) {
  __shared__ float wave_sums[kVectors][kGemvWaves];
  const unsigned int lane = threadIdx.x & (kGemvWavefront - 1U);
  const unsigned int wave = threadIdx.x >> 5U;
  const Packed* row =
      weights + (static_cast<std::size_t>(blockIdx.x) * packed_columns);

  float sums[kVectors];
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    sums[vector] = 0.0F;
  }
  for (std::uint32_t index = (wave * kGemvWavefront) + lane;
       index < packed_columns; index += kGemvWaves * kGemvWavefront) {
    const Packed weight = row[index];
#pragma unroll
    for (unsigned int vector = 0; vector < kVectors; ++vector) {
      sums[vector] +=
          PackedDot(weight, inputs[(vector * packed_columns) + index]);
    }
  }
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    float sum = sums[vector];
#pragma unroll
    for (int offset = kGemvWavefront / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down(sum, offset, kGemvWavefront);
    }
    if (lane == 0) {
      wave_sums[vector][wave] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x < kVectors) {
    float total = 0.0F;
#pragma unroll
    for (unsigned int index = 0; index < kGemvWaves; ++index) {
      total += wave_sums[threadIdx.x][index];
    }
    output[(threadIdx.x * rows) + blockIdx.x] = total;
  }
}

dim3 Grid(std::size_t count) {
  constexpr unsigned int block_size = 256;
  return dim3(static_cast<unsigned int>((count + block_size - 1) / block_size));
}

}  // namespace

void LaunchBfloat16PerHeadRMSNorm(float* values, const float* weight,
                                  std::size_t batch_size,
                                  std::uint32_t num_heads,
                                  std::uint32_t head_dim, float epsilon,
                                  hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || head_dim == 0) {
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  Bfloat16PerHeadRMSNormKernel<128><<<grid, 128, 0, stream>>>(
      values, weight, batch_size, num_heads, head_dim, epsilon);
}

void LaunchBfloat16RoPEAndRoundV(float* query, float* key, float* value,
                                 std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t num_key_value_heads,
                                 std::uint32_t head_dim,
                                 std::uint32_t start_position, float rope_theta,
                                 hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || num_key_value_heads == 0 ||
      head_dim == 0) {
    return;
  }
  const std::size_t half_rotary = head_dim / 2;
  const std::size_t query_pairs =
      static_cast<std::size_t>(num_heads) * half_rotary;
  const std::size_t value_elements =
      static_cast<std::size_t>(num_key_value_heads) * head_dim;
  constexpr std::size_t kBlockSize = 256;
  const std::size_t blocks =
      (std::max(query_pairs, value_elements) + kBlockSize - 1) / kBlockSize;
  const dim3 grid(static_cast<unsigned int>(blocks),
                  static_cast<unsigned int>(batch_size));
  Bfloat16RoPEAndRoundVKernel<<<grid, kBlockSize, 0, stream>>>(
      query, key, value, batch_size, num_heads, num_key_value_heads, head_dim,
      start_position, rope_theta);
}

bool LaunchBfloat16QkNormRoPE(float* query, float* key, float* value,
                              const float* query_weight,
                              const float* key_weight, std::size_t batch_size,
                              std::uint32_t num_heads,
                              std::uint32_t num_key_value_heads,
                              std::uint32_t head_dim,
                              std::uint32_t start_position, float rope_theta,
                              float epsilon, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  if (batch_size == 0 || num_heads == 0 || num_key_value_heads == 0 ||
      head_dim == 0 || head_dim > kThreads || head_dim % 2 != 0) {
    return false;
  }
  const dim3 grid(std::max(num_heads, num_key_value_heads),
                  static_cast<unsigned int>(batch_size));
  Bfloat16QkNormRoPEKernel<kThreads><<<grid, kThreads, 0, stream>>>(
      query, key, value, query_weight, key_weight, batch_size, num_heads,
      num_key_value_heads, head_dim, start_position, rope_theta, epsilon);
  return true;
}

void LaunchBfloat16ResidualAddRMSNorm(float* hidden, const float* update,
                                      const float* weight,
                                      void* output_bfloat16,
                                      std::size_t batch_size,
                                      std::size_t dimension, float epsilon,
                                      hipStream_t stream) {
  if (batch_size == 0 || dimension == 0) {
    return;
  }
  Bfloat16ResidualAddRMSNormKernel<<<static_cast<unsigned int>(batch_size), 256,
                                     0, stream>>>(
      hidden, update, weight, static_cast<hip_bfloat16*>(output_bfloat16),
      batch_size, dimension, epsilon);
}

void LaunchRoundBfloat16InPlace(float* values, std::size_t count,
                                hipStream_t stream) {
  if (count == 0) {
    return;
  }
  RoundBfloat16InPlaceKernel<<<Grid(count), 256, 0, stream>>>(values, count);
}

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16ToFloatKernel<<<Grid(count), 256, 0, stream>>>(
      static_cast<const hip_bfloat16*>(input_bfloat16), output, count);
}

void LaunchBfloat16ResidualAdd(const float* residual, const float* update,
                               float* output, std::size_t count,
                               hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16ResidualAddKernel<<<Grid(count), 256, 0, stream>>>(residual, update,
                                                             output, count);
}

void LaunchBfloat16SwiGlu(const float* gate, const float* up, float* output,
                          void* output_bfloat16, std::size_t count,
                          hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16SwiGluKernel<<<Grid(count), 256, 0, stream>>>(
      gate, up, output, static_cast<hip_bfloat16*>(output_bfloat16), count);
}

void LaunchBfloat16BiasSilu(const float* input, const void* bias_bfloat16,
                            float* output, void* output_bfloat16,
                            std::size_t rows, std::size_t columns,
                            hipStream_t stream) {
  const std::size_t count = rows * columns;
  if (count == 0) {
    return;
  }
  Bfloat16BiasSiluKernel<<<Grid(count), 256, 0, stream>>>(
      input, static_cast<const hip_bfloat16*>(bias_bfloat16), output,
      static_cast<hip_bfloat16*>(output_bfloat16), count, columns);
}

void LaunchBfloat16Bias(const float* input, const void* bias_bfloat16,
                        float* output, std::size_t rows, std::size_t columns,
                        hipStream_t stream) {
  const std::size_t count = rows * columns;
  if (count == 0) {
    return;
  }
  Bfloat16BiasKernel<<<Grid(count), 256, 0, stream>>>(
      input, static_cast<const hip_bfloat16*>(bias_bfloat16), output, count,
      columns);
}

void LaunchSumCodecEmbeddings(const float* embeddings,
                              const float* text_embedding, float* output,
                              std::size_t groups, std::size_t hidden_size,
                              hipStream_t stream) {
  if (groups == 0 || hidden_size == 0) {
    return;
  }
  SumCodecEmbeddingsKernel<<<Grid(hidden_size), 256, 0, stream>>>(
      embeddings, text_embedding, output, groups, hidden_size);
}

bool LaunchBfloat16Gemv(const void* weights_bfloat16,
                        const void* inputs_bfloat16, float* output,
                        std::size_t batch, std::size_t rows,
                        std::size_t columns, hipStream_t stream) {
  if (batch == 0 || batch > kGemvMaxBatch || rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const auto aligned = [&](std::size_t bytes) {
    return reinterpret_cast<std::uintptr_t>(weights_bfloat16) % bytes == 0 &&
           reinterpret_cast<std::uintptr_t>(inputs_bfloat16) % bytes == 0 &&
           (columns * sizeof(std::uint16_t)) % bytes == 0;
  };
  const unsigned int block = kGemvWaves * kGemvWavefront;
  const auto grid = static_cast<unsigned int>(rows);
  const auto row_count = static_cast<std::uint32_t>(rows);

  // Prefer 16-byte loads. Safetensors place every tensor on an 8-byte boundary,
  // so the narrower path is what runs against memory-mapped weights.
  const std::size_t lanes = aligned(sizeof(uint4))   ? 8
                            : aligned(sizeof(uint2)) ? 4
                                                     : 0;
  if (lanes == 0) {
    return false;
  }
  const auto packed = static_cast<std::uint32_t>(columns / lanes);

  const auto launch = [&]<unsigned int kVectors>() {
    if (lanes == 8) {
      Bfloat16GemvKernel<uint4, kVectors><<<grid, block, 0, stream>>>(
          static_cast<const uint4*>(weights_bfloat16),
          static_cast<const uint4*>(inputs_bfloat16), output, packed,
          row_count);
    } else {
      Bfloat16GemvKernel<uint2, kVectors><<<grid, block, 0, stream>>>(
          static_cast<const uint2*>(weights_bfloat16),
          static_cast<const uint2*>(inputs_bfloat16), output, packed,
          row_count);
    }
  };
  switch (batch) {
    case 1:
      launch.operator()<1>();
      return true;
    case 2:
      launch.operator()<2>();
      return true;
    case 3:
      launch.operator()<3>();
      return true;
    case 4:
      launch.operator()<4>();
      return true;
    default:
      return false;
  }
}

}  // namespace strix::models::qwen3_tts::hip

#endif  // defined(ENGINE_ENABLE_HIP)
