#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

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

}  // namespace strix::models::qwen3_tts::hip

#endif  // defined(ENGINE_ENABLE_HIP)
