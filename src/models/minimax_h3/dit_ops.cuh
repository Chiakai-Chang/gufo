#ifndef STRIX_MODELS_MINIMAX_H3_DIT_OPS_CUH_
#define STRIX_MODELS_MINIMAX_H3_DIT_OPS_CUH_

// MiniMax H3 DiT operation boundaries translated from antirez/h3.c
// h3_shaders.metal at 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT).
// This is a model-private HIP implementation for gfx1151. It intentionally
// keeps BF16 write boundaries and F32 reductions visible.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::minimax_h3::dit_ops {

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  std::uint32_t bits = __float_as_uint(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

static __global__ void F32ToBf16Kernel(const float* input,
                                       std::uint16_t* output,
                                       std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = FloatToBf16(input[index]);
  }
}

static __global__ void Bf16ToF32Kernel(const std::uint16_t* input,
                                       float* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = Bf16ToFloat(input[index]);
  }
}

inline void LaunchF32ToBf16(const float* input, std::uint16_t* output,
                            std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(F32ToBf16Kernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

inline void LaunchBf16ToF32(const std::uint16_t* input, float* output,
                            std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(Bf16ToF32Kernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void FirstMismatchKernel(const std::uint16_t* left,
                                           const std::uint16_t* right,
                                           std::size_t elements,
                                           unsigned long long* first) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements && left[index] != right[index]) {
    atomicMin(first, static_cast<unsigned long long>(index));
  }
}

inline void LaunchFirstMismatch(const std::uint16_t* left,
                                const std::uint16_t* right,
                                std::size_t elements,
                                unsigned long long* first,
                                hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(FirstMismatchKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, left, right, elements, first);
}

static __global__ void FillBf16PatternKernel(std::uint16_t* output,
                                             std::size_t elements,
                                             std::uint32_t seed) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  std::uint32_t value = static_cast<std::uint32_t>(index) ^ seed;
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  value ^= value >> 16U;
  const float scaled =
      (static_cast<float>(value & 0xFFFFU) / 32767.5F - 1.0F) * 0.75F;
  output[index] = FloatToBf16(scaled);
}

inline void LaunchFillBf16Pattern(std::uint16_t* output,
                                  std::size_t elements, std::uint32_t seed,
                                  hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(FillBf16PatternKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, output, elements, seed);
}

static __global__ void AddSubtractKernel(const std::uint16_t* left,
                                         const std::uint16_t* right,
                                         std::uint16_t* output,
                                         std::size_t elements, bool subtract) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float right_value = Bf16ToFloat(right[index]);
    output[index] = FloatToBf16(Bf16ToFloat(left[index]) +
                                (subtract ? -right_value : right_value));
  }
}

inline void LaunchAddSubtract(const std::uint16_t* left,
                              const std::uint16_t* right,
                              std::uint16_t* output, std::size_t elements,
                              bool subtract, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AddSubtractKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, left, right, output, elements,
                     subtract);
}

static __global__ void SiluKernel(const std::uint16_t* input,
                                  std::uint16_t* output,
                                  std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = Bf16ToFloat(input[index]);
    output[index] = FloatToBf16(value / (1.0F + expf(-value)));
  }
}

inline void LaunchSilu(const std::uint16_t* input, std::uint16_t* output,
                       std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(SiluKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void PatchProjectionKernel(
    const float* input, const float* weight, const float* bias,
    std::uint16_t* output, std::uint32_t rows, std::uint32_t input_width,
    std::uint32_t output_width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= output_width) {
    return;
  }
  float sum = bias == nullptr ? 0.0F : bias[column];
  const std::size_t input_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t weight_base =
      static_cast<std::size_t>(column) * input_width;
  for (std::uint32_t index = 0; index < input_width; ++index) {
    sum = fmaf(input[input_base + index], weight[weight_base + index], sum);
  }
  output[static_cast<std::size_t>(row) * output_width + column] =
      FloatToBf16(sum);
}

inline void LaunchPatchProjection(const float* input, const float* weight,
                                  const float* bias, std::uint16_t* output,
                                  std::uint32_t rows,
                                  std::uint32_t input_width,
                                  std::uint32_t output_width,
                                  hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(PatchProjectionKernel,
                     dim3((output_width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     rows, input_width, output_width);
}

static __global__ void FinalProjectionKernel(
    const std::uint16_t* input, const float* weight, const float* bias,
    float* output, std::uint32_t rows, std::uint32_t input_width,
    std::uint32_t output_width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= output_width) {
    return;
  }
  float sum = bias == nullptr ? 0.0F : bias[column];
  const std::size_t input_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t weight_base =
      static_cast<std::size_t>(column) * input_width;
  for (std::uint32_t index = 0; index < input_width; ++index) {
    sum = fmaf(Bf16ToFloat(input[input_base + index]),
               weight[weight_base + index], sum);
  }
  output[static_cast<std::size_t>(row) * output_width + column] = sum;
}

inline void LaunchFinalProjection(const std::uint16_t* input,
                                  const float* weight, const float* bias,
                                  float* output, std::uint32_t rows,
                                  std::uint32_t input_width,
                                  std::uint32_t output_width,
                                  hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  hipLaunchKernelGGL(FinalProjectionKernel,
                     dim3((output_width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     rows, input_width, output_width);
}

static __global__ void SwiGluKernel(const std::uint16_t* fused,
                                    std::uint16_t* output,
                                    std::uint32_t rows,
                                    std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * width * 2U;
  const float gate = Bf16ToFloat(fused[base + column]);
  const float up = Bf16ToFloat(fused[base + width + column]);
  output[static_cast<std::size_t>(row) * width + column] =
      FloatToBf16(gate / (1.0F + expf(-gate)) * up);
}

inline void LaunchSwiGlu(const std::uint16_t* fused, std::uint16_t* output,
                         std::uint32_t rows, std::uint32_t width,
                         hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(SwiGluKernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, fused, output, rows, width);
}

static __global__ void RmsNormKernel(const std::uint16_t* input,
                                     const std::uint16_t* weight,
                                     std::uint16_t* output,
                                     std::uint32_t rows,
                                     std::uint32_t width, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    output[base + column] =
        FloatToBf16(Bf16ToFloat(input[base + column]) * inverse *
                    Bf16ToFloat(weight[column]));
  }
}

inline void LaunchRmsNorm(const std::uint16_t* input,
                          const std::uint16_t* weight, std::uint16_t* output,
                          std::uint32_t rows, std::uint32_t width,
                          float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, output, rows, width, epsilon);
}

static __global__ void LayerNormKernel(const std::uint16_t* input,
                                       const std::uint16_t* weight,
                                       const std::uint16_t* bias,
                                       std::uint16_t* output,
                                       std::uint32_t rows,
                                       std::uint32_t width, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float sums[256];
  __shared__ float squares[256];
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local_sum = 0.0F;
  float local_square = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[base + column]);
    local_sum += value;
    local_square = fmaf(value, value, local_square);
  }
  sums[lane] = local_sum;
  squares[lane] = local_square;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      squares[lane] += squares[lane + stride];
    }
    __syncthreads();
  }
  const float mean = sums[0] / static_cast<float>(width);
  const float variance =
      fmaxf(0.0F, squares[0] / static_cast<float>(width) - mean * mean);
  const float inverse = rsqrtf(variance + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float normalized =
        (Bf16ToFloat(input[base + column]) - mean) * inverse;
    output[base + column] =
        FloatToBf16(normalized * Bf16ToFloat(weight[column]) +
                    (bias == nullptr ? 0.0F : Bf16ToFloat(bias[column])));
  }
}

inline void LaunchLayerNorm(const std::uint16_t* input,
                            const std::uint16_t* weight,
                            const std::uint16_t* bias, std::uint16_t* output,
                            std::uint32_t rows, std::uint32_t width,
                            float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, bias, output, rows, width, epsilon);
}

static __global__ void AdaLnKernel(
    const std::uint16_t* input, const std::uint16_t* weight,
    const std::uint16_t* modulation, const std::uint32_t* row_map,
    std::uint16_t* output, std::uint32_t rows, std::uint32_t width,
    std::uint32_t slots, std::uint32_t scale_slot,
    std::uint32_t shift_slot, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const std::size_t row_base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(input[row_base + column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  const std::size_t modulation_base =
      static_cast<std::size_t>(row_map[row]) * slots * width;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float normalized =
        Bf16ToFloat(input[row_base + column]) * inverse *
        Bf16ToFloat(weight[column]);
    const float scale = Bf16ToFloat(
        modulation[modulation_base +
                   static_cast<std::size_t>(scale_slot) * width + column]);
    const float shift = Bf16ToFloat(
        modulation[modulation_base +
                   static_cast<std::size_t>(shift_slot) * width + column]);
    output[row_base + column] =
        FloatToBf16(normalized * (1.0F + scale) + shift);
  }
}

inline void LaunchAdaLn(const std::uint16_t* input,
                        const std::uint16_t* weight,
                        const std::uint16_t* modulation,
                        const std::uint32_t* row_map, std::uint16_t* output,
                        std::uint32_t rows, std::uint32_t width,
                        std::uint32_t slots, std::uint32_t scale_slot,
                        std::uint32_t shift_slot, float epsilon,
                        hipStream_t stream) {
  hipLaunchKernelGGL(AdaLnKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, modulation, row_map, output, rows, width, slots,
                     scale_slot, shift_slot, epsilon);
}

static __global__ void GateKernel(
    const std::uint16_t* residual, const std::uint16_t* branch,
    const std::uint16_t* modulation, const std::uint32_t* row_map,
    std::uint16_t* output, std::uint32_t rows, std::uint32_t width,
    std::uint32_t slots, std::uint32_t gate_slot) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= width) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * width + column;
  const std::size_t modulation_index =
      static_cast<std::size_t>(row_map[row]) * slots * width +
      static_cast<std::size_t>(gate_slot) * width + column;
  output[index] =
      FloatToBf16(Bf16ToFloat(residual[index]) +
                  Bf16ToFloat(branch[index]) *
                      Bf16ToFloat(modulation[modulation_index]));
}

inline void LaunchGate(const std::uint16_t* residual,
                       const std::uint16_t* branch,
                       const std::uint16_t* modulation,
                       const std::uint32_t* row_map, std::uint16_t* output,
                       std::uint32_t rows, std::uint32_t width,
                       std::uint32_t slots, std::uint32_t gate_slot,
                       hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(GateKernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, residual, branch, modulation,
                     row_map, output, rows, width, slots, gate_slot);
}

static __global__ void GroupedQkvNormRopeKernel(
    const std::uint16_t* qkv, const std::uint16_t* query_weight,
    const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
    const std::uint16_t* rope_sin, std::uint16_t* query,
    std::uint16_t* key, std::uint16_t* value, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t head_dimension,
    std::uint32_t rope_half, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dimension = threadIdx.x;
  if (row >= sequence || head >= heads || dimension >= head_dimension) {
    return;
  }
  const std::size_t inner = static_cast<std::size_t>(heads) * head_dimension;
  const std::size_t row_base = static_cast<std::size_t>(row) * inner * 3U;
  const std::size_t query_base =
      row_base + static_cast<std::size_t>(head) * head_dimension * 3U;
  const std::size_t key_base = query_base + head_dimension;
  const std::size_t value_base = key_base + head_dimension;
  __shared__ float inverses[2];
  if (dimension == 0) {
    float query_sum = 0.0F;
    float key_sum = 0.0F;
    for (std::uint32_t index = 0; index < head_dimension; ++index) {
      const float query_value = Bf16ToFloat(qkv[query_base + index]);
      const float key_value = Bf16ToFloat(qkv[key_base + index]);
      query_sum = fmaf(query_value, query_value, query_sum);
      key_sum = fmaf(key_value, key_value, key_sum);
    }
    inverses[0] =
        rsqrtf(query_sum / static_cast<float>(head_dimension) + epsilon);
    inverses[1] =
        rsqrtf(key_sum / static_cast<float>(head_dimension) + epsilon);
  }
  __syncthreads();
  float query_value = Bf16ToFloat(qkv[query_base + dimension]) * inverses[0] *
                      Bf16ToFloat(query_weight[dimension]);
  float key_value = Bf16ToFloat(qkv[key_base + dimension]) * inverses[1] *
                    Bf16ToFloat(key_weight[dimension]);
  if (dimension < rope_half) {
    const std::uint32_t pair = dimension + rope_half;
    const float paired_query =
        Bf16ToFloat(qkv[query_base + pair]) * inverses[0] *
        Bf16ToFloat(query_weight[pair]);
    const float paired_key = Bf16ToFloat(qkv[key_base + pair]) * inverses[1] *
                             Bf16ToFloat(key_weight[pair]);
    const float cosine =
        Bf16ToFloat(rope_cos[static_cast<std::size_t>(row) * rope_half +
                             dimension]);
    const float sine =
        Bf16ToFloat(rope_sin[static_cast<std::size_t>(row) * rope_half +
                             dimension]);
    query_value = query_value * cosine - paired_query * sine;
    key_value = key_value * cosine - paired_key * sine;
  } else if (dimension < rope_half * 2U) {
    const std::uint32_t pair = dimension - rope_half;
    const float paired_query =
        Bf16ToFloat(qkv[query_base + pair]) * inverses[0] *
        Bf16ToFloat(query_weight[pair]);
    const float paired_key = Bf16ToFloat(qkv[key_base + pair]) * inverses[1] *
                             Bf16ToFloat(key_weight[pair]);
    const float cosine =
        Bf16ToFloat(rope_cos[static_cast<std::size_t>(row) * rope_half + pair]);
    const float sine =
        Bf16ToFloat(rope_sin[static_cast<std::size_t>(row) * rope_half + pair]);
    query_value = query_value * cosine + paired_query * sine;
    key_value = key_value * cosine + paired_key * sine;
  }
  const std::size_t output =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension +
      dimension;
  query[output] = FloatToBf16(query_value);
  key[output] = FloatToBf16(key_value);
  value[output] = qkv[value_base + dimension];
}

inline void LaunchGroupedQkvNormRope(
    const std::uint16_t* qkv, const std::uint16_t* query_weight,
    const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
    const std::uint16_t* rope_sin, std::uint16_t* query,
    std::uint16_t* key, std::uint16_t* value, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t head_dimension,
    std::uint32_t rope_half, float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(GroupedQkvNormRopeKernel, dim3(sequence, heads),
                     dim3(head_dimension), 0, stream, qkv, query_weight,
                     key_weight, rope_cos, rope_sin, query, key, value,
                     sequence, heads, head_dimension, rope_half, epsilon);
}

static __global__ void FullAttentionKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output,
    std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t head_dimension, float scale) {
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || head >= heads) {
    return;
  }
  extern __shared__ float scores[];
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * heads + head) * head_dimension;
  if (lane == 0) {
    float maximum = -INFINITY;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      const std::size_t key_base =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension;
      float dot = 0.0F;
      for (std::uint32_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        dot = fmaf(Bf16ToFloat(query[query_base + dimension]),
                   Bf16ToFloat(key[key_base + dimension]), dot);
      }
      scores[key_row] = dot * scale;
      maximum = fmaxf(maximum, scores[key_row]);
    }
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      scores[key_row] = expf(scores[key_row] - maximum);
      sum += scores[key_row];
    }
    const float inverse_sum = 1.0F / sum;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      scores[key_row] *= inverse_sum;
    }
  }
  __syncthreads();
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t key_row = 0; key_row < sequence; ++key_row) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension +
          dimension;
      sum = fmaf(scores[key_row], Bf16ToFloat(value[value_index]), sum);
    }
    output[query_base + dimension] = FloatToBf16(sum);
  }
}

inline void LaunchFullAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output,
    std::uint32_t sequence, std::uint32_t heads,
    std::uint32_t head_dimension, float scale, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  hipLaunchKernelGGL(FullAttentionKernel, dim3(sequence, heads),
                     dim3(kThreads), static_cast<std::size_t>(sequence) *
                                         sizeof(float),
                     stream, query, key, value, output, sequence, heads,
                     head_dimension, scale);
}

}  // namespace strix::minimax_h3::dit_ops

#endif  // STRIX_MODELS_MINIMAX_H3_DIT_OPS_CUH_
