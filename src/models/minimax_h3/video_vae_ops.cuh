#ifndef STRIX_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_CUH_
#define STRIX_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_CUH_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::minimax_h3::video_vae_ops {

static __global__ void AddBiasKernel(float* values, const float* bias,
                                     std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    values[static_cast<std::size_t>(row) * width + column] += bias[column];
  }
}

inline void LaunchAddBias(float* values, const float* bias, std::uint32_t rows,
                          std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(AddBiasKernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, values, bias, rows, width);
}

static __global__ void PrepareTileKernel(
    const float* normalized, const float* mean, const float* deviation,
    float* rows, std::uint32_t latent_time, std::uint32_t latent_height,
    std::uint32_t latent_width, std::uint32_t start_time, std::uint32_t start_y,
    std::uint32_t start_x, std::uint32_t tile_time, std::uint32_t tile_height,
    std::uint32_t tile_width, std::uint32_t channels) {
  const std::size_t elements =
      static_cast<std::size_t>(tile_time) * tile_height * tile_width * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t channel = index % channels;
  std::size_t remaining = index / channels;
  const std::uint32_t x = remaining % tile_width;
  remaining /= tile_width;
  const std::uint32_t y = remaining % tile_height;
  const std::uint32_t time = remaining / tile_height;
  const std::size_t source =
      (((static_cast<std::size_t>(channel) * latent_time +
         (start_time + time)) *
            latent_height +
        (start_y + y)) *
           latent_width +
       (start_x + x));
  rows[index] = fmaf(normalized[source], deviation[channel], mean[channel]);
}

inline void LaunchPrepareTile(
    const float* normalized, const float* mean, const float* deviation,
    float* rows, std::uint32_t latent_time, std::uint32_t latent_height,
    std::uint32_t latent_width, std::uint32_t start_time, std::uint32_t start_y,
    std::uint32_t start_x, std::uint32_t tile_time, std::uint32_t tile_height,
    std::uint32_t tile_width, std::uint32_t channels, hipStream_t stream) {
  const std::size_t elements =
      static_cast<std::size_t>(tile_time) * tile_height * tile_width * channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      PrepareTileKernel, dim3((elements + kThreads - 1U) / kThreads),
      dim3(kThreads), 0, stream, normalized, mean, deviation, rows, latent_time,
      latent_height, latent_width, start_time, start_y, start_x, tile_time,
      tile_height, tile_width, channels);
}

static __global__ void RopeKernel(float* cosine, float* sine,
                                  std::uint32_t patches, std::uint32_t sequence,
                                  std::uint32_t tile_time,
                                  std::uint32_t tile_height,
                                  std::uint32_t tile_width) {
  const std::uint32_t frequency = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= sequence || frequency >= 24) {
    return;
  }
  const std::size_t output = static_cast<std::size_t>(row) * 24 + frequency;
  if (row >= patches) {
    cosine[output] = 1.0F;
    sine[output] = 0.0F;
    return;
  }
  std::uint32_t remaining = row;
  const std::uint32_t x = remaining % tile_width;
  remaining /= tile_width;
  const std::uint32_t y = remaining % tile_height;
  const std::uint32_t time = remaining / tile_height;
  const std::uint32_t axis = frequency / 8U;
  const std::uint32_t frequency_in_axis = frequency % 8U;
  float coordinate = 0.0F;
  if (axis == 0) {
    coordinate = 2.0F * ((static_cast<float>(time) + 0.5F) /
                         static_cast<float>(tile_time)) -
                 1.0F;
  } else if (axis == 1) {
    coordinate = 2.0F * ((static_cast<float>(y) + 0.5F) /
                         static_cast<float>(tile_height)) -
                 1.0F;
  } else {
    coordinate = 2.0F * ((static_cast<float>(x) + 0.5F) /
                         static_cast<float>(tile_width)) -
                 1.0F;
  }
  constexpr float kTwoPi = 6.28318530717958647692F;
  const float inverse =
      1.0F / powf(100.0F, static_cast<float>(frequency_in_axis) * 0.125F);
  const float angle = kTwoPi * coordinate * inverse;
  cosine[output] = cosf(angle);
  sine[output] = sinf(angle);
}

inline void LaunchRope(float* cosine, float* sine, std::uint32_t patches,
                       std::uint32_t sequence, std::uint32_t tile_time,
                       std::uint32_t tile_height, std::uint32_t tile_width,
                       hipStream_t stream) {
  hipLaunchKernelGGL(RopeKernel, dim3(1, sequence), dim3(32), 0, stream, cosine,
                     sine, patches, sequence, tile_time, tile_height,
                     tile_width);
}

static __global__ void RmsNormKernel(const float* input, const float* weight,
                                     float* output, std::uint32_t rows,
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
    local = fmaf(input[base + column], input[base + column], local);
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
    output[base + column] = input[base + column] * inverse * weight[column];
  }
}

inline void LaunchRmsNorm(const float* input, const float* weight,
                          float* output, std::uint32_t rows,
                          std::uint32_t width, float epsilon,
                          hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(32), 0, stream, input,
                     weight, output, rows, width, epsilon);
}

static __global__ void LayerNormKernel(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       std::uint32_t rows, std::uint32_t width,
                                       float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float sums[256];
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local_sum = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    local_sum += input[base + column];
  }
  sums[lane] = local_sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    __syncthreads();
  }
  const float mean = sums[0] / static_cast<float>(width);
  float local_square = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float centered = input[base + column] - mean;
    local_square = fmaf(centered, centered, local_square);
  }
  sums[lane] = local_square;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    __syncthreads();
  }
  const float variance = sums[0] / static_cast<float>(width);
  const float inverse = rsqrtf(variance + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    output[base + column] =
        (input[base + column] - mean) * inverse * weight[column] + bias[column];
  }
}

inline void LaunchLayerNorm(const float* input, const float* weight,
                            const float* bias, float* output,
                            std::uint32_t rows, std::uint32_t width,
                            float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(32), 0, stream, input,
                     weight, bias, output, rows, width, epsilon);
}

static __global__ void VideoQkvRopeKernel(
    const float* qkv, const float* cosine, const float* sine, float* query,
    float* key, float* value, std::uint32_t rows, std::uint32_t heads,
    std::uint32_t head_dimension, std::uint32_t rope_half, float epsilon) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || head >= heads || lane >= head_dimension) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension * 3U;
  const float query_value = qkv[base + lane];
  const float key_value = qkv[base + head_dimension + lane];
  float query_square = 0.0F;
  float key_square = 0.0F;
  for (std::uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
    const float query_component = qkv[base + dimension];
    const float key_component = qkv[base + head_dimension + dimension];
    query_square = fmaf(query_component, query_component, query_square);
    key_square = fmaf(key_component, key_component, key_square);
  }
  const float query_inverse =
      rsqrtf(query_square / static_cast<float>(head_dimension) + epsilon);
  const float key_inverse =
      rsqrtf(key_square / static_cast<float>(head_dimension) + epsilon);
  float transformed_query = query_value * query_inverse;
  float transformed_key = key_value * key_inverse;
  if (lane < rope_half) {
    const std::uint32_t pair = lane + rope_half;
    const float pair_query = qkv[base + pair] * query_inverse;
    const float pair_key = qkv[base + head_dimension + pair] * key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + lane];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + lane];
    transformed_query = transformed_query * c - pair_query * s;
    transformed_key = transformed_key * c - pair_key * s;
  } else if (lane < rope_half * 2U) {
    const std::uint32_t pair = lane - rope_half;
    const float pair_query = qkv[base + pair] * query_inverse;
    const float pair_key = qkv[base + head_dimension + pair] * key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + pair];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + pair];
    transformed_query = transformed_query * c + pair_query * s;
    transformed_key = transformed_key * c + pair_key * s;
  }
  const std::size_t output =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension + lane;
  query[output] = transformed_query;
  key[output] = transformed_key;
  value[output] = qkv[base + head_dimension * 2U + lane];
}

inline void LaunchVideoQkvRope(const float* qkv, const float* cosine,
                               const float* sine, float* query, float* key,
                               float* value, std::uint32_t rows,
                               std::uint32_t heads,
                               std::uint32_t head_dimension,
                               std::uint32_t rope_half, float epsilon,
                               hipStream_t stream) {
  hipLaunchKernelGGL(VideoQkvRopeKernel, dim3(heads, rows),
                     dim3(head_dimension), 0, stream, qkv, cosine, sine, query,
                     key, value, rows, heads, head_dimension, rope_half,
                     epsilon);
}

// One workgroup owns one (query row, head). Scores are retained in dynamic
// shared memory so the value reduction does not materialize an NxN matrix.
static __global__ void FullAttentionKernel(const float* query, const float* key,
                                           const float* value, float* output,
                                           std::uint32_t rows,
                                           std::uint32_t heads,
                                           std::uint32_t head_dimension,
                                           float scale) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = scores + rows;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  if (query_row >= rows || head >= heads) {
    return;
  }
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * heads + head) * head_dimension;
  float local_max = -INFINITY;
  for (std::uint32_t key_row = lane; key_row < rows; key_row += blockDim.x) {
    const std::size_t key_base =
        (static_cast<std::size_t>(key_row) * heads + head) * head_dimension;
    float dot = 0.0F;
    for (std::uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
      dot = fmaf(query[query_base + dimension], key[key_base + dimension], dot);
    }
    dot *= scale;
    scores[key_row] = dot;
    local_max = fmaxf(local_max, dot);
  }
  reduction[lane] = local_max;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_sum = 0.0F;
  for (std::uint32_t key_row = lane; key_row < rows; key_row += blockDim.x) {
    const float probability = expf(scores[key_row] - maximum);
    scores[key_row] = probability;
    local_sum += probability;
  }
  reduction[lane] = local_sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    float accumulated = 0.0F;
    for (std::uint32_t key_row = 0; key_row < rows; ++key_row) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_row) * heads + head) * head_dimension +
          dimension;
      accumulated =
          fmaf(scores[key_row] * inverse_sum, value[value_index], accumulated);
    }
    output[query_base + dimension] = accumulated;
  }
}

inline void LaunchFullAttention(const float* query, const float* key,
                                const float* value, float* output,
                                std::uint32_t rows, std::uint32_t heads,
                                std::uint32_t head_dimension, float scale,
                                hipStream_t stream) {
  // gfx1151 uses 32-lane wavefronts. Keep the complete softmax reduction in one
  // wavefront; each lane emits two dimensions of the 64-wide attention head.
  constexpr std::uint32_t kThreads = 32;
  const std::size_t shared_bytes =
      (static_cast<std::size_t>(rows) + kThreads) * sizeof(float);
  hipLaunchKernelGGL(FullAttentionKernel, dim3(rows, heads), dim3(kThreads),
                     shared_bytes, stream, query, key, value, output, rows,
                     heads, head_dimension, scale);
}

static __global__ void SwiGluKernel(const float* fused, float* output,
                                    std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t base = static_cast<std::size_t>(row) * width * 2U;
    const float gate = fused[base + column];
    output[static_cast<std::size_t>(row) * width + column] =
        (gate / (1.0F + expf(-gate))) * fused[base + width + column];
  }
}

inline void LaunchSwiGlu(const float* fused, float* output, std::uint32_t rows,
                         std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(SwiGluKernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, fused, output, rows, width);
}

static __global__ void ScaleAddKernel(float* residual, const float* branch,
                                      const float* scale, std::uint32_t rows,
                                      std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    residual[index] = fmaf(branch[index], scale[column], residual[index]);
  }
}

inline void LaunchScaleAdd(float* residual, const float* branch,
                           const float* scale, std::uint32_t rows,
                           std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(
      ScaleAddKernel, dim3((width + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, residual, branch, scale, rows, width);
}

static __global__ void UnpackRgbKernel(const float* projected,
                                       const std::int32_t* frames, float* rgb,
                                       std::uint32_t frame_count,
                                       std::uint32_t tile_height,
                                       std::uint32_t tile_width,
                                       std::uint32_t output_frames) {
  const std::size_t elements = static_cast<std::size_t>(frame_count) *
                               tile_height * 16U * tile_width * 16U * 3U;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t channel = index % 3U;
  std::size_t remaining = index / 3U;
  const std::uint32_t x = remaining % (tile_width * 16U);
  remaining /= tile_width * 16U;
  const std::uint32_t y = remaining % (tile_height * 16U);
  const std::uint32_t selected = remaining / (tile_height * 16U);
  const std::uint32_t frame = static_cast<std::uint32_t>(frames[selected]);
  if (frame >= output_frames) {
    return;
  }
  std::uint32_t decoded_time = frame + 3U;
  if (output_frames == 22U && frame >= 17U) {
    decoded_time += 3U;
  }
  const std::uint32_t patch_time = decoded_time / 4U;
  const std::uint32_t within_time = decoded_time % 4U;
  const std::uint32_t patch_y = y / 16U;
  const std::uint32_t within_y = y % 16U;
  const std::uint32_t patch_x = x / 16U;
  const std::uint32_t within_x = x % 16U;
  const std::size_t patch =
      (static_cast<std::size_t>(patch_time) * tile_height + patch_y) *
          tile_width +
      patch_x;
  const std::size_t component =
      (((static_cast<std::size_t>(channel) * 4U + within_time) * 16U +
        within_y) *
           16U +
       within_x);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kDeviation[3] = {0.229F, 0.224F, 0.225F};
  const float value = fmaf(projected[patch * 3072U + component],
                           kDeviation[channel], kMean[channel]);
  rgb[index] = fminf(1.0F, fmaxf(0.0F, value));
}

inline void LaunchUnpackRgb(const float* projected, const std::int32_t* frames,
                            float* rgb, std::uint32_t frame_count,
                            std::uint32_t tile_height, std::uint32_t tile_width,
                            std::uint32_t output_frames, hipStream_t stream) {
  const std::size_t elements = static_cast<std::size_t>(frame_count) *
                               tile_height * 16U * tile_width * 16U * 3U;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(UnpackRgbKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, projected, frames, rgb,
                     frame_count, tile_height, tile_width, output_frames);
}

}  // namespace strix::minimax_h3::video_vae_ops

#endif  // STRIX_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_CUH_
