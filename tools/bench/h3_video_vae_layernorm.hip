// MiniMax H3 VisualVAE final LayerNorm launch ablation for gfx1151.
//
// The released 256-pixel tile contains 7 * 16 * 16 patch tokens plus five
// suffix tokens, each 2048 values wide. This harness compares the former
// one-row/one-wave launch with the candidate eight-row/eight-wave launch while
// keeping each row's arithmetic and reduction order identical.
//
//   nix develop -c tools/bench/build.sh h3_video_vae_layernorm
//   /tmp/h3_video_vae_layernorm

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expression)                                           \
  do {                                                                  \
    const hipError_t status_ = (expression);                            \
    if (status_ != hipSuccess) {                                        \
      std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(status_));                         \
      std::exit(1);                                                     \
    }                                                                   \
  } while (false)

namespace {

constexpr std::uint32_t kRows = (7U * 16U * 16U) + 5U;
constexpr std::uint32_t kWidth = 2048U;
constexpr float kEpsilon = 1.0e-5F;
constexpr int kWarmups = 100;
constexpr int kIterations = 1000;
constexpr int kSamples = 15;

__global__ void BaselineLayerNorm(const float* input, const float* weight,
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
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
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
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
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

__global__ void CandidateLayerNorm(const float* input, const float* weight,
                                   const float* bias, float* output,
                                   std::uint32_t rows, std::uint32_t width,
                                   float epsilon) {
  constexpr std::uint32_t kRowsPerBlock = 8U;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock + wave;
  const bool active = row < rows;
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local_sum = 0.0F;
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      local_sum += input[base + column];
    }
  }
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
    const float peer = __shfl_down(local_sum, stride, 32);
    if (lane < stride) {
      local_sum += peer;
    }
  }
  const float mean = __shfl(local_sum, 0, 32) / static_cast<float>(width);
  float local_square = 0.0F;
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      const float centered = input[base + column] - mean;
      local_square = fmaf(centered, centered, local_square);
    }
  }
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
    const float peer = __shfl_down(local_square, stride, 32);
    if (lane < stride) {
      local_square += peer;
    }
  }
  const float variance =
      __shfl(local_square, 0, 32) / static_cast<float>(width);
  const float inverse = rsqrtf(variance + epsilon);
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      output[base + column] =
          (input[base + column] - mean) * inverse * weight[column] +
          bias[column];
    }
  }
}

void LaunchBaseline(const float* input, const float* weight, const float* bias,
                    float* output, hipStream_t stream) {
  hipLaunchKernelGGL(BaselineLayerNorm, dim3(kRows), dim3(32), 0, stream, input,
                     weight, bias, output, kRows, kWidth, kEpsilon);
}

void LaunchCandidate(const float* input, const float* weight, const float* bias,
                     float* output, hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = 8U;
  hipLaunchKernelGGL(CandidateLayerNorm,
                     dim3((kRows + kRowsPerBlock - 1U) / kRowsPerBlock),
                     dim3(kRowsPerBlock * 32U), 0, stream, input, weight, bias,
                     output, kRows, kWidth, kEpsilon);
}

template<typename Launch>
double Measure(Launch&& launch, hipStream_t stream) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    launch();
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    launch();
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / kIterations;
}

double Median(std::vector<double> samples) {
  std::ranges::sort(samples);
  return samples[samples.size() / 2U];
}

}  // namespace

int main() {
  const std::size_t elements = static_cast<std::size_t>(kRows) * kWidth;
  std::vector<float> input(elements);
  std::vector<float> weight(kWidth);
  std::vector<float> bias(kWidth);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const int centered = static_cast<int>((index * 17U) % 1024U) - 512;
    input[index] = static_cast<float>(centered) * 0.001F;
  }
  for (std::size_t index = 0; index < kWidth; ++index) {
    weight[index] = 0.75F + static_cast<float>(index % 17U) * 0.01F;
    bias[index] =
        static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.002F;
  }

  float* device_input = nullptr;
  float* device_weight = nullptr;
  float* device_bias = nullptr;
  float* baseline_output = nullptr;
  float* candidate_output = nullptr;
  HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_weight, weight.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_bias, bias.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&baseline_output, input.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&candidate_output, input.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_weight, weight.data(),
                      weight.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_bias, bias.data(), bias.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  LaunchBaseline(device_input, device_weight, device_bias, baseline_output,
                 stream);
  LaunchCandidate(device_input, device_weight, device_bias, candidate_output,
                  stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> baseline(input.size());
  std::vector<float> candidate(input.size());
  HIP_CHECK(hipMemcpy(baseline.data(), baseline_output,
                      baseline.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(candidate.data(), candidate_output,
                      candidate.size() * sizeof(float), hipMemcpyDeviceToHost));
  if (std::memcmp(baseline.data(), candidate.data(),
                  baseline.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: LayerNorm outputs are not bit-identical\n");
    return 1;
  }

  std::vector<double> baseline_samples;
  std::vector<double> candidate_samples;
  baseline_samples.reserve(kSamples);
  candidate_samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    const auto baseline_launch = [&] {
      LaunchBaseline(device_input, device_weight, device_bias, baseline_output,
                     stream);
    };
    const auto candidate_launch = [&] {
      LaunchCandidate(device_input, device_weight, device_bias,
                      candidate_output, stream);
    };
    if ((sample & 1) == 0) {
      baseline_samples.push_back(Measure(baseline_launch, stream));
      candidate_samples.push_back(Measure(candidate_launch, stream));
    } else {
      candidate_samples.push_back(Measure(candidate_launch, stream));
      baseline_samples.push_back(Measure(baseline_launch, stream));
    }
  }

  const double baseline_ms = Median(baseline_samples);
  const double candidate_ms = Median(candidate_samples);
  std::printf(
      "H3 VisualVAE LayerNorm rows=%u width=%u baseline_ms=%.6f "
      "candidate_ms=%.6f speedup=%.3fx exact=yes\n",
      kRows, kWidth, baseline_ms, candidate_ms, baseline_ms / candidate_ms);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(candidate_output));
  HIP_CHECK(hipFree(baseline_output));
  HIP_CHECK(hipFree(device_bias));
  HIP_CHECK(hipFree(device_weight));
  HIP_CHECK(hipFree(device_input));
  return 0;
}
