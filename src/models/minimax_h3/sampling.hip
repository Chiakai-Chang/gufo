#if defined(ENGINE_ENABLE_HIP)

// Euler arithmetic adapted from antirez/h3.c h3_shaders.metal at
// 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT). This is an independent
// gfx1151 ROCm/HIP kernel that keeps the sampler state in device F32.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "src/models/minimax_h3/sampling.hpp"

namespace strix::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool ValidateTarget(std::string* error) {
  int device = 0;
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    SetError(error, std::string("hipGetDevice: ") + hipGetErrorString(status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) {
    SetError(error, std::string("hipGetDeviceProperties: ") +
                        hipGetErrorString(status));
    return false;
  }
  if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 Euler sampling supports only integrated gfx1151 "
             "Strix Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

__global__ void EulerUpdateKernel(float* sample, std::size_t sample_offset,
                                  const hip_bfloat16* last,
                                  const hip_bfloat16* previous,
                                  std::size_t elements, float delta,
                                  float extrapolation) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float last_value = static_cast<float>(last[index]);
  const float velocity =
      fmaf(extrapolation, last_value - static_cast<float>(previous[index]),
           last_value);
  const std::size_t sample_index = sample_offset + index;
  sample[sample_index] = fmaf(delta, velocity, sample[sample_index]);
}

}  // namespace

bool HipEulerUpdate(void* sample_f32, std::size_t sample_elements,
                    std::size_t sample_offset, const void* last_bf16,
                    const void* previous_bf16, std::size_t velocity_elements,
                    float delta, float extrapolation, void* stream,
                    std::string* error) {
  if (sample_f32 == nullptr || last_bf16 == nullptr ||
      previous_bf16 == nullptr || velocity_elements == 0 ||
      sample_offset > sample_elements ||
      velocity_elements > sample_elements - sample_offset ||
      velocity_elements >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) *
              256ULL ||
      !std::isfinite(delta) || delta <= 0.0F || !std::isfinite(extrapolation) ||
      extrapolation < -2.0F || extrapolation > 2.0F) {
    SetError(error, "invalid MiniMax H3 HIP Euler update");
    return false;
  }
  if (!ValidateTarget(error)) {
    return false;
  }
  constexpr unsigned int kThreads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((velocity_elements + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(EulerUpdateKernel, dim3(blocks), dim3(kThreads), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<float*>(sample_f32), sample_offset,
                     static_cast<const hip_bfloat16*>(last_bf16),
                     static_cast<const hip_bfloat16*>(previous_bf16),
                     velocity_elements, delta, extrapolation);
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    SetError(error, std::string("MiniMax H3 Euler kernel launch: ") +
                        hipGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace strix::minimax_h3

#endif  // ENGINE_ENABLE_HIP
