#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

void TestGpuRMSNorm() {
  const std::size_t dim = 256;
  std::vector<float> h_x(dim, 1.0F);
  std::vector<float> h_w(dim, 2.0F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_x, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_out, dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  // mean(x^2) = 1.0, rms = 1.0, out = (1.0 / 1.0) * 2.0 = 2.0
  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 2.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuResidualAdd() {
  const std::size_t dim = 128;
  std::vector<float> h_a(dim, 3.5F);
  std::vector<float> h_b(dim, 1.5F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchResidualAdd(d_a, d_b, d_out, dim);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 5.0F) < 1e-5F);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuGEMV() {
  const std::size_t M = 4;
  const std::size_t K = 8;
  std::vector<float> h_A(M * K, 1.0F);  // all ones
  std::vector<float> h_x(K, 2.0F);      // all twos
  std::vector<float> h_y(M, 0.0F);

  float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A, false, d_x, d_y, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(
      hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Each row has K=8 ones * 2.0 = 16.0
  for (std::size_t m = 0; m < M; ++m) {
    assert(std::abs(h_y[m] - 16.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

int main() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping GPU kernel tests.\n";
    return 0;
  }

  TestGpuRMSNorm();
  TestGpuResidualAdd();
  TestGpuGEMV();
  std::cout << "All Qwen HIP GPU kernel tests passed on gfx1151.\n";
  return 0;
}
#else
int main() {
  std::cout << "HIP disabled, skipping GPU kernel tests.\n";
  return 0;
}
#endif
