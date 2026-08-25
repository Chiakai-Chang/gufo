// Achievable-peak calibration for gfx1151 (Strix Halo).
//
// Establishes the roofline constants every kernel experiment is scored
// against: VALU FP32 FMA rate, WMMA INT8/BF16/FP16 matrix rates, LDS read
// bandwidth, and DRAM read/write/copy bandwidth. Compute loops are register
// resident with no memory traffic, so the numbers are issue-rate limited and
// reflect the clocks the part actually sustains rather than a spec sheet.
//
//   nix develop -c hipcc -O3 --offload-arch=gfx1151 -std=c++20 \
//       tools/bench/gfx1151_peak.hip -o /tmp/gfx1151_peak
//
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using bf16x16_t = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using f16x16_t = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using f32x8_t = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kIters = 4096;

// --- WMMA INT8: 16x16x16 per instruction, 4 independent chains for ILP -------
__global__ void PeakWmmaI8(int* sink, int reps, int gate) {
  int32x4_t a = {1, 2, 3, 4};
  int32x4_t b = {5, 6, 7, 8};
  a[0] += static_cast<int>(threadIdx.x);
  int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
  int32x8_t c1 = c0;
  int32x8_t c2 = c0;
  int32x8_t c3 = c0;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, true);
      c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1, true);
      c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2, true);
      c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3, true);
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = c0[0] + c1[0] + c2[0] + c3[0];
  }
}

// --- WMMA BF16: 16x16x16, fp32 accumulate -----------------------------------
__global__ void PeakWmmaBf16(float* sink, int reps, int gate) {
  bf16x16_t a;
  bf16x16_t b;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<short>(0x3f80 + (int)threadIdx.x % 3);
    b[i] = static_cast<short>(0x3f00);
  }
  f32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x8_t c1 = c0;
  f32x8_t c2 = c0;
  f32x8_t c3 = c0;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c0);
      c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c1);
      c2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c2);
      c3 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c3);
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = c0[0] + c1[0] + c2[0] + c3[0];
  }
}

// --- VALU FP32 FMA ----------------------------------------------------------
__global__ void PeakFmaF32(float* sink, int reps, int gate) {
  float a0 = static_cast<float>(threadIdx.x) * 1.0001F;
  float a1 = a0 + 1.0F;
  float a2 = a0 + 2.0F;
  float a3 = a0 + 3.0F;
  float a4 = a0 + 4.0F;
  float a5 = a0 + 5.0F;
  float a6 = a0 + 6.0F;
  float a7 = a0 + 7.0F;
  const float m = 1.0000001F;
  const float b = 0.0000001F;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      a0 = fmaf(a0, m, b);
      a1 = fmaf(a1, m, b);
      a2 = fmaf(a2, m, b);
      a3 = fmaf(a3, m, b);
      a4 = fmaf(a4, m, b);
      a5 = fmaf(a5, m, b);
      a6 = fmaf(a6, m, b);
      a7 = fmaf(a7, m, b);
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  }
}

// --- Mixed WMMA-INT8 + FP32 epilogue, the shape the Q8 GEMM inner loop has --
// `EpiPerWmma` FP32 ops (cvt + mul + fma triples) per 2 WMMA instructions.
template<int EpiTriples>
__global__ void PeakWmmaI8WithEpilogue(float* sink, int reps, int gate) {
  int32x4_t a = {1, 2, 3, 4};
  int32x4_t b = {5, 6, 7, 8};
  a[0] += static_cast<int>(threadIdx.x);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float dw = 1.0000001F;
  const float dx = 0.9999999F;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
#pragma unroll
      for (int l = 0; l < EpiTriples; ++l) {
        acc[l] += (dw * dx) * static_cast<float>(c[l]);
      }
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = acc[0] + acc[1] + acc[2] + acc[3];
  }
}

// --- DRAM bandwidth ---------------------------------------------------------
__global__ void StreamRead(const int32x4_t* __restrict__ src, int* sink,
                           std::size_t n, int gate) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  int32x4_t acc = {0, 0, 0, 0};
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    acc += src[i];
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = acc[1];
  }
}

__global__ void StreamWrite(int32x4_t* __restrict__ dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const int32x4_t v = {1, 2, 3, 4};
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    dst[i] = v;
  }
}

__global__ void StreamCopy(const int32x4_t* __restrict__ src,
                           int32x4_t* __restrict__ dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    dst[i] = src[i];
  }
}

// --- LDS read bandwidth -----------------------------------------------------
__global__ void PeakLdsRead(int* sink, int reps, int gate) {
  __shared__ int32x4_t buf[1024];
  for (unsigned i = threadIdx.x; i < 1024; i += blockDim.x) {
    buf[i] = int32x4_t{static_cast<int>(i), 1, 2, 3};
  }
  __syncthreads();
  int32x4_t acc = {0, 0, 0, 0};
  const unsigned base = threadIdx.x & 1023u;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      acc += buf[(base + (i * 37u)) & 1023u];
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    *sink = acc[1];
  }
}

template<typename Launch>
static double BenchSeconds(Launch&& launch, int iters) {
  launch();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return static_cast<double>(ms) / 1e3 / iters;
}

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s  arch=%s  WGPs=%d (CUs=%d)  clock=%.2f GHz\n",
              prop.name, prop.gcnArchName, prop.multiProcessorCount,
              prop.multiProcessorCount * 2, prop.clockRate / 1e6);

  const int cus = prop.multiProcessorCount * 2;  // HIP reports WGPs on RDNA
  // 8 waves per CU x 4 blocks to saturate every SIMD.
  const int blocks = cus * 8;
  const int threads = 256;
  const double waves = static_cast<double>(blocks) * threads / 32.0;

  int* d_i = nullptr;
  float* d_f = nullptr;
  HIP_OK(hipMalloc(&d_i, sizeof(int)));
  HIP_OK(hipMalloc(&d_f, sizeof(float)));

  std::printf("\n-- compute peaks (register resident, no memory traffic) --\n");
  {
    const double s = BenchSeconds(
        [&] { PeakWmmaI8<<<blocks, threads>>>(d_i, kIters, -1); }, 3);
    // 4 chains x 16 unrolled x kIters instructions, each 16x16x16 MACs = 8192 ops
    const double ops = waves * 4.0 * 16.0 * kIters * 8192.0;
    std::printf("  WMMA int8  16x16x16 : %8.2f TOPS\n", ops / s / 1e12);
  }
  {
    const double s = BenchSeconds(
        [&] { PeakWmmaBf16<<<blocks, threads>>>(d_f, kIters, -1); }, 3);
    const double ops = waves * 4.0 * 16.0 * kIters * 8192.0;
    std::printf("  WMMA bf16  16x16x16 : %8.2f TFLOPS\n", ops / s / 1e12);
  }
  {
    const double s = BenchSeconds(
        [&] { PeakFmaF32<<<blocks, threads>>>(d_f, kIters, -1); }, 3);
    const double ops = waves * 32.0 * 8.0 * 32.0 * kIters * 2.0;
    std::printf("  VALU fp32 fma       : %8.2f TFLOPS\n", ops / s / 1e12);
  }
  std::printf("\n-- WMMA int8 with the Q8 dequant epilogue (per 2 WMMA) --\n");
#define EPI(N)                                                              \
  {                                                                         \
    const double s = BenchSeconds(                                          \
        [&] { PeakWmmaI8WithEpilogue<N><<<blocks, threads>>>(d_f, kIters, -1); }, \
        3);                                                                 \
    const double ops = waves * 16.0 * kIters * 2.0 * 8192.0;                \
    std::printf("  %2d fp32 triples / 2 WMMA : %8.2f TOPS  (%.0f%% of pure)\n", \
                N, ops / s / 1e12, 100.0 * (ops / s / 1e12) / pure_i8);      \
  }
  {
    const double s = BenchSeconds(
        [&] { PeakWmmaI8<<<blocks, threads>>>(d_i, kIters, -1); }, 3);
    const double pure_i8 = waves * 4.0 * 16.0 * kIters * 8192.0 / s / 1e12;
    EPI(0)
    EPI(2)
    EPI(4)
    EPI(8)
  }
#undef EPI

  std::printf("\n-- LDS --\n");
  {
    const double s =
        BenchSeconds([&] { PeakLdsRead<<<blocks, threads>>>(d_i, kIters, -1); }, 3);
    const double bytes = waves * 32.0 * 32.0 * kIters * 16.0;
    std::printf("  LDS read            : %8.2f GB/s\n", bytes / s / 1e9);
  }

  std::printf("\n-- DRAM (unified memory) --\n");
  {
    const std::size_t bytes = 2ULL << 30;  // 2 GiB
    const std::size_t n = bytes / sizeof(int32x4_t);
    int32x4_t* a = nullptr;
    int32x4_t* b = nullptr;
    HIP_OK(hipMalloc(&a, bytes));
    HIP_OK(hipMalloc(&b, bytes));
    HIP_OK(hipMemset(a, 1, bytes));
    const int g = cus * 16;
    {
      const double s =
          BenchSeconds([&] { StreamRead<<<g, 256>>>(a, d_i, n, -1); }, 5);
      std::printf("  read                : %8.2f GB/s\n", bytes / s / 1e9);
    }
    {
      const double s = BenchSeconds([&] { StreamWrite<<<g, 256>>>(b, n); }, 5);
      std::printf("  write               : %8.2f GB/s\n", bytes / s / 1e9);
    }
    {
      const double s =
          BenchSeconds([&] { StreamCopy<<<g, 256>>>(a, b, n); }, 5);
      std::printf("  copy (r+w)          : %8.2f GB/s\n",
                  2.0 * bytes / s / 1e9);
    }
    HIP_OK(hipFree(a));
    HIP_OK(hipFree(b));
  }

  HIP_OK(hipFree(d_i));
  HIP_OK(hipFree(d_f));
  return 0;
}
