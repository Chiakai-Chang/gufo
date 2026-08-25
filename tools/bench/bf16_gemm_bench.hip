// Reference-library BF16 GEMM throughput at the Qwen3.8-27B prefill shapes.
//
// Answers whether "dequantize Q8_0 to BF16 then call a tuned library GEMM"
// beats a native W8A8 WMMA kernel on gfx1151. Both hipBLAS (rocBLAS/Tensile)
// and hipBLASLt are measured because their kernel selection differs.
//
//   nix develop -c hipcc -O3 --offload-arch=gfx1151 -std=c++20 \
//       tools/bench/bf16_gemm_bench.hip -lhipblas -lhipblaslt \
//       -o /tmp/bf16_gemm_bench
//
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bf16.h>
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

#define BLAS_OK(expr)                                                       \
  do {                                                                      \
    hipblasStatus_t st_ = (expr);                                           \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "%s:%d hipblas error %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(st_));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

template<typename Launch>
static float BenchMs(Launch&& launch, int iters) {
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
  return ms / static_cast<float>(iters);
}

struct Case {
  const char* name;
  int m;
  int k;
};

int main(int argc, char** argv) {
  int batch = 2048;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-b" && i + 1 < argc) {
      batch = std::atoi(argv[++i]);
    } else if (std::string(argv[i]) == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    }
  }

  const Case cases[] = {
      {"ffn_gate/up", 17408, 5120},
      {"ffn_down", 5120, 17408},
      {"ssm_qkv", 6144, 5120},
      {"ssm_out", 5120, 2048},
  };

  hipblasHandle_t blas = nullptr;
  BLAS_OK(hipblasCreate(&blas));
  hipblasLtHandle_t lt = nullptr;
  hipblasLtCreate(&lt);
  void* lt_ws = nullptr;
  const std::size_t lt_ws_bytes = 64ULL << 20;
  HIP_OK(hipMalloc(&lt_ws, lt_ws_bytes));

  constexpr double kPeak = 55.05;  // WMMA bf16 TFLOPS, measured
  std::printf("batch=%d iters=%d  (WMMA bf16 peak %.2f TFLOPS)\n", batch, iters,
              kPeak);
  std::printf("%-13s %-14s %9s %9s %7s\n", "case", "engine", "ms", "TFLOPS",
              "%peak");

  for (const auto& c : cases) {
    const std::size_t wn = static_cast<std::size_t>(c.m) * c.k;
    const std::size_t xn = static_cast<std::size_t>(batch) * c.k;
    const std::size_t yn = static_cast<std::size_t>(batch) * c.m;
    hip_bfloat16* d_w = nullptr;
    hip_bfloat16* d_x = nullptr;
    float* d_y = nullptr;
    HIP_OK(hipMalloc(&d_w, wn * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_x, xn * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_y, yn * sizeof(float)));
    HIP_OK(hipMemset(d_w, 0x3c, wn * sizeof(hip_bfloat16)));
    HIP_OK(hipMemset(d_x, 0x3c, xn * sizeof(hip_bfloat16)));

    const double flops = 2.0 * c.m * static_cast<double>(c.k) * batch;
    const float alpha = 1.0F;
    const float beta = 0.0F;

    // y[batch][m] = x[batch][k] * w[m][k]^T, all row major.
    // Column-major view: C(m, batch) = A(m,k) op_N? -- use w as (k x m) with
    // transpose so that C = W^T-free: C = op(A)*op(B) with A = w (k-major).
    {
      const float ms = BenchMs(
          [&] {
            BLAS_OK(hipblasGemmEx(
                blas, HIPBLAS_OP_T, HIPBLAS_OP_N, c.m, batch, c.k, &alpha, d_w,
                HIPBLAS_R_16B, c.k, d_x, HIPBLAS_R_16B, c.k, &beta, d_y,
                HIPBLAS_R_32F, c.m, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
          },
          iters);
      std::printf("%-13s %-14s %9.3f %9.2f %6.0f%%\n", c.name, "hipblas", ms,
                  flops / (ms * 1e-3) / 1e12,
                  100.0 * (flops / (ms * 1e-3) / 1e12) / kPeak);
    }

    if (lt != nullptr) {
      hipblasLtMatmulDesc_t op = nullptr;
      hipblasLtMatrixLayout_t la = nullptr;
      hipblasLtMatrixLayout_t lb = nullptr;
      hipblasLtMatrixLayout_t lc = nullptr;
      hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t ta = HIPBLAS_OP_T;
      hipblasOperation_t tb = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta,
                                      sizeof(ta));
      hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb,
                                      sizeof(tb));
      hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, c.k, c.m, c.k);
      hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, c.k, batch, c.k);
      hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, c.m, batch, c.m);

      hipblasLtMatmulPreference_t pref = nullptr;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t heur[8]{};
      int found = 0;
      hipblasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, heur,
                                      &found);
      if (found > 0) {
        double best = 1e30;
        int best_i = 0;
        for (int i = 0; i < found; ++i) {
          const float ms = BenchMs(
              [&] {
                hipblasLtMatmul(lt, op, &alpha, d_w, la, d_x, lb, &beta, d_y,
                                lc, d_y, lc, &heur[i].algo, lt_ws,
                                lt_ws_bytes, nullptr);
              },
              iters);
          if (ms < best) {
            best = ms;
            best_i = i;
          }
        }
        std::printf("%-13s %-14s %9.3f %9.2f %6.0f%%   (best of %d algos, #%d)\n",
                    c.name, "hipblaslt", best, flops / (best * 1e-3) / 1e12,
                    100.0 * (flops / (best * 1e-3) / 1e12) / kPeak, found,
                    best_i);
      } else {
        std::printf("%-13s %-14s      no heuristic\n", c.name, "hipblaslt");
      }
      hipblasLtMatmulPreferenceDestroy(pref);
      hipblasLtMatrixLayoutDestroy(la);
      hipblasLtMatrixLayoutDestroy(lb);
      hipblasLtMatrixLayoutDestroy(lc);
      hipblasLtMatmulDescDestroy(op);
    }

    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
  }

  HIP_OK(hipFree(lt_ws));
  if (lt != nullptr) {
    hipblasLtDestroy(lt);
  }
  BLAS_OK(hipblasDestroy(blas));
  return 0;
}
