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
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
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

#define BLAS_OK(expr)                                                      \
  do {                                                                     \
    hipblasStatus_t st_ = (expr);                                          \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "%s:%d hipblas error %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                 \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Hand-written blocked BF16 WMMA GEMM (opt-c180-bf16-blocked).
//
// Same tiling as the blocked W8A8 kernel in prefill_quant_gemm.hip -- BM weight
// rows x BN tokens per block, BK 16-element K steps per LDS stage, WM x WN waves
// each owning (BM/WM) x (BN/WN) -- but with one structural advantage: BF16 has no
// per-block scale, so the WMMA result accumulates straight into the output
// fragment. There is no epilogue at all. That matters because the W8A8 kernel's
// 3-VALU-per-output-element epilogue is exactly what caps it at 60% of peak.
//
//   y[batch][m] = x[batch][k] * w[m][k]^T,  w and x row major, y FP32.
//
// Operands are staged in WMMA fragment order, [k step][tile][16 rows][16 k], so
// a fragment read is 16 contiguous bf16 = one 32-byte load per lane.
// ---------------------------------------------------------------------------

using v16bf = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaBf16(v16bf a, v16bf b, v8f c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
  return c;
#endif
}

__device__ __forceinline__ v16bf LoadBfFrag(const short* p) {
  union {
    v16bf f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// KS is the number of bf16 per staged row: 16 payload plus PAD of bank padding.
template<int BM, int BN, int BK, int WM, int WN, int PAD>
__launch_bounds__(WM* WN * 32) __global__ void BlockedBf16Wmma(
    const short* __restrict__ w, const short* __restrict__ x,
    float* __restrict__ y, int batch, int m, int k) {
  constexpr int kThreads = WM * WN * 32;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16 + PAD;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert((BM * BK) % kThreads == 0 && (BN * BK) % kThreads == 0);
  constexpr int PFA = (BM * BK) / kThreads;
  constexpr int PFB = (BN * BK) / kThreads;

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_b[BK][TS][16][KS];
  static_assert(sizeof(s_a) >= static_cast<std::size_t>(WM) * WN * 256 * 4,
                "the weight stage doubles as the store transpose scratch");

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;
  const int ksteps = k / 16;

  v8f acc[WRS][WTS] = {};

  // Hoisted 64-bit row bases, clamped so an out-of-range row reads row m-1 and
  // is masked out at the store instead of branching in the K loop.
  const short* w_row[PFA];
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int r = r_block + (((p * kThreads) + tid) / BK);
    w_row[p] = w + (static_cast<std::size_t>((r < m) ? r : (m - 1)) * k);
  }
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] = x + (static_cast<std::size_t>((t < batch) ? t : (batch - 1)) * k);
  }

  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    __syncthreads();
    // Stage weights: each thread owns one (row, k step) 32-byte chunk.
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = (p * kThreads) + tid;
      const int row = idx / BK;
      const int ks = idx % BK;
      const int kk = ks0 + ks;
      const short* src = w_row[p] + (static_cast<std::size_t>(kk) * 16);
      short* dst = &s_a[ks][row / 16][row % 16][0];
      if (kk < ksteps) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(dst + 8) =
            *reinterpret_cast<const uint4*>(src + 8);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = (p * kThreads) + tid;
      const int tok = idx / BK;
      const int ks = idx % BK;
      const int kk = ks0 + ks;
      const short* src = x_row[p] + (static_cast<std::size_t>(kk) * 16);
      short* dst = &s_b[ks][tok / 16][tok % 16][0];
      if (kk < ksteps) {
        *reinterpret_cast<uint4*>(dst) =
            *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(dst + 8) =
            *reinterpret_cast<const uint4*>(src + 8);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16bf a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadBfFrag(&s_a[ks][(wr * WRS) + i][sl][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16bf b = LoadBfFrag(&s_b[ks][(wt * WTS) + j][sl][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          acc[i][j] = WmmaBf16(a[i], b, acc[i][j]);
        }
      }
    }
  }

  // C element l of a wave32 tile is row 2l + lane/16, column lane%16. The output
  // is [token][row], so transpose through the (now dead) weight stage.
  __syncthreads();
  const int hi = lane >> 4;
  float* scratch = reinterpret_cast<float*>(&s_a[0][0][0][0]) + (wave * 256);
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[(sl * 16) + (2 * l) + hi] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int r0 = r_block + (((wr * WRS) + i) * 16);
      const int t0 = t_block + (((wt * WTS) + j) * 16);
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t base =
            (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          y[base + (static_cast<std::size_t>(flat >> 4) * m) + (flat & 15)] =
              scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          const int tok = t0 + (flat >> 4);
          const int r = r0 + (flat & 15);
          if (tok < batch && r < m) {
            y[(static_cast<std::size_t>(tok) * m) + r] = scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

template<int BMv, int BNv, int BKv, int WMv, int WNv, int PADv>
struct Cfg {
  static constexpr int bm = BMv;
  static constexpr int bn = BNv;
  static constexpr int bk = BKv;
  static constexpr int wm = WMv;
  static constexpr int wn = WNv;
  static constexpr int pad = PADv;
};

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
      // The three tensors the Q8_K_XL artifact keeps at BF16, so these are the
      // only shapes where the library GEMM is actually on the production path.
      {"attn_q", 12288, 5120},
      {"attn_k/v", 1024, 5120},
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
    float* d_y_ref = nullptr;
    HIP_OK(hipMalloc(&d_y_ref, yn * sizeof(float)));
    {
      // Varied operands: a constant bit pattern would let a wrong kernel agree
      // with the reference by accident.
      std::vector<hip_bfloat16> h(wn > xn ? wn : xn);
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] = static_cast<hip_bfloat16>(
            0.7F * std::sin(0.0131F * static_cast<float>(i % 811)));
      }
      HIP_OK(hipMemcpy(d_w, h.data(), wn * sizeof(hip_bfloat16),
                       hipMemcpyHostToDevice));
      for (std::size_t i = 0; i < h.size(); ++i) {
        h[i] = static_cast<hip_bfloat16>(
            0.9F * std::cos(0.0177F * static_cast<float>(i % 643)));
      }
      HIP_OK(hipMemcpy(d_x, h.data(), xn * sizeof(hip_bfloat16),
                       hipMemcpyHostToDevice));
    }

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
                                lc, d_y, lc, &heur[i].algo, lt_ws, lt_ws_bytes,
                                nullptr);
              },
              iters);
          if (ms < best) {
            best = ms;
            best_i = i;
          }
        }
        std::printf(
            "%-13s %-14s %9.3f %9.2f %6.0f%%   (best of %d algos, #%d)\n",
            c.name, "hipblaslt", best, flops / (best * 1e-3) / 1e12,
            100.0 * (flops / (best * 1e-3) / 1e12) / kPeak, found, best_i);
      } else {
        std::printf("%-13s %-14s      no heuristic\n", c.name, "hipblaslt");
      }
      hipblasLtMatmulPreferenceDestroy(pref);
      hipblasLtMatrixLayoutDestroy(la);
      hipblasLtMatrixLayoutDestroy(lb);
      hipblasLtMatrixLayoutDestroy(lc);
      hipblasLtMatmulDescDestroy(op);
    }

    // Hand-written blocked BF16 WMMA, checked against whatever the reference
    // engine left in d_y before timing anything.
    HIP_OK(hipMemcpy(d_y_ref, d_y, yn * sizeof(float), hipMemcpyDeviceToDevice));
    std::vector<float> ref(yn);
    std::vector<float> got(yn);
    HIP_OK(hipMemcpy(ref.data(), d_y_ref, yn * sizeof(float),
                     hipMemcpyDeviceToHost));

    const auto run_blocked = [&](auto tag) {
      constexpr int BM = decltype(tag)::bm;
      constexpr int BN = decltype(tag)::bn;
      constexpr int BK = decltype(tag)::bk;
      constexpr int WM = decltype(tag)::wm;
      constexpr int WN = decltype(tag)::wn;
      constexpr int PAD = decltype(tag)::pad;
      if (c.m % BM != 0) {
        return;
      }
      const dim3 grid((batch + BN - 1) / BN, (c.m + BM - 1) / BM);
      const auto launch = [&] {
        BlockedBf16Wmma<BM, BN, BK, WM, WN, PAD>
            <<<grid, WM * WN * 32>>>(reinterpret_cast<const short*>(d_w),
                                     reinterpret_cast<const short*>(d_x), d_y,
                                     batch, static_cast<int>(c.m),
                                     static_cast<int>(c.k));
      };
      HIP_OK(hipMemset(d_y, 0, yn * sizeof(float)));
      launch();
      HIP_OK(hipGetLastError());
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipMemcpy(got.data(), d_y, yn * sizeof(float),
                       hipMemcpyDeviceToHost));
      double worst = 0.0;
      double mag = 0.0;
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < yn; ++i) {
        if (!std::isfinite(got[i])) {
          ++nonfinite;
          continue;
        }
        worst = std::fmax(worst, std::fabs(static_cast<double>(got[i]) -
                                          static_cast<double>(ref[i])));
        mag = std::fmax(mag, std::fabs(static_cast<double>(ref[i])));
      }
      const double rel = (nonfinite != 0)
                             ? 9.99
                             : ((mag > 0.0) ? (worst / mag) : worst);
      const float ms = BenchMs(launch, iters);
      char nm[48];
      std::snprintf(nm, sizeof(nm), "blk %dx%dx%d p%d", BM, BN, BK, PAD);
      std::printf("%-13s %-14s %9.3f %9.2f %6.0f%%   rel=%.1e\n", c.name, nm,
                  ms, flops / (ms * 1e-3) / 1e12,
                  100.0 * (flops / (ms * 1e-3) / 1e12) / kPeak, rel);
    };
    // Both BM*BK and BN*BK must be at least the thread count for one whole
    // 32-byte staging chunk per thread, which rules out BK=2 at 512 threads.
    run_blocked(Cfg<128, 128, 2, 4, 2, 0>{});
    run_blocked(Cfg<128, 128, 2, 4, 2, 8>{});
    run_blocked(Cfg<128, 128, 4, 4, 2, 0>{});
    run_blocked(Cfg<128, 128, 4, 4, 2, 8>{});
    run_blocked(Cfg<256, 128, 4, 8, 2, 0>{});
    run_blocked(Cfg<128, 256, 4, 4, 4, 0>{});
    // A 256x256 macro tile halves the re-read traffic of 128x128, and at BK=2 it
    // still fits two blocks per CU, so unlike 256x128 and 128x256 it does not pay
    // for the traffic with occupancy.
    run_blocked(Cfg<256, 256, 2, 4, 4, 0>{});

    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
    HIP_OK(hipFree(d_y_ref));
  }

  HIP_OK(hipFree(lt_ws));
  if (lt != nullptr) {
    hipblasLtDestroy(lt);
  }
  BLAS_OK(hipblasDestroy(blas));
  return 0;
}
