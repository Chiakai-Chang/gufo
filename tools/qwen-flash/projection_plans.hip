// Offline gfx1151 F16 alpha/beta and router projection sweep.
// Build with tools/bench/build.sh inside nix develop (pass this file's path).
// Run /tmp/projection_plans > /tmp/projection-plans.csv with no other GPU work.
// Each workspace-free candidate must match sampled F64 dots within 0.002
// and repeat exactly before its timing can qualify a production plan.
// hipBLASLt algorithm indices belong to the ROCm version pinned by Nix.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
void Check(hipError_t x) {
  if (x != hipSuccess) {
    fprintf(stderr, "%s\n", hipGetErrorString(x));
    exit(1);
  }
}
void Check(hipblasStatus_t x) {
  if (x != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "BLAS %d\n", int(x));
    exit(1);
  }
}
int main() {
  setbuf(stdout, nullptr);
  hipStream_t stream;
  Check(hipStreamCreate(&stream));
  hipblasLtHandle_t handle;
  Check(hipblasLtCreate(&handle));
  puts(
      "m,n,k,rank,algo,us,max_reference_error,max_first_difference,repeat_"
      "exact");
  for (int m : {96, 513})
    for (int n : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
      int k = 2560;
      std::vector<__half> a(size_t(m) * k), b(size_t(n) * k);
      unsigned rng = 123;
      for (auto* values : {&a, &b})
        for (auto& v : *values) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          v = __float2half_rn(float(int(rng % 8193) - 4096) / 4096.f);
        }
      __half *da, *db;
      float* dc;
      Check(hipMalloc(&da, a.size() * 2));
      Check(hipMalloc(&db, b.size() * 2));
      Check(hipMalloc(&dc, size_t(m) * n * 4));
      Check(hipMemcpy(da, a.data(), a.size() * 2, hipMemcpyHostToDevice));
      Check(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
      hipblasLtMatmulDesc_t desc;
      hipblasLtMatrixLayout_t la, lb, lc;
      Check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
      hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
      Check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                            &ta, sizeof(ta)));
      Check(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                            &tb, sizeof(tb)));
      Check(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k));
      Check(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, n, k));
      Check(hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, m, n, m));
      hipblasLtMatmulPreference_t pref;
      Check(hipblasLtMatmulPreferenceCreate(&pref));
      size_t ws = 32ULL << 20;
      Check(hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      std::array<hipblasLtMatmulHeuristicResult_t, 16> choices{};
      int found = 0;
      Check(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref,
                                            choices.size(), choices.data(),
                                            &found));
      float one = 1, zero = 0;
      std::vector<float> first, actual(size_t(m) * n), repeat(actual.size());
      hipEvent_t start, end;
      Check(hipEventCreate(&start));
      Check(hipEventCreate(&end));
      for (int i = 0; i < found; ++i) {
        auto& choice = choices[i];
        if (choice.state != HIPBLAS_STATUS_SUCCESS || choice.workspaceSize)
          continue;
        auto run = [&] {
          return hipblasLtMatmul(handle, desc, &one, da, la, db, lb, &zero, dc,
                                 lc, dc, lc, &choice.algo, nullptr, 0, stream);
        };
        Check(run());
        Check(hipStreamSynchronize(stream));
        Check(hipMemcpy(actual.data(), dc, actual.size() * 4,
                        hipMemcpyDeviceToHost));
        Check(hipEventRecord(start, stream));
        for (int r = 0; r < 3; ++r)
          Check(run());
        Check(hipEventRecord(end, stream));
        Check(hipEventSynchronize(end));
        float ms;
        Check(hipEventElapsedTime(&ms, start, end));
        Check(hipMemcpy(repeat.data(), dc, repeat.size() * 4,
                        hipMemcpyDeviceToHost));
        if (first.empty())
          first = actual;
        float diff = 0;
        for (size_t x = 0; x < actual.size(); ++x)
          diff = std::max(diff, std::abs(actual[x] - first[x]));
        double error = 0;
        for (int s = 0; s < 32; ++s) {
          int row = (s * 79) % n, col = (s * 127) % m;
          double expected = 0;
          for (int j = 0; j < k; ++j)
            expected += double(__half2float(a[size_t(col) * k + j])) *
                        double(__half2float(b[size_t(row) * k + j]));
          error = std::max(
              error,
              std::abs(expected - double(actual[size_t(row) * m + col])));
        }
        printf("%d,%d,%d,%d,%d,%.3f,%.8g,%.8g,%d\n", m, n, k, i,
               hipblaslt_ext::getIndexFromAlgo(choice.algo), ms * 1000 / 3,
               error, diff,
               std::memcmp(actual.data(), repeat.data(),
                           actual.size() * sizeof(float)) == 0);
        if (error > 0.002 ||
            std::memcmp(actual.data(), repeat.data(),
                        actual.size() * sizeof(float)) != 0 ||
            !std::all_of(actual.begin(), actual.end(),
                         [](float x) { return std::isfinite(x); }))
          return 3;
      }
      Check(hipEventDestroy(start));
      Check(hipEventDestroy(end));
      Check(hipblasLtMatmulPreferenceDestroy(pref));
      Check(hipblasLtMatrixLayoutDestroy(lc));
      Check(hipblasLtMatrixLayoutDestroy(lb));
      Check(hipblasLtMatrixLayoutDestroy(la));
      Check(hipblasLtMatmulDescDestroy(desc));
      Check(hipFree(dc));
      Check(hipFree(db));
      Check(hipFree(da));
    }
  Check(hipblasLtDestroy(handle));
  Check(hipStreamDestroy(stream));
}
