// Standalone WMMA iu8 C-fragment layout probe for gfx1151 (RDNA3.5, w32).
// Computes a known 16x16x32 int8 matmul (M=16, K=32, N=16) and dumps the
// int32x8 accumulator fragment per lane so the lane -> (row,col) mapping can
// be derived by matching against a CPU reference.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t Wmma(int32x4_t a, int32x4_t b,
                                          int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

// grid (1,1) block (32,1): one wave. M=16,K=32,N=16.
__global__ void WmmaProbeKernel(
    const int8_t* __restrict__ A,  // 16 x 32, row-major
    const int8_t* __restrict__ B,  // 16 x 32, row-major (N x K, token-major)
    int32_t* __restrict__ frag_out) {  // 32 lanes x 8
  const int lane = threadIdx.x;
  const int row = lane % 16;  // A row
  const int tok = lane % 16;  // B token (N)
  // A: 16x32 = 2 int32x4 per row (K0..15, K16..31)
  int32x4_t a0 = *reinterpret_cast<const int32x4_t*>(&A[row * 32 + 0]);
  int32x4_t a1 = *reinterpret_cast<const int32x4_t*>(&A[row * 32 + 16]);
  int32x4_t b0 = *reinterpret_cast<const int32x4_t*>(&B[tok * 32 + 0]);
  int32x4_t b1 = *reinterpret_cast<const int32x4_t*>(&B[tok * 32 + 16]);
  int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = Wmma(a0, b0, c);
  c = Wmma(a1, b1, c);
  // dump fragment
  for (int l = 0; l < 8; ++l)
    frag_out[lane * 8 + l] = c[l];
}

int main() {
  constexpr int M = 16, K = 32, N = 16;
  std::vector<int8_t> A(M * K), B(N * K);
  // LCG with fixed seed so int8 values are spread and dot products are
  // (with very high probability) unique across the 16x16 result grid.
  uint32_t s = 0x1234567u;
  auto nxt = [&]() {
    s = s * 1103515245u + 12345u;
    return (int)((s >> 16) & 0xff) - 128;  // -128..127
  };
  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k)
      A[r * K + k] = (int8_t)nxt();
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      B[n * K + k] = (int8_t)nxt();

  // CPU reference C[r][n] = sum_k A[r][k]*B[n][k]
  std::vector<int32_t> Cref(M * N, 0);
  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k)
        s += A[r * K + k] * B[n * K + k];
      Cref[r * N + n] = s;
    }

  int8_t *dA, *dB;
  int32_t* dFrag;
  hipMalloc(&dA, A.size());
  hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice);
  hipMalloc(&dB, B.size());
  hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice);
  hipMalloc(&dFrag, 32 * 8 * sizeof(int32_t));
  WmmaProbeKernel<<<1, 32>>>(dA, dB, dFrag);
  hipDeviceSynchronize();
  std::vector<int32_t> frag(32 * 8);
  hipMemcpy(frag.data(), dFrag, frag.size() * sizeof(int32_t),
            hipMemcpyDeviceToHost);

  std::printf("CPU reference C[r][n] (row r, col n):\n");
  for (int r = 0; r < M; ++r) {
    for (int n = 0; n < N; ++n)
      std::printf("%6d ", Cref[r * N + n]);
    std::printf("\n");
  }
  std::printf("\nPer-lane fragment (lane -> 8 int32):\n");
  for (int lane = 0; lane < 32; ++lane) {
    std::printf("lane %2d: ", lane);
    for (int l = 0; l < 8; ++l)
      std::printf("%6d ", frag[lane * 8 + l]);
    std::printf("\n");
  }

  // Inverse map: for each (r,n) find (lane,l) holding Cref[r][n].
  // Requires unique Cref values; verify first.
  bool unique = true;
  for (int i = 0; i < M * N && unique; ++i)
    for (int j = i + 1; j < M * N; ++j)
      if (Cref[i] == Cref[j]) {
        unique = false;
        break;
      }
  std::printf("\nCref unique=%d\n", unique);
  if (!unique) {
    std::printf("regenerate seed\n");
    return 0;
  }
  std::printf("\nInverse map (r,n) -> (lane,l):\n");
  for (int r = 0; r < M; ++r) {
    for (int n = 0; n < N; ++n) {
      int32_t v = Cref[r * N + n];
      int fl = -1, ll = -1;
      for (int lane = 0; lane < 32 && fl < 0; ++lane)
        for (int l = 0; l < 8; ++l)
          if (frag[lane * 8 + l] == v) {
            fl = lane;
            ll = l;
            break;
          }
      std::printf("(%2d,%2d)->L%02d:%d  ", r, n, fl, ll);
    }
    std::printf("\n");
  }
  // Also print forward: lane -> list of (r,n)
  std::printf("\nForward map lane -> (r,n) by l=0..7:\n");
  for (int lane = 0; lane < 32; ++lane) {
    std::printf("L%02d: ", lane);
    for (int l = 0; l < 8; ++l) {
      int32_t v = frag[lane * 8 + l];
      int rr = -1, nn = -1;
      for (int r = 0; r < M && rr < 0; ++r)
        for (int n = 0; n < N; ++n)
          if (Cref[r * N + n] == v) {
            rr = r;
            nn = n;
            break;
          }
      std::printf("(%2d,%2d) ", rr, nn);
    }
    std::printf("\n");
  }
  hipFree(dA);
  hipFree(dB);
  hipFree(dFrag);
  return 0;
}
#else
#include <cstdio>
int main() {
  std::printf("HIP not enabled\n");
  return 0;
}
#endif
