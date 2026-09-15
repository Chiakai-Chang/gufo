// Prototype grouped GEMM for the Qwen3.8-Flash-Next expert projections:
// int8 WMMA (gfx1151, wave32) over Q4_K weights and Q8 activations, with the
// per-32 scales applied in a float epilogue per k-step. Standalone benchmark
// against a CPU reference; see ../README.md for the measurements and the
// comparison with the vendored MMQ tier this was meant to replace.
//
//   hipcc -O3 --offload-arch=gfx1151 -DBNW=128 moe_gemm_i8_bench.hip
//   ./a.out E N K rows_per_expert [check]
// Prototype: grouped Q4_K (weights) x Q8 (activations) GEMM with int8 WMMA on gfx1151.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

struct __attribute__((packed)) block_q4_K { __half d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(block_q4_K)==144, "");

__device__ __forceinline__ void q4k_scale_min(const uint8_t* sc, unsigned g, int& s, int& m) {
  if (g < 4) { s = sc[g] & 0x3F; m = sc[g+4] & 0x3F; }
  else { s = (sc[g+4] & 0x0F) | ((sc[g-4] >> 6) << 4); m = (sc[g+4] >> 4) | ((sc[g] >> 6) << 4); }
}

struct Tile { int expert; int slot0; int rows; };

// Activations: q[slot][K] int8, d[slot][K/32] f32, sq[slot][K/32] f32 (sum of q in group).
// Activations: q[slot][K] int8, d[slot][K/32] f32, sq[slot][K/32] f32 (sum of q in group).
// Weights: the 144-byte Q4_K blocks of a 256-wide k chunk are staged in LDS
// once (row pitch 148 B: 37 words, coprime with the 32 banks) and every
// 16-wide B fragment is unpacked straight from those bytes into registers.
// Activations: q[slot][K] int8, d[slot][K/32] f32, sq[slot][K/32] f32 (sum of q in group).
// Weights are laid out chunk-major, [expert][k chunk of 256][n] blocks, so a
// lane loads its own column's 144-byte block straight into registers (the two
// half-waves fetch the same rows and share the cache lines). Activations and
// their row factors are staged in LDS per chunk; both are prefetched one chunk
// ahead so the eight k-steps of a chunk run without global loads or barriers.
template <int BM, int BN>
__global__ __launch_bounds__(BN * 2) void moe_gemm_i8(const Tile* tiles, const block_q4_K* W, const int8_t* Xq, const float* Xd, const float* Xs,
                                                      float* out, int N, int K) {
  constexpr int THREADS = BN * 2;
  constexpr int APITCH = 256 + 16;
  constexpr int MF = BM / 16;
  constexpr int A_U4 = BM * 16;
  constexpr int A_PER = (A_U4 + THREADS - 1) / THREADS;
  constexpr int F_PER = (BM * 8 + THREADS - 1) / THREADS;
  constexpr int BPITCH = 148;  // 37 words: the 16 rows of a half-wave hit distinct banks
  constexpr int W_U4 = BN * 9;
  constexpr int W_PER = (W_U4 + THREADS - 1) / THREADS;
  __shared__ __align__(16) int8_t As[BM * APITCH];
  __shared__ __align__(16) uint8_t Bb[BN * BPITCH];
  __shared__ __align__(16) float dxp[8][MF][2][8];
  __shared__ __align__(16) float rfp[8][MF][2][8];
  const Tile t = tiles[blockIdx.x];
  const int n0 = blockIdx.y * BN;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int KB = K / 32, NB = K / 256;
  const int col = wave * 16 + (lane & 15);
  const uint8_t* wchunk0 = reinterpret_cast<const uint8_t*>(W + (size_t)t.expert * NB * N + n0);
  float acc[MF][8];
  for (int i = 0; i < MF; ++i) for (int e = 0; e < 8; ++e) acc[i][e] = 0.0f;
  uint4 pa[A_PER];
  float pd[F_PER], pr[F_PER];
  uint4 pw[W_PER];  // next chunk's weight blocks, coalesced slices
  uint4 cw[9];      // current chunk's block of this lane's column
  auto prefetch = [&](int c) {
#pragma unroll
    for (int u = 0; u < A_PER; ++u) {
      const int idx = tid + u * THREADS;
      uint4 v = make_uint4(0,0,0,0);
      if (idx < A_U4) {
        const int row = idx >> 4, part = idx & 15;
        if (row < t.rows) v = *reinterpret_cast<const uint4*>(Xq + (size_t)(t.slot0 + row) * K + c * 256 + part * 16);
      }
      pa[u] = v;
    }
#pragma unroll
    for (int u = 0; u < F_PER; ++u) {
      const int idx = tid + u * THREADS;
      float d = 0.0f, sq = 0.0f;
      if (idx < BM * 8) {
        const int row = idx >> 3, grp = idx & 7;
        if (row < t.rows) { d = Xd[(size_t)(t.slot0 + row) * KB + c * 8 + grp]; sq = Xs[(size_t)(t.slot0 + row) * KB + c * 8 + grp]; }
      }
      pd[u] = d; pr[u] = d * sq;
    }
    // The chunk's BN blocks are one contiguous run of BN*144 bytes.
    const uint8_t* run = wchunk0 + (size_t)c * N * 144;
#pragma unroll
    for (int u = 0; u < W_PER; ++u) {
      const int idx = tid + u * THREADS;
      uint4 v = make_uint4(0,0,0,0);
      if (idx < W_U4) v = *reinterpret_cast<const uint4*>(run + (size_t)idx * 16);
      pw[u] = v;
    }
  };
  auto commit = [&]() {
#pragma unroll
    for (int u = 0; u < A_PER; ++u) {
      const int idx = tid + u * THREADS;
      if (idx < A_U4) { const int row = idx >> 4, part = idx & 15; *reinterpret_cast<uint4*>(&As[row * APITCH + part * 16]) = pa[u]; }
    }
#pragma unroll
    for (int u = 0; u < F_PER; ++u) {
      const int idx = tid + u * THREADS;
      if (idx < BM * 8) {
        const int row = idx >> 3, grp = idx & 7;
        dxp[grp][row >> 4][row & 1][(row & 15) >> 1] = pd[u];
        rfp[grp][row >> 4][row & 1][(row & 15) >> 1] = pr[u];
      }
    }
#pragma unroll
    for (int u = 0; u < W_PER; ++u) {
      const int idx = tid + u * THREADS;
      if (idx < W_U4) {
        const int row = idx / 9, part = idx % 9;
        uint32_t* dst = reinterpret_cast<uint32_t*>(&Bb[row * BPITCH + part * 16]);
        dst[0] = pw[u].x; dst[1] = pw[u].y; dst[2] = pw[u].z; dst[3] = pw[u].w;
      }
    }
  };
  prefetch(0);
#pragma unroll 1
  for (int c = 0; c < NB; ++c) {
    __syncthreads();
    commit();
    __syncthreads();
    if (c + 1 < NB) prefetch(c + 1);
    {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(&Bb[col * BPITCH]);
#pragma unroll
      for (int p = 0; p < 9; ++p) cw[p] = make_uint4(src[4*p], src[4*p+1], src[4*p+2], src[4*p+3]);
    }
    // Block words in registers: word 0 = d | dmin, 1..3 = scales, 4..35 = qs.
    auto word = [&](int k) -> uint32_t {
      const uint4& v = cw[k >> 2];
      switch (k & 3) { case 0: return v.x; case 1: return v.y; case 2: return v.z; default: return v.w; }
    };
    const float bd = __half2float(__ushort_as_half((unsigned short)(word(0) & 0xFFFFu)));
    const float bdmin = __half2float(__ushort_as_half((unsigned short)(word(0) >> 16)));
    auto sbyte = [&](int b) -> uint32_t { return (word(1 + b / 4) >> (8 * (b % 4))) & 0xFFu; };
#pragma unroll
    for (int grp = 0; grp < 8; ++grp) {
      uint32_t sc, mn;
      if (grp < 4) { sc = sbyte(grp) & 0x3F; mn = sbyte(grp + 4) & 0x3F; }
      else { sc = (sbyte(grp + 4) & 0x0F) | ((sbyte(grp - 4) >> 6) << 4); mn = (sbyte(grp + 4) >> 4) | ((sbyte(grp) >> 6) << 4); }
      const float cds = bd * (float)sc, cdm = bdmin * (float)mn;
      const int q0 = 4 + (grp / 2) * 8;
      const uint32_t shift = (grp & 1) ? 4u : 0u;
      v4i b0, b1;
      b0[0] = (word(q0 + 0) >> shift) & 0x0F0F0F0Fu; b0[1] = (word(q0 + 1) >> shift) & 0x0F0F0F0Fu;
      b0[2] = (word(q0 + 2) >> shift) & 0x0F0F0F0Fu; b0[3] = (word(q0 + 3) >> shift) & 0x0F0F0F0Fu;
      b1[0] = (word(q0 + 4) >> shift) & 0x0F0F0F0Fu; b1[1] = (word(q0 + 5) >> shift) & 0x0F0F0F0Fu;
      b1[2] = (word(q0 + 6) >> shift) & 0x0F0F0F0Fu; b1[3] = (word(q0 + 7) >> shift) & 0x0F0F0F0Fu;
#pragma unroll
      for (int i = 0; i < MF; ++i) {
        const int arow = i * 16 + (lane & 15);
        const v4i a0 = *reinterpret_cast<const v4i*>(&As[arow * APITCH + grp * 32]);
        const v4i a1 = *reinterpret_cast<const v4i*>(&As[arow * APITCH + grp * 32 + 16]);
        v8i ci = {0,0,0,0,0,0,0,0};
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, false, b0, ci, false);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, false, b1, ci, false);
        const float4 d0 = *reinterpret_cast<const float4*>(&dxp[grp][i][lane >> 4][0]);
        const float4 d1 = *reinterpret_cast<const float4*>(&dxp[grp][i][lane >> 4][4]);
        const float4 r0 = *reinterpret_cast<const float4*>(&rfp[grp][i][lane >> 4][0]);
        const float4 r1 = *reinterpret_cast<const float4*>(&rfp[grp][i][lane >> 4][4]);
        acc[i][0] += d0.x * cds * (float)ci[0] - cdm * r0.x;
        acc[i][1] += d0.y * cds * (float)ci[1] - cdm * r0.y;
        acc[i][2] += d0.z * cds * (float)ci[2] - cdm * r0.z;
        acc[i][3] += d0.w * cds * (float)ci[3] - cdm * r0.w;
        acc[i][4] += d1.x * cds * (float)ci[4] - cdm * r1.x;
        acc[i][5] += d1.y * cds * (float)ci[5] - cdm * r1.y;
        acc[i][6] += d1.z * cds * (float)ci[6] - cdm * r1.z;
        acc[i][7] += d1.w * cds * (float)ci[7] - cdm * r1.w;
      }
    }
  }
  for (int i = 0; i < MF; ++i)
    for (int e = 0; e < 8; ++e) {
      const int row = i * 16 + 2 * e + (lane >> 4);
      out[(size_t)(t.slot0 + row) * N + n0 + col] = acc[i][e];
    }
}

// ---------------- host side ----------------
static float f16(__half h) { return __half2float(h); }
static void cpu_ref(const std::vector<block_q4_K>& W, int E, int N, int K, const std::vector<int8_t>& Xq, const std::vector<float>& Xd, const std::vector<float>& Xs,
                    const std::vector<Tile>& tiles, std::vector<float>& out) {
  const int KB = K / 32;
  for (const Tile& t : tiles) for (int r = 0; r < t.rows; ++r) {
    const int slot = t.slot0 + r;
    for (int n = 0; n < N; ++n) {
      const block_q4_K* row = &W[((size_t)t.expert * N + n) * (K / 256)];
      double s = 0;
      for (int g = 0; g < KB; ++g) {
        const block_q4_K& b = row[g / 8]; const int grp = g % 8;
        int sc, mn; if (grp < 4) { sc = b.scales[grp] & 0x3F; mn = b.scales[grp+4] & 0x3F; } else { sc = (b.scales[grp+4] & 0x0F) | ((b.scales[grp-4] >> 6) << 4); mn = (b.scales[grp+4] >> 4) | ((b.scales[grp] >> 6) << 4); }
        const float dws = f16(b.d) * sc, dm = f16(b.dmin) * mn;
        int dot = 0, sumq = 0;
        for (int k = 0; k < 32; ++k) {
          const uint8_t qb = b.qs[(grp / 2) * 32 + k]; const int qw = (grp & 1) ? (qb >> 4) : (qb & 0xF);
          const int qx = Xq[(size_t)slot * K + g * 32 + k];
          dot += qx * qw; sumq += qx;
        }
        s += Xd[(size_t)slot * KB + g] * (dws * dot - dm * sumq);
      }
      out[(size_t)slot * N + n] = (float)s;
    }
  }
}

int main(int argc, char** argv) {
  const int E = argc > 1 ? atoi(argv[1]) : 512, N = argc > 2 ? atoi(argv[2]) : 640, K = argc > 3 ? atoi(argv[3]) : 2560;
  const int R = argc > 4 ? atoi(argv[4]) : 40; const bool check = argc > 5;
  const int BM = R <= 16 ? 16 : R <= 32 ? 32 : R <= 48 ? 48 : 64;
  std::mt19937 rng(1);
  std::vector<block_q4_K> W((size_t)E * N * (K / 256));
  for (auto& b : W) { b.d = __float2half(0.01f * (rng() % 100 + 1) / 100.f); b.dmin = __float2half(0.005f * (rng() % 100) / 100.f); for (auto& s : b.scales) s = rng() & 0xFF; for (auto& q : b.qs) q = rng() & 0xFF; }
  std::vector<Tile> tiles; int slot = 0;
  for (int e = 0; e < E; ++e) { for (int m0 = 0; m0 < R; m0 += BM) tiles.push_back({e, slot + m0, std::min(BM, R - m0)}); slot += ((R + BM - 1) / BM) * BM; }
  const size_t slots = slot, KB = K / 32;
  std::vector<int8_t> Xq(slots * K); std::vector<float> Xd(slots * KB), Xs(slots * KB);
  for (size_t s = 0; s < slots; ++s) for (size_t g = 0; g < KB; ++g) { int sum = 0; for (int k = 0; k < 32; ++k) { int q = (int)(rng() % 255) - 127; Xq[s * K + g * 32 + k] = q; sum += q; } Xd[s * KB + g] = 0.02f; Xs[s * KB + g] = sum; }
  block_q4_K* dW; int8_t* dXq; float *dXd, *dXs, *dout; Tile* dt;
  hipMalloc(&dW, W.size() * sizeof(block_q4_K)); hipMalloc(&dXq, Xq.size()); hipMalloc(&dXd, Xd.size() * 4); hipMalloc(&dXs, Xs.size() * 4); hipMalloc(&dout, slots * N * 4); hipMalloc(&dt, tiles.size() * sizeof(Tile));
  {
    // chunk-major copy for the device: [e][c][n]
    const int NB = K / 256; std::vector<block_q4_K> Wc(W.size());
    for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int c = 0; c < NB; ++c) Wc[((size_t)e * NB + c) * N + n] = W[((size_t)e * N + n) * NB + c];
    hipMemcpy(dW, Wc.data(), Wc.size() * sizeof(block_q4_K), hipMemcpyHostToDevice);
  }
  hipMemcpy(dXq, Xq.data(), Xq.size(), hipMemcpyHostToDevice);
  hipMemcpy(dXd, Xd.data(), Xd.size() * 4, hipMemcpyHostToDevice); hipMemcpy(dXs, Xs.data(), Xs.size() * 4, hipMemcpyHostToDevice); hipMemcpy(dt, tiles.data(), tiles.size() * sizeof(Tile), hipMemcpyHostToDevice);
#ifndef BNW
#define BNW 128
#endif
  dim3 grid(tiles.size(), N / BNW);
  auto launch = [&] {
    if (BM == 16) hipLaunchKernelGGL((moe_gemm_i8<16, BNW>), grid, dim3(BNW * 2), 0, 0, dt, dW, dXq, dXd, dXs, dout, N, K);
    else if (BM == 32) hipLaunchKernelGGL((moe_gemm_i8<32, BNW>), grid, dim3(BNW * 2), 0, 0, dt, dW, dXq, dXd, dXs, dout, N, K);
    else if (BM == 48) hipLaunchKernelGGL((moe_gemm_i8<48, BNW>), grid, dim3(BNW * 2), 0, 0, dt, dW, dXq, dXd, dXs, dout, N, K);
    else hipLaunchKernelGGL((moe_gemm_i8<64, BNW>), grid, dim3(BNW * 2), 0, 0, dt, dW, dXq, dXd, dXs, dout, N, K);
  };
  launch(); launch(); hipDeviceSynchronize();
  if (check) {
    std::vector<float> got(slots * N), ref(slots * N, 0.f); hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost);
    cpu_ref(W, E, N, K, Xq, Xd, Xs, tiles, ref);
    double maxd = 0, maxv = 0; for (const Tile& t : tiles) for (int r = 0; r < t.rows; ++r) for (int n = 0; n < N; ++n) { size_t i = (size_t)(t.slot0 + r) * N + n; maxd = std::max(maxd, (double)fabs(got[i] - ref[i])); maxv = std::max(maxv, (double)fabs(ref[i])); }
    printf("check: max|diff| %g (max |ref| %g)\n", maxd, maxv);
  }
  const int iters = 10; auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) launch();
  hipDeviceSynchronize();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  const double flop = 2.0 * E * R * N * K, wbytes = (double)W.size() * 144;
  printf("E=%d N=%d K=%d R=%d: %.3f ms  %.1f TOPS  weights %.0f MB @ %.0f GB/s\n", E, N, K, R, ms, flop / ms / 1e9, wbytes / 1e6, wbytes / ms / 1e6);
}
