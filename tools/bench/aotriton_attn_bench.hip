// AOTriton flash-attention viability check at the Qwen3.8-27B prefill shapes.
//
// The production prefill attention (`QwenTiledAttentionKernel`) is a correct
// online-softmax flash attention, but it computes QK^T and PV with
// `v_dot2_f32_f16` on the vector ALU instead of the WMMA matrix cores, which
// puts it at ~4.4 TFLOPS -- 8% of the measured 55 TFLOPS ceiling, 18% of prefill
// at depth 8192, and it is the term that grows quadratically with context.
// AOTriton ships pretuned gfx11xx `attn_fwd` variants including head_dim 256 in
// fp16 and bf16, and this repository already links `aotriton_v2` for MiniMax H3.
//
// Findings this tool established:
//   * GQA works (24 query heads over 4 KV heads) and both fp16 and bf16work.
//   * The v2 `is_causal=true` path is REJECTED on gfx11xx: only
//     `causal_type` None and WindowedAttention are compiled into the shipped
//     images (variant suffixes `_0_0`, `_0_1`, `_3_0`), so causal masking has to
//     go through the v3 API as a window.
//   * Hence this probes (window_left, window_right) against independent
//     top-left and bottom-right aligned references to pin the semantics.
//
//   nix develop -c tools/bench/build.sh aotriton_attn_bench
//
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
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

constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kGqa = kQHeads / kKvHeads;

// Reference: one block per (query, query head), fp32 online softmax over the
// visible keys. `key_offset` is how far ahead of its own index a query may see:
// bottom-right alignment uses seqlen_k - seqlen_q, top-left uses 0.
__global__ void ReferenceAttention(const __half* __restrict__ q,
                                   const __half* __restrict__ k,
                                   const __half* __restrict__ v,
                                   float* __restrict__ out, int seqlen_k,
                                   int key_offset) {
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int kv_head = head / kGqa;
  const int lane = static_cast<int>(threadIdx.x);
  const int last_key = key_offset + query;
  const float scale = 1.0F / 16.0F;

  __shared__ float red[256];
  __shared__ float s_score;

  float acc = 0.0F;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t q_row =
      (static_cast<std::size_t>(query) * kQHeads * kHeadDim) +
      (static_cast<std::size_t>(head) * kHeadDim);

  for (int key = 0; key <= last_key && key < seqlen_k; ++key) {
    const std::size_t kv_row =
        (static_cast<std::size_t>(key) * kKvHeads * kHeadDim) +
        (static_cast<std::size_t>(kv_head) * kHeadDim);
    red[lane] = __half2float(q[q_row + lane]) * __half2float(k[kv_row + lane]);
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (lane < s) {
        red[lane] += red[lane + s];
      }
      __syncthreads();
    }
    if (lane == 0) {
      s_score = red[0] * scale;
    }
    __syncthreads();

    const float next_max = fmaxf(running_max, s_score);
    const float rescale =
        isfinite(running_max) ? expf(running_max - next_max) : 0.0F;
    const float weight = expf(s_score - next_max);
    running_sum = (running_sum * rescale) + weight;
    running_max = next_max;
    acc = (acc * rescale) + (weight * __half2float(v[kv_row + lane]));
    __syncthreads();
  }

  out[q_row + lane] = (running_sum > 0.0F) ? (acc / running_sum) : 0.0F;
}

__global__ void HalfToFloat(const __half* in, float* out, std::size_t n) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = __half2float(in[i]);
  }
}

// AOTriton's persistent/causal kernels expect a real atomic counter tensor
// rather than a null one; a single zeroed 32-bit word is enough.
static std::uint32_t* g_atomic_counter = nullptr;
static double g_last_output_magnitude = 0.0;

struct AttnConfig {
  int kv_heads = kKvHeads;
  bool bf16 = false;
  int8_t causal_type = 0;
  std::int32_t window_left = 0;
  std::int32_t window_right = 0;
  int force_backend = -1;
};

static bool RunAotriton(const __half* q, const __half* k, const __half* v,
                        __half* out, float* lse, int seqlen_q, int seqlen_k,
                        const AttnConfig& cfg, hipStream_t stream,
                        bool quiet = false) {
  using aotriton::DType;
  using aotriton::TensorView;
  const DType dt = cfg.bf16 ? DType::kBFloat16 : DType::kFloat16;
  const std::array<std::uint64_t, 4> q_sizes{
      1, kQHeads, static_cast<std::uint64_t>(seqlen_q), kHeadDim};
  const std::array<std::uint64_t, 4> q_strides{
      static_cast<std::uint64_t>(seqlen_q) * kQHeads * kHeadDim, kHeadDim,
      kQHeads * kHeadDim, 1};
  const std::array<std::uint64_t, 4> kv_sizes{
      1, static_cast<std::uint64_t>(cfg.kv_heads),
      static_cast<std::uint64_t>(seqlen_k), kHeadDim};
  const std::array<std::uint64_t, 4> kv_strides{
      static_cast<std::uint64_t>(seqlen_k) * cfg.kv_heads * kHeadDim, kHeadDim,
      static_cast<std::uint64_t>(cfg.kv_heads) * kHeadDim, 1};

  aotriton::v3::flash::attn_fwd_params params;
  params.Q = TensorView<4>(reinterpret_cast<intptr_t>(q), q_sizes, q_strides, dt);
  params.K =
      TensorView<4>(reinterpret_cast<intptr_t>(k), kv_sizes, kv_strides, dt);
  params.V =
      TensorView<4>(reinterpret_cast<intptr_t>(v), kv_sizes, kv_strides, dt);
  params.B = TensorView<4>::get_null_tensor(dt);
  params.A = TensorView<2>::get_null_tensor(DType::kFloat32);
  params.Sm_scale = 1.0F / 16.0F;
  params.L = TensorView<2>(reinterpret_cast<intptr_t>(lse),
                           {kQHeads, static_cast<std::uint64_t>(seqlen_q)},
                           {static_cast<std::uint64_t>(seqlen_q), 1},
                           DType::kFloat32);
  params.Out =
      TensorView<4>(reinterpret_cast<intptr_t>(out), q_sizes, q_strides, dt);
  params.Max_seqlen_q = seqlen_q;
  params.Max_seqlen_k = seqlen_k;
  params.dropout_p = 0.0F;
  params.encoded_softmax = TensorView<4>::get_null_tensor(DType::kFloat32);
  params.persistent_atomic_counter = TensorView<0>(
      reinterpret_cast<intptr_t>(g_atomic_counter), aotriton::DType::kUInt32);
  params.causal_type = cfg.causal_type;
  params.varlen_type = 0;
  params.window_left = cfg.window_left;
  params.window_right = cfg.window_right;

  aotriton::v3::flash::attn_options options;
  options.force_backend_index = cfg.force_backend;
  const hipError_t status = aotriton::v3::flash::attn_fwd(
      params, aotriton::v3::flash::attn_fwd_params::kVersion,
      aotriton::Stream(stream), &options);
  if (status != hipSuccess) {
    if (!quiet) {
      std::printf("    attn_fwd failed: %s\n", hipGetErrorString(status));
    }
    return false;
  }
  return true;
}

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

struct Buffers {
  __half* q = nullptr;
  __half* k = nullptr;
  __half* v = nullptr;
  __half* out = nullptr;
  float* out_f32 = nullptr;
  float* ref = nullptr;
  float* lse = nullptr;
  std::size_t qn = 0;
  std::size_t kn = 0;
};

static Buffers Allocate(int seqlen_q, int seqlen_k) {
  Buffers b;
  b.qn = static_cast<std::size_t>(seqlen_q) * kQHeads * kHeadDim;
  b.kn = static_cast<std::size_t>(seqlen_k) * kKvHeads * kHeadDim;
  HIP_OK(hipMalloc(&b.q, b.qn * sizeof(__half)));
  HIP_OK(hipMalloc(&b.k, b.kn * sizeof(__half)));
  HIP_OK(hipMalloc(&b.v, b.kn * sizeof(__half)));
  HIP_OK(hipMalloc(&b.out, b.qn * sizeof(__half)));
  HIP_OK(hipMalloc(&b.out_f32, b.qn * sizeof(float)));
  HIP_OK(hipMalloc(&b.ref, b.qn * sizeof(float)));
  HIP_OK(hipMalloc(&b.lse, static_cast<std::size_t>(kQHeads) * seqlen_q *
                               sizeof(float)));

  std::vector<__half> hq(b.qn);
  std::vector<__half> hk(b.kn);
  std::vector<__half> hv(b.kn);
  std::uint32_t seed = 1234567U;
  auto rnd = [&seed]() {
    seed = (seed * 1664525U) + 1013904223U;
    return (static_cast<float>((seed >> 8) & 0xFFFFU) / 65535.0F) - 0.5F;
  };
  for (auto& x : hq) {
    x = __float2half(rnd());
  }
  for (auto& x : hk) {
    x = __float2half(rnd());
  }
  for (auto& x : hv) {
    x = __float2half(rnd());
  }
  HIP_OK(hipMemcpy(b.q, hq.data(), b.qn * sizeof(__half),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(b.k, hk.data(), b.kn * sizeof(__half),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(b.v, hv.data(), b.kn * sizeof(__half),
                   hipMemcpyHostToDevice));
  return b;
}

static void Free(Buffers& b) {
  HIP_OK(hipFree(b.q));
  HIP_OK(hipFree(b.k));
  HIP_OK(hipFree(b.v));
  HIP_OK(hipFree(b.out));
  HIP_OK(hipFree(b.out_f32));
  HIP_OK(hipFree(b.ref));
  HIP_OK(hipFree(b.lse));
}

// Returns max_abs error relative to the tensor magnitude.
static double CompareAgainstReference(Buffers& b, int seqlen_q, int seqlen_k,
                                      int key_offset) {
  const dim3 grid(static_cast<unsigned>(seqlen_q), kQHeads);
  ReferenceAttention<<<grid, kHeadDim>>>(b.q, b.k, b.v, b.ref, seqlen_k,
                                        key_offset);
  HalfToFloat<<<static_cast<unsigned>((b.qn + 255) / 256), 256>>>(
      b.out, b.out_f32, b.qn);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> want(b.qn);
  std::vector<float> got(b.qn);
  HIP_OK(hipMemcpy(want.data(), b.ref, b.qn * sizeof(float),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(got.data(), b.out_f32, b.qn * sizeof(float),
                   hipMemcpyDeviceToHost));
  double max_abs = 0.0;
  double magnitude = 0.0;
  double got_magnitude = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(want[i]) -
                                           static_cast<double>(got[i])));
    magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
    got_magnitude =
        std::fmax(got_magnitude, std::fabs(static_cast<double>(got[i])));
  }
  g_last_output_magnitude = got_magnitude;
  return (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    }
  }

  if (aotriton::v2::flash::check_gpu(aotriton::Stream()) != hipSuccess) {
    std::printf("aotriton: check_gpu rejected this device\n");
    return 1;
  }
  HIP_OK(hipMalloc(&g_atomic_counter, sizeof(std::uint32_t)));
  HIP_OK(hipMemset(g_atomic_counter, 0, sizeof(std::uint32_t)));
  std::printf("aotriton v3: heads_q=%d heads_kv=%d head_dim=%d\n", kQHeads,
              kKvHeads, kHeadDim);

  using aotriton::v3::flash::CausalType;
  using aotriton::v3::flash::WindowValue;

  // --- window semantics probe on an asymmetric shape, where the two causal
  // --- alignments differ. seqlen_q=256, seqlen_k=1024 -> bottom-right lets
  // --- query 0 see keys 0..768, top-left only key 0.
  {
    std::printf("\n-- causal window probe (seqlen_q=256 seqlen_k=1024) --\n");
    const int nq = 256;
    const int nk = 1024;
    Buffers b = Allocate(nq, nk);
    struct Probe {
      const char* name;
      int8_t causal;
      std::int32_t left;
      std::int32_t right;
    };
    const Probe probes[] = {
        {"none (full)", CausalType::None, 0, 0},
        {"window l=-1 r=0", CausalType::WindowedAttention, -1, 0},
        {"window l=nk r=0", CausalType::WindowedAttention, nk, 0},
        {"window l=BR r=0", CausalType::WindowedAttention,
         WindowValue::BottomRightAligned, 0},
        {"window l=TL r=0", CausalType::WindowedAttention,
         WindowValue::TopLeftAligned, 0},
        {"window l=-1 r=BR", CausalType::WindowedAttention, -1,
         WindowValue::BottomRightAligned},
        {"window l=-1 r=TL", CausalType::WindowedAttention, -1,
         WindowValue::TopLeftAligned},
        {"window l=0 r=0", CausalType::WindowedAttention, 0, 0},
        {"window l=16 r=0", CausalType::WindowedAttention, 16, 0},
        {"window l=nk r=nk", CausalType::WindowedAttention, nk, nk},
        {"window l=nk r=-1", CausalType::WindowedAttention, nk, -1},
        {"window l=-1 r=-1", CausalType::WindowedAttention, -1, -1},
    };
    for (const auto& p : probes) {
      AttnConfig cfg;
      cfg.causal_type = p.causal;
      cfg.window_left = p.left;
      cfg.window_right = p.right;
      HIP_OK(hipMemset(b.out, 0, b.qn * sizeof(__half)));
      const bool ok =
          RunAotriton(b.q, b.k, b.v, b.out, b.lse, nq, nk, cfg, nullptr, true);
      HIP_OK(hipDeviceSynchronize());
      if (!ok) {
        std::printf("  %-18s -> REJECTED\n", p.name);
        continue;
      }
      const double br = CompareAgainstReference(b, nq, nk, nk - nq);
      const double tl = CompareAgainstReference(b, nq, nk, 0);
      const double full = CompareAgainstReference(b, nq, nk, nk);
      std::printf("  %-18s -> OK   |out|=%.3e rel(BR)=%.2e rel(TL)=%.2e "
                  "rel(full)=%.2e  %s\n",
                  p.name, g_last_output_magnitude, br, tl, full,
                  br < 5e-3 ? "<== BOTTOM-RIGHT"
                            : (tl < 5e-3 ? "<== TOP-LEFT"
                                         : (full < 5e-3 ? "<== NON-CAUSAL" : "")));
    }
    std::printf("\n  -- windowed causal, forced backend index --\n");
    for (int backend = 0; backend < 6; ++backend) {
      AttnConfig cfg;
      cfg.causal_type = CausalType::WindowedAttention;
      cfg.window_left = nk;
      cfg.window_right = 0;
      cfg.force_backend = backend;
      HIP_OK(hipMemset(b.out, 0, b.qn * sizeof(__half)));
      const bool ok =
          RunAotriton(b.q, b.k, b.v, b.out, b.lse, nq, nk, cfg, nullptr, true);
      HIP_OK(hipDeviceSynchronize());
      if (!ok) {
        std::printf("    backend %d -> REJECTED\n", backend);
        continue;
      }
      const double br = CompareAgainstReference(b, nq, nk, nk - nq);
      std::printf("    backend %d -> OK |out|=%.3e rel(BR)=%.2e %s\n", backend,
                  g_last_output_magnitude, br,
                  br < 5e-3 ? "<== BOTTOM-RIGHT CAUSAL" : "");
    }
    Free(b);
  }

  // --- performance at the production shapes, using non-causal as the
  // --- throughput reference plus whichever causal window was accepted.
  struct Case {
    const char* name;
    int seqlen_q;
    int seqlen_k;
  };
  const Case cases[] = {
      {"pp512 d0", 512, 512},
      {"pp2048 d0", 2048, 2048},
      {"pp2048 d8192", 2048, 10240},
  };
  std::printf("\n-- throughput, CausalType::None (verified correct path) --\n");
  std::printf("   TFLOPS counts every query-key pair, since this is the full\n"
              "   non-causal product; the tiled kernel it would replace does\n"
              "   only the causal half.\n");
  std::printf("%-14s %10s %10s %8s %10s\n", "case", "ms", "TFLOPS", "%peak",
              "rel_err");
  for (const auto& c : cases) {
    Buffers b = Allocate(c.seqlen_q, c.seqlen_k);
    AttnConfig cfg;
    cfg.causal_type = CausalType::None;
    HIP_OK(hipMemset(b.out, 0, b.qn * sizeof(__half)));
    if (!RunAotriton(b.q, b.k, b.v, b.out, b.lse, c.seqlen_q, c.seqlen_k, cfg,
                     nullptr)) {
      std::printf("%-14s rejected\n", c.name);
      Free(b);
      continue;
    }
    HIP_OK(hipDeviceSynchronize());
    const double rel =
        CompareAgainstReference(b, c.seqlen_q, c.seqlen_k, c.seqlen_k);
    const float ms = BenchMs(
        [&] {
          RunAotriton(b.q, b.k, b.v, b.out, b.lse, c.seqlen_q, c.seqlen_k, cfg,
                      nullptr, true);
        },
        iters);
    const double pairs =
        static_cast<double>(c.seqlen_q) * static_cast<double>(c.seqlen_k);
    const double flops = 2.0 * 2.0 * pairs * kHeadDim * kQHeads;
    const double tflops = flops / (ms * 1e-3) / 1e12;
    std::printf("%-14s %10.3f %10.2f %7.0f%% %10.2e\n", c.name, ms, tflops,
                100.0 * tflops / 55.05, rel);
    Free(b);
  }
  return 0;
}
