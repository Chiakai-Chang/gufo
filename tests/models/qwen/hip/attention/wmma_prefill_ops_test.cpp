// Oracle test for the masked WMMA prefill attention (opt-c177-attn-wmma).
//
// The WMMA kernel replaces the tiled `v_dot2_f32_f16` kernel over the same
// visible key range, with the same FP16 K/V cache, the same causal rule and the
// same gate epilogue. Both carry FP16 operands, so they agree to FP16 output
// precision rather than exactly; a CPU reference in tools/bench puts each of
// them at 2.1e-4 against double precision, so the two agreeing to the same
// order is the correct expectation here.
//
// Covered: a depth-0 chunk, a batch that is not a multiple of the 32-query
// block, a chunk at depth, a depth that is not a multiple of the 16-key tile,
// and the log-sum-exp output path.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 24;
constexpr std::uint32_t kNumKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kMaxContext = 16384;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

void Compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& want, double tolerance) {
  double max_abs = 0.0;
  double magnitude = 0.0;
  std::size_t non_finite = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) {
      ++non_finite;
      continue;
    }
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(want[i])));
    magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
  }
  const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
  std::cout << "  " << label << ": max_abs=" << max_abs
            << " magnitude=" << magnitude << " rel=" << rel
            << " non_finite=" << non_finite << "\n";
  if (non_finite != 0 || rel > tolerance) {
    std::cerr << label
              << ": WMMA prefill attention disagrees with the tiled kernel\n";
    std::abort();
  }
}

void RunCase(std::uint32_t start_pos, std::size_t batch_size, bool want_lse) {
  std::cout << "wmma prefill attention: start_pos=" << start_pos
            << " batch=" << batch_size << " lse=" << (want_lse ? "yes" : "no")
            << "\n";

  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
  const std::size_t q_elements = batch_size * attention_size;
  const std::size_t lse_elements = batch_size * kNumHeads;

  Rng rng(0xA11CEU ^ (start_pos * 131U) ^
          static_cast<std::uint32_t>(batch_size));
  std::vector<float> h_q(q_elements);
  std::vector<float> h_gate(q_elements);
  std::vector<float> h_k(batch_size * kv_size);
  std::vector<float> h_v(batch_size * kv_size);
  std::vector<float> h_cache(total_kv * 2);
  for (auto& x : h_q) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_gate) {
    x = rng.Uniform(-2.0F, 2.0F);
  }
  for (auto& x : h_k) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_v) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_cache) {
    x = rng.Uniform(-1.0F, 1.0F);
  }

  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_cache = nullptr;
  void* d_cache_f16 = nullptr;
  float* d_out_ref = nullptr;
  float* d_out_new = nullptr;
  float* d_lse_ref = nullptr;
  float* d_lse_new = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_ref, lse_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_new, lse_elements * sizeof(float)));

  const auto upload = [](float* dst, const std::vector<float>& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  upload(d_q, h_q);
  upload(d_gate, h_gate);
  upload(d_k, h_k);
  upload(d_v, h_v);
  upload(d_cache, h_cache);
  HIP_CHECK(hipMemset(d_cache_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));

  auto* v_cache = d_cache + total_kv;
  auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
  float* lse_ref = want_lse ? d_lse_ref : nullptr;
  float* lse_new = want_lse ? d_lse_new : nullptr;

  // Reference: the tiled kernel, which also packs the FP16 cache.
  HIP_CHECK(hipMemset(d_out_ref, 0, q_elements * sizeof(float)));
  if (!strix::hip::LaunchBatchedAttentionTile(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_ref, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_ref, 0, false)) {
    std::cerr << "tiled attention rejected the production shape\n";
    std::abort();
  }

  // Candidate: the WMMA kernel over the same range. The cache is already
  // packed, so suppress the write and prove it reads the same bytes.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (!strix::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_new, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_new, 0,
          /*skip_kv_write=*/true)) {
    std::cerr << "WMMA attention rejected the production shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(q_elements);
  std::vector<float> got(q_elements);
  HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  Compare("context", got, ref, 2e-3);

  if (want_lse) {
    std::vector<float> lref(lse_elements);
    std::vector<float> lgot(lse_elements);
    HIP_CHECK(hipMemcpy(lref.data(), d_lse_ref, lse_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(lgot.data(), d_lse_new, lse_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    Compare("log-sum-exp", lgot, lref, 2e-3);
  }

  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out_ref, d_out_new,
                   d_lse_ref, d_lse_new}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      strix::test::GateHipDevice(strix::test::HipDeviceRequirement::kOptional,
                                 "Qwen WMMA prefill attention ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  RunCase(0, 256, false);
  // Not a multiple of the 32-query block, so the last block is partly out of
  // range.
  RunCase(0, 100, false);
  // At depth: the whole visible range, prefix and diagonal, in one pass.
  RunCase(1024, 512, false);
  // A depth that is not a multiple of the 16-key tile.
  RunCase(1500, 128, false);
  // The log-sum-exp path, which also suppresses the gate.
  RunCase(1024, 256, true);

  std::cout << "Qwen WMMA prefill attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen WMMA prefill attention ops test.\n";
  return 77;
#endif
}
