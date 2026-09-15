#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host, std::size_t extra_bytes = 0) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + extra_bytes + 4096),
           "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

/// Packed expert weights plus their dequantized values [e][m][k].
struct Experts {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

/// Random Q4_K blocks with the real 6-bit scale/min packing.
Experts MakeQ4K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 144);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 144;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qs = blk + 16;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code = (qs[base] >> shift) & 0xF;
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_1 blocks: d, m, 32 high bits, 16 packed low nibbles.
Experts MakeQ5_1(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 24);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 24;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half mn = __float2half(Uniform(&seed, 0.5F));
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &mn, 2);
    const std::uint32_t qh = NextRandom(&seed);
    std::memcpy(blk + 4, &qh, 4);
    std::uint8_t* qs = blk + 8;
    for (int i = 0; i < 16; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(mn);
    for (int j = 0; j < 32; ++j) {
      const int low = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
      const int code = low | (static_cast<int>((qh >> j) & 1U) << 4);
      w.values[b * 32 + j] = df * static_cast<float>(code) + mf;
    }
  }
  return w;
}

struct Result {
  double vs_mmq;
  double vs_ref;
  double scale;
};

/// Runs the MMQ reference and the WMMA route over the same routing and
/// reports the worst absolute errors against each other and against an F64
/// reference over the dequantized weights.
Result Run(q::WeightType type, std::size_t n_tokens, std::size_t used,
           std::size_t experts, std::size_t m, std::size_t k,
           std::uint32_t seed, bool check = true) {
  const Experts w = type == q::WeightType::kQ4_K
                        ? MakeQ4K(experts, m, k, seed)
                        : MakeQ5_1(experts, m, k, seed);
  const std::size_t slots = n_tokens * used;
  // Skewed top-k without replacement: the low experts see most rows.
  std::vector<std::int32_t> ids(slots);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (std::size_t t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < used) {
      const std::uint32_t r = NextRandom(&state);
      const auto e = static_cast<std::int32_t>(
          (check && r % 3 == 0 ? r / 3 % 4 : r / 3) % experts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * used);
  }
  std::vector<float> x(n_tokens * k);
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  std::vector<std::uint32_t> counts(experts, 0);
  for (std::int32_t e : ids) {
    ++counts[e];
  }
  std::uint32_t max_bucket = 0;
  for (std::uint32_t c : counts) {
    max_bucket = std::max(max_bucket, (c + 15u) / 16u * 16u);
  }

  std::uint8_t* d_w = Upload(w.packed);
  float* d_x = Upload(x);
  std::int32_t* d_ids = Upload(ids);
  std::uint32_t* d_counts = Upload(counts);
  const std::size_t compact = q::RoutedCompactRows(slots, experts);
  std::vector<std::int32_t> zeros(compact + experts + 1, 0);
  std::int32_t* d_bounds = Upload(std::vector<std::int32_t>(experts + 1, 0));
  std::int32_t* d_cursors = Upload(std::vector<std::int32_t>(experts, 0));
  std::int32_t* d_rows_token = Upload(zeros);
  std::int32_t* d_rows_slot = Upload(zeros);
  void* d_tiled = nullptr;
  CheckHip(hipMalloc(&d_tiled, q::Q8TiledBytes(n_tokens, k)), "hipMalloc");
  std::vector<float> zero_out(slots * m, 0.0F);
  float* d_mmq = Upload(zero_out);
  float* d_wmma = Upload(zero_out);

  const int rc =
      type == q::WeightType::kQ4_K
          ? qfn_mmq_q4_K_moe_raw(
                d_w, d_x, d_ids, d_mmq, static_cast<int>(m),
                static_cast<int>(k), static_cast<int>(n_tokens),
                static_cast<int>(experts), static_cast<int>(used), nullptr)
          : qfn_mmq_q5_1_moe_raw(
                d_w, d_x, d_ids, d_mmq, static_cast<int>(m),
                static_cast<int>(k), static_cast<int>(n_tokens),
                static_cast<int>(experts), static_cast<int>(used), nullptr);
  if (rc != 0) {
    throw std::runtime_error("MMQ routed GEMM failed");
  }
  // The WMMA route over the same assignments. `used == 1` (the down
  // projection's view) gathers slot rows; the gate/up view gathers tokens.
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, static_cast<std::uint32_t>(n_tokens),
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(experts), nullptr);
  q::QuantizeQ8Tiled(d_x, d_tiled, n_tokens, k, nullptr);
  if (!q::RoutedWmmaGemm(
          d_w, type, d_tiled, d_bounds, d_rows_token, d_rows_slot, d_wmma, m, k,
          static_cast<std::uint32_t>(experts), max_bucket, nullptr)) {
    throw std::runtime_error("routed WMMA GEMM rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "routed GEMMs");
  const auto mmq = Download(d_mmq, slots * m);
  const auto wmma = Download(d_wmma, slots * m);

  Result r{0.0, 0.0, 0.0};
  for (std::size_t s = 0; s < (check ? slots : 0); ++s) {
    const std::size_t t = s / used;
    const std::int32_t e = ids[s];
    for (std::size_t row = 0; row < m; ++row) {
      double ref = 0.0;
      const float* wr = w.values.data() + (e * m + row) * k;
      const float* xr = x.data() + t * k;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(wr[i]) * xr[i];
      }
      const float v = wmma[s * m + row];
      if (!std::isfinite(v)) {
        throw std::runtime_error("routed WMMA output is not finite");
      }
      r.vs_mmq = std::max(r.vs_mmq,
                          std::abs(static_cast<double>(mmq[s * m + row] - v)));
      r.vs_ref = std::max(r.vs_ref, std::abs(ref - v));
      r.scale = std::max(r.scale, std::abs(ref));
    }
  }
  std::cout << (type == q::WeightType::kQ4_K ? "Q4_K" : "Q5_1")
            << " routed n=" << n_tokens << " used=" << used << " E=" << experts
            << " m=" << m << " k=" << k << ": worst |WMMA - MMQ| " << r.vs_mmq
            << ", worst |WMMA - F64| " << r.vs_ref << " (scale " << r.scale
            << ", widest bucket " << max_bucket << ")\n";
  for (void* p :
       {static_cast<void*>(d_w), static_cast<void*>(d_x),
        static_cast<void*>(d_ids), static_cast<void*>(d_counts),
        static_cast<void*>(d_bounds), static_cast<void*>(d_cursors),
        static_cast<void*>(d_rows_token), static_cast<void*>(d_rows_slot),
        d_tiled, static_cast<void*>(d_mmq), static_cast<void*>(d_wmma)}) {
    (void)hipFree(p);
  }
  return r;
}

bool Ok(const Result& r) {
  // Both routes quantize the activations to 8 bits per 32-wide block but
  // with their own layouts and accumulation order, so they agree to a few
  // 1e-4 of the output scale; the F64 gap is the quantization itself.
  return r.vs_mmq < 3e-3 * r.scale && r.vs_ref < 1e-2 * r.scale;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    // Gate/up view: 64 experts, top-10, 640 x 2560 Q4_K.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 10, 64, 640, 2560, 0x1234ABCDU)) &&
         ok;
    // Down view: single-expert slot rows, 2560 x 640 Q5_1.
    ok = Ok(Run(q::WeightType::kQ5_1, 3000, 1, 64, 2560, 640, 0x0BADF00DU)) &&
         ok;
    // Ragged rows against the 128-row tile and a tiny batch.
    ok = Ok(Run(q::WeightType::kQ4_K, 40, 4, 8, 200, 512, 0xDEADBEEFU)) && ok;
    // Production shapes for profiling only (QFN_ROUTED_BENCH=1): the two
    // routes are launched three times each, no reference.
    if (std::getenv("QFN_ROUTED_BENCH") != nullptr) {
      for (int i = 0; i < 3; ++i) {
        (void)Run(q::WeightType::kQ4_K, 2048, 10, 512, 640, 2560, 0x1111U + i,
                  false);
        (void)Run(q::WeightType::kQ5_1, 20480, 1, 512, 2560, 640, 0x2222U + i,
                  false);
      }
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
