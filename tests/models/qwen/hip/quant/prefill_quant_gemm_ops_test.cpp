// Oracle test for the batched prefill W8A8 WMMA GEMM (opt-c163-blocked-w8a8).
//
// Covers the blocked macro-tile kernel behind
// LaunchBatchedQuantGEMMPreQuantized and the dual gate/up kernel, against an
// independent CPU reference that reproduces the exact Q8_0 x Q8_1 block dot
// product the kernels compute. Both the 128-token throughput configuration and
// the 64-token short-prompt configuration are exercised, along with shapes
// whose row count and batch are not multiples of the macro tile.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

struct HostQ8_0Block {
  std::uint16_t d;  // fp16 bits
  std::int8_t qs[32];
};
static_assert(sizeof(HostQ8_0Block) == 34, "block_q8_0 must be 34 bytes");

float Fp16ToFloat(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 15u) << 31u;
  const std::uint32_t exp = (bits >> 10u) & 0x1Fu;
  const std::uint32_t man = bits & 0x3FFu;
  std::uint32_t out = 0;
  if (exp == 0) {
    out = sign;  // zero / subnormal flushed, adequate for this fixture
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (man << 13u);
  } else {
    out = sign | ((exp + 112u) << 23u) | (man << 13u);
  }
  float f = 0.0F;
  __builtin_memcpy(&f, &out, sizeof(f));
  return f;
}

std::uint16_t FloatToFp16(float f) {
  std::uint32_t bits = 0;
  __builtin_memcpy(&bits, &f, sizeof(bits));
  const std::uint32_t sign = (bits >> 31u) & 1u;
  std::int32_t exp = static_cast<std::int32_t>((bits >> 23u) & 0xFFu) - 127;
  std::uint32_t man = bits & 0x7FFFFFu;
  if (exp < -14) {
    return static_cast<std::uint16_t>(sign << 15u);
  }
  if (exp > 15) {
    exp = 15;
    man = 0x7FFFFFu;
  }
  return static_cast<std::uint16_t>(
      (sign << 15u) | (static_cast<std::uint32_t>(exp + 15) << 10u) |
      (man >> 13u));
}

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  std::uint32_t Next() {
    state_ = (state_ * 1664525U) + 1013904223U;
    return state_;
  }
  float Uniform(float lo, float hi) {
    const float u = static_cast<float>(Next() & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }
  std::int8_t Int8() {
    return static_cast<std::int8_t>(static_cast<int>(Next() % 255U) - 127);
  }

private:
  std::uint32_t state_;
};

// Independent CPU reference: y[t][r] = sum over 32-element blocks of
// (d_w * d_x) * dot_int8, accumulated in fp32 in ascending block order.
std::vector<float> ReferenceGemm(const std::vector<HostQ8_0Block>& w,
                                 const std::vector<float>& x_scale,
                                 const std::vector<std::int8_t>& x_q,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k) {
  const std::size_t num_blocks = k / 32;
  std::vector<float> y(batch * m, 0.0F);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t r = 0; r < m; ++r) {
      float acc = 0.0F;
      for (std::size_t b = 0; b < num_blocks; ++b) {
        const HostQ8_0Block& blk = w[(r * num_blocks) + b];
        std::int32_t dot = 0;
        for (std::size_t j = 0; j < 32; ++j) {
          dot +=
              static_cast<std::int32_t>(blk.qs[j]) *
              static_cast<std::int32_t>(x_q[(((t * num_blocks) + b) * 32) + j]);
        }
        // The device epilogue contracts the scale multiply and the accumulate
        // into a single fma, so the reference must too.
        acc = std::fma(Fp16ToFloat(blk.d) * x_scale[(t * num_blocks) + b],
                       static_cast<float>(dot), acc);
      }
      y[(t * m) + r] = acc;
    }
  }
  return y;
}

// Individual outputs cancel to near zero, so the meaningful error measure is
// scaled by the magnitude of the tensor, not by each element.
void Compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& want) {
  double max_abs = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) {
      std::cerr << label << ": non-finite output at " << i << "\n";
      std::abort();
    }
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(want[i])));
    scale = std::fmax(scale, std::fabs(static_cast<double>(want[i])));
  }
  const double rel = (scale > 0.0) ? (max_abs / scale) : max_abs;
  std::cout << "  " << label << ": max_abs=" << max_abs
            << " magnitude=" << scale << " rel_to_magnitude=" << rel << "\n";
  if (rel >= 1e-5) {
    std::cerr << label << ": prefill quant GEMM mismatch against CPU oracle\n";
    std::abort();
  }
}

// Decodes the tiled Q8_1 activation buffer the production quantize kernel wrote
// into flat per-block scale/byte arrays. Driving the reference from the actual
// device activations isolates the GEMM under test from the quantizer's own
// rounding, so the comparison is exact.
void DecodeTiledActivations(const std::vector<std::uint8_t>& tiled,
                            std::size_t batch, std::size_t num_blocks,
                            std::vector<float>& scale_out,
                            std::vector<std::int8_t>& q_out) {
  constexpr std::size_t kTileBytes = 576;
  constexpr std::size_t kScaleOffset = 512;
  scale_out.assign(batch * num_blocks, 0.0F);
  q_out.assign(batch * num_blocks * 32, 0);
  for (std::size_t t = 0; t < batch; ++t) {
    const std::size_t tt = t / 16;
    const std::size_t tl = t % 16;
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const std::uint8_t* tile =
          tiled.data() + (((tt * num_blocks) + b) * kTileBytes);
      float d = 0.0F;
      std::memcpy(&d, tile + kScaleOffset + (tl * sizeof(float)), sizeof(d));
      scale_out[(t * num_blocks) + b] = d;
      for (std::size_t j = 0; j < 16; ++j) {
        q_out[(((t * num_blocks) + b) * 32) + j] =
            static_cast<std::int8_t>(tile[(tl * 16) + j]);
        q_out[(((t * num_blocks) + b) * 32) + 16 + j] =
            static_cast<std::int8_t>(tile[256 + (tl * 16) + j]);
      }
    }
  }
}

// Independent CPU quantizer, used only to confirm the device quantizer stays
// within one code of the reference rounding.
void QuantizeReference(const std::vector<float>& x, std::size_t batch,
                       std::size_t k, std::vector<float>& scale_out,
                       std::vector<std::int8_t>& q_out) {
  const std::size_t num_blocks = k / 32;
  scale_out.assign(batch * num_blocks, 0.0F);
  q_out.assign(batch * num_blocks * 32, 0);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t b = 0; b < num_blocks; ++b) {
      float max_abs = 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        max_abs = std::fmax(max_abs, std::fabs(x[(t * k) + (b * 32) + j]));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      scale_out[(t * num_blocks) + b] = d;
      for (std::size_t j = 0; j < 32; ++j) {
        q_out[(((t * num_blocks) + b) * 32) + j] = static_cast<std::int8_t>(
            std::lround(x[(t * k) + (b * 32) + j] * id));
      }
    }
  }
}

void RunCase(std::size_t batch, std::size_t m, std::size_t k, bool dual) {
  std::cout << "prefill quant GEMM: batch=" << batch << " m=" << m << " k=" << k
            << (dual ? " dual" : " single") << "\n";
  const std::size_t num_blocks = k / 32;
  Rng rng(0x9E3779B9U ^
          static_cast<std::uint32_t>((batch * 131) + (m * 17) + k));

  std::vector<HostQ8_0Block> h_w(m * num_blocks);
  std::vector<HostQ8_0Block> h_w2(m * num_blocks);
  for (std::size_t i = 0; i < h_w.size(); ++i) {
    h_w[i].d = FloatToFp16(rng.Uniform(-0.05F, 0.05F));
    h_w2[i].d = FloatToFp16(rng.Uniform(-0.05F, 0.05F));
    for (int j = 0; j < 32; ++j) {
      h_w[i].qs[j] = rng.Int8();
      h_w2[i].qs[j] = rng.Int8();
    }
  }

  std::vector<float> h_x(batch * k);
  for (auto& v : h_x) {
    v = rng.Uniform(-1.5F, 1.5F);
  }
  // The kernels read bf16 activations, so the reference must see the rounded
  // values, not the original fp32 ones.
  std::vector<std::uint16_t> h_x_bf16(h_x.size());
  for (std::size_t i = 0; i < h_x.size(); ++i) {
    h_x_bf16[i] = strix::test::FloatToBf16Bits(h_x[i]);
    h_x[i] = strix::test::Bf16BitsToFloat(h_x_bf16[i]);
  }

  void* d_w = nullptr;
  void* d_w2 = nullptr;
  void* d_x_bf16 = nullptr;
  void* d_q8 = nullptr;
  float* d_y = nullptr;
  float* d_y2 = nullptr;
  const std::size_t q8_bytes = (((batch + 15) / 16) * num_blocks * 576) + 4096;
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(HostQ8_0Block)));
  HIP_CHECK(hipMalloc(&d_w2, h_w2.size() * sizeof(HostQ8_0Block)));
  HIP_CHECK(hipMalloc(&d_x_bf16, h_x_bf16.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_q8, q8_bytes));
  HIP_CHECK(hipMalloc(&d_y, batch * m * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y2, batch * m * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(HostQ8_0Block),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(HostQ8_0Block),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x_bf16, h_x_bf16.data(),
                      h_x_bf16.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_y, 0, batch * m * sizeof(float)));
  HIP_CHECK(hipMemset(d_y2, 0, batch * m * sizeof(float)));

  strix::hip::LaunchQuantizeActivationQ8_1(d_x_bf16, d_q8, batch, k);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<std::uint8_t> h_q8(q8_bytes);
  HIP_CHECK(hipMemcpy(h_q8.data(), d_q8, q8_bytes, hipMemcpyDeviceToHost));
  std::vector<float> x_scale;
  std::vector<std::int8_t> x_q;
  DecodeTiledActivations(h_q8, batch, num_blocks, x_scale, x_q);

  // The device quantizer must agree with the reference to within one code.
  {
    std::vector<float> ref_scale;
    std::vector<std::int8_t> ref_q;
    QuantizeReference(h_x, batch, k, ref_scale, ref_q);
    int worst_code = 0;
    double worst_scale = 0.0;
    for (std::size_t i = 0; i < ref_q.size(); ++i) {
      worst_code = std::max(worst_code, std::abs(static_cast<int>(x_q[i]) -
                                                 static_cast<int>(ref_q[i])));
    }
    for (std::size_t i = 0; i < ref_scale.size(); ++i) {
      worst_scale =
          std::fmax(worst_scale, std::fabs(x_scale[i] - ref_scale[i]));
    }
    std::cout << "  quantizer: max_code_delta=" << worst_code
              << " max_scale_delta=" << worst_scale << "\n";
    if (worst_code > 1) {
      std::cerr << "device Q8_1 quantizer deviates by more than one code\n";
      std::abort();
    }
  }

  if (dual) {
    strix::hip::LaunchBatchedDualQuantGEMMPreQuantized(
        strix::core::GgmlType::kQ8_0, d_w, d_w2, d_q8, d_y, d_y2, batch, m, k);
  } else {
    strix::hip::LaunchBatchedQuantGEMMPreQuantized(strix::core::GgmlType::kQ8_0,
                                                   d_w, d_q8, d_y, batch, m, k);
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(batch * m);
  HIP_CHECK(hipMemcpy(got.data(), d_y, got.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  Compare("primary", got, ReferenceGemm(h_w, x_scale, x_q, batch, m, k));
  if (dual) {
    HIP_CHECK(hipMemcpy(got.data(), d_y2, got.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    Compare("secondary", got, ReferenceGemm(h_w2, x_scale, x_q, batch, m, k));
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_w2));
  HIP_CHECK(hipFree(d_x_bf16));
  HIP_CHECK(hipFree(d_q8));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_y2));
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      strix::test::GateHipDevice(strix::test::HipDeviceRequirement::kOptional,
                                 "Qwen prefill quant GEMM ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  // 128-token throughput configuration, macro-tile aligned.
  RunCase(128, 256, 256, false);
  // Short-prompt 64-token configuration.
  RunCase(64, 256, 256, false);
  // Row count and batch that are not multiples of the macro tile.
  RunCase(100, 200, 160, false);
  // Rows below one macro tile.
  RunCase(96, 64, 128, false);
  // Dual gate/up path.
  RunCase(128, 256, 256, true);

  std::cout << "Qwen prefill quant GEMM ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen prefill quant GEMM ops test.\n";
  return 77;
#endif
}
