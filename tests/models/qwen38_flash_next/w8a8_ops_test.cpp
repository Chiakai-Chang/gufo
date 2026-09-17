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

/// Row-major Q8_0 weights [m][k] and their dequantized values.
struct Q8Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<float> values;
};

Q8Weights MakeWeights(std::size_t m, std::size_t k, std::uint32_t seed) {
  Q8Weights w;
  w.blocks.resize(m * (k / 32) * 34);
  w.values.resize(m * k);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < k / 32; ++b) {
      float raw[32];
      float max_abs = 0.0F;
      for (float& v : raw) {
        v = Uniform(&seed, 1.0F);
        max_abs = std::max(max_abs, std::abs(v));
      }
      const __half d = __float2half(max_abs / 127.0F);
      const float df = __half2float(d);
      std::uint8_t* block = w.blocks.data() + (r * (k / 32) + b) * 34;
      std::memcpy(block, &d, 2);
      for (std::size_t i = 0; i < 32; ++i) {
        const auto qv = static_cast<std::int8_t>(
            std::lround(df != 0.0F ? raw[i] / df : 0.0F));
        block[2 + i] = static_cast<std::uint8_t>(qv);
        w.values[r * k + b * 32 + i] = df * static_cast<float>(qv);
      }
    }
  }
  return w;
}

double Run(std::size_t batch, std::size_t m, std::size_t k, std::uint32_t seed,
           bool check = true, std::size_t reference_tokens = 0) {
  const Q8Weights w = MakeWeights(m, k, seed);
  std::vector<float> x(batch * k);
  std::uint32_t state = seed ^ 0xABCDEF01U;
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_mmq = nullptr;
  float* d_w8 = nullptr;
  float* d_f16 = nullptr;
  __half* d_x_half = nullptr;
  void* d_tiled = nullptr;
  // The model's weights live in GTT (the carve-out VRAM is 0.5 GiB); a
  // bench allocation past that size lands the weights there too.
  const std::size_t w_alloc =
      check ? w.blocks.size() + 4096
            : std::max<std::size_t>(w.blocks.size() + 4096, 1u << 30);
  CheckHip(hipMalloc(&d_w, w_alloc), "hipMalloc");
  CheckHip(hipMalloc(&d_x, x.size() * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_mmq, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_w8, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_tiled, q::Q8TiledBytes(batch, k)), "hipMalloc");
  CheckHip(hipMalloc(&d_f16, batch * m * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_x_half, x.size() * 2), "hipMalloc");
  CheckHip(
      hipMemcpy(d_w, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "upload");
  CheckHip(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemset(d_w8, 0, batch * m * 4), "memset");
  CheckHip(hipMemset(d_f16, 0, batch * m * 4), "memset");
  if (qfn_mmq_q8_0_dense(d_w, d_x, d_mmq, static_cast<int>(m),
                         static_cast<int>(batch), static_cast<int>(k),
                         nullptr) != 0) {
    throw std::runtime_error("MMQ dense failed");
  }
  q::QuantizeQ8Tiled(d_x, d_tiled, batch, k, nullptr);
  for (int rep = 0; rep < (check ? 1 : 3); ++rep) {
    if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
      throw std::runtime_error("W8A8 GEMM rejected the shape");
    }
  }
  q::NarrowActivations(d_x, d_x_half, false, x.size(), nullptr);
  const int f16_reps = std::getenv("QFN_F16_REPS") != nullptr
                           ? std::atoi(std::getenv("QFN_F16_REPS"))
                           : 3;
  for (int rep = 0; rep < (check ? 1 : f16_reps); ++rep) {
    if (!q::DenseF16Gemm(d_w, d_x_half, d_f16, batch, m, k, nullptr)) {
      throw std::runtime_error("dense F16 GEMM rejected the shape");
    }
  }
  CheckHip(hipDeviceSynchronize(), "GEMMs");
  std::vector<float> mmq(batch * m);
  std::vector<float> w8(batch * m);
  std::vector<float> f16(batch * m);
  CheckHip(hipMemcpy(f16.data(), d_f16, f16.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(mmq.data(), d_mmq, mmq.size() * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(w8.data(), d_w8, w8.size() * 4, hipMemcpyDeviceToHost),
           "download");
  // Both routes quantize the activations per 32-wide block, so they agree
  // to accumulation order; the F64 reference over the dequantized weights
  // bounds the activation quantization itself.
  double worst_vs_mmq = 0.0;
  double worst_vs_ref = 0.0;
  double worst_f16 = 0.0;
  double ref_scale = 0.0;
  if (check) {
    for (std::size_t i = 0; i < w8.size(); ++i) {
      if (!std::isfinite(w8[i]) || !std::isfinite(f16[i])) {
        throw std::runtime_error("projection output is not finite");
      }
      worst_vs_mmq =
          std::max(worst_vs_mmq, std::abs(static_cast<double>(mmq[i] - w8[i])));
    }
    if (!q::W8A8Gemm(d_w, d_tiled, d_w8, batch, m, k, nullptr)) {
      throw std::runtime_error("W8A8 replay rejected the shape");
    }
    std::vector<float> replay(w8.size());
    CheckHip(hipMemcpy(replay.data(), d_w8, replay.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "replay download");
    if (replay != w8) {
      throw std::runtime_error("W8A8 replay changed the output");
    }
  }
  // Large production shapes still compare every output against MMQ. Sample
  // evenly spaced tokens for the more expensive independent F64 reference.
  const std::size_t samples =
      reference_tokens == 0 ? batch : std::min(batch, reference_tokens);
  for (std::size_t sample = 0; sample < (check ? samples : 0); ++sample) {
    const std::size_t t =
        samples > 1 ? sample * (batch - 1) / (samples - 1) : 0;
    for (std::size_t r = 0; r < m; ++r) {
      double ref = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(w.values[r * k + i]) * x[t * k + i];
      }
      const std::size_t idx = t * m + r;
      worst_vs_ref = std::max(worst_vs_ref, std::abs(ref - w8[idx]));
      worst_f16 = std::max(worst_f16, std::abs(ref - f16[idx]));
      ref_scale = std::max(ref_scale, std::abs(ref));
    }
  }
  std::cout << "W8A8 batch=" << batch << " m=" << m << " k=" << k
            << ": worst |W8A8 - MMQ| " << worst_vs_mmq
            << ", worst |W8A8 - F64| " << worst_vs_ref << ", worst |F16 - F64| "
            << worst_f16 << " (reference scale " << ref_scale << ")\n";
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_mmq);
  (void)hipFree(d_w8);
  (void)hipFree(d_tiled);
  (void)hipFree(d_f16);
  (void)hipFree(d_x_half);
  // The MMQ agreement is accumulation order; the F64 gap is the shared
  // 8-bit activation quantization, well under 1% of the output scale. The
  // F16 route's gap is its F16 activation rounding, a few ulps smaller.
  return worst_vs_mmq < 1e-3 ? std::max(worst_vs_ref, worst_f16) / ref_scale
                             : 1.0;
}

void CheckDecodeGrouping() {
  constexpr int rows = 64, cols = 2560, tokens = 8;
  const auto w = MakeWeights(rows, cols, 11);
  const auto gate = MakeWeights(rows, cols, 17);
  std::vector<float> x(tokens * cols);
  std::uint32_t seed = 37;
  for (auto& v : x)
    v = Uniform(&seed, 2.0F);
  void* dw = nullptr;
  void* dg = nullptr;
  float* dx = nullptr;
  void* dq = nullptr;
  float* out = nullptr;
  CheckHip(hipMalloc(&dw, w.blocks.size() + 4096), "decode weights");
  CheckHip(hipMalloc(&dg, gate.blocks.size() + 4096), "decode gate");
  CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "decode inputs");
  CheckHip(hipMalloc(&dq, qfn_mmq_q8_1_bytes(tokens, cols)),
           "decode quantized inputs");
  CheckHip(hipMalloc(&out, rows * tokens * sizeof(float)), "decode output");
  CheckHip(
      hipMemcpy(dw, w.blocks.data(), w.blocks.size(), hipMemcpyHostToDevice),
      "weights upload");
  CheckHip(hipMemcpy(dg, gate.blocks.data(), gate.blocks.size(),
                     hipMemcpyHostToDevice),
           "gate upload");
  CheckHip(
      hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "input upload");
  if (qfn_mmq_quantize_q8_1(dx, dq, tokens, cols, nullptr))
    throw std::runtime_error("decode input quantization failed");
  for (const bool gated : {false, true}) {
    for (int t = 0; t < tokens; ++t) {
      const auto* qrow = static_cast<const std::uint8_t*>(dq) +
                         t * qfn_mmq_q8_1_bytes(1, cols);
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, qrow,
                                      out + t * rows, rows, 1, cols, nullptr))
        throw std::runtime_error("scalar dense projection failed");
    }
    std::vector<float> scalar(tokens * rows), batch(tokens * rows);
    CheckHip(hipMemcpy(scalar.data(), out, scalar.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "scalar output");
    for (int n : {2, 3, 4, 8}) {
      if (qfn_mmq_q8_0_dense_vec_preq(dw, gated ? dg : nullptr, dq, out, rows,
                                      n, cols, nullptr))
        throw std::runtime_error("batched dense projection failed");
      CheckHip(hipMemcpy(batch.data(), out, n * rows * sizeof(float),
                         hipMemcpyDeviceToHost),
               "batch output");
      if (!std::equal(scalar.begin(), scalar.begin() + n * rows, batch.begin()))
        throw std::runtime_error("Q8 dense projection changed with grouping");
    }
  }
  for (void* ptr :
       {dw, dg, static_cast<void*>(dx), dq, static_cast<void*>(out)})
    CheckHip(hipFree(ptr), "decode test free");
}

}  // namespace

int main() {
  try {
    CheckDecodeGrouping();
    bool ok = true;
    // Ragged batch and rows against the 128-wide macro tiles, the 64-token
    // tile below 96, and the model's ssm_out / shexp_down widths.
    ok = Run(100, 320, 2560, 0x1234ABCDU) < 1e-2 && ok;
    ok = Run(37, 640, 2560, 0x0BADF00DU) < 1e-2 && ok;
    ok = Run(200, 200, 6144, 0xDEADBEEFU) < 1e-2 && ok;
    ok = Run(1025, 2560, 6144, 0x51A17U, true, 2) < 1e-2 && ok;
    // Production shapes for profiling only (QFN_W8A8_BENCH=1): three
    // launches each, no reference.
    if (std::getenv("QFN_W8A8_BENCH") != nullptr) {
      (void)Run(2048, 16384, 2560, 0x1111U, false);
      (void)Run(2048, 2560, 2560, 0x2222U, false);
      (void)Run(2048, 10240, 640, 0x3333U, false);
      (void)Run(2048, 320, 10240, 0x4444U, false);
      (void)Run(2048, 640, 2560, 0x5555U, false);
      (void)Run(2048, 2560, 640, 0x7777U, false);
      (void)Run(2048, 2560, 6144, 0x8888U, false);
      (void)Run(2048, 10240, 320, 0x9999U, false);
      (void)Run(2048, 13312, 2560, 0x6666U, false);
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
