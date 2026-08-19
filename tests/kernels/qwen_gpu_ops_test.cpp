#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

static inline std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

void TestGpuRMSNorm() {
  const std::size_t dim = 256;
  std::vector<float> h_x(dim, 1.0F);
  std::vector<float> h_w(dim, 2.0F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_x, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_out, dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  // mean(x^2) = 1.0, rms = 1.0, out = (1.0 / 1.0) * 2.0 = 2.0
  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 2.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuResidualAdd() {
  const std::size_t dim = 128;
  std::vector<float> h_a(dim, 3.5F);
  std::vector<float> h_b(dim, 1.5F);
  std::vector<float> h_out(dim, 0.0F);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchResidualAdd(d_a, d_b, d_out, dim);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 5.0F) < 1e-5F);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuGEMV() {
  const std::size_t M = 4;
  const std::size_t K = 8;
  std::vector<float> h_A(M * K, 1.0F);  // all ones
  std::vector<float> h_x(K, 2.0F);      // all twos
  std::vector<float> h_y(M, 0.0F);

  float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A, false, d_x, d_y, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(
      hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Each row has K=8 ones * 2.0 = 16.0
  for (std::size_t m = 0; m < M; ++m) {
    assert(std::abs(h_y[m] - 16.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestBatchedGEMM() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;

  std::vector<float> h_A(M * K, 1.5F);
  std::vector<float> h_X(batch * K);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t k = 0; k < K; ++k) {
      h_X[b * K + k] = static_cast<float>(b + 1);
    }
  }
  std::vector<float> h_Y(batch * M, 0.0F);

  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));

  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchBatchedGEMM(d_A, false, d_X, d_Y, batch, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t b = 0; b < batch; ++b) {
    const float expected =
        static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
    for (std::size_t m = 0; m < M; ++m) {
      const float diff = std::abs(h_Y[b * M + m] - expected);
      if (diff > 1e-4F) {
        std::cerr << "BatchedGEMM mismatch at b=" << b << " m=" << m
                  << " got=" << h_Y[b * M + m] << " expected=" << expected
                  << "\n";
        assert(false);
      }
    }
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_X));
  HIP_CHECK(hipFree(d_Y));
}

void TestHipblasGEMM() {
  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));

  constexpr std::size_t batch = 4;
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;

  // Test FP32
  {
    std::vector<float> h_A(M * K, 1.5F);
    std::vector<float> h_X(batch * K);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t k = 0; k < K; ++k) {
        h_X[b * K + k] = static_cast<float>(b + 1);
      }
    }
    std::vector<float> h_Y(batch * M, 0.0F);

    float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                        hipMemcpyHostToDevice));

    strix::hip::LaunchHipblasGEMM(handle, d_A, false, d_X, d_Y, batch, M, K,
                                  nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                        hipMemcpyDeviceToHost));

    for (std::size_t b = 0; b < batch; ++b) {
      const float expected =
          static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
      for (std::size_t m = 0; m < M; ++m) {
        const float diff = std::abs(h_Y[b * M + m] - expected);
        if (diff >= 1e-4F) {
          std::cerr << "Mismatch in FP32 HipblasGEMM at b=" << b << " m=" << m
                    << "\n";
          std::abort();
        }
      }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));
  }

  // Test BF16
  {
    std::vector<std::uint16_t> h_A(M * K);
    for (std::size_t i = 0; i < M * K; ++i) {
      h_A[i] = FloatToBf16Bits(1.5F);
    }
    std::vector<float> h_X(batch * K);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t k = 0; k < K; ++k) {
        h_X[b * K + k] = static_cast<float>(b + 1);
      }
    }
    std::vector<float> h_Y(batch * M, 0.0F);

    void* d_A = nullptr;
    float *d_X = nullptr, *d_Y = nullptr;
    void* d_x_bf16 = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_X, batch * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y, batch * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x_bf16, batch * K * sizeof(std::uint16_t)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), batch * K * sizeof(float),
                        hipMemcpyHostToDevice));

    strix::hip::LaunchHipblasGEMM(handle, d_A, true, d_X, d_Y, batch, M, K,
                                  d_x_bf16);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, batch * M * sizeof(float),
                        hipMemcpyDeviceToHost));

    for (std::size_t b = 0; b < batch; ++b) {
      const float expected =
          static_cast<float>(b + 1) * 1.5F * static_cast<float>(K);
      for (std::size_t m = 0; m < M; ++m) {
        const float diff = std::abs(h_Y[b * M + m] - expected);
        if (diff >= 1e-2F) {
          std::cerr << "Mismatch in BF16 HipblasGEMM at b=" << b << " m=" << m
                    << " got=" << h_Y[b * M + m] << " expected=" << expected
                    << "\n";
          std::abort();
        }
      }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));
    HIP_CHECK(hipFree(d_x_bf16));
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
}

void TestHipblasLtGEMM() {
  constexpr std::size_t batch = 32;
  constexpr std::size_t m = 256;
  constexpr std::size_t k = 64;

  std::vector<std::uint16_t> h_a(m * k, FloatToBf16Bits(1.5F));
  std::vector<std::uint16_t> h_x(batch * k);
  for (std::size_t b = 0; b < batch; ++b) {
    const auto value = FloatToBf16Bits(static_cast<float>(b + 1));
    std::fill_n(h_x.begin() + static_cast<std::ptrdiff_t>(b * k), k, value);
  }
  std::vector<float> h_y(batch * m, 0.0F);

  void* d_a = nullptr;
  void* d_x = nullptr;
  float* d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::HipblasLtGemm gemm;
  if (!gemm.RunBf16(d_a, d_x, d_y, batch, m, k)) {
    std::cerr << "hipBLASLt did not return a supported BF16 GEMM plan\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t b = 0; b < batch; ++b) {
    const float expected =
        static_cast<float>(b + 1) * 1.5F * static_cast<float>(k);
    for (std::size_t row = 0; row < m; ++row) {
      if (std::abs(h_y[b * m + row] - expected) >= 1e-2F) {
        std::cerr << "hipBLASLt mismatch at batch=" << b << " row=" << row
                  << " got=" << h_y[b * m + row] << " expected=" << expected
                  << '\n';
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_a));
}

void TestBatchedSSMConvEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim, 1.0F);
  std::vector<float> h_gate(batch * inner_size, 0.5F);

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_seq = nullptr, *d_state_batch = nullptr;
  float *d_conv_out_seq = nullptr, *d_conv_out_batch = nullptr;
  float *d_delta_seq = nullptr, *d_delta_batch = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_seq, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_batch, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_seq, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_batch, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_seq, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_batch, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_seq, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_seq, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_batch, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w, d_state_seq, d_conv_out_seq + t * qkv_dim,
        d_delta_seq, d_alpha + t * num_heads, d_beta + t * num_heads, d_ssm_a,
        d_ssm_dt, d_ssm_norm, d_gate + t * inner_size,
        d_out_seq + t * inner_size, 0, qkv_dim, num_key_heads, num_heads,
        key_dim, val_dim);
  }

  // Batched
  strix::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_batch, d_conv_out_batch, d_delta_batch, d_alpha,
      d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_batch, 0, batch,
      qkv_dim, num_key_heads, num_heads, key_dim, val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * inner_size);
  std::vector<float> res_batch(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "DeltaNet Seq vs Batch max diff: " << max_diff << "\n";
  if (max_diff >= 1e-4F) {
    std::cerr << "DeltaNet batched recurrence mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_seq));
  HIP_CHECK(hipFree(d_state_batch));
  HIP_CHECK(hipFree(d_conv_out_seq));
  HIP_CHECK(hipFree(d_conv_out_batch));
  HIP_CHECK(hipFree(d_delta_seq));
  HIP_CHECK(hipFree(d_delta_batch));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedAttentionEquivalence() {
  constexpr std::size_t batch = 2048;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 32;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;
  const std::size_t cache_size = 8 * num_kv_heads * max_context * head_dim * 2;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] = 0.08F * std::sin(static_cast<float>(index % 257) * 0.07F);
    h_gate[index] = 0.4F * std::cos(static_cast<float>(index % 193) * 0.05F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] = 0.07F * std::cos(static_cast<float>(index % 251) * 0.06F);
    h_v[index] = 0.2F * std::sin(static_cast<float>(index % 239) * 0.04F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_batch = nullptr,
        *d_cache_gemm = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr, *d_out_gemm = nullptr;
  float* d_scores = nullptr;
  hipblasHandle_t hipblas_handle = nullptr;

  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_batch, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_gemm, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_gemm, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scores, (num_heads / num_kv_heads) * batch *
                                     max_context * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_batch, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_gemm, 0, cache_size * sizeof(float)));

  const std::size_t total_k = 8 * num_kv_heads * max_context * head_dim;

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k, nullptr, nullptr,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  strix::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache_batch,
                                     d_cache_batch + total_k, nullptr, nullptr,
                                     d_out_batch, 0, 0, batch, max_context,
                                     num_heads, num_kv_heads, head_dim);
  strix::hip::LaunchBatchedAttentionGemm(
      hipblas_handle, d_q, d_k, d_v, d_gate, d_cache_gemm,
      d_cache_gemm + total_k, nullptr, nullptr, d_scores, d_out_gemm, 0, 0,
      batch, max_context, num_heads, num_kv_heads, head_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(q_size), res_batch(q_size), res_gemm(q_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_gemm.data(), d_out_gemm, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  float max_gemm_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
    const float gemm_diff = std::abs(res_seq[i] - res_gemm[i]);
    if (gemm_diff > max_gemm_diff)
      max_gemm_diff = gemm_diff;
  }
  std::cout << "Attention Seq vs Batch max diff: " << max_diff << "\n";
  std::cout << "Attention Seq vs GEMM max diff: " << max_gemm_diff << "\n";
  if (max_diff >= 1e-4F || max_gemm_diff >= 1e-4F) {
    std::cerr << "Batched attention mismatch\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_batch));
  HIP_CHECK(hipFree(d_cache_gemm));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_gemm));
  HIP_CHECK(hipFree(d_scores));
}

void TestAttentionBackendEquivalence() {
  constexpr std::size_t batch = 128;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 128;
  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = batch * attention_width;
  const std::size_t kv_size = batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_tile = nullptr, *d_cache_ck = nullptr;
  float *d_out_seq = nullptr, *d_out_tile = nullptr, *d_out_ck = nullptr;
  void *d_cache_tile_f16 = nullptr, *d_cache_ck_f16 = nullptr,
       *d_scratch_ck_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_tile, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_ck, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_cache_tile_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_cache_ck_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_scratch_ck_f16, 2 * q_size * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_tile, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ck, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_ck, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMemset(d_cache_ck_f16, 0, 2 * cache_elements * sizeof(hip_bfloat16)));

  for (std::size_t token = 0; token < batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }
  const bool tile_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!tile_launched) {
    std::cerr << "Tiled attention rejected the Qwen shape\n";
    std::abort();
  }
  const bool launched = strix::hip::LaunchBatchedAttentionCk(
      d_q, d_k, d_v, d_gate, d_cache_ck, d_cache_ck + cache_elements,
      d_cache_ck_f16,
      static_cast<std::uint16_t*>(d_cache_ck_f16) + cache_elements,
      d_scratch_ck_f16, d_out_ck, 0, 0, batch, max_context, num_heads,
      num_kv_heads, head_dim);
  if (!launched) {
    std::cerr << "Composable Kernel attention rejected the Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> sequential(q_size), tiled(q_size), ck(q_size);
  std::vector<float> sequential_cache(2 * cache_elements);
  std::vector<float> tile_cache(2 * cache_elements);
  std::vector<float> ck_cache(2 * cache_elements);
  HIP_CHECK(hipMemcpy(sequential.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck.data(), d_out_ck, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sequential_cache.data(), d_cache_seq,
                      sequential_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck_cache.data(), d_cache_ck,
                      ck_cache.size() * sizeof(float), hipMemcpyDeviceToHost));
  float max_tile_diff = 0.0F;
  float max_ck_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_tile_diff =
        std::max(max_tile_diff, std::abs(sequential[index] - tiled[index]));
    max_ck_diff =
        std::max(max_ck_diff, std::abs(sequential[index] - ck[index]));
  }
  float max_tile_cache_diff = 0.0F;
  float max_cache_diff = 0.0F;
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_tile_cache_diff =
        std::max(max_tile_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
    max_cache_diff = std::max(
        max_cache_diff, std::abs(sequential_cache[index] - ck_cache[index]));
  }
  std::cout << "Attention Seq vs tile max diff: " << max_tile_diff << "\n";
  std::cout << "Attention Seq vs tile cache max diff: " << max_tile_cache_diff
            << "\n";
  std::cout << "Attention Seq vs CK max diff: " << max_ck_diff << "\n";
  std::cout << "Attention Seq vs CK cache max diff: " << max_cache_diff << "\n";
  if (max_tile_diff >= 5e-3F || max_tile_cache_diff != 0.0F) {
    std::cerr << "Tiled attention mismatch\n";
    std::abort();
  }
  if (max_ck_diff >= 2e-3F || max_cache_diff != 0.0F) {
    std::cerr << "Composable Kernel attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_out_tile, 0, q_size * sizeof(float)));
  constexpr std::size_t chunk_size = batch / 2;
  const bool first_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, chunk_size, max_context, num_heads, num_kv_heads,
      head_dim);
  const bool second_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk_size * attention_width, d_k + chunk_size * kv_width,
      d_v + chunk_size * kv_width, d_gate + chunk_size * attention_width,
      d_cache_tile, d_cache_tile + cache_elements, d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile + chunk_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk_size), chunk_size, max_context,
      num_heads, num_kv_heads, head_dim);
  if (!first_chunk_launched || !second_chunk_launched) {
    std::cerr << "Tiled attention rejected a chunked Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_chunked_diff = 0.0F;
  float max_chunked_cache_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_chunked_diff =
        std::max(max_chunked_diff, std::abs(sequential[index] - tiled[index]));
  }
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_chunked_cache_diff =
        std::max(max_chunked_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
  }
  std::cout << "Attention Seq vs chunked tile max diff: " << max_chunked_diff
            << "\n";
  std::cout << "Attention Seq vs chunked tile cache max diff: "
            << max_chunked_cache_diff << "\n";
  if (max_chunked_diff >= 5e-3F || max_chunked_cache_diff != 0.0F) {
    std::cerr << "Chunked tiled attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_tile));
  HIP_CHECK(hipFree(d_cache_ck));
  HIP_CHECK(hipFree(d_cache_tile_f16));
  HIP_CHECK(hipFree(d_cache_ck_f16));
  HIP_CHECK(hipFree(d_scratch_ck_f16));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_tile));
  HIP_CHECK(hipFree(d_out_ck));
}

void TestLongContextDecodeAttention() {
  constexpr std::uint32_t position = 16384;
  constexpr std::uint32_t max_context = position + 129;
  constexpr std::uint32_t num_heads = 6;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 256;
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(num_kv_heads) * max_context * head_dim;

  std::vector<float> h_q(attention_width);
  std::vector<float> h_k(kv_width);
  std::vector<float> h_v(kv_width);
  std::vector<float> h_gate(attention_width);
  std::vector<float> h_cache(cache_elements);
  for (std::size_t index = 0; index < attention_width; ++index) {
    h_q[index] =
        0.1F * std::sin(static_cast<float>((index % 257) + 1) * 0.013F);
    h_gate[index] =
        0.4F * std::cos(static_cast<float>((index % 193) + 1) * 0.017F);
  }
  for (std::size_t index = 0; index < kv_width; ++index) {
    h_k[index] =
        0.08F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.2F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  for (std::size_t index = 0; index < cache_elements; ++index) {
    h_cache[index] =
        0.03F * std::sin(static_cast<float>((index % 509) + 1) * 0.011F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_k_cache = nullptr, *d_v_cache = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, attention_width * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_cache, h_cache.data(), cache_elements * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_cache, h_cache.data(), cache_elements * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchAttention(d_q, d_k, d_v, d_gate, d_k_cache, d_v_cache,
                              nullptr, nullptr, d_out, 0, position, max_context,
                              num_heads, num_kv_heads, head_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> output(attention_width);
  HIP_CHECK(hipMemcpy(output.data(), d_out, attention_width * sizeof(float),
                      hipMemcpyDeviceToHost));
  std::vector<float> reference(attention_width);
  std::vector<double> scores(static_cast<std::size_t>(position) + 1);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (std::uint32_t head = 0; head < num_heads; ++head) {
    const std::size_t query_offset = static_cast<std::size_t>(head) * head_dim;
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::uint32_t token = 0; token <= position; ++token) {
      double dot = 0.0;
      const std::size_t cache_offset =
          static_cast<std::size_t>(token) * head_dim;
      for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        const float key =
            token == position ? h_k[dim] : h_cache[cache_offset + dim];
        dot += static_cast<double>(h_q[query_offset + dim]) * key;
      }
      scores[token] = dot * scale;
      max_score = std::max(max_score, scores[token]);
    }

    double denominator = 0.0;
    for (double& score : scores) {
      score = std::exp(score - max_score);
      denominator += score;
    }
    for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
      double weighted_value = 0.0;
      for (std::uint32_t token = 0; token <= position; ++token) {
        const std::size_t cache_offset =
            static_cast<std::size_t>(token) * head_dim;
        const float value =
            token == position ? h_v[dim] : h_cache[cache_offset + dim];
        weighted_value += scores[token] * value;
      }
      const double sigmoid =
          1.0 / (1.0 + std::exp(-h_gate[query_offset + dim]));
      reference[query_offset + dim] =
          static_cast<float>((weighted_value / denominator) * sigmoid);
    }
  }

  bool has_nonzero = false;
  float max_reference_diff = 0.0F;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float value = output[index];
    max_reference_diff =
        std::max(max_reference_diff, std::abs(value - reference[index]));
    if (!std::isfinite(value)) {
      std::cerr << "Long-context decode attention produced non-finite output\n";
      std::abort();
    }
    has_nonzero = has_nonzero || std::abs(value) > 1e-8F;
  }
  std::cout << "Long-context decode attention max reference diff: "
            << max_reference_diff << "\n";
  if (max_reference_diff >= 5e-4F) {
    std::cerr << "Long-context decode attention mismatch\n";
    std::abort();
  }
  if (!has_nonzero) {
    std::cerr << "Long-context decode attention produced only zeroes\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_k_cache));
  HIP_CHECK(hipFree(d_v_cache));
  HIP_CHECK(hipFree(d_out));
}

void TestBatchedFusedProjectionsEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t qkv_size = 10240;
  constexpr std::size_t inner_size = 6144;
  constexpr std::size_t time_step_rank = 48;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_qkv_w(qkv_size * hidden_size, 0.01F);
  std::vector<float> h_gate_w(inner_size * hidden_size, 0.02F);
  std::vector<float> h_alpha_w(time_step_rank * hidden_size, 0.03F);
  std::vector<float> h_beta_w(time_step_rank * hidden_size, 0.04F);

  float *d_x = nullptr, *d_qkv_w = nullptr, *d_gate_w = nullptr,
        *d_alpha_w = nullptr, *d_beta_w = nullptr;
  float *d_qkv_seq = nullptr, *d_qkv_batch = nullptr;
  float *d_gate_seq = nullptr, *d_gate_batch = nullptr;
  float *d_alpha_seq = nullptr, *d_alpha_batch = nullptr;
  float *d_beta_seq = nullptr, *d_beta_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_w, qkv_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w, inner_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_alpha_w, time_step_rank * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_w, time_step_rank * hidden_size * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv_seq, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_batch, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_batch, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_batch, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_batch, batch * time_step_rank * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_w, h_qkv_w.data(),
                      qkv_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      inner_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha_w, h_alpha_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta_w, h_beta_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSSMInputProjections(
        d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false,
        d_x + t * hidden_size, d_qkv_seq + t * qkv_size,
        d_gate_seq + t * inner_size, d_alpha_seq + t * time_step_rank,
        d_beta_seq + t * time_step_rank, hidden_size, qkv_size, inner_size,
        time_step_rank);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSSMInputProjections(
      d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false, d_x,
      d_qkv_batch, d_gate_batch, d_alpha_batch, d_beta_batch, batch,
      hidden_size, qkv_size, inner_size, time_step_rank);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * qkv_size);
  std::vector<float> res_batch(batch * qkv_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_qkv_seq,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_qkv_batch,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SSM Projections Seq vs Batch max diff: " << max_diff
            << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_qkv_w));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_alpha_w));
  HIP_CHECK(hipFree(d_beta_w));
  HIP_CHECK(hipFree(d_qkv_seq));
  HIP_CHECK(hipFree(d_qkv_batch));
  HIP_CHECK(hipFree(d_gate_seq));
  HIP_CHECK(hipFree(d_gate_batch));
  HIP_CHECK(hipFree(d_alpha_seq));
  HIP_CHECK(hipFree(d_alpha_batch));
  HIP_CHECK(hipFree(d_beta_seq));
  HIP_CHECK(hipFree(d_beta_batch));
}

void TestBatchedFusedSwiGLUEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 2560;
  constexpr std::size_t intermediate_size = 9216;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_gate_w(intermediate_size * hidden_size, 0.01F);
  std::vector<float> h_up_w(intermediate_size * hidden_size, 0.02F);

  float *d_x = nullptr, *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_gate_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_up_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * intermediate_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSwiGLUGEMV(
        d_gate_w, false, d_up_w, false, d_x + t * hidden_size,
        d_out_seq + t * intermediate_size, intermediate_size, hidden_size);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSwiGLUGEMM(d_gate_w, false, d_up_w, false, d_x,
                                           d_out_batch, batch,
                                           intermediate_size, hidden_size);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * intermediate_size),
      res_batch(batch * intermediate_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SwiGLU Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedRoPEEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 16;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;

  std::vector<float> h_q(q_size), h_k(kv_size);
  for (std::size_t i = 0; i < q_size; ++i)
    h_q[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < kv_size; ++i)
    h_k[i] = std::cos(static_cast<float>(i + 1));

  float *d_q_seq = nullptr, *d_k_seq = nullptr;
  float *d_q_batch = nullptr, *d_k_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_q_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_seq, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_batch, kv_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_seq, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_seq, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_batch, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_batch, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchRoPE(d_q_seq + t * (num_heads * head_dim),
                           d_k_seq + t * (num_kv_heads * head_dim), num_heads,
                           num_kv_heads, head_dim, rotary_dim,
                           static_cast<std::uint32_t>(t), rope_theta);
  }

  // Batched
  strix::hip::LaunchBatchedRoPE(d_q_batch, d_k_batch, batch, num_heads,
                                num_kv_heads, head_dim, rotary_dim, 0,
                                rope_theta);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_seq(q_size), res_q_batch(q_size);
  std::vector<float> res_k_seq(kv_size), res_k_batch(kv_size);
  HIP_CHECK(hipMemcpy(res_q_seq.data(), d_q_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_batch.data(), d_q_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_seq.data(), d_k_seq, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_batch.data(), d_k_batch, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_q_diff = 0.0F, max_k_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    const float d = std::abs(res_q_seq[i] - res_q_batch[i]);
    if (d > max_q_diff)
      max_q_diff = d;
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    const float d = std::abs(res_k_seq[i] - res_k_batch[i]);
    if (d > max_k_diff)
      max_k_diff = d;
  }
  std::cout << "RoPE Seq vs Batch max Q diff: " << max_q_diff
            << " K diff: " << max_k_diff << "\n";
  assert(max_q_diff < 1e-4F);
  assert(max_k_diff < 1e-4F);

  HIP_CHECK(hipFree(d_q_seq));
  HIP_CHECK(hipFree(d_k_seq));
  HIP_CHECK(hipFree(d_q_batch));
  HIP_CHECK(hipFree(d_k_batch));
}

void TestBatchedPerHeadRMSNormEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr float eps = 1e-6F;

  const std::size_t total_size = batch * num_heads * head_dim;
  std::vector<float> h_x(total_size), h_w(head_dim);
  for (std::size_t i = 0; i < total_size; ++i)
    h_x[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < head_dim; ++i)
    h_w[i] = 1.0F + 0.1F * std::cos(static_cast<float>(i + 1));

  float *d_x_seq = nullptr, *d_x_batch = nullptr, *d_w = nullptr;
  HIP_CHECK(hipMalloc(&d_x_seq, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x_batch, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, head_dim * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x_seq, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x_batch, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));

  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchPerHeadRMSNorm(d_x_seq + t * (num_heads * head_dim), d_w,
                                     d_x_seq + t * (num_heads * head_dim),
                                     num_heads, head_dim, eps);
  }

  strix::hip::LaunchBatchedPerHeadRMSNorm(d_x_batch, d_w, d_x_batch, batch,
                                          num_heads, head_dim, eps);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(total_size), res_batch(total_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_x_seq, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_x_batch, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < total_size; ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "PerHeadRMSNorm Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x_seq));
  HIP_CHECK(hipFree(d_x_batch));
  HIP_CHECK(hipFree(d_w));
}

void TestBaselineToTiledKvCacheTransition() {
  constexpr std::size_t chunk0_size = 512;
  constexpr std::size_t chunk1_size = 1024;
  constexpr std::size_t total_batch = chunk0_size + chunk1_size;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = total_batch * attention_width;
  const std::size_t kv_size = total_batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);

  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_out_seq = nullptr;
  float *d_cache_trans = nullptr, *d_out_trans = nullptr;
  void* d_cache_trans_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_trans, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans_f16,
                      2 * cache_elements * sizeof(std::uint16_t)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_seq, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_trans, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans_f16, 0,
                      2 * cache_elements * sizeof(std::uint16_t)));

  // 1. Golden sequential reference token-by-token
  for (std::size_t token = 0; token < total_batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }

  // 2. Incremental transition:
  // Chunk 0 (0..512): LaunchBatchedAttention (Baseline)
  strix::hip::LaunchBatchedAttention(
      d_q, d_k, d_v, d_gate, d_cache_trans, d_cache_trans + cache_elements,
      d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans, 0, 0, chunk0_size, max_context, num_heads, num_kv_heads,
      head_dim);

  // Chunk 1 (512..1536): LaunchBatchedAttentionTile (Tiled FP16 at start_pos =
  // 512)
  const bool chunk1_ok = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk0_size * attention_width, d_k + chunk0_size * kv_width,
      d_v + chunk0_size * kv_width, d_gate + chunk0_size * attention_width,
      d_cache_trans, d_cache_trans + cache_elements, d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans + chunk0_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk0_size), chunk1_size, max_context,
      num_heads, num_kv_heads, head_dim);
  assert(chunk1_ok);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> golden(q_size);
  std::vector<float> transition(q_size);
  HIP_CHECK(hipMemcpy(golden.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(transition.data(), d_out_trans, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t token = 0; token < total_batch; ++token) {
    for (std::size_t i = 0; i < attention_width; ++i) {
      const float g = golden[token * attention_width + i];
      const float tr = transition[token * attention_width + i];
      const float diff = std::abs(g - tr);
      if (diff > 1e-3F || !std::isfinite(tr)) {
        std::cout << "Token " << token << " index " << i << " head "
                  << (i / head_dim) << " dim " << (i % head_dim)
                  << " golden=" << g << " trans=" << tr << " diff=" << diff
                  << "\n";
        goto done_diff;
      }
    }
  }
done_diff:
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_diff = std::max(max_diff, std::abs(golden[i] - transition[i]));
  }
  std::cout << "Baseline-to-Tiled transition max diff: " << max_diff << "\n";
  assert(max_diff < 5e-3F);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_cache_trans));
  HIP_CHECK(hipFree(d_out_trans));
  HIP_CHECK(hipFree(d_cache_trans_f16));
}

int main() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping GPU kernel tests.\n";
    return 0;
  }

  TestBaselineToTiledKvCacheTransition();
  TestGpuRMSNorm();
  TestGpuResidualAdd();
  TestGpuGEMV();
  TestBatchedGEMM();
  TestHipblasGEMM();
  TestHipblasLtGEMM();
  TestBatchedSSMConvEquivalence();
  TestBatchedAttentionEquivalence();
  TestAttentionBackendEquivalence();
  TestLongContextDecodeAttention();
  TestBatchedFusedProjectionsEquivalence();
  TestBatchedFusedSwiGLUEquivalence();
  TestBatchedRoPEEquivalence();
  TestBatchedPerHeadRMSNormEquivalence();
  std::cout << "All Qwen HIP GPU kernel tests passed on gfx1151.\n";
  return 0;
}
#else
int main() {
  std::cout << "HIP disabled, skipping GPU kernel tests.\n";
  return 0;
}
#endif
