#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
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

void TestBatchedSSMConvEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t qkv_dim = 8192;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i));
  }
  std::vector<float> h_weights(qkv_dim * 4);
  for (std::size_t i = 0; i < h_weights.size(); ++i) {
    h_weights[i] = 0.25F;
  }

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_seq = nullptr, *d_state_batch = nullptr;
  float *d_conv_out_seq = nullptr, *d_conv_out_batch = nullptr;
  float *d_delta_seq = nullptr, *d_delta_batch = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  const std::size_t delta_size = 32 * 128 * 128;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_seq, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_batch, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_seq, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_batch, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_seq, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_batch, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, 128 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * 4096 * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_seq, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_seq, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_batch, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMemset(d_ssm_a, 0, 32 * sizeof(float)));
  HIP_CHECK(hipMemset(d_ssm_dt, 0, 32 * sizeof(float)));
  HIP_CHECK(hipMemset(d_ssm_norm, 0, 128 * sizeof(float)));
  HIP_CHECK(hipMemset(d_gate, 0, batch * 4096 * sizeof(float)));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w, d_state_seq, d_conv_out_seq + t * qkv_dim,
        d_delta_seq, d_alpha + t * 32, d_beta + t * 32, d_ssm_a, d_ssm_dt,
        d_ssm_norm, d_gate + t * 4096, d_out_seq + t * 4096, 0, 32, 128, 128);
  }

  // Batched
  strix::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_batch, d_conv_out_batch, d_delta_batch, d_alpha,
      d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_batch, 0, batch, 32,
      128, 128);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * 4096), res_batch(batch * 4096);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, batch * 4096 * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * 4096 * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "DeltaNet Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

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
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 16;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 64;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;
  const std::size_t cache_size = 8 * num_kv_heads * max_context * head_dim * 2;

  std::vector<float> h_q(q_size, 0.1F);
  std::vector<float> h_k(kv_size, 0.1F);
  std::vector<float> h_v(kv_size, 0.2F);
  std::vector<float> h_gate(q_size, 0.5F);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_batch = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_batch, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, q_size * sizeof(float)));

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

  const std::size_t total_k = 8 * num_kv_heads * max_context * head_dim;

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  strix::hip::LaunchBatchedAttention(
      d_q, d_k, d_v, d_gate, d_cache_batch, d_cache_batch + total_k,
      d_out_batch, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(q_size), res_batch(q_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Attention Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_batch));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedFusedProjectionsEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 2560;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_qkv_w(8192 * hidden_size, 0.01F);
  std::vector<float> h_gate_w(4096 * hidden_size, 0.02F);
  std::vector<float> h_alpha_w(32 * hidden_size, 0.03F);
  std::vector<float> h_beta_w(32 * hidden_size, 0.04F);

  float *d_x = nullptr, *d_qkv_w = nullptr, *d_gate_w = nullptr,
        *d_alpha_w = nullptr, *d_beta_w = nullptr;
  float *d_qkv_seq = nullptr, *d_qkv_batch = nullptr;
  float *d_gate_seq = nullptr, *d_gate_batch = nullptr;
  float *d_alpha_seq = nullptr, *d_alpha_batch = nullptr;
  float *d_beta_seq = nullptr, *d_beta_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_w, 8192 * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w, 4096 * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_w, 32 * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_w, 32 * hidden_size * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv_seq, batch * 8192 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_batch, batch * 8192 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_seq, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_batch, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_seq, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_batch, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_seq, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_batch, batch * 32 * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_w, h_qkv_w.data(),
                      8192 * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      4096 * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha_w, h_alpha_w.data(),
                      32 * hidden_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta_w, h_beta_w.data(),
                      32 * hidden_size * sizeof(float), hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSSMInputProjections(
        d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false,
        d_x + t * hidden_size, d_qkv_seq + t * 8192, d_gate_seq + t * 4096,
        d_alpha_seq + t * 32, d_beta_seq + t * 32, hidden_size);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSSMInputProjections(
      d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false, d_x,
      d_qkv_batch, d_gate_batch, d_alpha_batch, d_beta_batch, batch,
      hidden_size);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * 8192), res_batch(batch * 8192);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_qkv_seq, batch * 8192 * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_qkv_batch,
                      batch * 8192 * sizeof(float), hipMemcpyDeviceToHost));

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

int main() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping GPU kernel tests.\n";
    return 0;
  }

  TestGpuRMSNorm();
  TestGpuResidualAdd();
  TestGpuGEMV();
  TestBatchedGEMM();
  TestHipblasGEMM();
  TestBatchedSSMConvEquivalence();
  TestBatchedAttentionEquivalence();
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
