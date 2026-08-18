#include "src/models/qwen_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "src/models/qwen_forward.hpp"

namespace strix::models {

QwenSsmCache::QwenSsmCache(std::uint32_t num_layers,
                           std::size_t conv_channels,
                           std::uint32_t num_heads,
                           std::uint32_t key_dim,
                           std::uint32_t val_dim)
    : num_layers_(num_layers),
      conv_channels_(conv_channels),
      num_heads_(num_heads),
      key_dim_(key_dim),
      val_dim_(val_dim) {
  const std::size_t total_conv =
      static_cast<std::size_t>(num_layers_) * conv_channels_ * 4;
  const std::size_t total_deltanet = static_cast<std::size_t>(num_layers_) *
                                     num_heads_ * key_dim_ * val_dim_;
  conv_states_.resize(total_conv, 0.0F);
  deltanet_states_.resize(total_deltanet, 0.0F);
}

void QwenSsmCache::Reset() noexcept {
  std::ranges::fill(conv_states_, 0.0F);
  std::ranges::fill(deltanet_states_, 0.0F);
}

std::span<float> QwenSsmCache::GetConvState(std::uint32_t layer) noexcept {
  const std::size_t offset =
      static_cast<std::size_t>(layer) * conv_channels_ * 4;
  return {&conv_states_[offset], conv_channels_ * 4};
}

std::span<float> QwenSsmCache::GetDeltaNetState(std::uint32_t layer,
                                                std::uint32_t head) noexcept {
  const std::size_t head_size =
      static_cast<std::size_t>(key_dim_) * static_cast<std::size_t>(val_dim_);
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * num_heads_ + head) * head_size;
  return {&deltanet_states_[offset], head_size};
}

namespace {

[[nodiscard]] inline float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

[[nodiscard]] inline float SiLU(float x) noexcept {
  return x * Sigmoid(x);
}

}  // namespace

void ForwardSSM(std::span<const float> x_normed,
                const QwenLayerWeights& layer,
                QwenSsmCache& ssm_cache,
                std::uint32_t layer_idx,
                std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept {
  const std::size_t hidden_size = x_normed.size();
  const std::size_t qkv_dim = 8192;
  const std::size_t gate_dim = 4096;
  const std::uint32_t num_heads = ssm_cache.NumHeads();  // 16
  const std::uint32_t key_dim = ssm_cache.KeyDim();      // 128
  const std::uint32_t val_dim = ssm_cache.ValDim();      // 256

  // 1. QKV, Gate, Alpha, and Beta Projections
  if (!layer.attn_qkv.empty()) {
    TensorGEMV(layer.attn_qkv, x_normed, qkv_dim, hidden_size, ssm_qkv_scratch);
  } else {
    std::ranges::fill(ssm_qkv_scratch, 0.0F);
  }

  if (!layer.attn_gate.empty()) {
    TensorGEMV(layer.attn_gate, x_normed, gate_dim, hidden_size,
               ssm_gate_scratch);
  } else {
    std::ranges::fill(ssm_gate_scratch, 0.0F);
  }

  std::vector<float> alpha_buf(32, 0.0F);
  std::vector<float> beta_buf(32, 0.0F);
  if (!layer.ssm_alpha.empty()) {
    TensorGEMV(layer.ssm_alpha, x_normed, 32, hidden_size, alpha_buf);
  }
  if (!layer.ssm_beta.empty()) {
    TensorGEMV(layer.ssm_beta, x_normed, 32, hidden_size, beta_buf);
  }

  // 2. 1D Causal Convolution with rolling conv state
  auto conv_state = ssm_cache.GetConvState(layer_idx);
  std::vector<float> conv_out(qkv_dim, 0.0F);

#pragma omp parallel for schedule(static)
  for (std::size_t c = 0; c < qkv_dim; ++c) {
    const std::size_t c_off = c * 4;
    // Shift history: [0, 1, 2] <- [1, 2, 3]
    conv_state[c_off + 0] = conv_state[c_off + 1];
    conv_state[c_off + 1] = conv_state[c_off + 2];
    conv_state[c_off + 2] = conv_state[c_off + 3];
    conv_state[c_off + 3] = ssm_qkv_scratch[c];

    // Compute convolution
    float dot = 0.0F;
    if (!layer.ssm_conv1d.empty()) {
      for (std::size_t k = 0; k < 4; ++k) {
        dot += conv_state[c_off + k] * layer.ssm_conv1d.Get(c * 4 + k);
      }
    } else {
      dot = ssm_qkv_scratch[c];
    }

    conv_out[c] = SiLU(dot);
  }

  // 3. Partition Q, K, V from conv_out
  const float* q_ptr = conv_out.data();
  const float* k_ptr = conv_out.data() + (num_heads * key_dim);
  const float* v_ptr = conv_out.data() + (num_heads * key_dim * 2);

  // 4. Per-Head Gated DeltaNet Matrix Recurrence
#pragma omp parallel for schedule(static)
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    auto s_matrix = ssm_cache.GetDeltaNetState(layer_idx, h);
    const float* q_h = q_ptr + (h * key_dim);
    const float* k_h = k_ptr + (h * key_dim);
    const float* v_h = v_ptr + (h * val_dim);
    float* o_h = ssm_out_scratch.data() + (h * val_dim);

    // Compute decay alpha_h and learning rate beta_h
    float dt = 0.0F;
    if (!layer.ssm_dt.empty()) {
      dt = layer.ssm_dt.Get(h % layer.ssm_dt.num_elements);
    }
    float a_val = -0.05F;
    if (!layer.ssm_a.empty()) {
      a_val = layer.ssm_a.Get(h % layer.ssm_a.num_elements);
    }
    const float alpha_h = Sigmoid(alpha_buf[h] + dt) * std::exp(-std::abs(a_val));
    const float beta_h = Sigmoid(beta_buf[h]);

    // Normalize key k_h
    float k_norm_sq = 0.0F;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      k_norm_sq += k_h[i] * k_h[i];
    }
    const float inv_k_norm = 1.0F / std::sqrt(k_norm_sq + 1e-6F);

    // Associative memory retrieval: u = S^T * k
    std::vector<float> u_h(val_dim, 0.0F);
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      const float k_val = k_h[i] * inv_k_norm;
      const float* s_row = &s_matrix[i * val_dim];
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        u_h[j] += s_row[j] * k_val;
      }
    }

    // Prediction error: error = v - u
    std::vector<float> err_h(val_dim, 0.0F);
    for (std::uint32_t j = 0; j < val_dim; ++j) {
      err_h[j] = v_h[j] - u_h[j];
    }

    // Delta update: S = alpha * S + beta * (k * err^T)
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      const float k_val = k_h[i] * inv_k_norm;
      float* s_row = &s_matrix[i * val_dim];
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        s_row[j] = (alpha_h * s_row[j]) + (beta_h * k_val * err_h[j]);
      }
    }

    // Query readout: o = S^T * q
    for (std::uint32_t j = 0; j < val_dim; ++j) {
      o_h[j] = 0.0F;
    }
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      const float q_val = q_h[i];
      const float* s_row = &s_matrix[i * val_dim];
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        o_h[j] += s_row[j] * q_val;
      }
    }
  }

  // 5. Per-Head Output RMSNorm (128 elements per head norm)
  if (!layer.ssm_norm.empty()) {
    const std::size_t norm_dim = layer.ssm_norm.num_elements;
    const std::size_t num_norm_heads = gate_dim / norm_dim;
    for (std::size_t h = 0; h < num_norm_heads; ++h) {
      auto slice = ssm_out_scratch.subspan(h * norm_dim, norm_dim);
      ForwardRMSNorm(slice, layer.ssm_norm, 1e-6F, slice);
    }
  }

  // 6. Gating: y = o * SiLU(gate)
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < gate_dim; ++i) {
    ssm_out_scratch[i] *= SiLU(ssm_gate_scratch[i]);
  }

  // 7. Linear Output Projection
  if (!layer.ssm_out.empty()) {
    TensorGEMV(layer.ssm_out, ssm_out_scratch.subspan(0, gate_dim), hidden_size,
               gate_dim, out);
  } else {
    std::ranges::fill(out, 0.0F);
  }
}

}  // namespace strix::models
