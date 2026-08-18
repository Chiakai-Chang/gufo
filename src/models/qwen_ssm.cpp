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

QwenSsmCache::QwenSsmCache(std::uint32_t num_layers, std::size_t conv_channels,
                           std::size_t ssm_dim)
    : num_layers_(num_layers),
      conv_channels_(conv_channels),
      ssm_dim_(ssm_dim) {
  const std::size_t total_conv =
      static_cast<std::size_t>(num_layers_) * conv_channels_ * 4;
  const std::size_t total_ssm =
      static_cast<std::size_t>(num_layers_) * ssm_dim_;
  conv_states_.resize(total_conv, 0.0F);
  ssm_states_.resize(total_ssm, 0.0F);
}

void QwenSsmCache::Reset() noexcept {
  std::ranges::fill(conv_states_, 0.0F);
  std::ranges::fill(ssm_states_, 0.0F);
}

std::span<float> QwenSsmCache::GetConvState(std::uint32_t layer) noexcept {
  const std::size_t offset =
      static_cast<std::size_t>(layer) * conv_channels_ * 4;
  return {&conv_states_[offset], conv_channels_ * 4};
}

std::span<float> QwenSsmCache::GetSsmState(std::uint32_t layer) noexcept {
  const std::size_t offset = static_cast<std::size_t>(layer) * ssm_dim_;
  return {&ssm_states_[offset], ssm_dim_};
}

void ForwardSSM(std::span<const float> x_normed, const QwenLayerWeights& layer,
                QwenSsmCache& ssm_cache, std::uint32_t layer_idx,
                std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept {
  const std::size_t hidden_size = x_normed.size();
  const std::size_t qkv_dim = 8192;
  const std::size_t gate_dim = 4096;

  // 1. QKV and Gate Projections
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

  // 2. 1D Causal Convolution with rolling conv state
  auto conv_state = ssm_cache.GetConvState(layer_idx);
  std::vector<float> conv_out(qkv_dim, 0.0F);

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
        dot += conv_state[c_off + k] * layer.ssm_conv1d.Get(c_off + k);
      }
    } else {
      dot = ssm_qkv_scratch[c];
    }

    // SiLU activation: x / (1 + exp(-x))
    const float exp_neg = std::exp(-dot);
    conv_out[c] = dot / (1.0F + exp_neg);
  }

  // 3. DeltaNet State-Space Recurrence
  auto ssm_state = ssm_cache.GetSsmState(layer_idx);

  for (std::size_t i = 0; i < gate_dim; ++i) {
    const float u = conv_out[i];
    const float v = conv_out[i + gate_dim];

    // Recurrent decay parameter A
    float decay = 0.95F;
    if (!layer.ssm_a.empty()) {
      const float a_val = layer.ssm_a.Get(i % layer.ssm_a.num_elements);
      decay = std::exp(-std::abs(a_val));
    }

    // Update recurrent state
    ssm_state[i] = ssm_state[i] * decay + u * v;

    // Gated output: SiLU(gate) * ssm_state
    const float g = ssm_gate_scratch[i];
    const float g_act = g / (1.0F + std::exp(-g));
    ssm_out_scratch[i] = g_act * ssm_state[i];
  }

  // 4. Output RMSNorm
  if (!layer.ssm_norm.empty()) {
    ForwardRMSNorm(ssm_out_scratch.subspan(0, gate_dim), layer.ssm_norm, 1e-6F,
                   ssm_out_scratch.subspan(0, gate_dim));
  }

  // 5. Linear Output Projection
  if (!layer.ssm_out.empty()) {
    TensorGEMV(layer.ssm_out, ssm_out_scratch.subspan(0, gate_dim), hidden_size,
               gate_dim, out);
  } else {
    std::ranges::fill(out, 0.0F);
  }
}

}  // namespace strix::models
