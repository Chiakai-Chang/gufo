#include "src/models/qwen_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen_oracles.hpp"

namespace strix::models {

void ForwardEmbedding(std::uint32_t token_id, std::span<const float> token_embd,
                      std::size_t hidden_size,
                      std::span<float> hidden_out) noexcept {
  const std::size_t offset = static_cast<std::size_t>(token_id) * hidden_size;
  if (offset + hidden_size <= token_embd.size() &&
      hidden_out.size() >= hidden_size) {
    std::copy_n(token_embd.data() + offset, hidden_size, hidden_out.data());
  }
}

void ForwardRMSNorm(std::span<const float> x, std::span<const float> weight,
                    float eps, std::span<float> out) noexcept {
  qwen::ReferenceRMSNorm(x, weight, eps, out);
}

void ForwardRoPE(std::span<float> q, std::span<float> k,
                 std::uint32_t num_heads, std::uint32_t num_kv_heads,
                 std::uint32_t head_dim, std::uint32_t pos,
                 float rope_theta) noexcept {
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const auto q_slice =
        q.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
    qwen::ReferenceRoPE(q_slice, pos, rope_theta, q_slice);
  }
  for (std::uint32_t h = 0; h < num_kv_heads; ++h) {
    const auto k_slice =
        k.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
    qwen::ReferenceRoPE(k_slice, pos, rope_theta, k_slice);
  }
}

void ForwardAttention(std::span<const float> q, std::span<const float> k,
                      std::span<const float> v, std::span<const float> o_weight,
                      QwenKvCache& kv_cache, std::uint32_t layer_idx,
                      std::uint32_t pos, std::uint32_t num_heads,
                      std::uint32_t num_kv_heads, std::uint32_t head_dim,
                      std::span<float> attn_scores_scratch,
                      std::span<float> attn_out) noexcept {
  // Store new K, V into cache
  for (std::uint32_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
    const auto k_src =
        k.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);
    const auto v_src =
        v.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);

    const auto k_dst = kv_cache.GetKeySlice(layer_idx, kv_h, pos);
    const auto v_dst = kv_cache.GetValueSlice(layer_idx, kv_h, pos);

    std::ranges::copy(k_src, k_dst.begin());
    std::ranges::copy(v_src, v_dst.begin());
  }

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  std::vector<float> context(static_cast<std::size_t>(num_heads) * head_dim,
                             0.0F);

  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const std::uint32_t kv_h = h / group_size;
    const auto q_head =
        q.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);

    // Compute scores for 0..pos
    float max_score = -1e30F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto k_p =
          kv_cache.GetKeySlice(layer_idx, kv_h, static_cast<std::uint32_t>(p));
      double dot = 0.0;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q_head[d]) * static_cast<double>(k_p[d]);
      }
      const auto score = static_cast<float>(dot * scale);
      attn_scores_scratch[p] = score;
      max_score = std::max(score, max_score);
    }

    // Softmax
    double sum_exp = 0.0;
    for (std::size_t p = 0; p < seq_len; ++p) {
      sum_exp +=
          std::exp(static_cast<double>(attn_scores_scratch[p] - max_score));
    }
    const double inv_sum = 1.0 / sum_exp;
    for (std::size_t p = 0; p < seq_len; ++p) {
      attn_scores_scratch[p] = static_cast<float>(
          std::exp(static_cast<double>(attn_scores_scratch[p] - max_score)) *
          inv_sum);
    }

    // Weighted sum of values
    auto ctx_head = std::span<float>(
        &context[static_cast<std::size_t>(h) * head_dim], head_dim);
    std::ranges::fill(ctx_head, 0.0F);

    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto v_p = kv_cache.GetValueSlice(layer_idx, kv_h,
                                              static_cast<std::uint32_t>(p));
      const auto weight = static_cast<double>(attn_scores_scratch[p]);
      for (std::size_t d = 0; d < head_dim; ++d) {
        ctx_head[d] =
            static_cast<float>(static_cast<double>(ctx_head[d]) +
                               (weight * static_cast<double>(v_p[d])));
      }
    }
  }

  // Output projection: context @ o_weight -> attn_out
  const std::size_t hidden_size =
      static_cast<std::size_t>(num_heads) * head_dim;
  if (!o_weight.empty()) {
    qwen::ReferenceGEMV(o_weight, context, hidden_size, hidden_size, attn_out);
  } else {
    std::copy_n(context.data(), hidden_size, attn_out.data());
  }
}

void ForwardFFN(std::span<const float> x, std::span<const float> gate_weight,
                std::span<const float> up_weight,
                std::span<const float> down_weight, std::size_t hidden_size,
                std::size_t intermediate_size, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> ffn_out) noexcept {
  if (!gate_weight.empty()) {
    qwen::ReferenceGEMV(gate_weight, x, intermediate_size, hidden_size,
                        gate_scratch);
  }
  if (!up_weight.empty()) {
    qwen::ReferenceGEMV(up_weight, x, intermediate_size, hidden_size,
                        up_scratch);
  }

  qwen::ReferenceSwiGLU(gate_scratch, up_scratch, act_scratch);

  if (!down_weight.empty()) {
    qwen::ReferenceGEMV(down_weight, act_scratch, hidden_size,
                        intermediate_size, ffn_out);
  }
}

void ForwardLayer(std::span<float> hidden, const QwenLayerWeights& layer,
                  const core::ModelConfig& config, QwenKvCache& kv_cache,
                  std::uint32_t layer_idx, std::uint32_t pos,
                  QwenScratchArena& arena) noexcept {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t head_dim = config.head_dim;
  const std::size_t q_size =
      static_cast<std::size_t>(config.num_attention_heads) * head_dim;
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * head_dim;

  // 1. Attention Pre-RMSNorm
  ForwardRMSNorm(hidden, layer.attn_norm, 1e-6F, arena.normed);

  // 2. Q, K, V Projections
  if (!layer.attn_q.empty()) {
    qwen::ReferenceGEMV(layer.attn_q, arena.normed, q_size, hidden_size,
                        arena.q);
  }
  if (!layer.attn_k.empty()) {
    qwen::ReferenceGEMV(layer.attn_k, arena.normed, kv_size, hidden_size,
                        arena.k);
  }
  if (!layer.attn_v.empty()) {
    qwen::ReferenceGEMV(layer.attn_v, arena.normed, kv_size, hidden_size,
                        arena.v);
  }

  // 3. RoPE
  ForwardRoPE(arena.q, arena.k, config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, pos,
              config.rope_theta);

  // 4. Attention + KV cache
  ForwardAttention(arena.q, arena.k, arena.v, layer.attn_output, kv_cache,
                   layer_idx, pos, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   arena.attn_scores, arena.attn_out);

  // 5. Residual Add
  for (std::size_t i = 0; i < hidden_size; ++i) {
    hidden[i] += arena.attn_out[i];
  }

  // 6. FFN Pre-RMSNorm
  ForwardRMSNorm(hidden, layer.ffn_norm, 1e-6F, arena.normed);

  // 7. SwiGLU FFN
  ForwardFFN(arena.normed, layer.ffn_gate, layer.ffn_up, layer.ffn_down,
             hidden_size, config.intermediate_size, arena.mlp_gate,
             arena.mlp_up, arena.mlp_act, arena.mlp_out);

  // 8. Residual Add
  for (std::size_t i = 0; i < hidden_size; ++i) {
    hidden[i] += arena.mlp_out[i];
  }
}

void ForwardModel(std::uint32_t token_id, std::uint32_t pos,
                  const QwenModelWeights& weights, QwenKvCache& kv_cache,
                  QwenScratchArena& arena,
                  std::span<float> logits_out) noexcept {
  ForwardEmbedding(token_id, weights.token_embd, weights.config.hidden_size,
                   arena.hidden);

  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    ForwardLayer(arena.hidden, weights.layers[l], weights.config, kv_cache, l,
                 pos, arena);
  }

  ForwardRMSNorm(arena.hidden, weights.output_norm, 1e-6F, arena.normed);

  if (!weights.output.empty()) {
    qwen::ReferenceGEMV(weights.output, arena.normed, weights.config.vocab_size,
                        weights.config.hidden_size, logits_out);
  }
}

std::uint32_t GreedyArgmax(std::span<const float> logits) noexcept {
  if (logits.empty()) {
    return 0;
  }
  std::size_t best_idx = 0;
  float best_val = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return static_cast<std::uint32_t>(best_idx);
}

}  // namespace strix::models
