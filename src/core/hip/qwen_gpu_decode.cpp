#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_executor.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

namespace strix::hip {
tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  if (pos >= arena_.GetMaxContext()) {
    throw std::length_error("token position exceeds the GPU context length");
  }

  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;

  // GPU parameter buffers
  const std::uint32_t* d_in_token = arena_.d_prompt_tokens + 0;
  const std::uint32_t* d_in_pos = arena_.d_prompt_tokens + 1;
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);

  // Copy token_id and pos to GPU device memory
  const std::uint32_t in_params[2] = {token_id, pos};
  HIP_CHECK(hipMemcpyAsync(arena_.d_prompt_tokens, in_params, sizeof(in_params),
                           hipMemcpyHostToDevice, arena_.stream));

  auto ExecuteStep = [&]() {
    // 1. Embedding lookup
    const bool embd_is_bf16 = weights_.token_embd.type == core::GgmlType::kBF16;
    LaunchEmbeddingLookup(weights_.token_embd.data, embd_is_bf16, d_in_token,
                          arena_.d_hidden, hidden_size, arena_.stream);

    // 2. Layer stack
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      const auto& layer = weights_.layers[l];

      // Pre-RMSNorm
      LaunchRMSNorm(arena_.d_hidden,
                    static_cast<const float*>(layer.attn_norm.data),
                    arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

      if (layer.is_full_attention) {
        // Full attention path
        const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
        const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
        const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
        const bool o_bf16 = layer.attn_output.type == core::GgmlType::kBF16;

        LaunchFusedQKVProjections(layer.attn_q.data, q_bf16, layer.attn_k.data,
                                  k_bf16, layer.attn_v.data, v_bf16,
                                  arena_.d_normed, arena_.d_ssm_qkv, arena_.d_k,
                                  arena_.d_v, q_projection_size, kv_size,
                                  hidden_size, arena_.stream);

        // De-interleave Q and Gate from attn_q projection
        LaunchUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                       config.num_attention_heads, config.head_dim,
                       arena_.stream);

        // QK-Norm
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena_.d_q, config.num_attention_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena_.d_k, config.num_key_value_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }

        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(arena_.d_q, arena_.d_k, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   config.rotary_dim, d_in_pos, config.rope_theta,
                   arena_.stream);

        // Softmax Attention + Gating
        const std::size_t total_k = config.FullAttentionLayerCount() *
                                    config.num_key_value_heads *
                                    arena_.GetMaxContext() * config.head_dim;
        const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
        LaunchAttention(
            arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
            arena_.d_kv_cache, arena_.d_kv_cache + total_k,
            arena_.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
            arena_.d_ssm_out, attn_layer_idx, d_in_pos, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena_.stream);

        // Output projection
        LaunchGEMV(layer.attn_output.data, o_bf16, arena_.d_ssm_out,
                   arena_.d_attn_out, hidden_size, attention_size,
                   arena_.stream);
      } else {
        // SSM path
        const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
        const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
        const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
        const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;
        const bool out_bf16 = layer.ssm_out.type == core::GgmlType::kBF16;

        LaunchFusedSSMInputProjections(
            layer.attn_qkv.data, qkv_bf16, layer.attn_gate.data, gate_bf16,
            layer.ssm_alpha.data, alpha_bf16, layer.ssm_beta.data, beta_bf16,
            arena_.d_normed, arena_.d_ssm_qkv, arena_.d_ssm_gate,
            arena_.d_alpha_buf, arena_.d_beta_buf, hidden_size, ssm_qkv_size,
            ssm_inner_size, time_step_rank, arena_.stream);

        LaunchSSMConvRecurrence(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, l, ssm_qkv_size, config.ssm_group_count,
            config.ssm_time_step_rank, config.ssm_state_size,
            config.SsmValueSize(), arena_.stream);

        LaunchGEMV(layer.ssm_out.data, out_bf16, arena_.d_ssm_out,
                   arena_.d_attn_out, hidden_size, ssm_inner_size,
                   arena_.stream);
      }

      // Residual Add
      LaunchResidualAdd(arena_.d_hidden, arena_.d_attn_out, arena_.d_hidden,
                        hidden_size, arena_.stream);

      // FFN Pre-RMSNorm
      LaunchRMSNorm(arena_.d_hidden,
                    static_cast<const float*>(layer.ffn_norm.data),
                    arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

      // Fused SwiGLU FFN
      const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
      const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;
      const bool ffn_d_bf16 = layer.ffn_down.type == core::GgmlType::kBF16;

      LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, ffn_g_bf16, layer.ffn_up.data,
                            ffn_u_bf16, arena_.d_normed, arena_.d_ffn_act,
                            intermediate_size, hidden_size, arena_.stream);

      LaunchGEMV(layer.ffn_down.data, ffn_d_bf16, arena_.d_ffn_act,
                 arena_.d_ffn_out, hidden_size, intermediate_size,
                 arena_.stream);

      // Residual Add
      LaunchResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                        hidden_size, arena_.stream);
    }

    if (compute_logits) {
      // 3. Final Output Norm
      LaunchRMSNorm(arena_.d_hidden,
                    static_cast<const float*>(weights_.output_norm.data),
                    arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

      // 4. LM Head Logits GEMV on final token
      const bool out_bf16 = weights_.output.type == core::GgmlType::kBF16;
      LaunchGEMV(weights_.output.data, out_bf16, arena_.d_normed,
                 arena_.d_logits, vocab_size, hidden_size, arena_.stream);

      // 5. Parallel GPU Argmax
      LaunchGPUArgmax(arena_.d_logits, d_out_token, vocab_size, arena_.stream);
    }
  };

  if (compute_logits && graph_executor_.IsEnabled()) {
    if (graph_executor_.IsCaptured()) {
      graph_executor_.Launch(arena_.stream);
    } else {
      const bool ok = graph_executor_.TryCapture(arena_.stream, ExecuteStep);
      if (ok) {
        graph_executor_.Launch(arena_.stream);
      } else {
        ExecuteStep();
      }
    }
  } else {
    ExecuteStep();
  }

  if (!compute_logits) {
    return 0;
  }

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
