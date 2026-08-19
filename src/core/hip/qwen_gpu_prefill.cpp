#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "src/core/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_executor.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

namespace strix::hip {
tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  if (prompt_tokens.empty()) {
    return 0;
  }
  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    next_token =
        ForwardPromptChunk(prompt_tokens.subspan(offset, chunk_size),
                           start_pos + static_cast<std::uint32_t>(offset),
                           compute_logits && is_last);
  }
  return next_token;
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  const bool embd_is_bf16 = weights_.token_embd.type == core::GgmlType::kBF16;
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data, embd_is_bf16,
                               arena_.d_prompt_tokens, arena_.d_hidden,
                               batch_size, hidden_size, arena_.stream);

  const bool do_profile = (std::getenv("STRIX_PROFILE") != nullptr);
  auto t_start = std::chrono::high_resolution_clock::now();
  double time_attn_proj = 0, time_ssm_recur = 0, time_ssm_out = 0;
  double time_ffn = 0, time_norm = 0;

  constexpr std::size_t hipblaslt_min_dimension = 1024;
  const auto launch_bf16_gemm = [&](const void* weights, const void* input,
                                    float* output, std::size_t m,
                                    std::size_t k) {
    const bool use_hipblaslt =
        m >= hipblaslt_min_dimension && k >= hipblaslt_min_dimension &&
        arena_.hipblaslt_gemm->RunBf16(weights, input, output, batch_size, m, k,
                                       arena_.stream);
    if (!use_hipblaslt) {
      LaunchHipblasGEMMBF16(arena_.hipblas_handle, weights, input, output,
                            batch_size, m, k, arena_.stream);
    }
  };

  // 3. Layer stack across all 32 layers
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_norm += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }

    if (layer.is_full_attention) {
      const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
      const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
      const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
      const bool o_bf16 = layer.attn_output.type == core::GgmlType::kBF16;

      if (q_bf16) {
        launch_bf16_gemm(layer.attn_q.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_qkv, q_projection_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_q.data, false,
                          arena_.d_normed, arena_.d_ssm_qkv, batch_size,
                          q_projection_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (k_bf16) {
        launch_bf16_gemm(layer.attn_k.data, arena_.d_scratch_bf16, arena_.d_k,
                         kv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_k.data, false,
                          arena_.d_normed, arena_.d_k, batch_size, kv_size,
                          hidden_size, arena_.d_scratch_bf16, arena_.stream);
      }

      if (v_bf16) {
        launch_bf16_gemm(layer.attn_v.data, arena_.d_scratch_bf16, arena_.d_v,
                         kv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_v.data, false,
                          arena_.d_normed, arena_.d_v, batch_size, kv_size,
                          hidden_size, arena_.d_scratch_bf16, arena_.stream);
      }

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      if (!layer.attn_q_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
            arena_.d_q, batch_size, config.num_attention_heads, config.head_dim,
            eps, arena_.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
            arena_.d_k, batch_size, config.num_key_value_heads, config.head_dim,
            eps, arena_.stream);
      }

      LaunchBatchedRoPE(arena_.d_q, arena_.d_k, batch_size,
                        config.num_attention_heads, config.num_key_value_heads,
                        config.head_dim, config.rotary_dim, start_pos,
                        config.rope_theta, arena_.stream);

      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena_.GetMaxContext() * config.head_dim;
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
      const std::size_t visible_context =
          static_cast<std::size_t>(start_pos) + batch_size;
      detail::DispatchPrefillAttention(
          visible_context,
          [&] {
            return LaunchBatchedAttentionTile(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_attention_kv_f16,
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
                    total_k,
                arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream);
          },
          [&] {
            return LaunchBatchedAttentionCk(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_attention_kv_f16,
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
                    total_k,
                arena_.d_scratch_bf16, arena_.d_ssm_out, attn_layer_idx,
                start_pos, batch_size, arena_.GetMaxContext(),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream);
          },
          [&] {
            LaunchBatchedAttention(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream);
          });

      if (o_bf16) {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        launch_bf16_gemm(layer.attn_output.data, arena_.d_scratch_bf16,
                         arena_.d_attn_out, hidden_size, attention_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_output.data, false,
                          arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                          hidden_size, attention_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    } else {
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;
      const bool out_bf16 = layer.ssm_out.type == core::GgmlType::kBF16;

      if (qkv_bf16) {
        launch_bf16_gemm(layer.attn_qkv.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_qkv, ssm_qkv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_qkv.data, false,
                          arena_.d_normed, arena_.d_ssm_qkv, batch_size,
                          ssm_qkv_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (gate_bf16) {
        launch_bf16_gemm(layer.attn_gate.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_gate, ssm_inner_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_gate.data, false,
                          arena_.d_normed, arena_.d_ssm_gate, batch_size,
                          ssm_inner_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (alpha_bf16) {
        LaunchHipblasGEMMBF16(arena_.hipblas_handle, layer.ssm_alpha.data,
                              arena_.d_scratch_bf16, arena_.d_alpha_buf,
                              batch_size, time_step_rank, hidden_size,
                              arena_.stream);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_alpha.data, false,
                          arena_.d_normed, arena_.d_alpha_buf, batch_size,
                          time_step_rank, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (beta_bf16) {
        LaunchHipblasGEMMBF16(arena_.hipblas_handle, layer.ssm_beta.data,
                              arena_.d_scratch_bf16, arena_.d_beta_buf,
                              batch_size, time_step_rank, hidden_size,
                              arena_.stream);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_beta.data, false,
                          arena_.d_normed, arena_.d_beta_buf, batch_size,
                          time_step_rank, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      LaunchBatchedSSMConvRecurrence(
          arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, arena_.d_conv_out,
          arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
          arena_.d_ssm_out, l, batch_size, ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena_.stream);

      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_recur +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      if (out_bf16) {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        launch_bf16_gemm(layer.ssm_out.data, arena_.d_scratch_bf16,
                         arena_.d_attn_out, hidden_size, ssm_inner_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_out.data, false,
                          arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                          hidden_size, ssm_inner_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_out +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    }

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch_size, hidden_size,
                             arena_.stream);

    // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);

    const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
    const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;
    const bool ffn_d_bf16 = layer.ffn_down.type == core::GgmlType::kBF16;

    if (ffn_g_bf16) {
      launch_bf16_gemm(layer.ffn_gate.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_gate, intermediate_size, hidden_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_gate.data, false,
                        arena_.d_normed, arena_.d_ffn_gate, batch_size,
                        intermediate_size, hidden_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    if (ffn_u_bf16) {
      launch_bf16_gemm(layer.ffn_up.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_up, intermediate_size, hidden_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_up.data, false,
                        arena_.d_normed, arena_.d_ffn_up, batch_size,
                        intermediate_size, hidden_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    LaunchBatchedSwiGLUActivation(
        arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
        arena_.d_scratch_bf16, batch_size * intermediate_size, arena_.stream);

    if (ffn_d_bf16) {
      launch_bf16_gemm(layer.ffn_down.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_out, hidden_size, intermediate_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_down.data, false,
                        arena_.d_ffn_act, arena_.d_ffn_out, batch_size,
                        hidden_size, intermediate_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                             batch_size, hidden_size, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_ffn += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }
  }

  if (do_profile) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\n[STRIX_PROFILE B=" << batch_size << "] Total: " << total_ms
              << " ms (" << (batch_size / (total_ms / 1000.0)) << " tok/s)\n"
              << "  - Norms:      " << time_norm << " ms\n"
              << "  - Input Proj: " << time_attn_proj << " ms\n"
              << "  - SSM Recur:  " << time_ssm_recur << " ms\n"
              << "  - SSM Out:    " << time_ssm_out << " ms\n"
              << "  - FFN (3 GEMM): " << time_ffn << " ms\n";
  }

  if (!compute_logits) {
    return 0;
  }

  // 4. Output Norm for final token
  const float* final_hidden =
      arena_.d_hidden + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  const bool out_bf16 = weights_.output.type == core::GgmlType::kBF16;
  LaunchGEMV(weights_.output.data, out_bf16, arena_.d_normed, arena_.d_logits,
             vocab_size, hidden_size, arena_.stream);

  // 6. GPU Argmax
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);
  LaunchGPUArgmax(arena_.d_logits, d_out_token, vocab_size, arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
