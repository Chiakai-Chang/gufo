#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/qwen_gpu_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen_oracles.hpp"

namespace strix::hip {

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context)
    : config_(config), max_context_(std::min(max_context, 4096U)) {
  HIP_CHECK(hipStreamCreate(&stream));
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));

  const std::size_t hidden_size = config_.hidden_size;
  const std::size_t intermediate_size = config_.intermediate_size;
  const std::size_t vocab_size = config_.vocab_size;
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t batch = max_batch_;

  HIP_CHECK(hipMalloc(&d_hidden, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, batch * 8192 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch * 1024 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch * 1024 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_attn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_act, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_qkv, batch * 8192 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out, batch * 8192 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_gate, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_out, batch * 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_buf, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_buf, batch * 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_prompt_tokens, batch * sizeof(std::uint32_t)));

  const std::size_t scratch_elements =
      batch *
      std::max<std::size_t>({intermediate_size, hidden_size, 8192, 12352});
  HIP_CHECK(
      hipMalloc(&d_scratch_bf16, scratch_elements * sizeof(hip_bfloat16)));

  // 8 full-attention layers in Qwen 3.5
  const std::size_t total_kv = 8 * num_kv_heads * max_context_ * head_dim;
  HIP_CHECK(hipMalloc(&d_kv_cache, total_kv * sizeof(float) * 2));

  const std::size_t total_conv = num_layers * 8192 * 4;
  HIP_CHECK(hipMalloc(&d_ssm_conv_state, total_conv * sizeof(float)));

  const std::size_t total_deltanet = num_layers * 16 * 128 * 256;
  HIP_CHECK(hipMalloc(&d_ssm_deltanet_state, total_deltanet * sizeof(float)));

  Reset();
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_logits = other.d_logits;
  d_kv_cache = other.d_kv_cache;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  stream = other.stream;
  hipblas_handle = other.hipblas_handle;
  d_scratch_bf16 = other.d_scratch_bf16;

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_logits = nullptr;
  other.d_kv_cache = nullptr;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.stream = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_logits = other.d_logits;
    d_kv_cache = other.d_kv_cache;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    stream = other.stream;
    hipblas_handle = other.hipblas_handle;
    d_scratch_bf16 = other.d_scratch_bf16;

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_logits = nullptr;
    other.d_kv_cache = nullptr;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.stream = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
  }
  return *this;
}

void QwenGpuArena::Reset() noexcept {
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t total_kv = 8 * num_kv_heads * max_context_ * head_dim * 2;
  const std::size_t total_conv = num_layers * 8192 * 4;
  const std::size_t total_deltanet = num_layers * 16 * 128 * 256;

  if (d_kv_cache != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_kv_cache, 0, total_kv * sizeof(float), stream));
  }
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_deltanet_state, 0,
                             total_deltanet * sizeof(float), stream));
  }
}

void QwenGpuArena::FreeAll() noexcept {
  if (d_hidden != nullptr)
    HIP_CHECK(hipFree(d_hidden));
  if (d_normed != nullptr)
    HIP_CHECK(hipFree(d_normed));
  if (d_q != nullptr)
    HIP_CHECK(hipFree(d_q));
  if (d_k != nullptr)
    HIP_CHECK(hipFree(d_k));
  if (d_v != nullptr)
    HIP_CHECK(hipFree(d_v));
  if (d_attn_out != nullptr)
    HIP_CHECK(hipFree(d_attn_out));
  if (d_ffn_gate != nullptr)
    HIP_CHECK(hipFree(d_ffn_gate));
  if (d_ffn_up != nullptr)
    HIP_CHECK(hipFree(d_ffn_up));
  if (d_ffn_act != nullptr)
    HIP_CHECK(hipFree(d_ffn_act));
  if (d_ffn_out != nullptr)
    HIP_CHECK(hipFree(d_ffn_out));
  if (d_ssm_qkv != nullptr)
    HIP_CHECK(hipFree(d_ssm_qkv));
  if (d_conv_out != nullptr)
    HIP_CHECK(hipFree(d_conv_out));
  if (d_ssm_gate != nullptr)
    HIP_CHECK(hipFree(d_ssm_gate));
  if (d_ssm_out != nullptr)
    HIP_CHECK(hipFree(d_ssm_out));
  if (d_alpha_buf != nullptr)
    HIP_CHECK(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    HIP_CHECK(hipFree(d_beta_buf));
  if (d_logits != nullptr)
    HIP_CHECK(hipFree(d_logits));
  if (d_kv_cache != nullptr)
    HIP_CHECK(hipFree(d_kv_cache));
  if (d_ssm_conv_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_conv_state));
  if (d_ssm_deltanet_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_deltanet_state));
  if (d_prompt_tokens != nullptr)
    HIP_CHECK(hipFree(d_prompt_tokens));
  if (d_scratch_bf16 != nullptr)
    HIP_CHECK(hipFree(d_scratch_bf16));
  if (hipblas_handle != nullptr)
    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  if (stream != nullptr)
    HIP_CHECK(hipStreamDestroy(stream));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_logits = nullptr;
  d_kv_cache = nullptr;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  hipblas_handle = nullptr;
  stream = nullptr;
}

QwenGpuExecutor::QwenGpuExecutor(
    models::QwenModelWeights weights,
    std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
    void* d_model_weights, std::uint32_t max_context)
    : weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      d_model_weights_(d_model_weights),
      arena_(weights_.config, max_context),
      h_logits_(weights_.config.vocab_size, 0.0F) {}

QwenGpuExecutor::~QwenGpuExecutor() {
  if (d_model_weights_ != nullptr) {
    hipFree(d_model_weights_);
    d_model_weights_ = nullptr;
  }
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  void* dev_base_ptr = nullptr;
  void* d_allocated_weights = nullptr;

  if (reader.GetData() != nullptr && reader.GetSize() > 0) {
    const auto malloc_err = hipMalloc(&dev_base_ptr, reader.GetSize());
    if (malloc_err == hipSuccess) {
      HIP_CHECK(hipMemcpy(dev_base_ptr, reader.GetData(), reader.GetSize(),
                          hipMemcpyHostToDevice));
      d_allocated_weights = dev_base_ptr;
    } else {
      const auto reg_err =
          hipHostRegister(const_cast<void*>(reader.GetData()), reader.GetSize(),
                          hipHostRegisterMapped | hipHostRegisterReadOnly);
      if (reg_err == hipSuccess) {
        HIP_CHECK(hipHostGetDevicePointer(
            &dev_base_ptr, const_cast<void*>(reader.GetData()), 0));
      }
    }
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(reader, error_msg);
  if (!weights_opt.has_value()) {
    if (d_allocated_weights != nullptr) {
      hipFree(d_allocated_weights);
    }
    return nullptr;
  }

  if (dev_base_ptr != nullptr) {
    auto remap = [&](models::QwenTensorRef& t) {
      if (t.data != nullptr) {
        const auto offset = static_cast<const std::uint8_t*>(t.data) -
                            static_cast<const std::uint8_t*>(reader.GetData());
        t.data = static_cast<const std::uint8_t*>(dev_base_ptr) + offset;
      }
    };
    remap(weights_opt->token_embd);
    remap(weights_opt->output_norm);
    remap(weights_opt->output);
    for (auto& l : weights_opt->layers) {
      remap(l.attn_norm);
      remap(l.attn_q);
      remap(l.attn_k);
      remap(l.attn_v);
      remap(l.attn_output);
      remap(l.attn_q_norm);
      remap(l.attn_k_norm);
      remap(l.attn_qkv);
      remap(l.attn_gate);
      remap(l.ssm_out);
      remap(l.ssm_conv1d);
      remap(l.ssm_alpha);
      remap(l.ssm_beta);
      remap(l.ssm_a);
      remap(l.ssm_dt);
      remap(l.ssm_norm);
      remap(l.ffn_norm);
      remap(l.ffn_gate);
      remap(l.ffn_up);
      remap(l.ffn_down);
    }
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(reader, error_msg);
  if (!tokenizer || tokenizer->GetVocabSize() <= 256) {
    auto bin_tok = tokenization::QwenTokenizer::CreateFromBinaryFile(
        "models/qwen_vocab.bin");
    if (bin_tok) {
      tokenizer = std::move(bin_tok);
    } else if (!tokenizer) {
      if (d_allocated_weights != nullptr) {
        hipFree(d_allocated_weights);
      }
      return nullptr;
    }
  }

  const std::uint32_t context_len =
      std::min(weights_opt->config.context_length > 0
                   ? weights_opt->config.context_length
                   : 4096U,
               4096U);
  return std::make_unique<QwenGpuExecutor>(std::move(*weights_opt),
                                           std::move(tokenizer),
                                           d_allocated_weights, context_len);
}

tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;

  // 1. Embedding lookup
  const bool embd_is_bf16 = weights_.token_embd.type == core::GgmlType::kBF16;
  LaunchEmbeddingLookup(weights_.token_embd.data, embd_is_bf16, token_id,
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

      const std::size_t q_dim =
          config.num_attention_heads * config.head_dim * 2;
      const std::size_t kv_dim = config.num_key_value_heads * config.head_dim;

      LaunchFusedQKVProjections(
          layer.attn_q.data, q_bf16, layer.attn_k.data, k_bf16,
          layer.attn_v.data, v_bf16, arena_.d_normed, arena_.d_ssm_qkv,
          arena_.d_k, arena_.d_v, q_dim, kv_dim, hidden_size, arena_.stream);

      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                     config.num_attention_heads, config.head_dim,
                     arena_.stream);

      // QK-Norm
      if (!layer.attn_q_norm.empty()) {
        LaunchPerHeadRMSNorm(arena_.d_q,
                             static_cast<const float*>(layer.attn_q_norm.data),
                             arena_.d_q, config.num_attention_heads,
                             config.head_dim, 1e-6F, arena_.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchPerHeadRMSNorm(arena_.d_k,
                             static_cast<const float*>(layer.attn_k_norm.data),
                             arena_.d_k, config.num_key_value_heads,
                             config.head_dim, 1e-6F, arena_.stream);
      }

      // RoPE
      LaunchRoPE(arena_.d_q, arena_.d_k, config.num_attention_heads,
                 config.num_key_value_heads, config.head_dim, config.rotary_dim,
                 pos, config.rope_theta, arena_.stream);

      // Softmax Attention + Gating
      const std::size_t total_k =
          8 * config.num_key_value_heads * 4096 * config.head_dim;
      const std::uint32_t attn_layer_idx = l / 4;
      LaunchAttention(arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                      arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                      arena_.d_ssm_out, attn_layer_idx, pos, 4096,
                      config.num_attention_heads, config.num_key_value_heads,
                      config.head_dim, arena_.stream);

      // Output projection
      LaunchGEMV(layer.attn_output.data, o_bf16, arena_.d_ssm_out,
                 arena_.d_attn_out, hidden_size, 4096, arena_.stream);
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
          arena_.d_alpha_buf, arena_.d_beta_buf, hidden_size, arena_.stream);

      LaunchSSMConvRecurrence(
          arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, arena_.d_conv_out,
          arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
          arena_.d_ssm_out, l, 32, 128, 128, arena_.stream);

      LaunchGEMV(layer.ssm_out.data, out_bf16, arena_.d_ssm_out,
                 arena_.d_attn_out, hidden_size, 4096, arena_.stream);
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
               arena_.d_ffn_out, hidden_size, intermediate_size, arena_.stream);

    // Residual Add
    LaunchResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                      hidden_size, arena_.stream);
  }

  if (!compute_logits) {
    return 0;
  }

  // 3. Final Output Norm
  LaunchRMSNorm(arena_.d_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

  // 4. LM Head Logits GEMV on final token
  const bool out_bf16 = weights_.output.type == core::GgmlType::kBF16;
  LaunchGEMV(weights_.output.data, out_bf16, arena_.d_normed, arena_.d_logits,
             vocab_size, hidden_size, arena_.stream);

  // 5. Parallel GPU Argmax and 4-byte host transfer
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);
  LaunchGPUArgmax(arena_.d_logits, d_out_token, vocab_size, arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    tokenization::TokenId next_token = 0;
    for (std::size_t p = 0; p < prompt_tokens.size(); ++p) {
      const bool is_last = (p + 1 == prompt_tokens.size());
      next_token = ForwardToken(prompt_tokens[p], static_cast<std::uint32_t>(p),
                                is_last);
    }
    return next_token;
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

  // 3. Layer stack across all 32 layers
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];

    // Pre-layer RMSNorm
    LaunchBatchedRMSNorm(
        arena_.d_hidden, static_cast<const float*>(layer.attn_norm.data),
        arena_.d_normed, batch_size, hidden_size, eps, arena_.stream);

    if (layer.is_full_attention) {
      const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
      const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
      const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
      const bool o_bf16 = layer.attn_output.type == core::GgmlType::kBF16;

      const std::size_t q_dim =
          config.num_attention_heads * config.head_dim * 2;
      const std::size_t kv_dim = config.num_key_value_heads * config.head_dim;

      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_q.data, q_bf16,
                        arena_.d_normed, arena_.d_ssm_qkv, batch_size, q_dim,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_k.data, k_bf16,
                        arena_.d_normed, arena_.d_k, batch_size, kv_dim,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_v.data, v_bf16,
                        arena_.d_normed, arena_.d_v, batch_size, kv_dim,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);

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
                        config.head_dim, config.rotary_dim, 0,
                        config.rope_theta, arena_.stream);

      const std::size_t total_k =
          8 * config.num_key_value_heads * 4096 * config.head_dim;
      const std::uint32_t attn_layer_idx = l / 4;
      LaunchBatchedAttention(
          arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
          arena_.d_kv_cache, arena_.d_kv_cache + total_k, arena_.d_ssm_out,
          attn_layer_idx, 0, batch_size, 4096, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, arena_.stream);

      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_output.data, o_bf16,
                        arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                        hidden_size, 4096, arena_.d_scratch_bf16,
                        arena_.stream);
    } else {
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;
      const bool out_bf16 = layer.ssm_out.type == core::GgmlType::kBF16;

      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_qkv.data, qkv_bf16,
                        arena_.d_normed, arena_.d_ssm_qkv, batch_size, 8192,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_gate.data, gate_bf16,
                        arena_.d_normed, arena_.d_ssm_gate, batch_size, 4096,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_alpha.data, alpha_bf16,
                        arena_.d_normed, arena_.d_alpha_buf, batch_size, 32,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_beta.data, beta_bf16,
                        arena_.d_normed, arena_.d_beta_buf, batch_size, 32,
                        hidden_size, arena_.d_scratch_bf16, arena_.stream);

      LaunchBatchedSSMConvRecurrence(
          arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, arena_.d_conv_out,
          arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
          arena_.d_ssm_out, l, batch_size, 32, 128, 128, arena_.stream);

      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_out.data, out_bf16,
                        arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                        hidden_size, 4096, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch_size, hidden_size,
                             arena_.stream);

    LaunchBatchedRMSNorm(
        arena_.d_hidden, static_cast<const float*>(layer.ffn_norm.data),
        arena_.d_normed, batch_size, hidden_size, eps, arena_.stream);

    const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
    const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;
    const bool ffn_d_bf16 = layer.ffn_down.type == core::GgmlType::kBF16;

    LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_gate.data, ffn_g_bf16,
                      arena_.d_normed, arena_.d_ffn_gate, batch_size,
                      intermediate_size, hidden_size, arena_.d_scratch_bf16,
                      arena_.stream);
    LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_up.data, ffn_u_bf16,
                      arena_.d_normed, arena_.d_ffn_up, batch_size,
                      intermediate_size, hidden_size, arena_.d_scratch_bf16,
                      arena_.stream);
    LaunchBatchedSwiGLUActivation(
        arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
        batch_size * intermediate_size, arena_.stream);

    LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_down.data, ffn_d_bf16,
                      arena_.d_ffn_act, arena_.d_ffn_out, batch_size,
                      hidden_size, intermediate_size, arena_.d_scratch_bf16,
                      arena_.stream);

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                             batch_size, hidden_size, arena_.stream);
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

std::vector<tokenization::TokenId> QwenGpuExecutor::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }

  arena_.Reset();

  // 1. Batched GPU prompt prefill
  tokenization::TokenId next_token = ForwardPromptBatch(prompt_tokens);

  std::size_t cur_pos = prompt_tokens.size();
  const auto eos_id = tokenizer_->GetEosTokenId();

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == eos_id || next_token == 151643U ||
        next_token == 248044U || next_token == 248046U) {
      break;
    }

    output_tokens.push_back(next_token);
    if (on_token) {
      const auto piece = tokenizer_->DecodeToken(next_token);
      if (!on_token(next_token, piece)) {
        break;
      }
    }

    next_token = ForwardToken(next_token, static_cast<std::uint32_t>(cur_pos));
    ++cur_pos;
  }

  return output_tokens;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
