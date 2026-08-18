#include "src/models/qwen_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::models {
namespace {

QwenTensorRef ExtractTensorRef(const core::GgufReader& reader,
                               std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

}  // namespace

std::optional<QwenModelWeights> QwenModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto config_opt = reader.ExtractModelConfig(error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }

  QwenModelWeights weights;
  weights.config = *config_opt;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    l.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (l.ffn_norm.empty()) {
      l.ffn_norm =
          ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    }

    // Check if full attention layer
    l.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    if (!l.attn_q.empty()) {
      l.is_full_attention = true;
      l.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      l.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      l.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
      l.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      l.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    } else {
      l.is_full_attention = false;
      l.attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
      l.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
      l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
      l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
      l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
      l.ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
      l.ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
      l.ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
      l.ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
    }

    l.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");
  }

  return weights;
}

QwenKvCache::QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
                         std::uint32_t max_context, std::uint32_t head_dim)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      max_context_(std::min(max_context, 8192U)),
      head_dim_(head_dim) {
  const std::size_t total_elements = static_cast<std::size_t>(num_layers_) *
                                     num_kv_heads_ * max_context_ * head_dim_;
  k_data_.resize(total_elements, 0.0F);
  v_data_.resize(total_elements, 0.0F);
}

std::span<float> QwenKvCache::GetKeySlice(std::uint32_t layer,
                                          std::uint32_t kv_head,
                                          std::uint32_t pos) noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&k_data_[offset], head_dim_};
}

std::span<const float> QwenKvCache::GetKeySlice(
    std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t pos) const noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&k_data_[offset], head_dim_};
}

std::span<float> QwenKvCache::GetValueSlice(std::uint32_t layer,
                                            std::uint32_t kv_head,
                                            std::uint32_t pos) noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&v_data_[offset], head_dim_};
}

std::span<const float> QwenKvCache::GetValueSlice(
    std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t pos) const noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&v_data_[offset], head_dim_};
}

QwenScratchArena::QwenScratchArena(const core::ModelConfig& config) {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t q_size =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t max_context =
      std::min(config.context_length > 0 ? config.context_length : 4096, 8192U);
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;

  const std::size_t total_size = (hidden_size * 3) + q_size + (kv_size * 2) +
                                 max_context + (intermediate_size * 4) +
                                 vocab_size;
  buffer.resize(total_size, 0.0F);

  std::size_t cur = 0;
  auto alloc_span = [&](std::size_t sz) {
    auto sp = std::span<float>(&buffer[cur], sz);
    cur += sz;
    return sp;
  };

  hidden = alloc_span(hidden_size);
  normed = alloc_span(hidden_size);
  q = alloc_span(q_size);
  k = alloc_span(kv_size);
  v = alloc_span(kv_size);
  attn_scores = alloc_span(max_context);
  attn_out = alloc_span(hidden_size);
  mlp_gate = alloc_span(intermediate_size);
  mlp_up = alloc_span(intermediate_size);
  mlp_act = alloc_span(intermediate_size);
  mlp_out = alloc_span(hidden_size);
  logits = alloc_span(vocab_size);
}

}  // namespace strix::models
