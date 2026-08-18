#include "src/models/qwen_state.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::models {
namespace {

std::span<const float> ExtractTensorSpan(const core::GgufReader& reader,
                                         std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {static_cast<const float*>(tensor->data), tensor->ElementCount()};
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

  weights.token_embd = ExtractTensorSpan(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorSpan(reader, "output_norm.weight");
  weights.output = ExtractTensorSpan(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorSpan(reader, prefix + "attn_norm.weight");
    l.attn_q = ExtractTensorSpan(reader, prefix + "attn_q.weight");
    l.attn_k = ExtractTensorSpan(reader, prefix + "attn_k.weight");
    l.attn_v = ExtractTensorSpan(reader, prefix + "attn_v.weight");
    l.attn_output = ExtractTensorSpan(reader, prefix + "attn_output.weight");

    l.ffn_norm = ExtractTensorSpan(reader, prefix + "ffn_norm.weight");
    l.ffn_gate = ExtractTensorSpan(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorSpan(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorSpan(reader, prefix + "ffn_down.weight");
  }

  return weights;
}

QwenKvCache::QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
                         std::uint32_t max_context, std::uint32_t head_dim)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      max_context_(max_context),
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
      config.context_length > 0 ? config.context_length : 32768;
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
