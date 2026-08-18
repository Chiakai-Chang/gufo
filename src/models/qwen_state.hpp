#ifndef STRIX_MODELS_QWEN_STATE_HPP_
#define STRIX_MODELS_QWEN_STATE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"

namespace strix::models {

/// Tensor weight references for a single transformer layer block.
struct QwenLayerWeights {
  std::span<const float> attn_norm;
  std::span<const float> attn_q;
  std::span<const float> attn_k;
  std::span<const float> attn_v;
  std::span<const float> attn_output;

  std::span<const float> ffn_norm;
  std::span<const float> ffn_gate;
  std::span<const float> ffn_up;
  std::span<const float> ffn_down;
};

/// Full model tensor references mapped directly from GGUF storage.
struct QwenModelWeights {
  core::ModelConfig config;
  std::span<const float> token_embd;
  std::vector<QwenLayerWeights> layers;
  std::span<const float> output_norm;
  std::span<const float> output;

  /// Loads and binds weights from a GgufReader.
  [[nodiscard]] static std::optional<QwenModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Contiguous Key-Value Cache for auto-regressive generation.
/// Memory layout: [num_layers, num_kv_heads, max_context_length, head_dim].
class QwenKvCache {
public:
  QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
              std::uint32_t max_context, std::uint32_t head_dim);

  void Reset() noexcept { current_pos_ = 0; }
  [[nodiscard]] std::uint32_t GetCurrentPos() const noexcept {
    return current_pos_;
  }
  void AdvancePos() noexcept { ++current_pos_; }

  [[nodiscard]] std::span<float> GetKeySlice(std::uint32_t layer,
                                             std::uint32_t kv_head,
                                             std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetKeySlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

  [[nodiscard]] std::span<float> GetValueSlice(std::uint32_t layer,
                                               std::uint32_t kv_head,
                                               std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetValueSlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

private:
  std::uint32_t num_layers_;
  std::uint32_t num_kv_heads_;
  std::uint32_t max_context_;
  std::uint32_t head_dim_;
  std::uint32_t current_pos_{0};
  std::vector<float> k_data_;
  std::vector<float> v_data_;
};

/// Preallocated activation scratch arena for zero-heap-allocation decode
/// passes.
struct QwenScratchArena {
  explicit QwenScratchArena(const core::ModelConfig& config);

  std::vector<float> buffer;
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> attn_scores;
  std::span<float> attn_out;
  std::span<float> mlp_gate;
  std::span<float> mlp_up;
  std::span<float> mlp_act;
  std::span<float> mlp_out;
  std::span<float> logits;
};

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_STATE_HPP_
