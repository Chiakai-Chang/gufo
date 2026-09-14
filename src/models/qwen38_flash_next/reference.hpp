#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_REFERENCE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

/// Single-token, float32, scalar reference of the Qwen3.8-Flash-Next graph.
/// It exists to pin the semantics of every operator the GPU runtime must
/// reproduce: hyper-connection mixing, PLE injection, Gated DeltaNet, gated
/// GQA with QSA block selection, softmax top-k MoE with the shared expert,
/// and the MTP draft block. Speed is irrelevant here.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, NgramTable* ngram,
                 std::uint32_t max_context);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution when non-empty. `hc_out` (hc_dim floats) receives
  /// the final wide residual, the MTP block's hidden input.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::span<float> hc_out = {},
                          std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct LinearState {
    std::vector<float> conv;   ///< [kernel-1][channels], oldest first
    std::vector<float> state;  ///< [v_heads][head_dim(v)][head_dim(k)]
  };
  struct AttentionState {
    std::vector<float> k;        ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;        ///< [pos][kv_heads][head_dim]
    std::vector<float> index_k;  ///< [pos][indexer_dim], raw
    std::vector<float> block_k;  ///< [block][indexer_dim], pooled+norm+rope
    std::uint32_t blocks{0};
  };

  void HcMix(const HcMixer& m, std::span<const float> res,
             std::span<float> mixed, std::span<float> inject);
  void HcCombine(std::span<float> res, std::span<const float> block_out,
                 std::span<const float> inject);
  bool Ple(const LayerWeights& l, std::int32_t token, std::span<float> res,
           std::string* error_msg);
  void LinearAttention(const LayerWeights& l, LinearState& s,
                       std::span<const float> x, std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::uint32_t pos,
                 std::span<float> out);
  void Moe(const LayerWeights& l, std::span<const float> x,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  NgramTable* ngram_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  NgramHistory ngram_history_;
  std::vector<float> ple_conv_history_;  ///< [PleConvHistory()][hc_dim]
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_REFERENCE_HPP_
