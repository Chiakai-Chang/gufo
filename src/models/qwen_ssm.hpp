#ifndef STRIX_MODELS_QWEN_SSM_HPP_
#define STRIX_MODELS_QWEN_SSM_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen_state.hpp"

namespace strix::models {

/// SSM recurrent state cache across sequence positions for all SSM layers.
class QwenSsmCache {
public:
  QwenSsmCache(std::uint32_t num_layers, std::size_t conv_channels,
               std::size_t ssm_dim);

  void Reset() noexcept;

  /// Returns rolling conv state buffer for given layer [conv_channels * 4]
  [[nodiscard]] std::span<float> GetConvState(std::uint32_t layer) noexcept;

  /// Returns recurrent SSM state matrix for given layer [ssm_dim]
  [[nodiscard]] std::span<float> GetSsmState(std::uint32_t layer) noexcept;

private:
  std::uint32_t num_layers_;
  std::size_t conv_channels_;
  std::size_t ssm_dim_;
  std::vector<float> conv_states_;
  std::vector<float> ssm_states_;
};

/// Computes Qwen 3.5 Linear SSM operator for one token position:
/// 1. 1D Causal Convolution with rolling conv state
/// 2. Gated DeltaNet state-space recurrence
/// 3. Gating and output projection to out
void ForwardSSM(std::span<const float> x_normed, const QwenLayerWeights& layer,
                QwenSsmCache& ssm_cache, std::uint32_t layer_idx,
                std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept;

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_SSM_HPP_
