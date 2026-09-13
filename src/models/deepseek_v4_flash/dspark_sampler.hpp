#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_DSPARK_SAMPLER_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_DSPARK_SAMPLER_HPP_

#include <cstdint>
#include <span>

#include "src/core/sampling.hpp"
#include "src/models/deepseek_v4_flash/runtime/model.h"

namespace gufo::models::deepseek_v4_flash {

/// Lends a request sampler to DSpark cycles.
///
/// The runtime draws every verified row through `Sample`; each draw is the
/// token autoregressive decoding would produce there, and the runtime emits
/// every draw in order (a rejected draw becomes the next cycle's anchor), so
/// the draw enters the working history at once and later rows see the right
/// penalties. The owner records emitted tokens on its own sampler and copies
/// the RNG state back after each cycle with `rng_state()`.
class DsparkSamplerBridge {
public:
  explicit DsparkSamplerBridge(const sampling::SamplerState& sampler)
      : working_(sampler),
        hook_{.ctx = this, .sample = &Sample, .accept = &Accept} {}

  DsparkSamplerBridge(const DsparkSamplerBridge&) = delete;
  DsparkSamplerBridge& operator=(const DsparkSamplerBridge&) = delete;

  [[nodiscard]] const ds4_dspark_sampler* hook() const { return &hook_; }
  [[nodiscard]] std::uint64_t rng_state() const {
    return working_.rng_state();
  }

private:
  static int Sample(void* ctx, const float* logits,
                    std::uint32_t vocabulary_size) {
    auto& working = static_cast<DsparkSamplerBridge*>(ctx)->working_;
    const auto token =
        working.Sample(std::span<const float>(logits, vocabulary_size));
    working.Accept(token);
    return static_cast<int>(token);
  }

  static void Accept(void* ctx, int token) {
    static_cast<DsparkSamplerBridge*>(ctx)->working_.Accept(
        static_cast<sampling::TokenId>(token));
  }

  sampling::SamplerState working_;
  ds4_dspark_sampler hook_;
};

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_DSPARK_SAMPLER_HPP_
