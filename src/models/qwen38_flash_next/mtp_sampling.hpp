#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <cstdint>
#include <span>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

enum class DraftDecision { kAccept, kReject, kStop };

/// The draft is deterministic, so an exact target sample decides acceptance.
/// A rejected draw is deferred to the next cycle: restore its RNG so that
/// stopping at this prefix consumes exactly the draws made by ordinary decode.
template<typename IsStop>
DraftDecision VerifyDraft(std::span<const float> logits, std::int32_t draft,
                          sampling::SamplerState& sampler, IsStop is_stop) {
  const auto rng = sampler.rng_state();
  const auto token = sampler.Sample(logits);
  if (is_stop(static_cast<std::int32_t>(token))) {
    return DraftDecision::kStop;
  }
  if (token != static_cast<sampling::TokenId>(draft)) {
    sampler.SetRngState(rng);
    return DraftDecision::kReject;
  }
  sampler.Accept(token);
  return DraftDecision::kAccept;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
