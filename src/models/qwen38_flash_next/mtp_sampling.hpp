#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

enum class DraftDecision { kAccept, kReject, kStop };

inline constexpr std::size_t kMtpCandidates = 64;

struct MtpCandidateLogits {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> logits{};
  std::size_t size{0};
};

struct MtpProposal {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> probabilities{};
  std::size_t size{0};
  sampling::TokenId token{0};
  float probability{0.0F};
};

/// The proposal has bounded support; the target distribution stays complete.
/// Integer probability masses sum to 2^24, so the exported F32 q sums to
/// exactly one and is exactly the distribution used to draw the proposal.
inline MtpProposal SampleMtpProposal(const MtpCandidateLogits& candidates,
                                     const sampling::SamplerState& sampler,
                                     std::uint64_t* rng) {
  if (candidates.size == 0 || candidates.size > kMtpCandidates) {
    throw std::invalid_argument("invalid MTP candidate count");
  }
  std::vector<sampling::TokenId> history;
  history.reserve(sampler.history().size());
  const auto ids = std::span(candidates.ids).first(candidates.size);
  for (const auto token : sampler.history()) {
    const auto it = std::find(ids.begin(), ids.end(), token);
    history.push_back(it == ids.end()
                          ? std::numeric_limits<sampling::TokenId>::max()
                          : static_cast<sampling::TokenId>(it - ids.begin()));
  }
  const auto distribution = sampling::BuildDistribution(
      std::span(candidates.logits).first(candidates.size), sampler.config(),
      history);
  constexpr std::uint32_t units = 1U << 24;
  std::array<std::uint32_t, kMtpCandidates> mass{};
  MtpProposal proposal;
  proposal.size = distribution.entries().size();
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto& entry = distribution.entries()[i];
    proposal.ids[i] = candidates.ids[entry.token];
    mass[i] = static_cast<std::uint32_t>(std::floor(entry.value * units));
    total += mass[i];
  }
  mass[0] += units - total;
  const auto draw = static_cast<std::uint32_t>(sampling::Uniform(rng) * units);
  std::uint32_t cumulative = 0;
  bool selected = false;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    proposal.probabilities[i] = static_cast<float>(mass[i]) / units;
    cumulative += mass[i];
    if (!selected && draw < cumulative) {
      proposal.token = proposal.ids[i];
      proposal.probability = proposal.probabilities[i];
      selected = true;
    }
  }
  return proposal;
}

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
