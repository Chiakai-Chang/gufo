#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckLengthController() {
  qfn::MtpLengthController controller(7);
  const auto initial = controller.Choose(7);
  Require(controller.Choose(0) == 0, "empty draft budget was ignored");
  Require(controller.Choose(1) == 1, "one-draft budget was ignored");
  controller.Observe(0, 0);
  Require(controller.Choose(7) == initial, "empty cycle changed policy");

  for (unsigned i = 0; i < 12; ++i) {
    const auto length = controller.Choose(7);
    controller.Observe(length, length);
  }
  Require(controller.Choose(7) == 7,
          "fully accepted prefixes did not grow to the maximum length");
  Require(controller.Choose(2) == 2, "remaining output budget was ignored");
  for (unsigned i = 0; i < 12; ++i) {
    controller.Observe(0, controller.Choose(7));
  }
  Require(controller.Choose(7) == 1,
          "repeated rejection did not shorten the draft");
  controller.Reset();
  Require(controller.Choose(7) == initial, "reset retained old acceptance");
  controller.Observe(5, 6);
  qfn::MtpLengthController restored(7);
  Require(restored.Restore(controller.State()), "controller restore failed");
  for (unsigned budget = 0; budget <= 8; ++budget) {
    Require(restored.Choose(budget) == controller.Choose(budget),
            "restored policy changes draft widths");
  }
  Require(!restored.Restore({-1.0F, 1.0F}) && !restored.Restore({0.0F, 0.0F}),
          "invalid controller state accepted");

  // The unverified suffix is not additional evidence of failure.
  qfn::MtpLengthController short_tail(7);
  qfn::MtpLengthController long_tail(7);
  for (unsigned accepted : {2U, 0U, 1U, 2U, 1U, 0U}) {
    short_tail.Observe(accepted, accepted + 1);
    long_tail.Observe(accepted, 7);
    for (unsigned budget = 0; budget <= 9; ++budget) {
      Require(short_tail.Choose(budget) == long_tail.Choose(budget),
              "unverified suffix biased the acceptance estimate");
    }
  }
  for (unsigned limit = 1; limit <= 7; ++limit) {
    qfn::MtpLengthController a(limit);
    qfn::MtpLengthController b(limit);
    for (unsigned cycle = 0; cycle < 32; ++cycle) {
      const auto budget = cycle % 10;
      const auto length = a.Choose(budget);
      Require(length == b.Choose(budget) && length <= budget &&
                  length <= limit && (budget == 0 || length > 0),
              "controller violated a limit or changed on replay");
      const unsigned accepted = length == 0 ? 0 : cycle % (length + 1);
      a.Observe(accepted, length);
      b.Observe(accepted, length);
    }
  }
}

void CheckCompactProposals() {
  const std::array<float, 5> target_logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    sampling::SamplerState sampler(test.config, history);
    const auto p = sampler.Distribution(target_logits);
    for (std::uint64_t seed = 0; seed < 16; ++seed) {
      auto rng = seed;
      auto replay_rng = seed;
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &rng);
      const auto replay =
          qfn::SampleMtpProposal(candidates, sampler, &replay_rng);
      Require(q.token == replay.token && q.ids == replay.ids &&
                  q.probabilities == replay.probabilities && rng == replay_rng,
              "compact proposal is not reproducible");
      double sum = 0.0;
      std::array<double, 5> draft{};
      for (std::size_t i = 0; i < q.size; ++i) {
        const double mass = q.probabilities[i] * double(1U << 24);
        Require(mass >= 0.0 && mass == std::floor(mass),
                "proposal probability is not an exact discrete mass");
        draft[q.ids[i]] += q.probabilities[i];
        sum += q.probabilities[i];
      }
      Require(
          sum == 1.0 && q.probability > 0.0F && draft[q.token] == q.probability,
          "sampled q differs from the exported distribution");
      double accepted = 0.0, residual = 0.0;
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        accepted += std::min(probability, draft[i]);
        residual += std::max(probability - draft[i], 0.0);
      }
      Require(std::abs(accepted + residual - 1.0) < 1e-12,
              "acceptance and residual mass do not conserve the target");
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        const double correction =
            residual > 0.0
                ? (1.0 - accepted) * std::max(probability - draft[i], 0.0) /
                      residual
                : 0.0;
        Require(std::abs(std::min(probability, draft[i]) + correction -
                         probability) < 1e-12,
                "bounded draft support changes target output probability");
      }
    }
  }
}

void CheckSampledOutputFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  constexpr unsigned trials = 100000;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 73});
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &draft_rng);
      auto token = q.token;
      if (sampler.Uniform() * q.probability >= p.probability(token)) {
        token = p.SampleResidual(std::span(q.ids).first(q.size),
                                 std::span(q.probabilities).first(q.size),
                                 sampler.mutable_rng_state());
      }
      ++counts[token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      Require(std::abs(double(counts[token]) / trials - p.probability(token)) <
                  0.008,
              "proposal and correction draws distort target frequencies");
    }
  }
}

int main() {
  try {
    CheckLengthController();
    CheckCompactProposals();
    CheckSampledOutputFrequencies();
    const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
    const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      if (test.config.uses_random_sampling())
        continue;
      for (unsigned accepted = 0; accepted < 4; ++accepted) {
        sampling::SamplerState ar(test.config, history);
        auto speculative = ar;
        for (unsigned i = 0; i < accepted; ++i) {
          const auto token = ar.Sample(logits);
          ar.Accept(token);
          Require(qfn::VerifyDraft(logits, static_cast<std::int32_t>(token),
                                   speculative, [](auto) { return false; }) ==
                      qfn::DraftDecision::kAccept,
                  "matching target sample was rejected");
        }
        const auto before = speculative.rng_state();
        const auto saved_history = std::vector(speculative.history().begin(),
                                               speculative.history().end());
        const auto token = ar.Sample(logits);
        Require(qfn::VerifyDraft(logits, (token + 1) % logits.size(),
                                 speculative, [](auto) { return false; }) ==
                    qfn::DraftDecision::kReject,
                "mismatching target sample was accepted");
        Require(speculative.rng_state() == before &&
                    std::vector(speculative.history().begin(),
                                speculative.history().end()) == saved_history,
                "rejected suffix consumed RNG or history");
        Require(speculative.Sample(logits) == token,
                "deferred rejection did not reproduce the AR token");
        ar.Accept(token);
        speculative.Accept(token);
        Require(ar.rng_state() == speculative.rng_state() &&
                    std::vector(ar.history().begin(), ar.history().end()) ==
                        std::vector(speculative.history().begin(),
                                    speculative.history().end()),
                "accepted prefix differs from AR state");
        auto stop_reference = speculative;
        const auto stop_token = stop_reference.Sample(logits);
        Require(qfn::VerifyDraft(
                    logits, (stop_token + 1) % logits.size(), speculative,
                    [&](auto t) {
                      return t == static_cast<std::int32_t>(stop_token);
                    }) == qfn::DraftDecision::kStop,
                "target stop must end verification even when draft differs");
        Require(speculative.rng_state() == stop_reference.rng_state(),
                "stop draw was discarded");
      }
    }
    std::cout
        << "MTP policy replay, greedy prefix/RNG/stops and compact proposal "
           "probabilities pass all sampling configurations\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
