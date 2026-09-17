#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#include <array>
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

int main() {
  try {
    CheckLengthController();
    const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
    const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
    for (const auto& test : gufo::test::QwenSamplingCases()) {
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
        << "MTP controller respects budgets, censoring and replay; all 23 "
           "sampling strategies preserve prefix, RNG and stops\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
