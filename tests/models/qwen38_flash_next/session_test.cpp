#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

void CheckServingSampling(const std::shared_ptr<qfn::Model>& model) {
  namespace server = gufo::server;
  using Backend = server::InferenceBackend;
  Backend ar, mtp;
  std::string error;
  Require(ar.load(model, &error, 6145, 2), error);
  server::TextSpeculativeConfig options;
  options.backend = server::TextSpeculativeBackend::kMtp;
  Require(mtp.load(model, &error, 6145, 2, {}, {}, options), error);
  struct Reference {
    gufo::test::SamplingCase test;
    std::string prompt;
    Backend::Result ar, mtp;
  };
  std::vector<Reference> references;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    const std::string prompt = "Sampling " + std::string(test.name) +
                               ": Continue red, blue, blue, red,";
    auto ordinary = ar.complete(prompt, 8, test.config);
    auto speculative = mtp.complete(prompt, 8, test.config);
    // Exercise the serving handoff against the model session directly.
    // Replaying HTTP alone would not detect a consistently dropped residual.
    auto direct = model->CreateSession(6145, &error);
    const auto encoded = model->Tokenize(prompt);
    Require(direct && direct->Sync(encoded, &error), error);
    const std::vector<sampling::TokenId> history(encoded.begin(),
                                                 encoded.end());
    sampling::SamplerState direct_sampler(test.config, history);
    std::vector<sampling::TokenId> direct_tokens;
    while (direct_tokens.size() < 8) {
      qfn::Session::DecodeResult step;
      Require(direct->DecodeStep(8 - direct_tokens.size(), direct_sampler,
                                 &step, &error),
              error);
      direct_tokens.insert(direct_tokens.end(), step.tokens.begin(),
                           step.tokens.end());
      if (step.stop)
        break;
      Require(!step.tokens.empty(),
              "direct serving reference made no progress");
    }
    const auto direct_stats = direct->Statistics();
    Require(direct_tokens == speculative.tokens &&
                direct_stats.drafted == speculative.draft_tokens &&
                direct_stats.accepted == speculative.draft_accepted_tokens,
            "serving lost sampling state: " + std::string(test.name));
    Require(!ordinary.tokens.empty() && ordinary.draft_tokens == 0 &&
                speculative.draft_tokens > 0,
            "sampling strategy did not exercise AR and MTP");
    if (!test.config.uses_random_sampling()) {
      Require(ordinary.tokens == speculative.tokens,
              "serving greedy AR/MTP mismatch: " + std::string(test.name));
    }
    for (const bool use_mtp : {false, true}) {
      auto& backend = use_mtp ? mtp : ar;
      const auto& expected = use_mtp ? speculative : ordinary;
      const auto replay = backend.complete(prompt, 8, test.config);
      Require(
          replay.tokens == expected.tokens &&
              replay.draft_tokens == expected.draft_tokens &&
              replay.draft_accepted_tokens == expected.draft_accepted_tokens,
          "serving replay changed sampling or draft acceptance");
    }
    references.push_back(
        {test, prompt, std::move(ordinary), std::move(speculative)});
    std::cout << "serving sampling=" << test.name << " replay_exact=1\n"
              << std::flush;
  }
  // Every strategy also runs in an interleaved pair. Flash-Next currently
  // serializes model execution; this tests independent request histories.
  for (const bool use_mtp : {false, true}) {
    auto& backend = use_mtp ? mtp : ar;
    for (std::size_t offset = 0; offset < references.size(); offset += 2) {
      std::array<std::future<Backend::Result>, 2> pending;
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto* saved = &references[(offset + row) % references.size()];
        pending[row] = std::async(std::launch::async, [&backend, saved] {
          return backend.complete(saved->prompt, 8, saved->test.config);
        });
      }
      for (std::size_t row = 0; row < pending.size(); ++row) {
        const auto& saved = references[(offset + row) % references.size()];
        const auto& expected = use_mtp ? saved.mtp : saved.ar;
        const auto actual = pending[row].get();
        Require(
            actual.tokens == expected.tokens &&
                actual.draft_tokens == expected.draft_tokens &&
                actual.draft_accepted_tokens == expected.draft_accepted_tokens,
            "interleaving changed serving sampling or acceptance");
      }
    }
    // Greedy/AR prefixes are budget-independent. Sampled MTP consumes
    // proposal/rejection draws, so each budget must replay its own result.
    for (const auto index : {std::size_t{0}, references.size() - 2}) {
      const auto& saved = references[index];
      for (const auto budget : {1U, 2U, 3U}) {
        const auto actual =
            backend.complete(saved.prompt, budget, saved.test.config);
        if (use_mtp && saved.test.config.uses_random_sampling()) {
          const auto replay =
              backend.complete(saved.prompt, budget, saved.test.config);
          Require(
              actual.tokens.size() <= budget &&
                  actual.tokens == replay.tokens &&
                  actual.draft_tokens == replay.draft_tokens &&
                  actual.draft_accepted_tokens == replay.draft_accepted_tokens,
              "sampled MTP budget or replay differs");
        } else {
          Require(actual.tokens.size() == std::min<std::size_t>(
                                              budget, saved.ar.tokens.size()) &&
                      std::equal(actual.tokens.begin(), actual.tokens.end(),
                                 saved.ar.tokens.begin()),
                  "output budget changed the AR/greedy prefix");
        }
      }
    }
    std::cout << "serving strategies=" << references.size()
              << " mtp=" << use_mtp << " C2_and_budgets_exact=1\n"
              << std::flush;
  }
}

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr
        << "Usage: session_test --model FIRST.gguf --mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = 6145, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model != nullptr, error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(4096);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    const std::array<sampling::SamplingConfig, 4> configs{{
        {.seed = 1},
        {.temperature = 0.7F, .seed = 1},
        {.temperature = 1.0F, .top_p = 0.95F, .seed = 73},
        {.temperature = 0.8F,
         .top_k = 40,
         .top_p = 0.9F,
         .min_p = 0.01F,
         .min_keep = 2,
         .seed = 1,
         .repeat_penalty = 1.1F,
         .repeat_last_n = 16,
         .frequency_penalty = 0.2F,
         .presence_penalty = 0.1F},
    }};
    struct Replay {
      unsigned depth;
      std::size_t config;
      std::vector<float> prefill_logits;
      std::vector<float> final_logits;
      std::vector<std::int32_t> tokens;
      std::uint64_t rng;
    };
    std::vector<Replay> replays;
    for (const auto depth : {16U, 4096U}) {
      const auto prefix = std::span(prompt).first(depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (std::size_t c = 0; c < configs.size(); ++c) {
        const bool sampled = configs[c].uses_random_sampling();
        auto ar = model->CreateSession(6145, &error);
        auto mtp = model->CreateSession(6145, &error);
        Require(ar && mtp, error);
        Require(ar->Sync(prefix, &error), error);
        // Prefill the other session before decoding either: retained target
        // hidden rows must remain private to their session.
        Require(mtp->Sync(prefix, &error), error);
        RequireExact(ar->Logits(), mtp->Logits(),
                     "interleaving changed prefill logits");
        Replay saved{depth, c,  {ar->Logits().begin(), ar->Logits().end()},
                     {},    {}, 0};
        sampling::SamplerState a(configs[c], history), b(configs[c], history);
        std::vector<std::int32_t> reference, candidate;
        std::vector<std::size_t> interleaved_widths;
        const std::size_t token_count = c <= 1 ? 32 : 8;
        while (candidate.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - candidate.size(), b, &decoded,
                                  &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty MTP step");
          interleaved_widths.push_back(decoded.tokens.size());
          candidate.insert(candidate.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
          while (reference.size() < candidate.size()) {
            if (sampled) {
              // Independent target replay checks every committed token's
              // recurrent/cache state, including residual corrections.
              const auto token = candidate[reference.size()];
              Require(ar->Evaluate(token, &error), error);
              reference.push_back(token);
              a.Accept(static_cast<sampling::TokenId>(token));
              continue;
            }
            qfn::Session::DecodeResult single;
            Require(ar->DecodeStep(1, a, &single, &error, false), error);
            reference.insert(reference.end(), single.tokens.begin(),
                             single.tokens.end());
          }
          double squared = 0.0;
          float max_diff = 0.0F;
          for (std::size_t i = 0; i < ar->Logits().size(); ++i) {
            Require(std::isfinite(ar->Logits()[i]) &&
                        std::isfinite(mtp->Logits()[i]),
                    "non-finite logits");
            const float d = ar->Logits()[i] - mtp->Logits()[i];
            squared += static_cast<double>(d) * d;
            max_diff = std::max(max_diff, std::abs(d));
          }
          std::cout << "frontier depth=" << depth << " config=" << c
                    << " tokens=" << candidate.size()
                    << " width=" << decoded.tokens.size()
                    << " max_diff=" << max_diff
                    << " rmse=" << std::sqrt(squared / ar->Logits().size())
                    << " ar_rng=" << a.rng_state()
                    << " mtp_rng=" << b.rng_state() << '\n';
          if (reference != candidate) {
            std::cout << "AR:";
            for (auto t : reference)
              std::cout << ' ' << t;
            std::cout << "\nMTP:";
            for (auto t : candidate)
              std::cout << ' ' << t;
            std::cout << '\n';
          }
          std::cout << std::flush;
          Require(reference == candidate,
                  "AR/MTP token mismatch at depth " + std::to_string(depth) +
                      " config " + std::to_string(c) + " token " +
                      std::to_string(reference.size()));
          Require(max_diff == 0.0F, "AR/MTP frontier logits differ");
          if (!sampled) {
            Require(a.rng_state() == b.rng_state(),
                    "greedy AR/MTP RNG mismatch");
          }
          Require(ar->Position() == mtp->Position(),
                  "AR/MTP position mismatch");
        }
        saved.final_logits.assign(ar->Logits().begin(), ar->Logits().end());
        saved.tokens = reference;
        saved.rng = b.rng_state();
        // Greedy and seeded sampling exercise fresh-load determinism; the
        // remaining configurations cover filters and penalties above.
        if (c <= 1)
          replays.push_back(std::move(saved));
        const auto stats = mtp->Statistics();
        Require(mtp->Sync(prefix, &error), error);
        Require(mtp->Position() == depth, "prefix reset failed");
        sampling::SamplerState replay_sampler(configs[c], history);
        std::vector<std::int32_t> replay;
        std::vector<std::size_t> isolated_widths;
        while (replay.size() < token_count) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(token_count - replay.size(), replay_sampler,
                                  &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty replay step");
          isolated_widths.push_back(decoded.tokens.size());
          replay.insert(replay.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        const auto after = mtp->Statistics();
        Require(replay == candidate && isolated_widths == interleaved_widths &&
                    after.drafted - stats.drafted == stats.drafted &&
                    after.accepted - stats.accepted == stats.accepted,
                "interleaving changed MTP proposals or acceptance");
        std::cout << "depth=" << depth << " config=" << c
                  << " tokens=" << token_count
                  << " exact=1 drafted=" << stats.drafted
                  << " accepted=" << stats.accepted << '\n'
                  << std::flush;
      }
    }
    // Recreate device weights, executor and plans. Reverse request order so
    // a previous shape cannot silently determine this request's arithmetic.
    model.reset();
    model = qfn::Model::Load(
        argv[2],
        {.max_context = 6145, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model != nullptr, error);
    std::reverse(replays.begin(), replays.end());
    for (const auto& saved : replays) {
      const auto prefix = std::span(prompt).first(saved.depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (const bool speculative : {false, true}) {
        const bool sampled = configs[saved.config].uses_random_sampling();
        auto session = model->CreateSession(6145, &error);
        Require(session && session->Sync(prefix, &error), error);
        RequireExact(saved.prefill_logits, session->Logits(),
                     "fresh load changed prefill logits");
        sampling::SamplerState sampler(configs[saved.config], history);
        std::vector<std::int32_t> tokens;
        constexpr std::array<std::size_t, 4> budgets{2, 8, 1, 4};
        std::size_t step = 0;
        while (tokens.size() < saved.tokens.size()) {
          if (sampled && !speculative) {
            const auto token = saved.tokens[tokens.size()];
            Require(session->Evaluate(token, &error), error);
            tokens.push_back(token);
            continue;
          }
          const auto budget = sampled ? saved.tokens.size() - tokens.size()
                              : speculative ? budgets[step++ % budgets.size()]
                                            : 1;
          qfn::Session::DecodeResult decoded;
          Require(session->DecodeStep(
                      std::min(budget, saved.tokens.size() - tokens.size()),
                      sampler, &decoded, &error, false),
                  error);
          Require(!decoded.tokens.empty(), "empty fresh-load replay step");
          tokens.insert(tokens.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        }
        Require(tokens == saved.tokens,
                "fresh load or draft width changed tokens");
        RequireExact(saved.final_logits, session->Logits(),
                     "fresh load or draft width changed final logits");
        if (!sampled || speculative) {
          Require(sampler.rng_state() == saved.rng,
                  "fresh load or draft width changed RNG");
        }
        std::cout << "fresh_load depth=" << saved.depth
                  << " config=" << saved.config << " mtp=" << speculative
                  << " tokens=" << tokens.size() << " bitwise_exact=1\n"
                  << std::flush;
      }
    }
    CheckServingSampling(model);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
