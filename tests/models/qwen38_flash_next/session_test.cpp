#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
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
    auto model = qfn::Model::Load(argv[2],
                                  {.max_context = 4608,
                                   .mtp_model_path = argv[4],
                                   .max_batch = 512,
                                   .max_draft_tokens = 7,
                                   .draft_vocab = 65536},
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
    for (const auto depth : {16U, 4096U}) {
      const auto prefix = std::span(prompt).first(depth);
      const std::vector<sampling::TokenId> history(prefix.begin(),
                                                   prefix.end());
      for (std::size_t c = 0; c < configs.size(); ++c) {
        auto ar = model->CreateSession(4608, &error);
        auto mtp = model->CreateSession(4608, &error);
        Require(ar && mtp, error);
        Require(ar->Sync(prefix, &error), error);
        // Prefill the other session before decoding either: retained target
        // hidden rows must remain private to their session.
        Require(mtp->Sync(prefix, &error), error);
        sampling::SamplerState a(configs[c], history), b(configs[c], history);
        std::vector<std::int32_t> reference, candidate;
        std::vector<std::size_t> interleaved_widths;
        while (candidate.size() < 8) {
          qfn::Session::DecodeResult decoded;
          Require(
              mtp->DecodeStep(8 - candidate.size(), b, &decoded, &error, false),
              error);
          Require(!decoded.tokens.empty(), "empty MTP step");
          interleaved_widths.push_back(decoded.tokens.size());
          candidate.insert(candidate.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
          while (reference.size() < candidate.size()) {
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
          Require(a.rng_state() == b.rng_state(), "AR/MTP RNG mismatch");
          Require(ar->Position() == mtp->Position(),
                  "AR/MTP position mismatch");
        }
        const auto stats = mtp->Statistics();
        Require(mtp->Sync(prefix, &error), error);
        Require(mtp->Position() == depth, "prefix reset failed");
        sampling::SamplerState replay_sampler(configs[c], history);
        std::vector<std::int32_t> replay;
        std::vector<std::size_t> isolated_widths;
        while (replay.size() < 8) {
          qfn::Session::DecodeResult decoded;
          Require(mtp->DecodeStep(8 - replay.size(), replay_sampler, &decoded,
                                  &error, false),
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
                  << " tokens=8 exact=1 drafted=" << stats.drafted
                  << " accepted=" << stats.accepted << '\n'
                  << std::flush;
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
