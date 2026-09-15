// Runs the ROCm executor over a prompt, optionally next to the float32
// reference, and reports agreement per position plus greedy continuations.
//
//   gpu_probe --model FIRST_SHARD.gguf [--mtp MTP.gguf] --prompt TEXT [--warm]
//             [--generate N] [--reference] [--batch T] [--context N]
//             [--dump logits.bin]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/tokenizer.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

struct Stats {
  std::uint32_t argmax_a;
  std::uint32_t argmax_b;
  double kl;
  double max_abs;
};

Stats Compare(const float* a, const float* b, std::size_t n) {
  std::vector<double> pa(n);
  std::vector<double> pb(n);
  const float ma = *std::max_element(a, a + n);
  const float mb = *std::max_element(b, b + n);
  double sa = 0.0;
  double sb = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] = std::exp(static_cast<double>(a[i] - ma));
    pb[i] = std::exp(static_cast<double>(b[i] - mb));
    sa += pa[i];
    sb += pb[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] /= sa;
    pb[i] /= sb;
    kl += pa[i] * (std::log(pa[i] + 1e-30) - std::log(pb[i] + 1e-30));
  }
  return {static_cast<std::uint32_t>(std::max_element(a, a + n) - a),
          static_cast<std::uint32_t>(std::max_element(b, b + n) - b), kl,
          max_abs};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string mtp_path;
  std::string prompt = "The capital of France is";
  std::string dump_path;
  int generate = 0;
  bool reference = false;
  bool repeat_check = false;
  bool warm = false;
  int spec = 0;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp") {
      mtp_path = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--generate") {
      generate = std::atoi(next().c_str());
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--repeat") {
      repeat_check = true;
    } else if (arg == "--warm") {
      warm = true;
    } else if (arg == "--spec") {
      spec = std::atoi(next().c_str());
    } else if (arg == "--batch") {
      batch = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--context") {
      context = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--dump") {
      dump_path = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  std::unique_ptr<gufo::core::GgufReader> mtp_reader;
  std::optional<q::MtpWeights> mtp;
  if (!mtp_path.empty()) {
    mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
    if (!mtp_reader) {
      std::fprintf(stderr, "mtp open failed: %s\n", error.c_str());
      return 1;
    }
    mtp = q::MtpWeights::Bind(*mtp_reader, weights->config, &error);
    if (!mtp) {
      std::fprintf(stderr, "mtp bind failed: %s\n", error.c_str());
      return 1;
    }
  }
  auto tokenizer = q::Tokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = weights->config;
  std::unique_ptr<q::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram =
        q::NgramTable::Open(q::ShardPath(model_path, t.shard), t.file_offset,
                            t.rows, c.ple_head_dim, t.type, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  auto device = q::rocm::DeviceModel::Upload(
      *weights, model_path, mtp ? &*mtp : nullptr, mtp_path, &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  auto secs = [&](auto since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         since)
        .count();
  };
  std::printf("uploaded %.2f GiB in %.1f s\n",
              static_cast<double>(device->resident_bytes()) / (1ULL << 30),
              secs(t0));
  q::rocm::Executor::Options options;
  options.max_batch = batch;
  options.max_logit_rows = std::min<std::uint32_t>(batch, 64);
  options.max_speculative = static_cast<std::uint32_t>(std::max(spec, 1));
  if (spec > 0 && !device->has_mtp()) {
    std::fprintf(stderr, "--spec needs --mtp\n");
    return 2;
  }
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  auto session = executor->CreateSession(context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
  }

  // Prefill in batches; keep the logits of the whole prompt when it fits the
  // logit window so the reference can be compared position by position. The
  // draft block follows every chunk: MTP position i consumes token i+1 and
  // the trunk residual of position i.
  std::vector<float> gpu_logits;
  std::size_t logit_rows = 0;
  std::vector<float> draft_logits(c.vocab_size);
  if (warm) {
    // An untimed pass first: GEMM tuning and arena growth happen once.
    for (std::size_t off = 0; off < tokens.size(); off += batch) {
      const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
      std::vector<float> out(c.vocab_size);
      if (!executor->Forward(
              *session, std::span<const std::int32_t>(tokens.data() + off, n),
              1, out.data(), false, &error)) {
        std::fprintf(stderr, "warm-up failed: %s\n", error.c_str());
        return 1;
      }
    }
    session->Reset();
  }
  t0 = std::chrono::steady_clock::now();
  for (std::size_t off = 0; off < tokens.size(); off += batch) {
    const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(n, options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), false, &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
    if (spec > 0) {
      // Tokens off+1 .. off+n (the next chunk's first token, or the first
      // generated token for the last chunk) against hidden rows 0 .. n-1.
      std::vector<std::int32_t> next(
          tokens.begin() + static_cast<std::ptrdiff_t>(off + 1),
          tokens.begin() + static_cast<std::ptrdiff_t>(off + n));
      if (last) {
        next.push_back(static_cast<std::int32_t>(
            std::max_element(gpu_logits.end() - c.vocab_size,
                             gpu_logits.end()) -
            (gpu_logits.end() - c.vocab_size)));
      } else {
        next.push_back(tokens[off + n]);
      }
      if (!executor->MtpForward(*session, next, 0, draft_logits.data(),
                                &error)) {
        std::fprintf(stderr, "draft prefill failed: %s\n", error.c_str());
        return 1;
      }
    }
  }
  const double prefill_s = secs(t0);
  std::printf("prefill %zu tokens in %.3f s (%.1f tok/s)\n", tokens.size(),
              prefill_s, static_cast<double>(tokens.size()) / prefill_s);
  if (repeat_check && tokens.size() <= batch) {
    // Determinism check: the same prefill on a fresh session must reproduce
    // the logits bit for bit.
    std::unique_ptr<q::rocm::Session> fresh;
    q::rocm::Session* again = session.get();
    if (std::getenv("QFN_REPEAT_FRESH") != nullptr) {
      fresh = executor->CreateSession(context, &error);
      again = fresh.get();
    } else {
      session->Reset();
    }
    std::vector<float> out(gpu_logits.size());
    if (!again || !executor->Forward(*again, tokens,
                                     static_cast<std::uint32_t>(logit_rows),
                                     out.data(), false, &error)) {
      std::fprintf(stderr, "repeat failed: %s\n", error.c_str());
      return 1;
    }
    double max_abs = 0.0;
    std::size_t first_diff = out.size();
    for (std::size_t i = 0; i < out.size(); ++i) {
      const double d = std::fabs(static_cast<double>(out[i] - gpu_logits[i]));
      if (d > 0.0 && first_diff == out.size()) {
        first_diff = i;
      }
      max_abs = std::max(max_abs, d);
    }
    std::printf(
        "repeat: max|d| %.6f first diff at row %zu\n", max_abs,
        first_diff == out.size() ? logit_rows : first_diff / c.vocab_size);
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
  }

  if (reference) {
    q::ReferenceModel ref(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()));
    std::vector<float> ref_logits(c.vocab_size);
    const std::size_t first = tokens.size() - logit_rows;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!ref.Step(tokens[i], ref_logits, {}, &error)) {
        std::fprintf(stderr, "reference failed: %s\n", error.c_str());
        return 1;
      }
      if (i < first) {
        continue;
      }
      const Stats st =
          Compare(ref_logits.data(),
                  gpu_logits.data() + (i - first) * c.vocab_size, c.vocab_size);
      std::printf("pos %3zu ref %6u gpu %6u %s  KL %.5f  max|d| %.3f\n", i,
                  st.argmax_a, st.argmax_b,
                  st.argmax_a == st.argmax_b ? "OK  " : "DIFF", st.kl,
                  st.max_abs);
    }
  }

  auto argmax = [&](const float* row) {
    return static_cast<std::int32_t>(std::max_element(row, row + c.vocab_size) -
                                     row);
  };
  std::vector<std::uint32_t> generated;
  std::vector<float> logits(c.vocab_size);
  std::copy_n(gpu_logits.data() + (logit_rows - 1) * c.vocab_size, c.vocab_size,
              logits.begin());
  std::int32_t next_token = argmax(logits.data());
  t0 = std::chrono::steady_clock::now();
  if (spec > 0) {
    // Greedy draft-verify cycles: the trunk runs [x1, d1..dK-1] and keeps
    // the accepted prefix; the draft block re-syncs on the trunk's hidden
    // rows of the accepted tokens and proposes the next chain.
    std::size_t cycles = 0;
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    std::vector<float> verify(static_cast<std::size_t>(spec) * c.vocab_size);
    while (generated.size() < static_cast<std::size_t>(generate)) {
      std::vector<std::int32_t> chain{next_token};
      std::int32_t draft = argmax(draft_logits.data());
      for (int j = 1; j < spec; ++j) {
        chain.push_back(draft);
        if (j + 1 < spec &&
            !executor->MtpForward(*session,
                                  std::span<const std::int32_t>(&draft, 1), -1,
                                  draft_logits.data(), &error)) {
          std::fprintf(stderr, "draft failed: %s\n", error.c_str());
          return 1;
        }
        draft = argmax(draft_logits.data());
      }
      const auto k = static_cast<std::uint32_t>(chain.size());
      const std::uint32_t base = session->position();
      if (!executor->Forward(*session, chain, k, verify.data(), true, &error)) {
        std::fprintf(stderr, "verify failed: %s\n", error.c_str());
        return 1;
      }
      std::uint32_t keep = 1;
      while (keep < k &&
             argmax(verify.data() + (keep - 1) * c.vocab_size) == chain[keep]) {
        ++keep;
      }
      next_token = argmax(verify.data() + (keep - 1) * c.vocab_size);
      if (!executor->Rollback(*session, keep, &error)) {
        std::fprintf(stderr, "rollback failed: %s\n", error.c_str());
        return 1;
      }
      for (std::uint32_t i = 0; i < keep; ++i) {
        generated.push_back(static_cast<std::uint32_t>(chain[i]));
      }
      ++cycles;
      drafted += k - 1;
      accepted += keep - 1;
      // Re-sync the draft block: positions base .. base+keep-1 consume
      // chain[1..keep-1] and the new token, with the trunk's true hidden.
      std::vector<std::int32_t> resync(chain.begin() + 1, chain.begin() + keep);
      resync.push_back(next_token);
      executor->MtpRewind(*session, base);
      if (!executor->MtpForward(*session, resync, 0, draft_logits.data(),
                                &error)) {
        std::fprintf(stderr, "draft resync failed: %s\n", error.c_str());
        return 1;
      }
    }
    const double decode_s = secs(t0);
    std::printf(
        "spec decode %zu tokens in %.3f s (%.2f tok/s), %zu cycles, "
        "acceptance %.1f%% (%zu/%zu)\n",
        generated.size(), decode_s, generated.size() / decode_s, cycles,
        drafted > 0 ? 100.0 * accepted / drafted : 0.0, accepted, drafted);
  } else {
    for (int i = 0; i < generate; ++i) {
      generated.push_back(static_cast<std::uint32_t>(next_token));
      if (!executor->Forward(*session,
                             std::span<const std::int32_t>(&next_token, 1), 1,
                             logits.data(), false, &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
      }
      next_token = argmax(logits.data());
    }
    if (generate > 0) {
      const double decode_s = secs(t0);
      std::printf("decode %d tokens in %.3f s (%.2f tok/s)\n", generate,
                  decode_s, generate / decode_s);
    }
  }
  if (!generated.empty()) {
    std::printf("generated: %s\n", tokenizer->Decode(generated).c_str());
  }
  return 0;
}
