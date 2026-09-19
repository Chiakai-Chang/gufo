// Dumps trunk prefill logits and optionally compares them with the F32 oracle.
// Use the production CLI for generation and the session test for replay.
//
// gpu_probe --model FIRST_SHARD.gguf --prompt TEXT [--reference]
//           [--batch T] [--context N] [--dump logits.bin]
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
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
  std::string prompt = "The capital of France is";
  std::string dump_path;
  bool reference = false;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--batch" || arg == "--context") {
      const auto value = next();
      auto& count = arg == "--batch" ? batch : context;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
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
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
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

  auto device =
      q::rocm::DeviceModel::Upload(*weights, model_path, nullptr, {}, &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  q::rocm::Executor::Options options;
  options.max_batch = batch;
  options.max_logit_rows = std::min<std::uint32_t>(batch, 64);
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
  if (tokens.empty() || tokens.size() > context) {
    std::fprintf(stderr, "prompt must contain 1..%u tokens\n", context);
    return 2;
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    if (!dump) {
      std::fprintf(stderr, "cannot open logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  std::vector<float> gpu_logits;
  std::size_t logit_rows = 0;
  for (std::size_t off = 0; off < tokens.size(); off += batch) {
    const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(n, options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), q::rocm::Executor::ForwardMode::kPrefill,
            &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
    dump.flush();
    if (!dump) {
      std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
      return 1;
    }
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

  return 0;
}
