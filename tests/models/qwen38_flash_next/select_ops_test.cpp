#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's indexer: 4 heads of 128, 4-token blocks, a 2048-token budget.
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kBudget = 2048 / kRatio;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

enum class ScoreLayout { kTight, kAligned };

bool Run(std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t seed,
         bool zero_queries = false, std::uint32_t tied_high_blocks = 0,
         bool sample_scores = false, ScoreLayout layout = ScoreLayout::kTight) {
  const std::uint32_t max_context = start_pos + n_tokens + 64;
  const std::uint32_t blocks_needed = (max_context + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (blocks_needed + 31) / 32;
  const std::uint32_t max_blocks =
      layout == ScoreLayout::kAligned ? mask_words * 32 : blocks_needed;
  const std::size_t q_count =
      static_cast<std::size_t>(n_tokens) * kHeads * kDim;
  // Zero queries make every score tie at zero: the budget must then fill
  // in index order.
  auto qv = zero_queries ? std::vector<float>(q_count, 0.0F)
                         : MakeValues(q_count, seed, 1.0F);
  auto blocks = MakeValues(static_cast<std::size_t>(max_blocks) * kDim,
                           seed ^ 0x9999U, 1.0F);
  if (tied_high_blocks != 0) {
    std::fill(qv.begin(), qv.end(), 0.0F);
    std::fill(blocks.begin(), blocks.end(), 0.0F);
    for (std::size_t i = 0; i < qv.size(); i += kDim)
      qv[i] = 1.0F;
    for (std::uint32_t b = 0; b < tied_high_blocks; ++b)
      blocks[static_cast<std::size_t>(b) * kDim] = 1.0F;
  }
  std::vector<__half> q_half(qv.size()), blocks_half(blocks.size());
  const auto half = [](float value) { return __float2half_rn(value); };
  std::transform(qv.begin(), qv.end(), q_half.begin(), half);
  std::transform(blocks.begin(), blocks.end(), blocks_half.begin(), half);
  __half* d_q = Upload(q_half);
  __half* d_blocks = Upload(blocks_half);
  std::uint32_t* d_pos = Upload(std::vector<std::uint32_t>{start_pos});
  std::uint32_t* d_mask = Upload(std::vector<std::uint32_t>(
      static_cast<std::size_t>(n_tokens) * mask_words, 0xA5A5A5A5U));
  float* d_scores = Upload(std::vector<float>(
      static_cast<std::size_t>(n_tokens) * max_blocks, -1.0F));
  q::SelectBlocks(d_q, d_blocks, d_mask, d_scores, n_tokens, d_pos, 0, kHeads,
                  kDim, kRatio, kBudget, mask_words, max_blocks, nullptr);
  CheckHip(hipDeviceSynchronize(), "selection");
  const auto mask =
      Download(d_mask, static_cast<std::size_t>(n_tokens) * mask_words);
  const auto scores =
      Download(d_scores, static_cast<std::size_t>(n_tokens) * max_blocks);
  q::SelectBlocks(d_q, d_blocks, d_mask, d_scores, n_tokens, d_pos, 0, kHeads,
                  kDim, kRatio, kBudget, mask_words, max_blocks, nullptr);
  if (Download(d_mask, mask.size()) != mask) {
    throw std::runtime_error("selection replay changed the mask");
  }

  double worst_score = 0.0;
  std::size_t mismatches = 0;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / kRatio;
    const std::uint32_t* words =
        mask.data() + static_cast<std::size_t>(t) * mask_words;
    std::vector<bool> expect(max_blocks, false);
    if (complete <= kBudget) {
      // Below the budget every block is visible.
      for (std::uint32_t w = 0; w < mask_words; ++w) {
        mismatches += words[w] == 0xFFFFFFFFU ? 0 : 1;
      }
      continue;
    }
    // Reference scores in double; the GPU value must agree closely.
    for (std::uint32_t b = 0; b < complete; ++b) {
      // The large selection case checks every mask bit and samples score
      // arithmetic already covered exhaustively by the smaller cases.
      if (sample_scores && b % 1024 != 0 && b + 1 != complete)
        continue;
      double total = 0.0;
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double dot = 0.0;
        for (std::uint32_t i = 0; i < kDim; ++i) {
          dot +=
              static_cast<double>(
                  qv[(static_cast<std::size_t>(t) * kHeads + h) * kDim + i]) *
              blocks[static_cast<std::size_t>(b) * kDim + i];
        }
        total += std::max(dot, 0.0);
      }
      const float got = scores[static_cast<std::size_t>(t) * max_blocks + b];
      const double err = std::abs(total - got) / std::max(1.0, total);
      worst_score = std::max(worst_score, err);
    }
    // Selection contract on the GPU's own scores: the budget highest,
    // blocks strictly above the threshold first, ties by lowest index.
    std::vector<float> sorted(
        scores.begin() + static_cast<std::size_t>(t) * max_blocks,
        scores.begin() + static_cast<std::size_t>(t) * max_blocks + complete);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float threshold = sorted[kBudget - 1];
    std::uint32_t remaining = kBudget;
    for (std::uint32_t b = 0; b < complete; ++b) {
      if (scores[static_cast<std::size_t>(t) * max_blocks + b] > threshold) {
        expect[b] = true;
        --remaining;
      }
    }
    for (std::uint32_t b = 0; b < complete && remaining > 0; ++b) {
      if (scores[static_cast<std::size_t>(t) * max_blocks + b] == threshold) {
        expect[b] = true;
        --remaining;
      }
    }
    for (std::uint32_t b = 0; b < max_blocks; ++b) {
      const bool got = ((words[b / 32] >> (b % 32)) & 1U) != 0;
      mismatches += got == expect[b] ? 0 : 1;
    }
  }
  std::cout << "block selection n=" << n_tokens << " start=" << start_pos
            << ": worst score error " << worst_score << ", mask mismatches "
            << mismatches << '\n';
  (void)hipFree(d_q);
  (void)hipFree(d_blocks);
  (void)hipFree(d_pos);
  (void)hipFree(d_mask);
  (void)hipFree(d_scores);
  // The scores ride F16 fragments on the matrix cores, so they agree with
  // the F64 reference to about 1e-3; the mask is exact over the GPU scores.
  return worst_score < 1e-2 && mismatches == 0;
}

void CheckPooling(unsigned start, unsigned capacity) {
  constexpr unsigned tokens = 7;
  const unsigned first = start / kRatio;
  const unsigned blocks = (start + tokens) / kRatio + 3;
  constexpr unsigned rotary = 64;
  constexpr float theta = 1000000.0F, eps = 1e-6F;
  const auto raw = MakeValues((start + tokens) * kDim, 0x31415926U, 1.0F);
  const auto gamma = MakeValues(kDim, 0x27182818U, 1.0F);
  const std::vector<__half> initial(blocks * kDim, __float2half(3.0F));
  // Keep only the newest raw rows. Store the next batch through the same
  // ring operation used by the executor, including a wrap within a block
  // of speculative tokens.
  std::vector<float> ring(capacity * kDim, -123.0F);
  for (unsigned t = start > capacity ? start - capacity : 0; t < start; ++t)
    std::copy_n(raw.data() + std::size_t{t} * kDim, kDim,
                ring.data() + std::size_t{t & (capacity - 1)} * kDim);
  auto* d_raw = Upload(ring);
  auto* d_input =
      Upload(std::vector<float>(raw.begin() + start * kDim, raw.end()));
  auto* d_gamma = Upload(gamma);
  auto* d_blocks = Upload(initial);
  auto* d_first = Upload(std::vector<unsigned>{first});
  auto* d_start = Upload(std::vector<unsigned>{start});
  const auto run = [&] {
    q::StoreRows(d_input, d_raw, tokens, kDim, d_start, capacity, nullptr);
    q::PoolIndexerBlocks(d_raw, d_gamma, d_blocks, d_first, d_start, tokens, 4,
                         kRatio, kDim, rotary, theta, eps, capacity, nullptr);
  };
  run();
  const auto actual = Download(d_blocks, initial.size());
  run();
  const auto replay = Download(d_blocks, initial.size());
  if (std::memcmp(actual.data(), replay.data(),
                  actual.size() * sizeof(__half)) != 0) {
    throw std::runtime_error("pooling replay changed the block keys");
  }
  double worst = 0.0;
  for (unsigned block = 0; block < blocks; ++block) {
    if (block < first || block >= (start + tokens) / kRatio) {
      for (unsigned i = 0; i < kDim; ++i) {
        if (__half2float(actual[block * kDim + i]) != 3.0F) {
          throw std::runtime_error("pooling wrote an incomplete or old block");
        }
      }
      continue;
    }
    std::vector<double> values(kDim);
    double squares = 0.0;
    for (unsigned i = 0; i < kDim; ++i) {
      for (unsigned r = 0; r < kRatio; ++r)
        values[i] += raw[(block * kRatio + r) * kDim + i] / double(kRatio);
      squares += values[i] * values[i];
    }
    const double scale = 1.0 / std::sqrt(squares / kDim + eps);
    for (unsigned i = 0; i < kDim; ++i)
      values[i] *= scale * gamma[i];
    for (unsigned i = 0; i < rotary / 2; ++i) {
      const double angle =
          block * kRatio * std::pow(double(theta), -2.0 * i / rotary);
      const auto a = values[i], b = values[i + rotary / 2];
      values[i] = a * std::cos(angle) - b * std::sin(angle);
      values[i + rotary / 2] = a * std::sin(angle) + b * std::cos(angle);
    }
    for (unsigned i = 0; i < kDim; ++i) {
      const double value = __half2float(actual[block * kDim + i]);
      const double error =
          std::abs(value - values[i]) / std::max(1.0, std::abs(values[i]));
      if (!std::isfinite(value) || error > 1e-3) {
        throw std::runtime_error("pooled F16 key disagrees with FP64 formula");
      }
      worst = std::max(worst, error);
    }
  }
  for (void* pointer :
       {static_cast<void*>(d_raw), static_cast<void*>(d_input),
        static_cast<void*>(d_gamma), static_cast<void*>(d_blocks),
        static_cast<void*>(d_first), static_cast<void*>(d_start)})
    CheckHip(hipFree(pointer), "free pooling input");
  std::cout << "pooled F16 keys: FP64 error " << worst
            << ", boundaries and replay exact\n";
}

}  // namespace

int main() {
  try {
    CheckPooling(29, 64);
    CheckPooling(29, 16);     // batch wraps the raw-key ring
    CheckPooling(65533, 16);  // many wraps; absolute rotary position retained
    bool ok = true;
    ok = Run(100, 20000, 0x1234ABCDU) && ok;     // deep, ragged group
    ok = Run(7, 131069, 0x2468ACE0U) && ok;
    ok = Run(129, 131069, 0xC0FFEE01U, false, 0, true,
             ScoreLayout::kAligned) && ok;
    ok = Run(129, 131069, 0, true, 0, true) && ok;  // deep, partial word
    ok = Run(1, 9001, 0x0BADF00DU) && ok;        // decode
    ok = Run(40, 2040, 0xDEADBEEFU) && ok;       // straddles the budget
    ok = Run(3, 6000, 0x5EED5EEDU, true) && ok;  // all scores tie at zero
    // A threshold tie group that fits exactly, then one that needs the
    // lowest-index prefix. Both must produce the same CPU-sorted mask.
    ok = Run(1, 9001, 0, false, kBudget) && ok;
    ok = Run(1, 9001, 0, false, kBudget + 7) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
