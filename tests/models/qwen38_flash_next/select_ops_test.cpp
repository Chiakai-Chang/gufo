#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

bool Run(std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t seed) {
  const std::uint32_t max_context = start_pos + n_tokens + 64;
  const std::uint32_t max_blocks = (max_context + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count =
      static_cast<std::size_t>(n_tokens) * kHeads * kDim;
  const auto qv = MakeValues(q_count, seed, 1.0F);
  const auto blocks = MakeValues(static_cast<std::size_t>(max_blocks) * kDim,
                                 seed ^ 0x9999U, 1.0F);
  float* d_q = Upload(qv);
  float* d_blocks = Upload(blocks);
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
    std::vector<double> ref(complete);
    for (std::uint32_t b = 0; b < complete; ++b) {
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
      ref[b] = total;
      const float got = scores[static_cast<std::size_t>(t) * max_blocks + b];
      worst_score =
          std::max(worst_score, std::abs(total - got) / std::max(1.0, total));
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
  return worst_score < 1e-4 && mismatches == 0;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = Run(100, 20000, 0x1234ABCDU) && ok;  // deep, ragged group
    ok = Run(1, 9001, 0x0BADF00DU) && ok;     // decode
    ok = Run(40, 2040, 0xDEADBEEFU) && ok;    // straddles the budget
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
