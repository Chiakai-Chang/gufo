#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's linear-attention geometry: 16 key heads, 48 value heads,
// 128-wide state, kernel-4 causal conv.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kKHeads = 16;
constexpr std::uint32_t kVHeads = 48;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kKernel = 4;
constexpr std::uint32_t kChannels = 2 * kKHeads * kDim + kVHeads * kDim;
constexpr std::uint32_t kZ = kVHeads * kDim;
constexpr float kEps = 1e-6F;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

double WorstRelative(const std::vector<float>& reference,
                     const std::vector<float>& candidate) {
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double scale =
        std::max(1e-3, std::abs(static_cast<double>(reference[i])));
    worst = std::max(
        worst,
        std::abs(static_cast<double>(reference[i] - candidate[i])) / scale);
  }
  return worst;
}

}  // namespace

int main() {
  try {
    constexpr std::size_t kQkvCount =
        static_cast<std::size_t>(kTokens) * kChannels;
    constexpr std::size_t kStateCount =
        static_cast<std::size_t>(kVHeads) * kDim * kDim;
    constexpr std::size_t kConvState =
        static_cast<std::size_t>(kKernel - 1) * kChannels;
    constexpr std::size_t kScratch =
        static_cast<std::size_t>(kTokens + kKernel) * kChannels;
    constexpr std::size_t kQn =
        static_cast<std::size_t>(kTokens) * kKHeads * kDim;
    constexpr std::size_t kRaw =
        static_cast<std::size_t>(kTokens) * kVHeads * kDim;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kZ;

    const auto qkv = MakeValues(kQkvCount, 0x1234ABCDU, 1.0F);
    const auto z = MakeValues(kOut, 0x0BADF00DU, 2.0F);
    const auto alpha_beta = MakeValues(
        static_cast<std::size_t>(kTokens) * 2 * kVHeads, 0xBADC0FFEU, 2.0F);
    const auto conv_w = MakeValues(
        static_cast<std::size_t>(kChannels) * kKernel, 0xDEADBEEFU, 0.5F);
    const auto a = MakeValues(kVHeads, 0xC0FFEE11U, 1.0F, -1.5F);
    const auto dt = MakeValues(kVHeads, 0xFEEDFACEU, 1.0F);
    const auto norm_w = MakeValues(kDim, 0x600DCAFEU, 0.5F, 1.0F);
    const auto conv_state = MakeValues(kConvState, 0x13579BDFU, 1.0F);
    const auto state = MakeValues(kStateCount, 0x2468ACE0U, 0.1F);

    HipBuffer<float> d_qkv(kQkvCount);
    HipBuffer<float> d_z(kOut);
    HipBuffer<float> d_alpha_beta(alpha_beta.size());
    HipBuffer<float> d_conv_w(conv_w.size());
    HipBuffer<float> d_a(kVHeads);
    HipBuffer<float> d_dt(kVHeads);
    HipBuffer<float> d_norm_w(kDim);
    HipBuffer<float> d_scratch(kScratch);
    HipBuffer<float> d_qn(kQn);
    HipBuffer<float> d_kn(kQn);
    HipBuffer<float> d_raw(kRaw);
    Upload(&d_qkv, qkv);
    Upload(&d_z, z);
    Upload(&d_alpha_beta, alpha_beta);
    Upload(&d_conv_w, conv_w);
    Upload(&d_a, a);
    Upload(&d_dt, dt);
    Upload(&d_norm_w, norm_w);

    std::vector<float> outs[2];
    std::vector<float> states[2];
    for (int route = 0; route < 2; ++route) {
      HipBuffer<float> d_conv_state(kConvState);
      HipBuffer<float> d_state(kStateCount);
      HipBuffer<float> d_out(kOut);
      Upload(&d_conv_state, conv_state);
      Upload(&d_state, state);
      q::GatedDeltaNet(
          d_qkv.get(), kChannels, d_z.get(), kZ, d_alpha_beta.get(),
          d_conv_w.get(), d_a.get(), d_dt.get(), d_norm_w.get(),
          d_conv_state.get(), d_scratch.get(), d_qn.get(), d_kn.get(),
          d_raw.get(), d_state.get(), d_out.get(), nullptr, nullptr, nullptr,
          kTokens, kKHeads, kVHeads, kDim, kKernel, route == 1, kEps, nullptr);
      CheckHip(hipDeviceSynchronize(), "GDN synchronization");
      outs[route] = Download(&d_out, kOut);
      states[route] = Download(&d_state, kStateCount);
      // The causal conv's output (the scratch's first rows) against a CPU
      // reference over the same history.
      const auto conv_out = Download(&d_scratch, kScratch);
      double worst_conv = 0.0;
      for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t c = 0; c < kChannels; ++c) {
          double acc = 0.0;
          for (std::uint32_t k = 0; k < kKernel; ++k) {
            const std::int32_t src = static_cast<std::int32_t>(t + k) -
                                     static_cast<std::int32_t>(kKernel - 1);
            const double v =
                src >= 0
                    ? qkv[static_cast<std::size_t>(src) * kChannels + c]
                    : conv_state[static_cast<std::size_t>(kKernel - 1 + src) *
                                     kChannels +
                                 c];
            acc += static_cast<double>(conv_w[c * kKernel + k]) * v;
          }
          const double ref = acc / (1.0 + std::exp(-acc));
          worst_conv = std::max(
              worst_conv,
              std::abs(ref -
                       conv_out[static_cast<std::size_t>(t) * kChannels + c]) /
                  std::max(1e-3, std::abs(ref)));
        }
      }
      std::cout << "GDN conv vs CPU worst relative error " << worst_conv
                << '\n';
      if (worst_conv > 1e-4) {
        return 1;
      }
    }
    for (float v : outs[1]) {
      if (!std::isfinite(v)) {
        std::cerr << "row-split output is not finite\n";
        return 1;
      }
    }
    const double worst_out = WorstRelative(outs[0], outs[1]);
    const double worst_state = WorstRelative(states[0], states[1]);
    std::cout << "GDN row-split worst relative error: output " << worst_out
              << ", state " << worst_state << '\n';
    if (worst_out > 1e-3 || worst_state > 1e-3) {
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
