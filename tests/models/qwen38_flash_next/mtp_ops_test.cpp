#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

struct DeviceDelete {
  void operator()(void* pointer) const { (void)hipFree(pointer); }
};

template<typename T>
auto Allocate(std::size_t count) {
  T* pointer = nullptr;
  CheckHip(hipMalloc(&pointer, count * sizeof(T)), "allocate");
  return std::unique_ptr<T, DeviceDelete>(pointer);
}

void CheckArgmax(std::uint32_t vocab) {
  constexpr std::uint32_t rows = 10;
  constexpr std::int32_t guard = -1234567;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
  for (std::uint32_t t = 0; t < rows; ++t) {
    float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
    for (std::uint32_t i = 0; i < vocab; ++i) {
      row[i] = -static_cast<float>((i * 7919U + t * 31U) % 65521U);
    }
    switch (t) {
      case 0:
        row[vocab - 1] = 10.0F;
        break;
      case 1:
        row[0] = row[vocab - 1] = 10.0F;
        break;
      case 2:
        row[vocab / 3] = row[vocab - 1] = inf;
        break;
      case 3:
        std::fill_n(row, vocab, -inf);
        break;
      case 4:
        row[0] = nan;
        break;
      case 5:
        row[vocab - 1] = nan;
        row[vocab / 2] = 10.0F;
        break;
      case 6:
        std::fill_n(row, vocab, nan);
        break;
      case 7:
        std::fill_n(row, vocab, 0.0F);
        break;
      case 8:
        row[vocab / 2] = row[vocab - 1] = 10.0F;
        break;
      default:
        break;
    }
  }
  auto device_logits = Allocate<float>(logits.size());
  auto scratch = Allocate<q::ArgmaxCandidate>(rows * q::kArgmaxParts);
  auto output = Allocate<std::int32_t>(rows + 2);
  std::vector<std::int32_t> actual(rows + 2, guard);
  CheckHip(hipMemcpy(output.get(), actual.data(),
                     actual.size() * sizeof(actual[0]), hipMemcpyHostToDevice),
           "initialize output");
  // Reuse the captured selection with changed logits, as the draft executor
  // does.
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  CheckHip(hipStreamCreate(&stream), "create stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "capture");
  q::Argmax(device_logits.get(), scratch.get(), output.get() + 1, rows, vocab,
            stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "finish capture");
  CheckHip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
           "instantiate");
  for (int replay = 0; replay < 2; ++replay) {
    CheckHip(hipMemcpyAsync(device_logits.get(), logits.data(),
                            logits.size() * sizeof(float),
                            hipMemcpyHostToDevice, stream),
             "upload logits");
    CheckHip(hipGraphLaunch(executable, stream), "select");
    CheckHip(hipMemcpyAsync(actual.data(), output.get(),
                            actual.size() * sizeof(actual[0]),
                            hipMemcpyDeviceToHost, stream),
             "download tokens");
    CheckHip(hipStreamSynchronize(stream), "synchronize");
    if (actual.front() != guard || actual.back() != guard) {
      throw std::runtime_error("argmax overwrote output guard");
    }
    for (std::uint32_t t = 0; t < rows; ++t) {
      const float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
      const auto expected =
          static_cast<std::int32_t>(std::max_element(row, row + vocab) - row);
      if (actual[t + 1] != expected) {
        throw std::runtime_error(
            "argmax mismatch: vocab=" + std::to_string(vocab) + " row=" +
            std::to_string(t) + " expected=" + std::to_string(expected) +
            " actual=" + std::to_string(actual[t + 1]));
      }
    }
    std::reverse(logits.begin(), logits.end());
  }
  CheckHip(hipGraphExecDestroy(executable), "destroy executable");
  CheckHip(hipGraphDestroy(graph), "destroy graph");
  CheckHip(hipStreamDestroy(stream), "destroy stream");
  std::cout << "MTP argmax vocab=" << vocab
            << ": CPU oracle and graph replay passed\n";
}

}  // namespace

int main() {
  try {
    CheckArgmax(1);
    CheckArgmax(257);
    CheckArgmax(248320);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
