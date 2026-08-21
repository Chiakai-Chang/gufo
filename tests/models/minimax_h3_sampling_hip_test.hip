#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/sampling.hpp"

namespace {

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool Allocate(std::size_t bytes) {
    return hipMalloc(&data_, bytes) == hipSuccess;
  }
  [[nodiscard]] void* data() const noexcept { return data_; }

private:
  void* data_{nullptr};
};

class Stream {
public:
  Stream() = default;
  ~Stream() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  bool Create() {
    return hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) ==
           hipSuccess;
  }
  [[nodiscard]] hipStream_t get() const noexcept { return stream_; }

private:
  hipStream_t stream_{nullptr};
};

bool CheckTarget() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return false;
  }
  return std::string(properties.gcnArchName).starts_with("gfx1151") &&
         properties.integrated != 0;
}

bool CopyToDevice(void* destination, const void* source, std::size_t bytes,
                  hipStream_t stream) {
  return hipMemcpyAsync(destination, source, bytes, hipMemcpyHostToDevice,
                        stream) == hipSuccess;
}

bool RunEulerCase(std::array<float, 7>* output) {
  constexpr std::array<float, 7> sample = {-7.0F, -8.0F, 10.0F, 20.0F,
                                           30.0F, 40.0F, -9.0F};
  constexpr std::array<float, 4> last_values = {1.0F, -2.0F, 0.5F, 4.0F};
  constexpr std::array<float, 4> previous_values = {0.5F, -1.0F, 0.25F, 5.0F};
  std::array<hip_bfloat16, 4> last{};
  std::array<hip_bfloat16, 4> previous{};
  for (std::size_t index = 0; index < last.size(); ++index) {
    last[index] = hip_bfloat16(last_values[index]);
    previous[index] = hip_bfloat16(previous_values[index]);
  }

  DeviceBuffer device_sample;
  DeviceBuffer device_last;
  DeviceBuffer device_previous;
  Stream stream;
  if (!device_sample.Allocate(sizeof(sample)) ||
      !device_last.Allocate(sizeof(last)) ||
      !device_previous.Allocate(sizeof(previous)) || !stream.Create() ||
      !CopyToDevice(device_sample.data(), sample.data(), sizeof(sample),
                    stream.get()) ||
      !CopyToDevice(device_last.data(), last.data(), sizeof(last),
                    stream.get()) ||
      !CopyToDevice(device_previous.data(), previous.data(), sizeof(previous),
                    stream.get())) {
    return false;
  }

  std::string error;
  if (!strix::minimax_h3::HipEulerUpdate(device_sample.data(), sample.size(), 2,
                                         device_last.data(),
                                         device_previous.data(), last.size(),
                                         0.25F, 0.5F, stream.get(), &error)) {
    std::cerr << error << '\n';
    return false;
  }
  if (hipMemcpyAsync(output->data(), device_sample.data(), sizeof(sample),
                     hipMemcpyDeviceToHost, stream.get()) != hipSuccess ||
      hipStreamSynchronize(stream.get()) != hipSuccess) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckTarget()) {
    std::cerr << "MiniMax H3 sampling HIP test requires integrated gfx1151\n";
    return 77;
  }
  std::array<float, 7> first{};
  std::array<float, 7> second{};
  if (!RunEulerCase(&first) || !RunEulerCase(&second)) {
    std::cerr << "MiniMax H3 HIP Euler execution failed\n";
    return 1;
  }
  constexpr std::array<float, 7> expected = {
      -7.0F, -8.0F, 10.3125F, 19.375F, 30.15625F, 40.875F, -9.0F};
  if (std::memcmp(first.data(), expected.data(), sizeof(expected)) != 0) {
    std::cerr << "MiniMax H3 HIP Euler output differs from the pinned oracle\n";
    return 1;
  }
  if (std::memcmp(first.data(), second.data(), sizeof(first)) != 0) {
    std::cerr << "MiniMax H3 HIP Euler output is not byte stable\n";
    return 1;
  }
  std::string error;
  void* const invalid_device_pointer =
      reinterpret_cast<void*>(std::uintptr_t{1});
  if (strix::minimax_h3::HipEulerUpdate(
          invalid_device_pointer, 1, 0, invalid_device_pointer,
          invalid_device_pointer, 1, 0.0F, 0.0F, nullptr, &error)) {
    std::cerr << "MiniMax H3 HIP Euler accepted a zero-delta step\n";
    return 1;
  }
  if (strix::minimax_h3::HipEulerUpdate(
          invalid_device_pointer, 1, 1, invalid_device_pointer,
          invalid_device_pointer, 1, 0.25F, 0.0F, nullptr, &error)) {
    std::cerr << "MiniMax H3 HIP Euler accepted an out-of-bounds range\n";
    return 1;
  }
  std::cout << "MiniMax H3 gfx1151 GPU-state Euler checks passed\n";
  return 0;
}
