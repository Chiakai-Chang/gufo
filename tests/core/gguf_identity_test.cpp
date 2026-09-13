#include "src/core/gguf_identity.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::size_t kWindow = 4096;

/// Minimal GGUF v3 image: one string metadata entry and two tensors whose
/// payloads span `first_bytes` and `second_bytes`.
std::vector<std::uint8_t> BuildImage(std::string_view name_value,
                                     std::string_view first_tensor_name,
                                     std::size_t first_bytes,
                                     std::size_t second_bytes) {
  std::vector<std::uint8_t> buffer;
  auto append = [&buffer](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
  };
  auto pod = [&append](auto value) { append(&value, sizeof(value)); };
  auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    append(value.data(), value.size());
  };

  append("GGUF", 4);
  pod(static_cast<std::uint32_t>(3));
  pod(static_cast<std::uint64_t>(2));  // tensor count
  pod(static_cast<std::uint64_t>(1));  // metadata count
  string("general.name");
  pod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
  string(name_value);

  string(first_tensor_name);
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(first_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(0));
  string("second");
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(second_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(first_bytes));

  while (buffer.size() % 32 != 0) {
    buffer.push_back(0);
  }
  const std::size_t payload_base = buffer.size();
  buffer.resize(payload_base + first_bytes + second_bytes);
  for (std::size_t index = 0; index < first_bytes + second_bytes; ++index) {
    buffer[payload_base + index] = static_cast<std::uint8_t>(index * 7U + 3U);
  }
  return buffer;
}

std::size_t PayloadBase(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return static_cast<std::size_t>(
      static_cast<const std::uint8_t*>(reader->GetTensors()[0].data) -
      image.data());
}

std::string Identity(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return gufo::core::GgufSampledIdentityHex(*reader);
}

void TestDeterministic() {
  const auto image = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const std::string first = Identity(image);
  Expect(first.size() == 64, "identity is 64 hex characters");
  Expect(first == Identity(image), "identity is deterministic");
  Expect(first ==
             Identity(BuildImage("model", "first", 64 * kWindow, 2 * kWindow)),
         "identity depends on bytes, not on the buffer instance");
}

void TestHeaderChangesIdentity() {
  const auto base = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  Expect(Identity(base) != Identity(BuildImage("model-v2", "first",
                                               64 * kWindow, 2 * kWindow)),
         "metadata change alters identity");
  Expect(Identity(base) != Identity(BuildImage("model", "renamed", 64 * kWindow,
                                               2 * kWindow)),
         "tensor name change alters identity");
  Expect(Identity(base) !=
             Identity(BuildImage("model", "first", 64 * kWindow, 4 * kWindow)),
         "tensor extent change alters identity");
}

void TestSampledPayloadWindows() {
  const std::size_t first_bytes = 64 * kWindow;
  auto image = BuildImage("model", "first", first_bytes, 2 * kWindow);
  const std::string base = Identity(image);
  const std::size_t payload = PayloadBase(image);

  auto flipped = [&](std::size_t offset) {
    auto copy = image;
    copy[payload + offset] ^= 0xFFU;
    return Identity(copy);
  };

  Expect(flipped(0) != base, "start window is sampled");
  Expect(flipped(first_bytes - 1) != base, "end window is sampled");
  const std::size_t middle = ((first_bytes - kWindow) / 2) & ~(kWindow - 1);
  Expect(flipped(middle + 17) != base, "middle window is sampled");
  Expect(flipped(kWindow + 100) == base,
         "bytes outside the sampled windows do not alter identity");

  // The short second tensor is hashed whole.
  Expect(flipped(first_bytes + kWindow + 5) != base,
         "small tensors are hashed in full");
}

}  // namespace

int main() {
  TestDeterministic();
  TestHeaderChangesIdentity();
  TestSampledPayloadWindows();
  std::cout << "gguf_identity_test passed\n";
  return 0;
}
