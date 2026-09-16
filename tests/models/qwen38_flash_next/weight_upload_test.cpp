#include "src/models/qwen38_flash_next/kernels/rocm/weight_upload.hpp"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using gufo::models::qwen38_flash_next::rocm::WeightUpload;

namespace {

void Require(bool ok, const std::string& message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

struct Files {
  std::filesystem::path directory;
  Files() {
    char path[] = "/tmp/qfn-weight-upload-XXXXXX";
    const char* result = ::mkdtemp(path);
    Require(result != nullptr, "mkdtemp");
    directory = result;
  }
  ~Files() { std::filesystem::remove_all(directory); }
};

struct Device {
  void* data{nullptr};
  explicit Device(std::size_t size) {
    Require(hipMalloc(&data, size) == hipSuccess, "hipMalloc");
  }
  ~Device() { (void)hipFree(data); }
};

}  // namespace

int main() {
  Files files;
  std::array<std::filesystem::path, 2> paths{files.directory / "first",
                                             files.directory / "second"};
  std::array<std::vector<std::uint8_t>, 2> source;
  source[0].resize((35U << 20) + 4079);
  source[1].resize((2U << 20) + 39);
  for (std::size_t shard = 0; shard < source.size(); ++shard) {
    for (std::size_t i = 0; i < source[shard].size(); ++i) {
      source[shard][i] = static_cast<std::uint8_t>((i * 1315423911ULL) ^
                                                   (i >> 13) ^ (shard * 79));
    }
    std::ofstream out(paths[shard], std::ios::binary);
    out.write(reinterpret_cast<const char*>(source[shard].data()),
              static_cast<std::streamsize>(source[shard].size()));
    Require(out.good(), "write shard");
  }

  constexpr std::size_t guard = 64;
  std::vector<std::uint8_t> expected(guard, 0xa5);
  struct Slice {
    std::uint32_t shard;
    std::size_t offset;
    std::size_t size;
    std::size_t destination;
  };
  std::vector<Slice> slices;
  const auto slice = [&](std::uint32_t shard, std::size_t offset,
                         std::size_t size) {
    slices.push_back({shard, offset, size, expected.size()});
    expected.insert(expected.end(), source[shard].begin() + offset,
                    source[shard].begin() + offset + size);
    expected.insert(expected.end(), guard, 0xa5);
  };
  // Several staging chunks, unaligned starts, and the partial last page at
  // EOF exercise aligned direct reads without copying surrounding bytes.
  slice(0, 19, source[0].size() - 19);
  slice(1, 33, source[1].size() - 33);
  for (std::size_t i = 0; i < 100; ++i) {
    slice(static_cast<std::uint32_t>(i % 2), i * 4093 + 7, i * 71 + 1);
  }
  Device device(expected.size());
  Require(hipMemset(device.data, 0xa5, expected.size()) == hipSuccess,
          "initialize guards");
  std::string error;
  auto upload = WeightUpload::Create(paths, &error);
  Require(upload != nullptr, error);
  for (const auto& s : slices) {
    Require(
        upload->Copy(s.shard, s.offset, s.size,
                     static_cast<char*>(device.data) + s.destination, &error),
        error);
  }
  Require(upload->Copy(1, source[1].size(), 0, nullptr, &error),
          "empty range at EOF");
  Require(upload->Finish(&error), error);
  std::vector<std::uint8_t> actual(expected.size());
  Require(hipMemcpy(actual.data(), device.data, actual.size(),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "download uploaded bytes");
  Require(actual == expected, "all shard bytes and destination guards match");

  // A rejected range cancels queued reads and still drains active transfers.
  for (int i = 0; i < 32; ++i) {
    Require(upload->Copy(0, 0, 4096, static_cast<char*>(device.data) + i * 4096,
                         &error),
            error);
  }
  Require(!upload->Copy(1, std::numeric_limits<std::uint64_t>::max(), 1,
                        device.data, &error),
          "reject overflowing file offset");
  Require(!upload->Finish(&error), "failed upload reports failure after drain");
  upload.reset();

  upload = WeightUpload::Create(paths, &error);
  Require(upload != nullptr, "reopen shards");
  std::filesystem::resize_file(paths[1], 17);
  Require(upload->Copy(1, 4096, 8192, device.data, &error),
          "queue a range from the original file extent");
  Require(!upload->Finish(&error), "a truncated shard fails its pending read");
  std::puts(
      "PASS: byte-exact upload, EOF, guards, bounds and failed-read drain");
}
