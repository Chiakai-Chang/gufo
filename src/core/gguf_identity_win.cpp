// Windows implementation of GgufIdentityHex. Same digest as the POSIX file:
// SHA-256 over every byte of every region. Only the cache key and the cache
// location differ: the key uses the NTFS volume serial, file index, size and
// last-write time, and the cache lives under %LOCALAPPDATA%\gufo.
#include "src/core/gguf_identity.hpp"

#include <windows.h>
#include <io.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace gufo::core {
namespace {

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  hash.Update({reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

std::string FileStamp(int fd, std::size_t size) {
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  BY_HANDLE_FILE_INFORMATION info{};
  if (h == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(h, &info))
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  const unsigned long long file_size =
      (static_cast<unsigned long long>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
  if (file_size != size)
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  const unsigned long long index =
      (static_cast<unsigned long long>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
  const unsigned long long mtime =
      (static_cast<unsigned long long>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      info.ftLastWriteTime.dwLowDateTime;
  return std::to_string(info.dwVolumeSerialNumber) + ':' + std::to_string(index) + ':' +
         std::to_string(file_size) + ':' + std::to_string(mtime);
}

std::filesystem::path CacheDirectory() {
  const char* base = std::getenv("LOCALAPPDATA");
  if (!base || !*base)
    return {};
  std::filesystem::path path = std::filesystem::path(base) / "gufo" / "gguf-sha256-v1";
  std::error_code error;
  std::filesystem::create_directories(path, error);
  return error ? std::filesystem::path{} : path;
}

std::string ReadDigest(const std::filesystem::path& dir, const std::string& key) {
  if (dir.empty())
    return {};
  std::ifstream in(dir / key, std::ios::binary);
  std::array<char, 65> bytes{};
  if (!in.read(bytes.data(), bytes.size()) || bytes.back() != '\n' ||
      !std::all_of(bytes.begin(), bytes.end() - 1, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return {};
  return {bytes.data(), bytes.size() - 1};
}

void StoreDigest(const std::filesystem::path& dir, const std::string& key,
                 const std::string& digest) {
  if (dir.empty())
    return;
  const auto temp = dir / ("." + key + "." + std::to_string(GetCurrentProcessId()));
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out << digest << '\n';
    if (!out)
      return;
  }
  std::error_code error;
  std::filesystem::rename(temp, dir / key, error);
  if (error)
    std::filesystem::remove(temp, error);
}

void HashFile(crypto::Sha256Hasher& hash, int fd, std::size_t size) {
  constexpr std::size_t kChunk = 8 * 1024 * 1024;
  constexpr std::size_t kReaders = 4;
  struct Slot {
    std::vector<std::uint8_t> buffer;
    std::future<void> read;
    std::size_t size{0};
  };
  std::array<Slot, kReaders> slots;
  std::size_t next = 0;
  const auto queue = [&](Slot& slot) {
    if (next == size)
      return;
    slot.buffer.resize(kChunk);
    const auto offset = next;
    slot.size = std::min(kChunk, size - next);
    next += slot.size;
    slot.read = std::async(std::launch::async, [&slot, fd, offset] {
      const auto count = pread(fd, slot.buffer.data(), slot.size, static_cast<off_t>(offset));
      if (count < 0 || static_cast<std::size_t>(count) != slot.size)
        throw std::runtime_error("cannot read complete GGUF for identity");
    });
  };
  for (auto& slot : slots)
    queue(slot);
  for (std::size_t index = 0; slots[index].read.valid(); index = (index + 1) % slots.size()) {
    auto& slot = slots[index];
    slot.read.get();
    hash.Update({slot.buffer.data(), slot.size});
    queue(slot);
  }
}

std::string RegionDigest(const GgufMappedRegion& region) {
  std::string stamp;
  if (region.file_descriptor >= 0)
    stamp = FileStamp(region.file_descriptor, region.size);
  const auto cache = stamp.empty() ? std::filesystem::path{} : CacheDirectory();
  crypto::Sha256Hasher key_hash;
  HashString(key_hash, stamp);
  const auto key = key_hash.FinishHex();
  std::string digest = ReadDigest(cache, key);
  const bool cached = !digest.empty();
  if (!cached) {
    crypto::Sha256Hasher hash;
    constexpr std::size_t chunk = 8 * 1024 * 1024;
    if (region.file_descriptor >= 0) {
      HashFile(hash, region.file_descriptor, region.size);
    } else {
      const auto* bytes = static_cast<const std::uint8_t*>(region.data);
      for (std::size_t offset = 0; offset < region.size;) {
        const auto length = std::min(chunk, region.size - offset);
        hash.Update({bytes + offset, length});
        offset += length;
      }
    }
    digest = hash.FinishHex();
  }
  if (!stamp.empty() && FileStamp(region.file_descriptor, region.size) != stamp)
    throw std::runtime_error("GGUF file changed during identity lookup");
  if (!cached)
    StoreDigest(cache, key, digest);
  return digest;
}

}  // namespace

std::string GgufIdentityHex(const GgufReader& reader) {
  std::vector<std::string> stamps;
  for (const auto& region : reader.GetMappedRegions())
    stamps.push_back(region.file_descriptor < 0
                         ? std::string{}
                         : FileStamp(region.file_descriptor, region.size));
  crypto::Sha256Hasher hash;
  HashString(hash, kGgufIdentityScheme);
  for (const auto& region : reader.GetMappedRegions()) {
    HashString(hash, ":" + std::to_string(region.size) + ":");
    HashString(hash, RegionDigest(region));
  }
  for (std::size_t index = 0; index < stamps.size(); ++index) {
    const auto& region = reader.GetMappedRegions()[index];
    if (!stamps[index].empty() &&
        stamps[index] != FileStamp(region.file_descriptor, region.size))
      throw std::runtime_error("GGUF artifact changed during identity lookup");
  }
  return hash.FinishHex();
}

}  // namespace gufo::core
