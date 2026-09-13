#include "src/core/gguf_identity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace gufo::core {
namespace {

constexpr std::size_t kSampleWindowBytes = 4096;

class IdentityStream {
public:
  void Tag(std::string_view tag) {
    Bytes(std::as_bytes(std::span(tag)));
    const std::uint8_t terminator = 0;
    hasher_.Update(std::span(&terminator, 1));
  }

  void U64(std::uint64_t value) {
    std::array<std::uint8_t, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
    hasher_.Update(encoded);
  }

  void I64(std::int64_t value) { U64(static_cast<std::uint64_t>(value)); }
  void F64(double value) { U64(std::bit_cast<std::uint64_t>(value)); }

  void String(std::string_view value) {
    U64(value.size());
    Bytes(std::as_bytes(std::span(value)));
  }

  void Bytes(std::span<const std::byte> bytes) {
    hasher_.Update(std::span(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  }

  [[nodiscard]] std::string FinishHex() { return hasher_.FinishHex(); }

private:
  crypto::Sha256Hasher hasher_;
};

void StreamMetadataValue(IdentityStream& stream,
                         const GgufMetadataValue& entry) {
  stream.U64(static_cast<std::uint64_t>(entry.type));
  stream.U64(entry.value.index());
  std::visit(
      [&stream](const auto& value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, std::uint64_t>) {
          stream.U64(value);
        } else if constexpr (std::is_same_v<Value, std::int64_t>) {
          stream.I64(value);
        } else if constexpr (std::is_same_v<Value, double>) {
          stream.F64(value);
        } else if constexpr (std::is_same_v<Value, bool>) {
          stream.U64(value ? 1 : 0);
        } else if constexpr (std::is_same_v<Value, std::string_view>) {
          stream.String(value);
        } else if constexpr (std::is_same_v<Value,
                                            std::vector<std::string_view>>) {
          stream.U64(value.size());
          for (const auto item : value) {
            stream.String(item);
          }
        } else if constexpr (std::is_same_v<Value,
                                            std::vector<std::uint64_t>>) {
          stream.U64(value.size());
          for (const auto item : value) {
            stream.U64(item);
          }
        } else if constexpr (std::is_same_v<Value, std::vector<std::int64_t>>) {
          stream.U64(value.size());
          for (const auto item : value) {
            stream.I64(item);
          }
        } else if constexpr (std::is_same_v<Value, std::vector<double>>) {
          stream.U64(value.size());
          for (const auto item : value) {
            stream.F64(item);
          }
        }
      },
      entry.value);
}

using SampleWindow = std::span<const std::byte>;

/// Start, middle, and end windows of one tensor's payload extent. Extents
/// shorter than three windows are sampled whole.
void AppendTensorWindows(std::vector<SampleWindow>& windows,
                         const std::byte* begin, std::size_t extent) {
  if (extent <= 3 * kSampleWindowBytes) {
    windows.emplace_back(begin, extent);
    return;
  }
  const std::size_t middle =
      ((extent - kSampleWindowBytes) / 2) & ~(kSampleWindowBytes - 1);
  windows.emplace_back(begin, kSampleWindowBytes);
  windows.emplace_back(begin + middle, kSampleWindowBytes);
  windows.emplace_back(begin + extent - kSampleWindowBytes, kSampleWindowBytes);
}

/// Asks the kernel to fetch the sampled pages before they are touched.
/// Faulting them one at a time would serialize thousands of reads and drag the
/// device's readahead window (megabytes) in behind each 4 KiB sample.
void PrefetchWindows(const GgufReader& reader,
                     std::span<const SampleWindow> windows) {
  for (const SampleWindow& window : windows) {
    reader.PrefetchMapped(window.data(), window.size());
  }
}

}  // namespace

std::string GgufSampledIdentityHex(const GgufReader& reader) {
  IdentityStream stream;
  stream.Tag(kGgufSampledIdentityScheme);

  stream.Tag("header");
  stream.U64(reader.GetVersion());
  stream.U64(reader.GetAlignment());
  const auto regions = reader.GetMappedRegions();
  stream.U64(regions.size());
  for (const GgufMappedRegion& region : regions) {
    stream.U64(region.size);
  }

  stream.Tag("metadata");
  const auto keys = reader.GetMetadataKeys();
  stream.U64(keys.size());
  for (const std::string_view key : keys) {
    stream.String(key);
    StreamMetadataValue(stream, *reader.FindMetadata(key));
  }

  const auto tensors = reader.GetTensors();
  stream.Tag("tensors");
  stream.U64(tensors.size());
  for (const GgufTensorInfo& tensor : tensors) {
    stream.String(tensor.name);
    stream.U64(tensor.dimensions.size());
    for (const std::uint64_t dimension : tensor.dimensions) {
      stream.U64(dimension);
    }
    stream.U64(static_cast<std::uint64_t>(tensor.type));
    stream.U64(tensor.offset);
  }

  // A tensor's byte extent is the span from its payload to the next payload
  // in the same region (or the region end). Deriving it from the layout keeps
  // this independent of per-type size tables, including custom quantizations.
  std::vector<SampleWindow> windows;
  std::vector<std::uint64_t> extents;
  for (const GgufMappedRegion& region : regions) {
    const auto* region_begin = static_cast<const std::byte*>(region.data);
    const std::byte* region_end = region_begin + region.size;
    std::vector<const std::byte*> starts;
    for (const GgufTensorInfo& tensor : tensors) {
      const auto* payload = static_cast<const std::byte*>(tensor.data);
      if (payload != nullptr && payload >= region_begin &&
          payload < region_end) {
        starts.push_back(payload);
      }
    }
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    extents.push_back(starts.size());
    for (std::size_t index = 0; index < starts.size(); ++index) {
      const std::byte* next =
          index + 1 < starts.size() ? starts[index + 1] : region_end;
      const auto extent = static_cast<std::size_t>(next - starts[index]);
      extents.push_back(extent);
      AppendTensorWindows(windows, starts[index], extent);
    }
  }

  PrefetchWindows(reader, windows);
  stream.Tag("payload");
  for (const std::uint64_t extent : extents) {
    stream.U64(extent);
  }
  for (const SampleWindow& window : windows) {
    stream.Bytes(window);
  }
  return stream.FinishHex();
}

}  // namespace gufo::core
