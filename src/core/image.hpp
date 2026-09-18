#ifndef GUFO_CORE_IMAGE_HPP_
#define GUFO_CORE_IMAGE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

inline constexpr std::size_t kMaxEncodedImageBytes = 20 * 1024 * 1024;
inline constexpr std::size_t kMaxImagePixels = 32 * 1024 * 1024;

struct Image {
  std::uint32_t width{0};
  std::uint32_t height{0};
  /// Packed, unpremultiplied RGB8, in display orientation.
  std::vector<std::uint8_t> pixels;
};

/// PNG and JPEG only. Decoding checks dimensions before allocating pixels.
[[nodiscard]] Image DecodeImage(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::vector<std::uint8_t> ReadImageFile(
    const std::filesystem::path& path);
/// OpenAI image_url transport: base64 data URLs or HTTPS (bounded in
/// size/time).
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url);

}  // namespace gufo::core
#endif
