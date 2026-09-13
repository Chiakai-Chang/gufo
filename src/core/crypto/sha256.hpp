#ifndef GUFO_CORE_CRYPTO_SHA256_HPP_
#define GUFO_CORE_CRYPTO_SHA256_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace gufo::crypto {

/// Incremental SHA-256 over a byte stream.
class Sha256Hasher {
public:
  void Update(std::span<const std::uint8_t> bytes) noexcept;
  [[nodiscard]] std::array<std::uint8_t, 32> Finish() noexcept;
  /// Finishes the digest and renders it as 64 lowercase hex characters.
  [[nodiscard]] std::string FinishHex();

private:
  void Transform() noexcept;

  std::array<std::uint32_t, 8> state_ = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U,
                                         0xA54FF53AU, 0x510E527FU, 0x9B05688CU,
                                         0x1F83D9ABU, 0x5BE0CD19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{0};
  std::uint64_t total_bytes_{0};
};

[[nodiscard]] std::string Sha256Hex(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string Sha256FileHex(const std::filesystem::path& path);

}  // namespace gufo::crypto

#endif  // GUFO_CORE_CRYPTO_SHA256_HPP_
