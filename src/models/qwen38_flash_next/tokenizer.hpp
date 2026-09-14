#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_TOKENIZER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

using TokenId = std::uint32_t;

constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;
constexpr TokenId kDefaultEosTokenId = 248046U;  // <|im_end|>
constexpr TokenId kDefaultEndoftextId = 248044U;  // <|endoftext|>

/// Configuration options for the tokenizer encoding pass.
struct TokenizerOptions {
  bool add_bos{false};
  bool add_eos{false};
  bool parse_special_tokens{true};
};

struct VocabularyLoadOptions {
  bool eager_decoded_tokens{true};
};

/// Deterministic, zero-allocation-on-query Qwen BPE Tokenizer.
class Tokenizer {
public:
  ~Tokenizer() = default;

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&& other) noexcept = default;
  Tokenizer& operator=(Tokenizer&& other) noexcept = default;

  /// Creates a tokenizer by extracting vocabulary and merges from GGUF
  /// metadata.
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Creates a tokenizer from a binary vocabulary file.
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromBinaryFile(
      const std::string& path, std::string* error_msg = nullptr);

  /// Creates a tokenizer directly from token and merge lists (useful for tests
  /// and custom models).
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      const std::unordered_map<std::string, TokenId>& special_tokens = {},
      std::string* error_msg = nullptr,
      VocabularyLoadOptions load_options = {});

  /// Encodes a UTF-8 text string into token IDs.
  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text, const TokenizerOptions& options = {}) const;

  /// Decodes a sequence of token IDs into a UTF-8 string.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;

  /// Decodes a single token ID into its string representation.
  [[nodiscard]] std::string_view DecodeToken(TokenId token_id) const noexcept;
  /// Decodes a single token by value. This remains exact when eager decoded
  /// token construction was disabled at load time.
  [[nodiscard]] std::string DecodeTokenCopy(TokenId token_id) const;

  [[nodiscard]] std::size_t GetVocabSize() const noexcept {
    return id_to_token_.size();
  }

  [[nodiscard]] TokenId GetEosTokenId() const noexcept { return eos_token_id_; }
  [[nodiscard]] TokenId GetBosTokenId() const noexcept { return bos_token_id_; }
  [[nodiscard]] TokenId GetPadTokenId() const noexcept { return pad_token_id_; }

  [[nodiscard]] std::optional<TokenId> FindSpecialToken(
      std::string_view token_str) const noexcept;
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;

private:
  enum class PreTokenizer : std::uint8_t {
    kNone,
    kQwen35,
  };

  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^
             static_cast<std::size_t>(p.second);
    }
  };

  Tokenizer() = default;

  void InitializeByteTokens(bool eager_decoded_tokens = true);
  std::vector<TokenId> BpeMergeChunk(std::string_view chunk) const;
  std::vector<TokenId> BpeEncodeText(std::string_view text) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, std::uint32_t, PairHash>
      merge_ranks_;
  std::unordered_map<std::string, TokenId> special_token_to_id_;
  std::unordered_map<TokenId, bool> is_special_token_;

  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_token_id_{kInvalidTokenId};
  TokenId eos_token_id_{kDefaultEosTokenId};
  TokenId pad_token_id_{kDefaultEndoftextId};
  PreTokenizer pre_tokenizer_{PreTokenizer::kNone};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_TOKENIZER_HPP_
