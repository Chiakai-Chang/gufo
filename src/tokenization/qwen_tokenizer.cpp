#include "src/tokenization/qwen_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace strix::tokenization {

namespace {

std::string FormatHexByteToken(std::uint8_t byte_val) {
  char buf[8];
  static_cast<void>(std::snprintf(buf, sizeof(buf), "<0x%02X>", byte_val));
  return {buf};
}

constexpr int HexCharToInt(char c) noexcept {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

}  // namespace

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF metadata missing 'tokenizer.ggml.tokens'";
    }
    return nullptr;
  }

  std::vector<std::string> tokens;
  tokens.reserve(token_views.size());
  for (const auto& tv : token_views) {
    tokens.emplace_back(tv);
  }

  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  std::vector<std::string> merges;
  merges.reserve(merge_views.size());
  for (const auto& mv : merge_views) {
    merges.emplace_back(mv);
  }

  std::unordered_map<std::string, TokenId> special_tokens;
  // Standard Qwen special tokens
  const std::array<std::pair<std::string_view, TokenId>, 6> default_specials = {
      std::make_pair("<|endoftext|>", kDefaultQwenEndoftextId),
      std::make_pair("<|im_start|>", 151644U),
      std::make_pair("<|im_end|>", kDefaultQwenEosTokenId),
      std::make_pair("<|object_ref_start|>", 151646U),
      std::make_pair("<think>", 151667U),
      std::make_pair("</think>", 151668U),
  };

  for (const auto& [name, id] : default_specials) {
    if (id < tokens.size()) {
      special_tokens[std::string(name)] = id;
    }
  }

  auto tokenizer =
      CreateFromVocabulary(tokens, merges, special_tokens, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }

  if (auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    tokenizer->eos_token_id_ = *eos;
  }
  if (auto bos = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    tokenizer->bos_token_id_ = *bos;
  }
  if (auto pad = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    tokenizer->pad_token_id_ = *pad;
  }

  return tokenizer;
}

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    const std::unordered_map<std::string, TokenId>& special_tokens,
    std::string* error_msg) {
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Vocabulary cannot be empty";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<QwenTokenizer>(new QwenTokenizer());
  tokenizer->id_to_token_.reserve(tokens.size());
  tokenizer->token_to_id_.reserve(tokens.size());

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->id_to_token_.push_back(tokens[i]);
    tokenizer->token_to_id_[tokens[i]] = static_cast<TokenId>(i);
  }

  // Parse BPE merges
  tokenizer->merge_ranks_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto& merge = merges[rank];
    const auto space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const std::string part1 = merge.substr(0, space_pos);
    const std::string part2 = merge.substr(space_pos + 1);

    auto it1 = tokenizer->token_to_id_.find(part1);
    auto it2 = tokenizer->token_to_id_.find(part2);
    if (it1 != tokenizer->token_to_id_.end() &&
        it2 != tokenizer->token_to_id_.end()) {
      tokenizer->merge_ranks_[{it1->second, it2->second}] = rank;
    }
  }

  // Register special tokens
  for (const auto& [name, id] : special_tokens) {
    tokenizer->special_token_to_id_[name] = id;
    tokenizer->is_special_token_[id] = true;
    if (id < tokenizer->id_to_token_.size()) {
      tokenizer->id_to_token_[id] = name;
      tokenizer->token_to_id_[name] = id;
    }
  }

  tokenizer->InitializeByteTokens();
  return tokenizer;
}

void QwenTokenizer::InitializeByteTokens() {
  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    const std::string direct_char(1, static_cast<char>(byte_val));
    auto it = token_to_id_.find(direct_char);
    if (it != token_to_id_.end()) {
      byte_tokens_[b] = it->second;
    } else {
      const std::string hex_token = FormatHexByteToken(byte_val);
      auto hex_it = token_to_id_.find(hex_token);
      if (hex_it != token_to_id_.end()) {
        byte_tokens_[b] = hex_it->second;
      } else {
        byte_tokens_[b] = kInvalidTokenId;
      }
    }
  }
}

std::vector<TokenId> QwenTokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty()) {
    return {};
  }

  // Initial tokenization at byte level
  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());

  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId) {
      word_tokens.push_back(tid);
    }
  }

  if (word_tokens.size() <= 1) {
    return word_tokens;
  }

  // Iteratively merge the highest-ranked adjacent pairs
  while (word_tokens.size() >= 2) {
    std::optional<std::uint32_t> best_rank;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < word_tokens.size() - 1; ++i) {
      auto it = merge_ranks_.find({word_tokens[i], word_tokens[i + 1]});
      if (it != merge_ranks_.end()) {
        if (!best_rank.has_value() || it->second < *best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }
    }

    if (!best_rank.has_value()) {
      break;
    }

    const std::string merged_str = id_to_token_[word_tokens[best_idx]] +
                                   id_to_token_[word_tokens[best_idx + 1]];
    auto merged_it = token_to_id_.find(merged_str);
    if (merged_it == token_to_id_.end()) {
      break;
    }

    word_tokens[best_idx] = merged_it->second;
    word_tokens.erase(word_tokens.begin() +
                      static_cast<std::ptrdiff_t>(best_idx + 1));
  }

  return word_tokens;
}

std::vector<TokenId> QwenTokenizer::Encode(
    std::string_view text, const TokenizerOptions& options) const {
  std::vector<TokenId> tokens;
  if (text.empty()) {
    if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
      tokens.push_back(bos_token_id_);
    }
    if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
      tokens.push_back(eos_token_id_);
    }
    return tokens;
  }

  if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
    tokens.push_back(bos_token_id_);
  }

  if (!options.parse_special_tokens || special_token_to_id_.empty()) {
    const auto chunk_tokens = BpeMergeChunk(text);
    tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
  } else {
    // Scan text for special token delimiters
    std::size_t pos = 0;
    while (pos < text.size()) {
      std::size_t next_special_pos = std::string_view::npos;
      std::string_view matched_special;
      TokenId matched_id = kInvalidTokenId;

      for (const auto& [special_str, id] : special_token_to_id_) {
        const auto found = text.find(special_str, pos);
        if (found != std::string_view::npos) {
          if (next_special_pos == std::string_view::npos ||
              found < next_special_pos) {
            next_special_pos = found;
            matched_special = special_str;
            matched_id = id;
          }
        }
      }

      if (next_special_pos == std::string_view::npos) {
        // No more special tokens; encode remainder
        const auto chunk = text.substr(pos);
        const auto chunk_tokens = BpeMergeChunk(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        break;
      }

      if (next_special_pos > pos) {
        const auto chunk = text.substr(pos, next_special_pos - pos);
        const auto chunk_tokens = BpeMergeChunk(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
      }

      tokens.push_back(matched_id);
      pos = next_special_pos + matched_special.size();
    }
  }

  if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
    tokens.push_back(eos_token_id_);
  }

  return tokens;
}

std::string QwenTokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string result;
  for (const TokenId id : tokens) {
    if (id < id_to_token_.size()) {
      const auto& tok = id_to_token_[id];
      // Check if it is a byte token <0xXX>
      if (tok.size() == 6 && tok.starts_with("<0x") && tok.ends_with('>')) {
        const int h1 = HexCharToInt(tok[3]);
        const int h2 = HexCharToInt(tok[4]);
        if (h1 >= 0 && h2 >= 0) {
          const auto byte_val = static_cast<std::uint8_t>((h1 << 4) | h2);
          result.push_back(static_cast<char>(byte_val));
          continue;
        }
      }
      result.append(tok);
    }
  }
  return result;
}

std::string_view QwenTokenizer::DecodeToken(TokenId token_id) const noexcept {
  if (token_id < id_to_token_.size()) {
    return id_to_token_[token_id];
  }
  return "";
}

std::optional<TokenId> QwenTokenizer::FindSpecialToken(
    std::string_view token_str) const noexcept {
  auto it = special_token_to_id_.find(std::string(token_str));
  if (it != special_token_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool QwenTokenizer::IsSpecialToken(TokenId id) const noexcept {
  return is_special_token_.contains(id);
}

}  // namespace strix::tokenization
