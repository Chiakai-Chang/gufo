#ifndef STRIX_CORE_MODEL_CONFIG_HPP_
#define STRIX_CORE_MODEL_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace strix::core {

/// Unified architectural configuration parsed dynamically from GGUF metadata or
/// JSON.
struct ModelConfig {
  std::string architecture{"qwen35"};
  std::string model_name{"qwen3.5-4b-text"};
  std::uint32_t num_layers{36};           ///< 36 for Qwen3.5-4B (64 for 27B)
  std::uint32_t hidden_size{2560};        ///< 2560 for 4B (5120 for 27B)
  std::uint32_t intermediate_size{9728};  ///< 9728 for 4B (17920 for 27B)
  std::uint32_t num_attention_heads{20};  ///< 20 for 4B (40 for 27B)
  std::uint32_t num_key_value_heads{4};   ///< 4 for 4B (8 for 27B)
  std::uint32_t head_dim{128};       ///< 128 (standardized across 4B & 27B)
  std::uint32_t vocab_size{248320};  ///< 248320 for Qwen3.5/3.8
  std::uint32_t context_length{32768};
  std::uint32_t full_attention_interval{
      4};  ///< 3 linear attention + 1 full attention
  std::uint32_t linear_key_value_heads{16};  ///< 16 for 4B (32 for 27B)
  std::uint32_t linear_head_dim{128};
  std::uint32_t mtp_num_layers{1};
  float rope_theta{1000000.0F};
  float rope_scale{1.0F};
  bool is_text_only{true};

  /// Returns true if this configuration conforms to the Qwen3.5/3.8 repeating
  /// block structure.
  [[nodiscard]] constexpr bool IsValidQwen() const noexcept {
    return head_dim == 128 && num_layers > 0 && hidden_size > 0 &&
           intermediate_size > 0 && num_attention_heads > 0 &&
           num_key_value_heads > 0 && vocab_size > 0 &&
           full_attention_interval == 4 && is_text_only;
  }
};

}  // namespace strix::core

#endif  // STRIX_CORE_MODEL_CONFIG_HPP_
