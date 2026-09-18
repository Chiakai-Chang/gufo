#ifndef GUFO_CORE_MODEL_KIND_HPP_
#define GUFO_CORE_MODEL_KIND_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace gufo::core {

/// Implementation families grouping models sharing kernel architectures,
/// recurring layer patterns, and execution graphs.
enum class ModelFamily : std::uint8_t {
  kQwen35,     ///< Qwen3.5 family (3 x Gated DeltaNet + 1 x full attention)
  kMiniMaxH3,  ///< MiniMax H3 joint text/audio/video diffusion family
};

[[nodiscard]] constexpr std::string_view ToString(ModelFamily family) noexcept {
  switch (family) {
    case ModelFamily::kQwen35:
      return "qwen3.5";
    case ModelFamily::kMiniMaxH3:
      return "minimax-h3";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::optional<ModelFamily> ParseModelFamily(
    std::string_view name) noexcept {
  if (name == "qwen3.5" || name == "qwen3_5" || name == "qwen35") {
    return ModelFamily::kQwen35;
  }
  if (name == "minimax-h3" || name == "minimax_h3" || name == "h3") {
    return ModelFamily::kMiniMaxH3;
  }
  return std::nullopt;
}

/// Curated compiled model kinds supported natively by the engine.
/// Dynamic or arbitrary architecture loading is rejected; all supported
/// kinds are statically enumerated here.
enum class ModelKind : std::uint8_t {
  kQwen35_4B_Text,      ///< Qwen3.5-4B bring-up and rapid-iteration model
  kQwen38_27B_Text,     ///< Qwen3.8-27B production text model
  kMiniMaxH3Fl2vaBf16,  ///< Pinned MiniMax H3 Base FL2VA BF16 artifact
};

[[nodiscard]] constexpr std::string_view ToString(ModelKind kind) noexcept {
  switch (kind) {
    case ModelKind::kQwen35_4B_Text:
      return "qwen3.5-4b-text";
    case ModelKind::kQwen38_27B_Text:
      return "qwen3.8-27b-text";
    case ModelKind::kMiniMaxH3Fl2vaBf16:
      return "minimax-h3-fl2va-bf16";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::optional<ModelKind> ParseModelKind(
    std::string_view name) noexcept {
  if (name == "qwen3.5-4b-text") {
    return ModelKind::kQwen35_4B_Text;
  }
  if (name == "qwen3.8-27b-text") {
    return ModelKind::kQwen38_27B_Text;
  }
  if (name == "minimax-h3-fl2va-bf16") {
    return ModelKind::kMiniMaxH3Fl2vaBf16;
  }
  return std::nullopt;
}

/// Deployment role distinguishing iteration fixtures from production models.
enum class ModelRole : std::uint8_t {
  kIterationFixture,  ///< Rapid bring-up, test oracles, and CI fixtures
  kProduction,        ///< Production serving target
};

[[nodiscard]] constexpr std::string_view ToString(ModelRole role) noexcept {
  switch (role) {
    case ModelRole::kIterationFixture:
      return "iteration_fixture";
    case ModelRole::kProduction:
      return "production";
  }
  return "unknown";
}

/// Declared capabilities for a compiled model descriptor.
/// Input/output modalities implemented by each model runtime.
struct ModelCapabilities {
  bool text_input{true};
  bool text_output{true};
  bool vision_input{false};
  bool video_input{false};
  bool audio_input{false};
  bool video_output{false};
  bool audio_output{false};
  bool mtp_speculative{false};

  [[nodiscard]] constexpr bool IsTextOnly() const noexcept {
    return text_input && text_output && !vision_input && !video_input &&
           !audio_input && !video_output && !audio_output;
  }

  [[nodiscard]] constexpr bool SupportsVision() const noexcept {
    return vision_input || video_input;
  }

  [[nodiscard]] constexpr bool SupportsVideo() const noexcept {
    return video_input || video_output;
  }

  [[nodiscard]] constexpr bool SupportsAudio() const noexcept {
    return audio_input || audio_output;
  }

  [[nodiscard]] constexpr bool ProducesVideo() const noexcept {
    return video_output;
  }

  [[nodiscard]] constexpr bool ProducesAudio() const noexcept {
    return audio_output;
  }

  [[nodiscard]] constexpr bool SupportsMtp() const noexcept {
    return mtp_speculative;
  }
};

/// Structural architectural metadata pinned for a compiled model kind.
struct ArchitectureParameters {
  std::uint32_t num_layers{0};
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t num_attention_heads{0};
  std::uint32_t num_key_value_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t vocab_size{0};
  std::uint32_t default_context_tokens{0};
  std::uint32_t full_attention_interval{0};  ///< e.g. 4 (3 linear + 1 full)
  std::uint32_t linear_key_value_heads{0};
  std::uint32_t linear_head_dim{0};
  std::uint32_t mtp_num_layers{0};
};

/// Immutable descriptor identifying a compiled model kind, its capabilities,
/// implementation family, and architecture metadata.
struct ModelDescriptor {
  ModelKind kind{ModelKind::kQwen35_4B_Text};
  ModelFamily family{ModelFamily::kQwen35};
  ModelRole role{ModelRole::kIterationFixture};
  std::string_view canonical_id;
  std::string_view display_name;
  std::string_view family_name;
  std::string_view source_repo;
  std::string_view pinned_revision;
  std::string_view architecture_pattern;
  std::string_view default_tokenizer_id;
  std::string_view default_chat_template_id;
  ModelCapabilities capabilities;
  ArchitectureParameters arch;
};

}  // namespace gufo::core

#endif  // GUFO_CORE_MODEL_KIND_HPP_
