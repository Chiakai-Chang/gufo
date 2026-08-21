#include "src/core/model_registry.hpp"

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix::core {

namespace {

constexpr std::array<ModelDescriptor, 3> kCompiledModels = {
    ModelDescriptor{
        .kind = ModelKind::kQwen35_4B_Text,
        .family = ModelFamily::kQwen35,
        .role = ModelRole::kIterationFixture,
        .canonical_id = "qwen3.5-4b-text",
        .display_name = "Qwen3.5-4B Text",
        .family_name = "qwen3.5",
        .source_repo = "https://huggingface.co/Qwen/Qwen3.5-4B",
        .pinned_revision = "main",
        .architecture_pattern = "3 x Gated DeltaNet + 1 x full attention",
        .default_tokenizer_id = "qwen35",
        .default_chat_template_id = "qwen35-chat-v1",
        .capabilities =
            ModelCapabilities{
                .text_input = true,
                .text_output = true,
                .vision_input = false,
                .video_input = false,
                .audio_input = false,
                .video_output = false,
                .audio_output = false,
                .mtp_speculative = true,
            },
        .arch =
            ArchitectureParameters{
                .num_layers = 36,
                .hidden_size = 2560,
                .intermediate_size = 9728,
                .num_attention_heads = 20,
                .num_key_value_heads = 4,
                .head_dim = 128,
                .vocab_size = 248320,
                .default_context_tokens = 32768,
                .full_attention_interval = 4,
                .linear_key_value_heads = 16,
                .linear_head_dim = 128,
                .mtp_num_layers = 1,
            },
    },
    ModelDescriptor{
        .kind = ModelKind::kQwen38_27B_Text,
        .family = ModelFamily::kQwen35,
        .role = ModelRole::kProduction,
        .canonical_id = "qwen3.8-27b-text",
        .display_name = "Qwen3.8-27B Text",
        .family_name = "qwen3.5",
        .source_repo = "https://huggingface.co/Qwen/Qwen3.8-27B",
        .pinned_revision = "main",
        .architecture_pattern = "3 x Gated DeltaNet + 1 x full attention",
        .default_tokenizer_id = "qwen35",
        .default_chat_template_id = "qwen38-chat-v1",
        .capabilities =
            ModelCapabilities{
                .text_input = true,
                .text_output = true,
                .vision_input = false,
                .video_input = false,
                .audio_input = false,
                .video_output = false,
                .audio_output = false,
                .mtp_speculative = true,
            },
        .arch =
            ArchitectureParameters{
                .num_layers = 64,
                .hidden_size = 5120,
                .intermediate_size = 17920,
                .num_attention_heads = 40,
                .num_key_value_heads = 8,
                .head_dim = 128,
                .vocab_size = 248320,
                .default_context_tokens = 32768,
                .full_attention_interval = 4,
                .linear_key_value_heads = 32,
                .linear_head_dim = 128,
                .mtp_num_layers = 1,
            },
    },
    ModelDescriptor{
        .kind = ModelKind::kMiniMaxH3Fl2vaBf16,
        .family = ModelFamily::kMiniMaxH3,
        .role = ModelRole::kProduction,
        .canonical_id = "minimax-h3-fl2va-bf16",
        .display_name = "MiniMax H3 Base FL2VA BF16",
        .family_name = "minimax-h3",
        .source_repo = "https://huggingface.co/MiniMaxAI/MiniMax-H3",
        .pinned_revision = "42ed227ee7df40d41602854ae760620d6eb651fe",
        .architecture_pattern =
            "Qwen3-VL layer-50 text state + 50-block joint audio/video DiT",
        .default_tokenizer_id = "minimax-h3-qwen2",
        .default_chat_template_id = "minimax-h3-chat-v1",
        .capabilities =
            ModelCapabilities{
                .text_input = true,
                .text_output = false,
                .vision_input = false,
                .video_input = false,
                .audio_input = false,
                .video_output = true,
                .audio_output = true,
                .mtp_speculative = false,
            },
        .arch =
            ArchitectureParameters{
                .num_layers = 50,
                .hidden_size = 5376,
                .intermediate_size = 14336,
                .num_attention_heads = 56,
                .num_key_value_heads = 56,
                .head_dim = 128,
                .vocab_size = 151936,
                .default_context_tokens = 512,
                .full_attention_interval = 1,
                .linear_key_value_heads = 0,
                .linear_head_dim = 0,
                .mtp_num_layers = 0,
            },
    },
};

}  // namespace

std::span<const ModelDescriptor> ModelRegistry::GetAllModels() noexcept {
  return kCompiledModels;
}

const ModelDescriptor* ModelRegistry::FindByKind(ModelKind kind) noexcept {
  for (const auto& model : kCompiledModels) {
    if (model.kind == kind) {
      return &model;
    }
  }
  return nullptr;
}

const ModelDescriptor* ModelRegistry::FindById(std::string_view id) noexcept {
  if (id.empty()) {
    return nullptr;
  }
  for (const auto& model : kCompiledModels) {
    if (model.canonical_id == id) {
      return &model;
    }
  }
  return nullptr;
}

std::optional<ModelKind> ModelRegistry::ResolveKind(
    std::string_view id) noexcept {
  const auto* model = FindById(id);
  if (model != nullptr) {
    return model->kind;
  }
  return std::nullopt;
}

std::vector<const ModelDescriptor*> ModelRegistry::FindByFamily(
    ModelFamily family) {
  std::vector<const ModelDescriptor*> results;
  results.reserve(kCompiledModels.size());
  for (const auto& model : kCompiledModels) {
    if (model.family == family) {
      results.push_back(&model);
    }
  }
  return results;
}

bool ModelRegistry::IsRegistered(std::string_view id) noexcept {
  return FindById(id) != nullptr;
}

bool ModelRegistry::ValidateArtifactClaim(std::string_view claimed_kind,
                                          ModelKind expected_kind) noexcept {
  const auto* model = FindById(claimed_kind);
  if (model == nullptr) {
    return false;
  }
  return model->kind == expected_kind;
}

}  // namespace strix::core
