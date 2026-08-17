#include "src/core/model_registry.hpp"

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace strix::core {

namespace {

constexpr std::array<ModelDescriptor, 2> kCompiledModels = {
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
