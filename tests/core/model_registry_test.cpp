#include "src/core/model_registry.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

void TestMiniMaxH3Descriptor() {
  using strix::core::ModelFamily;
  using strix::core::ModelKind;
  using strix::core::ModelRegistry;
  using strix::core::ModelRole;

  const auto parsed = strix::core::ParseModelKind("minimax-h3-fl2va-bf16");
  Expect(parsed == ModelKind::kMiniMaxH3Fl2vaBf16, "MiniMax H3 kind parses");
  Expect(strix::core::ToString(*parsed) == "minimax-h3-fl2va-bf16",
         "MiniMax H3 kind round trips");
  Expect(strix::core::ParseModelFamily("minimax-h3") == ModelFamily::kMiniMaxH3,
         "MiniMax H3 family parses");

  const auto* descriptor =
      ModelRegistry::FindByKind(ModelKind::kMiniMaxH3Fl2vaBf16);
  Expect(descriptor != nullptr, "MiniMax H3 descriptor is registered");
  Expect(descriptor->role == ModelRole::kProduction,
         "MiniMax H3 is a production model");
  Expect(
      descriptor->pinned_revision == "42ed227ee7df40d41602854ae760620d6eb651fe",
      "MiniMax H3 revision is immutable");
  Expect(descriptor->capabilities.text_input, "MiniMax H3 accepts text input");
  Expect(!descriptor->capabilities.text_output,
         "MiniMax H3 does not declare text output");
  Expect(!descriptor->capabilities.vision_input &&
             !descriptor->capabilities.video_input &&
             !descriptor->capabilities.audio_input,
         "Initial MiniMax H3 artifact is text-input only");
  Expect(descriptor->capabilities.ProducesVideo() &&
             descriptor->capabilities.ProducesAudio(),
         "MiniMax H3 declares synchronized video and audio output");
  Expect(!descriptor->capabilities.IsTextOnly(),
         "MiniMax H3 is not a text-only capability");
  Expect(descriptor->arch.num_layers == 50 &&
             descriptor->arch.hidden_size == 5376 &&
             descriptor->arch.intermediate_size == 14336,
         "MiniMax H3 DiT dimensions are pinned");
  Expect(descriptor->arch.num_attention_heads == 56 &&
             descriptor->arch.head_dim == 128,
         "MiniMax H3 attention dimensions are pinned");
  Expect(descriptor->arch.vocab_size == 151936,
         "MiniMax H3 tokenizer vocabulary is pinned");
  Expect(ModelRegistry::ValidateArtifactClaim("minimax-h3-fl2va-bf16",
                                              ModelKind::kMiniMaxH3Fl2vaBf16),
         "Matching MiniMax H3 artifact claim is accepted");
  Expect(!ModelRegistry::ValidateArtifactClaim("qwen3.8-27b-text",
                                               ModelKind::kMiniMaxH3Fl2vaBf16),
         "Mismatched model artifact claim is rejected");
}

void TestRegistryIsClosed() {
  const auto models = strix::core::ModelRegistry::GetAllModels();
  Expect(models.size() == 3, "Exactly three compiled model kinds are present");
  Expect(strix::core::ParseModelKind("minimax-h3-ref2va-bf16") == std::nullopt,
         "Ref2VA is not a compiled model kind");
  Expect(
      strix::core::ParseModelKind("minimax-h3-regenerate-2k") == std::nullopt,
      "2K regeneration is not a compiled model kind");
  Expect(strix::core::ParseModelKind("minimax-h3") == std::nullopt,
         "Ambiguous H3 artifact names are rejected");
}

}  // namespace

int main() {
  TestMiniMaxH3Descriptor();
  TestRegistryIsClosed();
  std::cout << "All model registry tests passed.\n";
  return 0;
}
