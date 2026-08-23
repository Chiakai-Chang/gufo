#include "src/models/qwen3_tts/loader.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/models/qwen3_tts/config.hpp"

namespace {

namespace qwen3_tts = strix::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_loader_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Optional argv: model dir. When absent, skips if default location missing.
  const std::string model_dir =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";

  if (!std::filesystem::exists(model_dir + "/config.json")) {
    std::cerr << "SKIP qwen3_tts_loader_test: model not present at "
              << model_dir << '\n';
    return 77;
  }

  auto config = qwen3_tts::LoadModelConfigFromPath(model_dir);
  Check(config.has_value(), "config parse");
  Check(config->tts_model_type == "custom_voice",
        "tts_model_type == custom_voice");
  Check(config->talker.hidden_size == 2048, "talker hidden 2048");
  Check(config->talker.num_hidden_layers == 28, "talker layers 28");
  Check(config->talker.num_attention_heads == 16, "talker heads 16");
  Check(config->talker.num_key_value_heads == 8, "talker kv heads 8");
  Check(config->talker.intermediate_size == 6144, "talker ffn 6144");
  Check(config->talker.codec_eos_token_id == 2150, "codec eos 2150");
  Check(config->code_predictor.num_hidden_layers == 5, "cp layers 5");
  Check(config->code_predictor.hidden_size == 1024, "cp hidden 1024");
  Check(config->speech_tokenizer.output_sample_rate == 24000,
        "speech tokenizer sample rate 24000");
  Check(config->speech_tokenizer.num_quantizers == 16,
        "speech tokenizer quantizers 16");
  Check(config->speech_tokenizer.latent_dim == 1024,
        "speech decoder latent 1024");
  Check(config->speech_tokenizer.codebook_dim == 512,
        "speech decoder codebook dim 512");
  Check(config->speech_tokenizer.decoder_dim == 1536,
        "speech decoder channel width 1536");
  Check(config->speech_tokenizer.hidden_size == 512 &&
            config->speech_tokenizer.intermediate_size == 1024,
        "speech decoder transformer dimensions");
  Check(config->speech_tokenizer.num_hidden_layers == 8 &&
            config->speech_tokenizer.num_attention_heads == 16 &&
            config->speech_tokenizer.head_dim == 64,
        "speech decoder transformer topology");
  Check(config->speech_tokenizer.upsample_rates ==
            std::vector<std::uint32_t>({8, 5, 4, 3}),
        "speech decoder waveform upsample rates");
  Check(config->speech_tokenizer.upsampling_ratios ==
            std::vector<std::uint32_t>({2, 2}),
        "speech decoder latent upsample ratios");

  const auto spk = config->talker.spk_id.find("vivian");
  Check(spk != config->talker.spk_id.end() && spk->second == 3065,
        "vivian spk id 3065");
  const auto dialect = config->talker.spk_dialect.find("eric");
  Check(dialect != config->talker.spk_dialect.end() &&
            dialect->second == "sichuan_dialect",
        "Eric dialect is preserved");

  auto loaded = qwen3_tts::LoadModelDirectory(model_dir);
  Check(loaded.ok, "load model dir: " + loaded.error);
  Check(loaded.store != nullptr, "store present");
  Check(loaded.store->size() > 850, "talker and speech tokenizer tensors");

  const auto* text_emb =
      loaded.store->Find("talker.model.text_embedding.weight");
  Check(text_emb != nullptr, "text_embedding present");
  Check(text_emb->shape.size() == 2 && text_emb->shape[0] == 151936,
        "text_embedding rows 151936");
  Check(text_emb->data != nullptr, "text_embedding payload mapped");
  const auto* waveform_weight =
      loaded.store->Find("decoder.decoder.6.conv.weight");
  Check(waveform_weight != nullptr && waveform_weight->shape.size() == 3,
        "speech decoder output convolution present");

  std::cout << "PASS qwen3_tts_loader_test (tensors=" << loaded.store->size()
            << ")\n";
  return 0;
}
