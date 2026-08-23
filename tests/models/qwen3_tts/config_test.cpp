#include "src/models/qwen3_tts/config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

namespace qwen3_tts = strix::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_config_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main() {
  const auto config = qwen3_tts::ParseModelConfig(R"({
    "model_type": "qwen3_tts",
    "tokenizer_type": "qwen3_tts_tokenizer_12hz",
    "tts_model_size": "1b7",
    "tts_model_type": "custom_voice",
    "im_start_token_id": 151644,
    "im_end_token_id": 151645,
    "tts_pad_token_id": 151671,
    "tts_bos_token_id": 151672,
    "tts_eos_token_id": 151673,
    "talker_config": {
      "hidden_size": 2048,
      "num_hidden_layers": 28,
      "max_position_embeddings": 32768,
      "spk_id": {"vivian": 3065, "eric": 3062},
      "codec_language_id": {"english": 2050},
      "spk_is_dialect": {
        "vivian": false,
        "eric": "sichuan_dialect"
      },
      "rope_scaling": {
        "mrope_section": [24, 20, 20],
        "interleaved": true
      },
      "code_predictor_config": {
        "hidden_size": 1024,
        "num_hidden_layers": 5,
        "max_position_embeddings": 65536
      }
    }
  })");
  Check(config.has_value(), "valid configuration parses");
  Check(config->model_type == "qwen3_tts" &&
            config->tts_model_type == "custom_voice",
        "model identity is preserved");
  Check(config->talker.spk_id.at("vivian") == 3065,
        "speaker token is preserved");
  Check(!config->talker.spk_dialect.at("vivian").has_value(),
        "non-dialect speaker is represented explicitly");
  Check(config->talker.spk_dialect.at("eric") == "sichuan_dialect",
        "dialect name is preserved");
  Check(
      config->talker.mrope_section == std::vector<std::uint32_t>({24, 20, 20}),
      "mRoPE sections are preserved");
  Check(config->code_predictor.max_position_embeddings == 65536,
        "code predictor dimensions are independent");
  Check(!qwen3_tts::ParseModelConfig("{").has_value(),
        "malformed JSON fails closed");

  std::cout << "PASS qwen3_tts_config_test\n";
  return 0;
}
