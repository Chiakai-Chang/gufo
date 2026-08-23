#include "src/server/tts_service.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/talker_runtime.hpp"
#include "src/models/qwen3_tts/tokenizer.hpp"

namespace strix::server {

struct TtsService::Impl {
  explicit Impl(TtsServiceOptions supplied) : options(std::move(supplied)) {
    std::optional<models::qwen3_tts::ModelConfig> config;
    if (options.validate_model) {
      config = models::qwen3_tts::LoadModelConfigFromPath(
          options.model_root.string());
      if (!config.has_value() || config->model_type != "qwen3_tts" ||
          config->tts_model_type != "custom_voice" ||
          config->tokenizer_type != "qwen3_tts_tokenizer_12hz") {
        initialization_error =
            "Qwen3-TTS service requires a 12Hz CustomVoice model";
        return;
      }
      if (options.voices.empty()) {
        for (const auto& [name, unused] : config->talker.spk_id) {
          (void)unused;
          options.voices.push_back(name);
        }
      }
    }
    if (!options.runner) {
      if (options.native_context_tokens == 0) {
        initialization_error =
            "Qwen3-TTS native context capacity must be positive";
        return;
      }
      if (!models::qwen3_tts::Tokenizer::Load(options.model_root, &tokenizer,
                                              &initialization_error)) {
        return;
      }
      talker = models::qwen3_tts::hip::TalkerHipRuntime::Create(
          options.model_root.string(), options.native_context_tokens,
          &initialization_error);
      if (talker == nullptr) {
        return;
      }
      speech_decoder = models::qwen3_tts::hip::SpeechDecoderHipRuntime::Create(
          options.model_root.string(), &initialization_error);
      if (speech_decoder == nullptr) {
        return;
      }
      options.runner =
          [this](const models::qwen3_tts::SynthesisRequest& request,
                 const models::qwen3_tts::CancellationCheck& is_cancelled,
                 models::qwen3_tts::SynthesisResult* result,
                 std::string* error) {
            return GenerateNative(request, is_cancelled, result, error);
          };
    }
    std::ranges::sort(options.voices);
    ready = true;
  }

  bool GenerateNative(const models::qwen3_tts::SynthesisRequest& request,
                      const models::qwen3_tts::CancellationCheck& is_cancelled,
                      models::qwen3_tts::SynthesisResult* result,
                      std::string* error) {
    if (result == nullptr) {
      if (error != nullptr) {
        *error = "Qwen3-TTS native result must not be null";
      }
      return false;
    }
    *result = {};
    if (is_cancelled && is_cancelled()) {
      if (error != nullptr) {
        *error = "Qwen3-TTS native generation cancelled";
      }
      return false;
    }
    std::vector<std::uint32_t> input_ids;
    if (!tokenizer.EncodeAssistantPrompt(request.text, &input_ids, error)) {
      return false;
    }
    std::vector<std::uint32_t> instruction_ids;
    if (!request.instruct.empty() &&
        !tokenizer.EncodeInstructionPrompt(request.instruct, &instruction_ids,
                                           error)) {
      return false;
    }
    models::qwen3_tts::hip::CustomVoicePromptOutput prompt;
    if (!talker->BuildCustomVoicePrompt(input_ids, instruction_ids,
                                        request.speaker, request.language,
                                        &prompt, error)) {
      return false;
    }
    if (request.max_new_tokens >
        options.native_context_tokens -
            std::min(prompt.tokens, options.native_context_tokens)) {
      if (error != nullptr) {
        *error =
            "Qwen3-TTS prompt plus generated tokens exceeds native "
            "context capacity";
      }
      return false;
    }
    models::qwen3_tts::hip::TalkerGenerationOutput generated;
    const models::qwen3_tts::hip::TalkerSamplingOptions sampling{
        .sample = !request.greedy,
        .seed = request.seed,
    };
    if (!talker->Generate(prompt, request.max_new_tokens, sampling, &generated,
                          error)) {
      return false;
    }
    if (generated.frames == 0 || generated.code_groups != 16 ||
        generated.codes.size() != generated.frames * generated.code_groups) {
      if (error != nullptr) {
        *error = "Qwen3-TTS native talker produced invalid codec frames";
      }
      return false;
    }
    if (is_cancelled && is_cancelled()) {
      if (error != nullptr) {
        *error = "Qwen3-TTS native generation cancelled";
      }
      return false;
    }
    models::qwen3_tts::hip::SpeechDecoderOutput audio;
    if (!speech_decoder->Decode(generated.codes, generated.frames, &audio,
                                nullptr, error)) {
      return false;
    }
    models::qwen3_tts::SynthesisResult native;
    native.sample_rate = audio.sample_rate;
    native.code_groups = static_cast<std::uint32_t>(generated.code_groups);
    native.samples = std::move(audio.samples);
    native.codes.reserve(generated.codes.size());
    for (const std::uint32_t code : generated.codes) {
      native.codes.push_back(static_cast<std::int32_t>(code));
    }
    *result = std::move(native);
    return true;
  }

  TtsServiceOptions options;
  models::qwen3_tts::Tokenizer tokenizer;
  std::unique_ptr<models::qwen3_tts::hip::TalkerHipRuntime> talker;
  std::unique_ptr<models::qwen3_tts::hip::SpeechDecoderHipRuntime>
      speech_decoder;
  std::string initialization_error;
  std::mutex generation_mutex;
  bool ready{false};
};

TtsService::TtsService(TtsServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

TtsService::~TtsService() = default;

bool TtsService::ready() const noexcept {
  return impl_->ready;
}

std::string TtsService::initialization_error() const {
  return impl_->initialization_error;
}

std::string TtsService::model_id() const {
  return impl_->options.model_id;
}

std::string TtsService::backend_name() const {
  return "native-hip";
}

std::vector<std::string> TtsService::voices() const {
  return impl_->options.voices;
}

bool TtsService::Synthesize(
    const models::qwen3_tts::SynthesisRequest& request,
    const models::qwen3_tts::CancellationCheck& is_cancelled,
    models::qwen3_tts::SynthesisResult* result, std::string* error) {
  if (!impl_->ready) {
    if (error != nullptr) {
      *error = impl_->initialization_error;
    }
    return false;
  }
  const std::lock_guard<std::mutex> lock(impl_->generation_mutex);
  return impl_->options.runner(request, is_cancelled, result, error);
}

}  // namespace strix::server
