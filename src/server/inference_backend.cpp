#include "src/server/inference_backend.hpp"

namespace strix::server {

bool InferenceBackend::load(const std::string& model_path, std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);

  std::string load_err;
  auto reader = core::GgufReader::OpenFile(model_path, &load_err);
  if (!reader) {
    if (error != nullptr)
      *error = "Failed to open GGUF: " + load_err;
    return false;
  }

  std::string gen_err;
  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &gen_err);
  if (!generator) {
    if (error != nullptr)
      *error = "Failed to create generator: " + gen_err;
    return false;
  }

  reader_ = std::move(reader);
  generator_ = std::move(generator);
  model_id_ = generator_->GetConfig().model_name;
  return true;
}

std::string InferenceBackend::model_id() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return model_id_;
}

InferenceBackend::Result InferenceBackend::complete(std::string_view prompt,
                                                    std::size_t max_tokens,
                                                    float temperature) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!generator_)
    return {};

  const auto& tokenizer = generator_->GetTokenizer();
  const auto prompt_tokens = tokenizer.Encode(prompt);

  models::GenerationOptions opts;
  opts.max_new_tokens = max_tokens > 0 ? max_tokens : 1;
  opts.temperature = temperature;

  const auto toks = generator_->Generate(prompt_tokens, opts);
  const auto text = tokenizer.Decode(toks);
  return {text, prompt_tokens.size(), toks.size()};
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, float temperature) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!generator_)
    return {};

  const auto& tokenizer = generator_->GetTokenizer();
  const auto prompt_tokens_opt =
      tokenization::QwenChatTemplate::RenderAndTokenize(tokenizer, messages);
  if (!prompt_tokens_opt.has_value() || prompt_tokens_opt->empty()) {
    return {};
  }
  const auto& prompt_tokens = *prompt_tokens_opt;

  models::GenerationOptions opts;
  opts.max_new_tokens = max_tokens > 0 ? max_tokens : 1;
  opts.temperature = temperature;

  const auto toks = generator_->Generate(prompt_tokens, opts);
  const auto text = tokenizer.Decode(toks);
  return {text, prompt_tokens.size(), toks.size()};
}

std::size_t InferenceBackend::count_tokens(std::string_view text) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!generator_)
    return 0;
  return generator_->GetTokenizer().Encode(text).size();
}

}  // namespace strix::server
