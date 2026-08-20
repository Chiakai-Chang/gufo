#include "src/models/deepseek_v4_flash/engine.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

extern "C" {
#include "src/models/deepseek_v4_flash/vendor/antirez/ds4.h"
}

namespace strix::models::deepseek_v4_flash {
namespace {

constexpr std::size_t kErrorCapacity = 512;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

std::vector<int> TakeTokens(ds4_tokens* tokens) {
  std::vector<int> result;
  if (tokens->v != nullptr && tokens->len > 0) {
    result.assign(tokens->v, tokens->v + tokens->len);
  }
  ds4_tokens_free(tokens);
  return result;
}

}  // namespace

struct Model::Impl {
  ds4_engine* engine = nullptr;
};

struct Session::Impl {
  ds4_session* session = nullptr;
  CancellationCheck is_cancelled;
};

struct SessionSnapshot::Impl {
  ds4_session_snapshot snapshot{};
};

Model::Model(std::unique_ptr<Impl> impl, ModelOptions options)
    : impl_(std::move(impl)), options_(options) {}

Model::~Model() {
  if (impl_ != nullptr && impl_->engine != nullptr) {
    ds4_engine_close(impl_->engine);
  }
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  if (model_path.empty()) {
    AssignError(error_msg, "DeepSeek V4 Flash model path is empty");
    return nullptr;
  }
  if (options.max_context < 2) {
    AssignError(error_msg, "DeepSeek V4 Flash context must be at least 2");
    return nullptr;
  }
  if (options.power_percent < 1 || options.power_percent > 100) {
    AssignError(error_msg, "DeepSeek V4 Flash power must be in [1, 100]");
    return nullptr;
  }
  if (options.max_context >
      static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg, "DeepSeek V4 Flash context exceeds engine limits");
    return nullptr;
  }

  ds4_engine_options engine_options{};
  engine_options.model_path = model_path.c_str();
  engine_options.backend = DS4_BACKEND_ROCM;
  engine_options.context_size = static_cast<int>(options.max_context);
  engine_options.prefill_chunk = options.prefill_chunk;
  engine_options.power_percent = options.power_percent;
  engine_options.warm_weights = options.warm_weights;
  engine_options.placement_ctx_hint = static_cast<int>(options.max_context);
  engine_options.placement_session_count_hint = 1;

  auto impl = std::make_unique<Impl>();
  if (ds4_engine_open(&impl->engine, &engine_options) != 0 ||
      impl->engine == nullptr) {
    AssignError(error_msg, "failed to load DeepSeek V4 Flash GGUF");
    return nullptr;
  }

  return std::shared_ptr<Model>(new Model(std::move(impl), options));
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context < 2 || max_context > options_.max_context ||
      max_context >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg,
                "DeepSeek V4 Flash session context is outside model limits");
    return nullptr;
  }

  auto impl = std::make_unique<Session::Impl>();
  if (ds4_session_create(&impl->session, impl_->engine,
                         static_cast<int>(max_context)) != 0 ||
      impl->session == nullptr) {
    AssignError(error_msg, "failed to create DeepSeek V4 Flash session");
    return nullptr;
  }

  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(impl)));
}

std::vector<int> Model::Tokenize(std::string_view text) const {
  const std::string owned(text);
  ds4_tokens tokens{};
  ds4_tokenize_text(impl_->engine, owned.c_str(), &tokens);
  return TakeTokens(&tokens);
}

std::vector<int> Model::EncodeChat(std::string_view system_prompt,
                                   std::string_view user_prompt) const {
  const std::string system(system_prompt);
  const std::string prompt(user_prompt);
  ds4_tokens tokens{};
  ds4_encode_chat_prompt(impl_->engine, system.c_str(), prompt.c_str(),
                         DS4_THINK_NONE, &tokens);
  return TakeTokens(&tokens);
}

std::vector<int> Model::EncodeChat(
    std::span<const ChatMessage> messages) const {
  ds4_tokens tokens{};
  ds4_chat_begin(impl_->engine, &tokens);
  for (const auto& message : messages) {
    ds4_chat_append_message(impl_->engine, &tokens, message.role.c_str(),
                            message.content.c_str());
  }
  ds4_chat_append_assistant_prefix(impl_->engine, &tokens, DS4_THINK_NONE);
  return TakeTokens(&tokens);
}

std::string Model::DecodeToken(int token) const {
  std::size_t length = 0;
  char* text = ds4_token_text(impl_->engine, token, &length);
  if (text == nullptr) {
    return {};
  }
  std::string result(text, length);
  std::free(text);
  return result;
}

bool Model::IsStopToken(int token) const {
  return ds4_token_is_stop(impl_->engine, token);
}

int Model::EosToken() const {
  return ds4_token_eos(impl_->engine);
}

int Model::VocabSize() const {
  return ds4_engine_vocab_size(impl_->engine);
}

std::string Model::ModelName() const {
  const char* name = ds4_engine_model_name(impl_->engine);
  return name != nullptr ? std::string(name) : std::string{};
}

std::uint32_t Model::PrefillChunk() const {
  return ds4_engine_prefill_chunk(impl_->engine);
}

std::uint32_t Model::MaxContext() const noexcept {
  return options_.max_context;
}

Session::Session(std::shared_ptr<Model> model, std::unique_ptr<Impl> impl)
    : model_(std::move(model)), impl_(std::move(impl)) {}

Session::~Session() {
  if (impl_ != nullptr && impl_->session != nullptr) {
    ds4_session_free(impl_->session);
  }
}

bool Session::Sync(std::span<const int> prompt, std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "DeepSeek V4 Flash prompt is empty");
    return false;
  }
  if (prompt.size() >= static_cast<std::size_t>(ContextSize())) {
    AssignError(error_msg,
                "DeepSeek V4 Flash prompt does not fit in session context");
    return false;
  }
  if (prompt.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    AssignError(error_msg, "DeepSeek V4 Flash prompt is too large");
    return false;
  }

  ds4_tokens tokens{
      .v = const_cast<int*>(prompt.data()),
      .len = static_cast<int>(prompt.size()),
      .cap = static_cast<int>(prompt.size()),
  };
  std::array<char, kErrorCapacity> error{};
  const int result =
      ds4_session_sync(impl_->session, &tokens, error.data(), error.size());
  if (result != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "DeepSeek V4 Flash prefill failed");
    return false;
  }
  return true;
}

int Session::SelectNext(float temperature, std::uint64_t* rng_state, int top_k,
                        float top_p, float min_p) const {
  if (temperature <= 0.0F) {
    return ds4_session_argmax(impl_->session);
  }
  if (rng_state == nullptr) {
    return -1;
  }
  return ds4_session_sample(impl_->session, temperature, top_k, top_p, min_p,
                            rng_state);
}

int Session::SelectNextExcluding(int excluded_token) const {
  return ds4_session_argmax_excluding(impl_->session, excluded_token);
}

bool Session::Evaluate(int token, std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_eval(impl_->session, token, error.data(), error.size()) !=
      0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "DeepSeek V4 Flash decode failed");
    return false;
  }
  return true;
}

std::vector<float> Session::CopyLogits(std::string* error_msg) const {
  const int vocab_size = model_->VocabSize();
  if (vocab_size <= 0) {
    AssignError(error_msg, "DeepSeek V4 Flash vocabulary size is invalid");
    return {};
  }
  std::vector<float> logits(static_cast<std::size_t>(vocab_size));
  if (ds4_session_copy_logits(impl_->session, logits.data(), vocab_size) !=
      vocab_size) {
    AssignError(error_msg, "failed to copy DeepSeek V4 Flash logits");
    return {};
  }
  return logits;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  auto impl = std::make_unique<SessionSnapshot::Impl>();
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_save_snapshot(impl_->session, &impl->snapshot, error.data(),
                                error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "failed to snapshot DeepSeek session");
    return nullptr;
  }
  return std::unique_ptr<SessionSnapshot>(new SessionSnapshot(std::move(impl)));
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  std::array<char, kErrorCapacity> error{};
  if (ds4_session_load_snapshot(impl_->session, &snapshot.impl_->snapshot,
                                error.data(), error.size()) != 0) {
    AssignError(error_msg, error[0] != '\0'
                               ? error.data()
                               : "failed to restore DeepSeek session");
    return false;
  }
  return true;
}

void Session::SetCancellationCheck(CancellationCheck is_cancelled) {
  impl_->is_cancelled = std::move(is_cancelled);
  ds4_session_cancel_fn callback = nullptr;
  if (impl_->is_cancelled) {
    callback = +[](void* user_data) -> bool {
      const auto* impl = static_cast<const Impl*>(user_data);
      return impl->is_cancelled && impl->is_cancelled();
    };
  }
  ds4_session_set_cancel(impl_->session, callback, impl_.get());
}

int Session::Position() const {
  return ds4_session_pos(impl_->session);
}

int Session::ContextSize() const {
  return ds4_session_ctx(impl_->session);
}

std::uint64_t Session::PayloadBytes() const {
  return ds4_session_payload_bytes(impl_->session);
}

SessionSnapshot::SessionSnapshot(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SessionSnapshot::~SessionSnapshot() {
  if (impl_ != nullptr) {
    ds4_session_snapshot_free(&impl_->snapshot);
  }
}

std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return impl_ != nullptr ? impl_->snapshot.len : 0;
}

}  // namespace strix::models::deepseek_v4_flash
