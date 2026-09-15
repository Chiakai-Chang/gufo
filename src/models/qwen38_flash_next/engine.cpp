#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(ShardPath(model_path, t.shard), t.file_offset,
                                 t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ = core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, model_path,
                                         m->mtp_weights_.get(),
                                         options.mtp_model_path, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  exec.max_logit_rows = std::max<std::uint32_t>(1, options.max_draft_tokens + 1);
  exec.max_speculative = exec.max_logit_rows;
  exec.draft_rows = options.draft_vocab;
  m->executor_ = rocm::Executor::Create(*m->device_, m->ngram_.get(), exec,
                                        error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == EosToken() ||
         token == static_cast<std::int32_t>(tokenizer_->GetPadTokenId());
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

bool Model::HasMtp() const noexcept { return device_->has_mtp(); }

std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept { return weights_->config; }

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)), session_(std::move(session)) {
  logits_.resize(model_->VocabSize());
  draft_logits_.resize(model_->VocabSize());
  verify_logits_.resize(static_cast<std::size_t>(model_->options_.max_draft_tokens + 1) *
                        model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept { return session_->position(); }
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  session_->Reset();
  tokens_.clear();
  pending_ = -1;
  hidden_base_ = 0;
  draft_ready_ = false;
  model_->executor_->MtpRewind(*session_, 0);
}

std::int32_t Session::Argmax(const float* row, std::size_t count) const noexcept {
  const std::size_t n = count > 0 ? count : model_->VocabSize();
  return static_cast<std::int32_t>(std::max_element(row, row + n) - row);
}

bool Session::DraftCatchUp(std::int32_t next_token, std::string* error_msg) {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. hc_keep holds the hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  if (draft_ready_ && next_token != pending_) {
    // The draft consumed a token the caller did not choose.
    exec.MtpRewind(*session_, size - 1);
    draft_ready_ = false;
  }
  if (draft_ready_) {
    return true;
  }
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    draft_ready_ = true;
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  std::vector<std::int32_t> replay(tokens_.begin() + mp + 1, tokens_.end());
  replay.push_back(next_token);
  if (!exec.MtpForward(*session_, replay, static_cast<std::int32_t>(mp - hidden_base_),
                       draft_logits_.data(), error_msg)) {
    return false;
  }
  draft_ready_ = true;
  return true;
}

bool Session::Feed(std::span<const std::int32_t> tokens,
                   std::string* error_msg) {
  rocm::Executor& exec = *model_->executor_;
  for (std::size_t off = 0; off < tokens.size(); off += exec.max_batch()) {
    const std::size_t n = std::min<std::size_t>(exec.max_batch(), tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (model_->HasMtp() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], error_msg)) {
      return false;
    }
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), false, error_msg)) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(tokens_.size());
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
    draft_ready_ = false;
  }
  pending_ = Argmax(logits_.data());
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  return Feed(prompt.subspan(common), error_msg);
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  return Feed(std::span<const std::int32_t>(&token, 1), error_msg);
}

bool Session::SpeculativeStep(std::size_t max_tokens,
                              std::vector<std::int32_t>* emitted,
                              std::string* error_msg) {
  if (tokens_.empty() || pending_ < 0) {
    AssignError(error_msg, "speculative step needs a synced prompt");
    return false;
  }
  rocm::Executor& exec = *model_->executor_;
  const std::uint32_t max_draft = model_->options_.max_draft_tokens;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t width =
      std::min<std::size_t>({max_tokens, room, static_cast<std::size_t>(max_draft) + 1});
  if (!model_->HasMtp() || width < 2) {
    if (width == 0) {
      AssignError(error_msg, "no room for another token");
      return false;
    }
    const std::int32_t token = pending_;
    if (!Evaluate(token, error_msg)) {
      return false;
    }
    emitted->push_back(token);
    return true;
  }

  // Bring the draft block up to the pending token, then chain drafts.
  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  if (!DraftCatchUp(pending_, error_msg)) {
    return false;
  }
  std::vector<std::int32_t> chain{pending_};
  const std::size_t draft_rows = exec.DraftRows();
  std::int32_t draft = Argmax(draft_logits_.data(), draft_rows);
  while (chain.size() < width) {
    chain.push_back(draft);
    if (chain.size() < width) {
      if (!exec.MtpForward(*session_, std::span<const std::int32_t>(&draft, 1),
                           -1, draft_logits_.data(), error_msg)) {
        return false;
      }
      draft = Argmax(draft_logits_.data(), draft_rows);
    }
  }

  // Verify: row i predicts chain[i+1]; the first token is the trunk's own
  // choice and always stands.
  const auto k = static_cast<std::uint32_t>(chain.size());
  const std::size_t vocab = model_->VocabSize();
  if (!exec.Forward(*session_, chain, k, verify_logits_.data(), true, error_msg)) {
    return false;
  }
  std::uint32_t keep = 1;
  while (keep < k && Argmax(verify_logits_.data() + (keep - 1) * vocab) ==
                         chain[keep]) {
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg)) {
    return false;
  }
  std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab, logits_.begin());
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  emitted->insert(emitted->end(), chain.begin(), chain.begin() + keep);
  pending_ = Argmax(logits_.data());
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;

  // Re-sync the draft block on the trunk's true hidden rows of the kept
  // tokens, ending on the new pending token.
  exec.MtpRewind(*session_, base);
  draft_ready_ = false;
  return DraftCatchUp(pending_, error_msg);
}

std::int32_t Session::SelectNext(float temperature, std::uint64_t* rng_state,
                                 int top_k, float top_p, float min_p) const {
  const std::size_t vocab = logits_.size();
  if (temperature <= 0.0F || rng_state == nullptr) {
    return Argmax(logits_.data());
  }
  std::vector<std::uint32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0U);
  const std::size_t keep = top_k > 0 ? std::min<std::size_t>(top_k, vocab) : vocab;
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep),
                    order.end(), [&](std::uint32_t a, std::uint32_t b) {
                      return logits_[a] > logits_[b];
                    });
  order.resize(keep);
  std::vector<double> probs(keep);
  const float max_logit = logits_[order[0]];
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    probs[i] = std::exp(static_cast<double>(logits_[order[i]] - max_logit) /
                        temperature);
    sum += probs[i];
  }
  std::size_t count = keep;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    probs[i] /= sum;
    if (probs[i] < min_p * probs[0] || (cumulative >= top_p && i > 0)) {
      count = i;
      break;
    }
    cumulative += probs[i];
  }
  double total = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    total += probs[i];
  }
  // xorshift64*
  *rng_state ^= *rng_state >> 12;
  *rng_state ^= *rng_state << 25;
  *rng_state ^= *rng_state >> 27;
  const double u = static_cast<double>((*rng_state * 2685821657736338717ULL) >> 11) /
                   static_cast<double>(1ULL << 53) * total;
  double acc = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    acc += probs[i];
    if (u < acc) {
      return static_cast<std::int32_t>(order[i]);
    }
  }
  return static_cast<std::int32_t>(order[count - 1]);
}

}  // namespace gufo::models::qwen38_flash_next
