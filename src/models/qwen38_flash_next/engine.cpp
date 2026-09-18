#include "src/models/qwen38_flash_next/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// gfx1151 pp4096 at depths 0/4096: the 512/1024/2048/4096 sweep favored
// 2048; larger chunks used more scratch without improving throughput.
constexpr std::uint32_t kPrefillChunkTokens = 2048;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  std::array<float, 2> draft_policy;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size) {
  return sizeof(SessionSnapshotHeader) +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
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
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
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
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  m->device_ =
      rocm::DeviceModel::Upload(*m->weights_, model_path, m->mtp_weights_.get(),
                                options.mtp_model_path, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.max_batch = m->PrefillCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
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
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
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

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(kPrefillChunkTokens, options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  session_->Reset();
  tokens_.clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return model_->HasMtp()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes =
      SessionSnapshotHostBytes(token_count, model_->VocabSize());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  if (!model_->executor_->SaveSnapshot(
          *session_, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.token_count > ContextSize() ||
      payload.size() !=
          SessionSnapshotHostBytes(header.token_count, header.vocab_size) +
              header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  rocm::Executor::SnapshotInfo info;
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  stats_ = {};
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_) {
    return false;
  }
  std::memcpy(destination.data(), data_.get(), size_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates) {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  std::vector<std::int32_t> replay(tokens_.begin() + mp + 1, tokens_.end());
  replay.push_back(next_token);
  if (!exec.MtpForward(
          *session_, replay, static_cast<std::int32_t>(mp - hidden_base_),
          {.token = propose && candidates == nullptr ? &draft_token_ : nullptr,
           .candidates = candidates},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::Feed(std::span<const std::int32_t> tokens,
                   std::string* error_msg) {
  rocm::Executor& exec = *model_->executor_;
  for (std::size_t off = 0; off < tokens.size(); off += exec.max_batch()) {
    const std::size_t n =
        std::min<std::size_t>(exec.max_batch(), tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (model_->HasMtp() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), false, error_msg)) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(tokens_.size());
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
  }
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

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
};

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  const std::size_t width =
      model_->HasMtp() && cap > 1
          ? 1 + draft_length_.Choose(static_cast<std::uint32_t>(cap - 1))
          : cap;
  if (width == 0) {
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  if (!model_->HasMtp() || width < 2) {
    if (model_->HasMtp() && !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy = sampler.config().can_use_unmodified_argmax();
  const bool gpu_verification = sampled || gpu_greedy;
  if (!DraftCatchUp(anchor, !defer_head, error_msg,
                    sampled && !defer_head ? &pending->candidates : nullptr)) {
    return false;
  }
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler;
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  if (!defer_head) {
    while (pending->chain.size() < width) {
      AppendDraft(*pending);
      if (pending->chain.size() < width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = sampled ? &pending->candidates : nullptr},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), error_msg)) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      std::int32_t token = 0;
      bool accepted = false;
      if (!exec.VerifyMtpProposal(keep - 1, proposals[keep - 1], sampler,
                                  &token, &accepted, error_msg)) {
        return false;
      }
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg,
                     gpu_verification ? logits_.data() : nullptr)) {
    return false;
  }
  if (!gpu_verification) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(keep - 1, result->stop ? keep - 1 : k - 1);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg)) {
    return false;
  }
  if (pending.chain.empty()) {
    return true;
  }
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  if (!model_->executor_->Forward(*session_, pending.chain,
                                  pending.chain.size(), logits,
                                  pending.speculative, error_msg)) {
    return false;
  }
  return FinishDecode(request, pending, error_msg);
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "decode batch must contain 1..8 sessions");
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    if (request.session == nullptr || request.sampler == nullptr ||
        request.result == nullptr || request.max_tokens == 0 ||
        request.session->tokens_.empty() ||
        request.session->model_ != requests.front().session->model_) {
      AssignError(error_msg, "invalid decode batch request");
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (requests[j].session == request.session ||
          requests[j].sampler == request.sampler ||
          requests[j].result == request.result) {
        AssignError(error_msg, "decode batch requests must be independent");
        return false;
      }
    }
  }
  if (requests.size() == 1) {
    const auto& r = requests.front();
    return r.session->DecodeStep(r.max_tokens, *r.sampler, r.result, error_msg,
                                 r.stop_at_eos);
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->PrepareDecode(r, &pending[i], error_msg, true)) {
      return false;
    }
  }
  // Advance one proposal round across ready sessions. Their draft bodies,
  // probability distributions and RNG streams remain private; only the
  // vocabulary projection shares weight reads.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (p.speculative && p.chain.size() < p.width) {
        heads.push_back({requests[i].session->session_.get(),
                         {.token = p.sampled ? nullptr : &p.draft,
                          .candidates = p.sampled ? &p.candidates : nullptr}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!p.speculative || p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      if (p.chain.size() < p.width &&
          !exec.MtpForward(*requests[i].session->session_,
                           std::span<const std::int32_t>(&p.draft, 1), -1, {},
                           error_msg)) {
        return false;
      }
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (items.empty()) {
    return true;
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    if (p.chain.empty()) {
      continue;
    }
    auto& session = *requests[i].session;
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    if (!exec.SelectBatchLogits(offset, p.chain.size(), logits, error_msg) ||
        !session.FinishDecode(requests[i], p, error_msg)) {
      return false;
    }
    offset += p.chain.size();
  }
  return true;
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "advance batch must contain 1..8 sessions");
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (r.session == nullptr || r.token < 0 ||
        static_cast<std::uint32_t>(r.token) >= r.session->model_->VocabSize() ||
        r.session->model_ != requests.front().session->model_ ||
        r.session->Position() >= r.session->ContextSize()) {
      AssignError(error_msg, "invalid advance batch request");
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (requests[j].session == r.session) {
        AssignError(error_msg, "advance batch contains a duplicate session");
        return false;
      }
    }
  }
  if (requests.size() == 1) {
    return requests.front().session->Evaluate(requests.front().token,
                                              error_msg);
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    if (r.session->model_->HasMtp() &&
        !r.session->DraftCatchUp(r.token, false, error_msg)) {
      return false;
    }
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), error_msg)) {
      return false;
    }
    session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
    session.tokens_.push_back(requests[i].token);
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
