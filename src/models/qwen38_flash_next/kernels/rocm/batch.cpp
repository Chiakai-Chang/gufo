#include <algorithm>
#include <array>
#include <stdexcept>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kBatchSessions = 8;
constexpr std::uint32_t kDecodeRows = 8;

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool Check(hipError_t status, std::string* error) {
  return status == hipSuccess || Fail(error, std::string("batched forward: ") +
                                                 hipGetErrorString(status));
}

// A cancelled row stops touching its private state. Shared projections retain
// their row layout so cancellation cannot change a peer's arithmetic/order.
template<class Item>
bool AnyActive(std::span<const Item> items) {
  bool active = false;
  for (const auto& item : items)
    active |= item.session->CheckCancellation(nullptr);
  return active;
}

WeightType EmbeddingType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next embedding format");
  }
}

}  // namespace

Executor::Scratch Executor::RowScratch(const Scratch& b,
                                       std::uint32_t offset) const {
  const Config& c = config();
  const std::size_t r = offset;
  auto s = b;
  s.tokens += r;
  s.res += r * c.HcDim();
  s.xn += r * c.HcDim();
  s.xn_half += r * c.HcDim();
  s.lo += r * c.hc_low_rank;
  s.hc_gate += r * c.HcDim();
  s.mixed += r * c.hidden_size;
  s.inject += r * c.hc_count * HcInjectParts(c.hidden_size);
  s.block_out += r * c.hidden_size;
  s.qkv += r * c.SsmConvChannels();
  s.z += r * c.SsmValueDim();
  s.qkvz += r * (c.SsmConvChannels() + c.SsmValueDim());
  s.alpha_beta += r * 2 * c.ssm_num_v_heads;
  s.qn += r * c.SsmKeyDim();
  s.kn += r * c.SsmKeyDim();
  s.gdn_raw += r * c.SsmValueDim();
  s.gdn_out += r * c.SsmValueDim();
  s.qg += r * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim());
  s.q += r * c.AttentionQDim();
  s.attn_gate += r * c.AttentionQDim();
  s.k += r * c.AttentionKvDim();
  s.v += r * c.AttentionKvDim();
  s.iq += r * c.indexer_heads * c.indexer_head_dim;
  s.ik += r * c.indexer_head_dim;
  s.mask += r * mask_words_;
  s.ctx += r * c.AttentionQDim();
  if (c.ple_layer >= 0) {
    s.ple_emb += r * c.PleEmbeddingDim();
    s.ple_key += r * c.HcDim();
    s.ple_value += r * c.hidden_size;
    s.ple_query += r * c.HcDim();
    s.ple_gated += r * c.HcDim();
    s.ple_norm += r * c.HcDim();
    s.ple_conv += r * c.HcDim();
  }
  s.router += r * (c.num_experts + 1);
  s.ids += r * c.num_experts_used;
  s.weights += r * c.num_experts_used;
  s.gate_e += r * c.num_experts_used * c.expert_ff;
  s.up_e += r * c.num_experts_used * c.expert_ff;
  s.down_e += r * c.num_experts_used * c.hidden_size;
  s.shexp_gate += r * c.shared_expert_ff;
  s.shexp_up += r * c.shared_expert_ff;
  s.shexp_out += r * c.hidden_size;
  if (has_mtp()) {
    s.mtp_h += r * c.HcDim();
    s.mtp_embd += r * c.hidden_size;
    // This allocation aliases qkvz, but has a different row stride.
    s.mtp_eproj += r * c.hidden_size;
    s.mtp_res += r * c.HcDim();
  }
  // Activation staging, attention partials and convolution staging have
  // one consumer at a time on stream_. They remain at the allocation base.
  return s;
}

void Executor::UseScratch(const Scratch& scratch) const {
  s_ = scratch;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  xn_half_ = false;
  moe_pending_ = false;
}

bool Executor::AllocateBatch(std::string* error) const {
  return (batch_logits_ != nullptr ||
          Check(hipMalloc(&batch_logits_,
                          static_cast<std::size_t>(kBatchSessions) *
                              std::min(kDecodeRows, options_.max_logit_rows) *
                              config().vocab_size * sizeof(float)),
                error)) &&
         (batch_q8_ != nullptr ||
          Check(hipMalloc(&batch_q8_,
                          qfn_mmq_q8_1_bytes(
                              32, std::max(2560U, config().hidden_size))),
                error));
}

bool Executor::HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                          float* mixed, float* inject, std::uint32_t rows,
                          std::span<const std::uint32_t> offsets,
                          std::string* error) const {
  const auto& c = config();
  xn_half_ = false;
  if (!normed)
    RmsNormRows(res, m.norm.f32(), s_.xn, rows, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  if (!DenseBatch(m.down, s_.xn, s_.lo, rows, error))
    return false;
  SiluScale(s_.lo, 1.0F / static_cast<float>(c.hc_count),
            std::size_t{rows} * c.hc_low_rank, stream_);
  if (!DenseBatch(m.up, s_.lo, s_.hc_gate, rows, error))
    return false;
  const bool fused_inject = inject != nullptr && !m.inject.empty() &&
                            m.inject.type == core::GgmlType::kF32;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  // Every request contains at most eight decode rows. Keep its scalar
  // epilogue/reduction even when the concatenated batch exceeds eight.
  HcMixEpilogue(s_.xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                mixed, inject, rows, c.hidden_size, c.hc_count, stream_);
  inject_parts_ = fused_inject ? HcInjectParts(c.hidden_size) : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    // Quantized inject projections use compact rows within each request,
    // while RowScratch reserves the full partial-sum stride between requests.
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      const auto begin = offsets[i];
      const auto end = i + 1 < offsets.size() ? offsets[i + 1] : rows;
      if (!Dense(m.inject, s_.xn + std::size_t{begin} * c.HcDim(),
                 inject + std::size_t{begin} * c.hc_count *
                              HcInjectParts(c.hidden_size),
                 end - begin, error))
        return false;
    }
  }
  return true;
}

bool Executor::MtpForwardBatch(std::span<const MtpBatchItem> items,
                               std::string* error) const {
  const Config& c = config();
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "invalid MTP body batch");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto n = item.tokens.size();
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || n == 0 || n > kDecodeRows ||
        (item.hidden_row < 0 && n != 1) ||
        (item.hidden_row >= 0 &&
         static_cast<std::uint32_t>(item.hidden_row) + n >
             options_.max_speculative) ||
        item.session->mtp_.position + n > item.session->max_context_) {
      return Fail(error, "invalid MTP body request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session)
        return Fail(error, "MTP body requests must be independent");
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size)
        return Fail(error, "MTP token out of range");
    }
    offsets[i] = rows;
    rows += n;
  }
  if (rows > options_.max_batch)
    return Fail(error, "MTP body batch exceeds executor capacity");
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    const auto& item = items.front();
    return MtpForward(*item.session, item.tokens, item.hidden_row, {}, error);
  }
  if (!AllocateBatch(error) ||
      (batch_controls_ == nullptr &&
       !Check(hipHostMalloc(&batch_controls_,
                            kBatchSessions * sizeof(Session::Control)),
              error))) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& session = *item.session;
    batch_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, item.hidden_row,
                          session.mtp_.blocks};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
  }
  const Scratch base = s_;
  const auto& l = model_->mtp();
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error))
      return false;
    EmbedTokens(model_->token_embd().data,
                EmbeddingType(model_->token_embd().type), base.tokens,
                base.mtp_embd, rows, c.hidden_size, 1, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      if (!Check(hipMemcpyAsync(session.control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error))
        return false;
      RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n,
                  c.hidden_size, 1, c.rms_eps, stream_);
      MtpHidden(session.mtp_.target_hidden, session.mtp_.h,
                &session.control_->hidden_row, s_.mtp_h, n, c.HcDim(), stream_);
      RmsNormRows(s_.mtp_h, l.nextn_hnorm.f32(), s_.mtp_h, n, c.HcDim(), 1,
                  c.rms_eps, stream_);
    }
    // Keep each session's vector/matrix arithmetic while sharing weight
    // reads among adjacent vector-sized inputs. Project the embedding once.
    for (std::size_t i = 0; i < items.size();) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      auto end = i + 1;
      if (n * c.hc_count <= kDecodeRows) {
        while (end < items.size() &&
               items[end].tokens.size() * c.hc_count <= kDecodeRows)
          ++end;
      }
      const auto end_row = end == items.size() ? rows : offsets[end];
      const auto count = end_row - offsets[i];
      UseScratch(RowScratch(base, offsets[i]));
      if (n * c.hc_count <= kDecodeRows) {
        if (!DenseBatch(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, count,
                        error) ||
            !DenseBatch(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res,
                        count * c.hc_count, error))
          return false;
      } else if (!Dense(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n,
                        error) ||
                 !Dense(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
                        error)) {
        return false;
      }
      MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, count, c.hidden_size,
                      c.hc_count, stream_);
      i = end;
    }
    UseScratch(base);
    if (!HcMixBatch(l.hc_attn, base.mtp_res, false, base.mixed, base.inject,
                    rows, std::span(offsets).first(items.size()), error))
      return false;
    if (!l.attn_qkv.empty()) {
      if (!DenseBatch(l.attn_qkv, base.mixed, base.qg, rows, error))
        return false;
    } else if (!DenseBatch(l.attn_q, base.mixed, base.qg, rows, error) ||
               !DenseBatch(l.attn_k, base.mixed, base.k, rows, error) ||
               !DenseBatch(l.attn_v, base.mixed, base.v, rows, error)) {
      return false;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      auto view = RowScratch(base, offsets[i]);
      if (l.attn_qkv.empty())
        view.qg = base.qg + std::size_t{offsets[i]} * 2 * c.AttentionQDim();
      UseScratch(view);
      if (!session.CheckCancellation(nullptr)) {
        if (!Check(hipMemsetAsync(
                       s_.ctx, 0,
                       std::size_t{n} * c.AttentionQDim() * sizeof(float),
                       stream_),
                   error))
          return false;
        continue;
      }
      ++session.mutation_epoch_;
      Session::AttentionState attention;
      attention.rope = session.vision_input_.rope();
      attention.k_cache = session.mtp_.k_cache;
      attention.v_cache = session.mtp_.v_cache;
      attention.index_k = session.mtp_.index_k;
      attention.block_k = session.mtp_.block_k;
      const bool sparse = session.mtp_.position + n > c.indexer_top_k;
      const auto complete = (session.mtp_.position + n) / c.compress_ratio;
      const auto pool = sparse ? complete - session.mtp_.blocks : 0;
      if (!Attention(l, attention, s_.mixed, s_.block_out, n,
                     &session.control_->mtp_position,
                     &session.control_->mtp_blocks, session.mtp_.position, pool,
                     session.max_context_, sparse, error, false, true, false))
        return false;
    }
    UseScratch(base);
    if (!DenseBatch(l.attn_out, base.ctx, base.block_out, rows, error))
      return false;
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      Combine(s_.mtp_res, l.hc_ffn.norm.f32(), n);
    }
    UseScratch(base);
    if (!HcMixBatch(l.hc_ffn, base.mtp_res, true, base.mixed, base.inject, rows,
                    std::span(offsets).first(items.size()), error))
      return false;
    if (!AnyActive(items))
      return true;
    for (std::uint32_t row = 0; row < rows; row += kDecodeRows) {
      UseScratch(RowScratch(base, row));
      if (!Moe(l, s_.mixed, s_.block_out, std::min(kDecodeRows, rows - row),
               error))
        return false;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      Combine(s_.mtp_res, nullptr, n);
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (!Check(hipMemcpyAsync(items[i].session->mtp_.h,
                                s_.mtp_res + std::size_t{n - 1} * c.HcDim(),
                                c.HcDim() * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error))
        return false;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (const auto& item : items) {
    if (item.session->Cancelled())
      continue;
    item.session->mtp_.position += item.tokens.size();
    if (item.session->mtp_.position > c.indexer_top_k)
      item.session->mtp_.blocks =
          item.session->mtp_.position / c.compress_ratio;
  }
  return true;
}

bool Executor::MtpHeads(std::span<const MtpHeadItem> items,
                        std::string* error) const {
  selected_logits_ = nullptr;
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions ||
      items.size() > options_.max_batch) {
    return Fail(error, "invalid MTP head batch");
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || item.session->mtp_.position == 0 ||
        item.output.trace != nullptr ||
        (item.output.token == nullptr && item.output.candidates == nullptr)) {
      return Fail(error, "invalid MTP head request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "MTP head requests must be independent");
      }
    }
  }
  const auto& output = model_->output();
  const auto& head = model_->mtp().nextn_head;
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    for (const auto& item : items) {
      if (!item.session->CheckCancellation(nullptr))
        continue;
      if (!MtpHead(head, item.session->mtp_.h, item.output.token != nullptr,
                   item.output.candidates != nullptr, error) ||
          !Check(hipStreamSynchronize(stream_), error)) {
        return false;
      }
      if (item.output.token != nullptr)
        *item.output.token = *mtp_token_host_;
      if (item.output.candidates != nullptr)
        *item.output.candidates = *mtp_candidates_host_;
    }
    return true;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (batch_candidates_host_ == nullptr) {
    if (!Check(hipHostMalloc(&batch_candidates_host_,
                             kBatchSessions * sizeof(MtpCandidateLogits)),
               error))
      return false;
    for (std::uint32_t i = 0; i < kBatchSessions; ++i)
      std::construct_at(batch_candidates_host_ + i);
  }
  const auto n = static_cast<std::uint32_t>(items.size());
  const auto count = std::min<std::size_t>(output.rows, kMtpCandidates);
  const Scratch base = s_;
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    for (std::uint32_t i = 0; i < n; ++i) {
      RmsNormRows(items[i].session->mtp_.h, head.norm.f32(),
                  base.xn + std::size_t{i} * config().HcDim(), 1,
                  config().HcDim(), config().hc_count, config().rms_eps,
                  stream_);
    }
    UseScratch(base);
    if (!HcMixBatch(head, nullptr, true, base.mixed, nullptr, n, {}, error))
      return false;
    if (!AnyActive(items))
      return true;
    if (!DenseBatch(output, base.mixed, batch_logits_, n, error))
      return false;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (items[i].output.candidates == nullptr) {
        Argmax(batch_logits_ + std::size_t(i) * output.rows, s_.mtp_argmax,
               s_.mtp_token, 1, output.rows, stream_);
        if (!Check(hipMemcpyAsync(batch_candidates_host_[i].ids.data(),
                                  s_.mtp_token, sizeof(std::int32_t),
                                  hipMemcpyDeviceToHost, stream_),
                   error))
          return false;
        continue;
      }
      MtpTopCandidates(batch_logits_ + std::size_t(i) * output.rows, s_.mtp_ids,
                       s_.mtp_scratch_ids, s_.mtp_scores, output.rows, stream_);
      if (!Check(hipMemcpyAsync(batch_candidates_host_ + i, s_.mtp_ids,
                                offsetof(MtpCandidateLogits, logits) +
                                    count * sizeof(float),
                                hipMemcpyDeviceToHost, stream_),
                 error)) {
        return false;
      }
      batch_candidates_host_[i].size = count;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    if (items[i].output.token != nullptr)
      *items[i].output.token = batch_candidates_host_[i].ids[0];
    if (items[i].output.candidates != nullptr)
      *items[i].output.candidates = batch_candidates_host_[i];
  }
  return true;
}

bool Executor::DenseBatch(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
  // Wide output matrices benefit from sharing weight rows across waves.
  // The smaller output projection with its long K sweep stays at eight.
  if (rows > kDecodeRows && w.type == core::GgmlType::kQ8_0 && w.cols == 2560 &&
      w.rows >= 8192) {
    constexpr std::uint32_t chunk = 32;
    for (std::uint32_t r = 0; r < rows;) {
      const auto n = std::min(chunk, rows - r);
      const auto padded = (n + kDecodeRows - 1) / kDecodeRows * kDecodeRows;
      if (n > kDecodeRows && padded != n &&
          !Check(hipMemsetAsync(static_cast<std::uint8_t*>(batch_q8_) +
                                    qfn_mmq_q8_1_bytes(n, w.cols),
                                0, qfn_mmq_q8_1_bytes(padded - n, w.cols),
                                stream_),
                 error)) {
        return false;
      }
      if (qfn_mmq_quantize_q8_1(x + static_cast<std::size_t>(r) * w.cols,
                                batch_q8_, n, w.cols, stream_) != 0 ||
          qfn_mmq_q8_0_dense_vec_preq(
              w.data, nullptr, batch_q8_,
              out + static_cast<std::size_t>(r) * w.rows, w.rows, n, w.cols,
              stream_) != 0) {
        return Fail(error, "batched Q8 projection failed");
      }
      r += n;
    }
    return true;
  }
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    if (!Dense(w, x + static_cast<std::size_t>(r) * w.cols,
               out + static_cast<std::size_t>(r) * w.rows,
               std::min(kDecodeRows, rows - r), error)) {
      return false;
    }
  }
  return true;
}

bool Executor::ForwardBatch(std::span<const BatchItem> items,
                            std::string* error) const {
  selected_logits_ = nullptr;
  const Config& c = config();
  if (items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "decode batch must contain 1..8 sessions");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::array<bool, kBatchSessions> sparse{};
  std::array<std::uint32_t, kBatchSessions> complete{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        item.tokens.empty() || item.tokens.size() > kDecodeRows ||
        item.tokens.size() > options_.max_logit_rows ||
        (item.speculative && item.tokens.size() > options_.max_speculative) ||
        item.session->position_ + item.tokens.size() >
            item.session->max_context_) {
      return Fail(error, "invalid session or chain in decode batch");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "decode batch contains a duplicate session");
      }
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        return Fail(error, "token out of range");
      }
    }
    offsets[i] = rows;
    rows += static_cast<std::uint32_t>(item.tokens.size());
    const auto end = item.session->position_ + item.tokens.size();
    sparse[i] = c.compress_ratio > 0 && end > c.indexer_top_k;
    complete[i] = c.compress_ratio > 0 ? end / c.compress_ratio : 0;
  }
  if (rows > options_.max_batch) {
    return Fail(error, "decode batch exceeds executor capacity");
  }
  if (!AnyActive(items))
    return true;
  for (const auto& item : items) {
    if (item.speculative &&
        !EnsureRollback(*item.session, item.tokens.size() - 1, error))
      return false;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (batch_controls_ == nullptr &&
      !Check(hipHostMalloc(&batch_controls_,
                           kBatchSessions * sizeof(Session::Control)),
             error)) {
    return false;
  }
  if (ple_pending_ && !WaitPle(error)) {
    return false;
  }
  batch_rows_ = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& session = *item.session;
    ++session.mutation_epoch_;
    session.spec_base_ = session.position_;
    session.spec_tokens_ = item.speculative ? item.tokens.size() : 0;
    batch_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, -1, session.mtp_.blocks};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
    if (c.ple_layer >= 0) {
      for (std::size_t t = 0; t < item.tokens.size(); ++t) {
        HashNgramRows(
            c, session.ngram_, item.tokens.subspan(t, 1),
            std::span(host_rows_)
                .subspan((offsets[i] + t) * c.ple_heads, c.ple_heads));
        if (item.speculative && t + 1 < item.tokens.size()) {
          session.ngram_snapshots_[t] = session.ngram_;
        }
      }
    }
  }
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr ||
        !ngram_->StartRead(std::span(host_rows_).first(rows * c.ple_heads),
                           std::span(host_emb_, static_cast<std::size_t>(rows) *
                                                    c.PleEmbeddingDim()))) {
      return Fail(error, "batched n-gram read could not start");
    }
    ple_pending_ = true;
  }

  const Scratch base = s_;
  // Every exit restores the ordinary single-session scratch view. Enqueued
  // work is drained before a caller can reuse its pinned staging buffers.
  const auto body = [&]() -> bool {
    if (!AnyActive(items))
      return true;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!Check(hipMemcpyAsync(items[i].session->control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error)) {
        return false;
      }
    }
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data,
                EmbeddingType(model_->token_embd().type), base.tokens, base.res,
                rows, c.hidden_size, c.hc_count, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      items[i].session->vision_input_.Inject(
          base.res + std::size_t{offsets[i]} * c.HcDim(),
          items[i].session->position_, items[i].tokens.size(), c.hidden_size,
          c.hc_count, stream_);
    }
    const auto& layers = model_->layers();
    bool normed = false;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!AnyActive(items))
        return true;
      const auto& l = layers[il];
      if (c.IsPleLayer(il)) {
        if (!WaitPle(error) ||
            !Check(hipMemcpyAsync(base.ple_emb, host_emb_,
                                  static_cast<std::size_t>(rows) *
                                      c.PleEmbeddingDim() * sizeof(float),
                                  hipMemcpyHostToDevice, stream_),
                   error)) {
          return false;
        }
      }
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        const auto n = static_cast<std::uint32_t>(item.tokens.size());
        UseScratch(RowScratch(base, offsets[i]));
        if (!item.session->Cancelled() && c.IsPleLayer(il) &&
            !Ple(l, *item.session, n, s_.res, item.speculative, error, true)) {
          return false;
        }
      }
      UseScratch(base);
      if (!HcMixBatch(l.hc_attn, base.res, normed, base.mixed, base.inject,
                      rows, std::span(offsets).first(items.size()), error))
        return false;
      if (l.linear) {
        if ((!l.ssm_in.empty()
                 ? !DenseBatch(l.ssm_in, base.mixed, base.qkvz, rows, error)
                 : (!DenseBatch(l.ssm_qkv, base.mixed, base.qkv, rows, error) ||
                    !DenseBatch(l.ssm_gate, base.mixed, base.z, rows,
                                error))) ||
            !DenseBatch(l.ssm_alpha_beta, base.mixed, base.alpha_beta, rows,
                        error)) {
          return false;
        }
      } else if (!l.attn_qkv.empty()) {
        if (!DenseBatch(l.attn_qkv, base.mixed, base.qg, rows, error)) {
          return false;
        }
      } else {
        // Separate Q/gate has a shorter row stride than the stacked buffer.
        // This model normally stacks QKV; keep the separate shape explicit.
        if (!DenseBatch(l.attn_q, base.mixed, base.qg, rows, error) ||
            !DenseBatch(l.attn_k, base.mixed, base.k, rows, error) ||
            !DenseBatch(l.attn_v, base.mixed, base.v, rows, error)) {
          return false;
        }
      }
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        auto& session = *item.session;
        const auto n = static_cast<std::uint32_t>(item.tokens.size());
        auto view = RowScratch(base, offsets[i]);
        if (!l.linear && l.attn_qkv.empty()) {
          view.qg = base.qg + static_cast<std::size_t>(offsets[i]) * 2 *
                                  c.AttentionQDim();
        }
        UseScratch(view);
        if (!session.CheckCancellation(nullptr)) {
          auto* output = l.linear ? s_.gdn_out : s_.ctx;
          const auto width = l.linear ? c.SsmValueDim() : c.AttentionQDim();
          if (!Check(hipMemsetAsync(output, 0,
                                    std::size_t{n} * width * sizeof(float),
                                    stream_),
                     error))
            return false;
          continue;
        }
        if (l.linear) {
          if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out,
                               n, item.speculative, error, true, false)) {
            return false;
          }
        } else {
          const auto pool = sparse[i] && complete[i] > session.blocks_
                                ? complete[i] - session.blocks_
                                : 0;
          if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                         &session.control_->position, &session.control_->blocks,
                         session.position_, pool, session.max_context_,
                         sparse[i], error, false, true, false)) {
            return false;
          }
        }
      }
      UseScratch(base);
      if (!DenseBatch(l.linear ? l.ssm_out : l.attn_out,
                      l.linear ? base.gdn_out : base.ctx, base.block_out, rows,
                      error)) {
        return false;
      }
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
        UseScratch(RowScratch(base, offsets[i]));
        Combine(s_.res, l.hc_ffn.norm.f32(), n);
      }
      UseScratch(base);
      if (!HcMixBatch(l.hc_ffn, base.res, true, base.mixed, base.inject, rows,
                      std::span(offsets).first(items.size()), error))
        return false;
      for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
        UseScratch(RowScratch(base, r));
        if (!Moe(l, s_.mixed, s_.block_out, std::min(kDecodeRows, rows - r),
                 error)) {
          return false;
        }
      }
      const float* next_norm =
          il + 1 < c.num_layers
              ? (c.IsPleLayer(il + 1) ? nullptr
                                      : layers[il + 1].hc_attn.norm.f32())
              : model_->hc_head().norm.f32();
      for (std::size_t i = 0; i < items.size(); ++i) {
        UseScratch(RowScratch(base, offsets[i]));
        Combine(s_.res, next_norm, items[i].tokens.size());
      }
      normed = next_norm != nullptr;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      if (!items[i].session->Cancelled() && items[i].session->mtp_enabled_ &&
          !Check(hipMemcpyAsync(
                     items[i].session->mtp_.target_hidden, s_.res,
                     static_cast<std::size_t>(n) * c.HcDim() * sizeof(float),
                     hipMemcpyDeviceToDevice, stream_),
                 error)) {
        return false;
      }
    }
    UseScratch(base);
    return HcMixBatch(model_->hc_head(), base.res, false, base.mixed, nullptr,
                      rows, {}, error) &&
           DenseBatch(model_->output(), base.mixed, batch_logits_, rows, error);
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    items[i].session->position_ += items[i].tokens.size();
    if (sparse[i]) {
      items[i].session->blocks_ = complete[i];
    }
  }
  batch_rows_ = rows;
  return true;
}

bool Executor::SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                 float* logits, std::string* error) const {
  if (rows == 0 || rows > options_.max_logit_rows || offset > batch_rows_ ||
      rows > batch_rows_ - offset) {
    return Fail(error, "logit rows outside the completed decode batch");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * config().vocab_size;
  selected_logits_ =
      batch_logits_ + static_cast<std::size_t>(offset) * config().vocab_size;
  if (logits != nullptr) {
    if (!Check(hipMemcpyAsync(logits_host_, selected_logits_,
                              count * sizeof(float), hipMemcpyDeviceToHost,
                              stream_),
               error) ||
        !Check(hipStreamSynchronize(stream_), error)) {
      return false;
    }
    std::copy_n(logits_host_, count, logits);
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
