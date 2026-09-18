#include <algorithm>
#include <array>
#include <stdexcept>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

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
  s.iq_half += r * c.indexer_heads * c.indexer_head_dim;
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

bool Executor::DenseBatch(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
  // Wide output matrices benefit from sharing weight rows across waves.
  // The smaller output projection with its long K sweep stays at eight.
  if (rows > kDecodeRows && w.type == core::GgmlType::kQ8_0 && w.cols == 2560 &&
      w.rows >= 8192) {
    constexpr std::uint32_t chunk = 32;
    if (batch_q8_ == nullptr &&
        !Check(hipMalloc(&batch_q8_, qfn_mmq_q8_1_bytes(chunk, 2560)), error)) {
      return false;
    }
    for (std::uint32_t r = 0; r < rows;) {
      auto n = std::min(chunk, rows - r);
      if (n > kDecodeRows) {
        n = n / kDecodeRows * kDecodeRows;
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
  if (batch_logits_ == nullptr &&
      !Check(hipMalloc(&batch_logits_,
                       static_cast<std::size_t>(kBatchSessions) *
                           std::min(kDecodeRows, options_.max_logit_rows) *
                           c.vocab_size * sizeof(float)),
             error)) {
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
    session.spec_base_ = session.position_;
    session.spec_tokens_ = item.speculative ? item.tokens.size() : 0;
    batch_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, -1};
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
    const auto& layers = model_->layers();
    bool normed = false;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
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
        if ((c.IsPleLayer(il) && !Ple(l, *item.session, n, s_.res,
                                      item.speculative, error, true)) ||
            !HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, n, error)) {
          return false;
        }
      }
      UseScratch(base);
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
        if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, n, error)) {
          return false;
        }
      }
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
      if (has_mtp() &&
          !Check(hipMemcpyAsync(
                     items[i].session->mtp_.target_hidden, s_.res,
                     static_cast<std::size_t>(n) * c.HcDim() * sizeof(float),
                     hipMemcpyDeviceToDevice, stream_),
                 error)) {
        return false;
      }
      if (!HcMix(model_->hc_head(), s_.res, false, s_.mixed, nullptr, n,
                 error)) {
        return false;
      }
    }
    UseScratch(base);
    return DenseBatch(model_->output(), base.mixed, batch_logits_, rows, error);
  };
  const bool ok = body();
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
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
  if (!Check(hipMemcpyAsync(s_.logits,
                            batch_logits_ + static_cast<std::size_t>(offset) *
                                                config().vocab_size,
                            count * sizeof(float), hipMemcpyDeviceToDevice,
                            stream_),
             error)) {
    return false;
  }
  if (logits != nullptr) {
    if (!Check(hipMemcpyAsync(logits_host_, s_.logits, count * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               error) ||
        !Check(hipStreamSynchronize(stream_), error)) {
      return false;
    }
    std::copy_n(logits_host_, count, logits);
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
