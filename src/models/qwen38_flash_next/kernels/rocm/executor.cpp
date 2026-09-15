#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

// The tier asks its host for producer-emitted Q8_1 activations (a DeepSeek
// decode fusion). This runtime never registers any, so every lookup misses
// and the tier quantizes the activation itself.
extern "C" int qfn_cuda_q8_fold_take_q81(const void* /*src*/,
                                         std::uint64_t /*in_dim*/,
                                         const void** /*q81*/) {
  return 0;
}

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

using core::GgmlType;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template <typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hipMalloc(&p, bytes) != hipSuccess) {
    AssignError(error_msg, "hipMalloc of " + std::to_string(bytes) +
                               " bytes failed");
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  return static_cast<T*>(p);
}

WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    default:
      return WeightType::kF32;
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;

}  // namespace

Session::~Session() {
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::Reset() {
  position_ = 0;
  spec_tokens_ = 0;
  ngram_.Reset();
  mtp_.position = 0;
  const Config& c = owner_->config();
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      (void)hipMemset(l.conv_state, 0,
                static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                    c.SsmConvChannels() * sizeof(float));
      (void)hipMemset(l.state, 0,
                static_cast<std::size_t>(c.ssm_num_v_heads) * c.ssm_head_dim *
                    c.ssm_head_dim * sizeof(float));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    (void)hipMemset(ple_history_, 0,
              static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim() *
                  sizeof(float));
  }
}

Executor::~Executor() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  for (void* p : {static_cast<void*>(host_emb_), static_cast<void*>(control_host_),
                  static_cast<void*>(tokens_host_), static_cast<void*>(logits_host_),
                  static_cast<void*>(counts_host_)}) {
    if (p != nullptr) {
      (void)hipHostFree(p);
    }
  }
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {
  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  const std::uint32_t vocab = model.config().vocab_size;
  if (options.draft_rows == 0 || options.draft_rows > vocab) {
    e->options_.draft_rows = vocab;
  }
  if (qfn_mmq_init(0) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  const std::size_t T = e->options_.max_batch;
  e->blaslt_ = BlasLt::Create(e->stream_, static_cast<std::uint32_t>(T),
                              error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  s.tokens = Alloc<std::int32_t>(a, T, error_msg);
  s.x_half = Alloc<std::uint16_t>(a, T * model.max_half_cols(), error_msg);
  for (void*& slot : s.x_q8) {
    slot = Alloc<std::uint8_t>(
        a, qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                              static_cast<int>(model.max_q8_cols())),
        error_msg);
  }
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.lo = f32(T * c.hc_low_rank);
  s.hc_gate = f32(T * hc_dim);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  s.qkv = f32(T * c.SsmConvChannels());
  s.z = f32(T * c.SsmValueDim());
  s.qkvz = f32(T * (c.SsmConvChannels() + c.SsmValueDim()));
  s.alpha_beta = f32(T * 2 * c.ssm_num_v_heads);
  s.conv_scratch = f32((T + c.ssm_conv_kernel) * c.SsmConvChannels());
  s.qn = f32(T * c.SsmKeyDim());
  s.kn = f32(T * c.SsmKeyDim());
  s.gdn_raw = f32(T * c.SsmValueDim());
  s.gdn_out = f32(T * c.SsmValueDim());
  s.qg = f32(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = f32(T * c.AttentionQDim());
  s.attn_gate = f32(T * c.AttentionQDim());
  s.k = f32(T * c.AttentionKvDim());
  s.v = f32(T * c.AttentionKvDim());
  s.iq = f32(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = f32(T * c.indexer_head_dim);
  const std::uint32_t max_blocks =
      (c.context_length + c.compress_ratio - 1) / c.compress_ratio;
  e->mask_words_ = (max_blocks + 31) / 32;
  s.mask = Alloc<std::uint32_t>(a, T * e->mask_words_, error_msg);
  s.scores = f32(static_cast<std::size_t>(e->select_chunk_) * max_blocks);
  s.ctx = f32(T * c.AttentionQDim());
  // Wide batches score every key of a dense window with GEMMs; past the
  // sparse budget the per-token kernel gathers the selected blocks instead.
  s.score_kv = static_cast<std::uint32_t>(
      std::max<std::size_t>(c.indexer_top_k, c.compress_ratio) + T);
  s.q_half = Alloc<__half>(a, T * c.AttentionQDim(), error_msg);
  s.attn_scores =
      f32(static_cast<std::size_t>(c.num_heads) * T * s.score_kv);
  s.probs = Alloc<__half>(a, static_cast<std::size_t>(c.num_heads) * T * s.score_kv,
                          error_msg);
  if (c.ple_layer >= 0) {
    s.ple_emb = f32(T * c.PleEmbeddingDim());
    s.ple_key = f32(T * hc_dim);
    s.ple_value = f32(T * hidden);
    s.ple_query = f32(T * hc_dim);
    s.ple_gated = f32(T * hc_dim);
    s.ple_norm = f32(T * hc_dim);
    s.ple_conv = f32(T * hc_dim);
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    void* pinned = nullptr;
    if (!Check(hipHostMalloc(&pinned, T * c.PleEmbeddingDim() * sizeof(float)),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    e->host_rows_.resize(T * c.ple_heads);
  }
  s.router = f32(T * (c.num_experts + 1));
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  s.weights = f32(slots);
  s.gate_e = f32(slots * c.expert_ff);
  s.up_e = f32(slots * c.expert_ff);
  s.down_e = f32(slots * hidden);
  s.shexp_gate = f32(T * c.shared_expert_ff);
  s.shexp_up = f32(T * c.shared_expert_ff);
  s.shexp_out = f32(T * hidden);
  s.logits = f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(hipHostMalloc(&control, sizeof(Session::Control)),
               "pinned control buffer", error_msg) ||
        !Check(hipHostMalloc(&tokens, T * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(hipHostMalloc(&logits, static_cast<std::size_t>(
                                          e->options_.max_logit_rows) *
                                          c.vocab_size * sizeof(float)),
               "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(hipHostMalloc(&counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
  }
  if (const char* env = std::getenv("QFN_GRAPHS"); env != nullptr) {
    e->graphs_enabled_ = std::string_view(env) != "0";
  }
  if (std::getenv("QFN_TRACE") != nullptr) {
    e->trace_ = f32(static_cast<std::size_t>(c.num_layers) * 8 + 16);
  }
  // Every row of the batch is kept: the draft block consumes the trunk's
  // hidden state of each prompt position during its own prefill.
  s.hc_keep = f32(T * hc_dim);
  if (model.has_mtp()) {
    s.mtp_h = f32(T * hc_dim);
    s.mtp_embd = f32(T * hidden);
    s.mtp_concat = f32(T * hc_dim * 2);
    s.mtp_res = f32(T * hc_dim);
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  return e;
}

std::unique_ptr<Session> Executor::CreateSession(std::uint32_t max_context,
                                                 std::string* error_msg) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  const Config& c = config();
  if (max_context == 0 || max_context > c.context_length) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  s->linear_.resize(c.num_layers);
  s->attention_.resize(c.num_layers);
  s->ngram_snapshots_.resize(options_.max_speculative);
  auto& a = s->allocations_;
  const std::size_t kv_row = c.AttentionKvDim();
  const std::size_t spec = options_.max_speculative;
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      auto& l = s->linear_[il];
      l.conv_state = Alloc<float>(a, conv_elems, error_msg);
      l.state = Alloc<float>(a, state_elems, error_msg);
      l.conv_snapshots = Alloc<float>(a, spec * conv_elems, error_msg);
      l.state_snapshots = Alloc<float>(a, spec * state_elems, error_msg);
    } else {
      auto& at = s->attention_[il];
      at.k_cache = Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                                 error_msg);
      at.v_cache = Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                                 error_msg);
      at.index_k = Alloc<float>(
          a, static_cast<std::size_t>(max_context) * c.indexer_head_dim, error_msg);
      at.block_k = Alloc<float>(
          a, static_cast<std::size_t>(max_context / c.compress_ratio + 1) *
                 c.indexer_head_dim,
          error_msg);
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist = static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    s->ple_history_ = Alloc<float>(a, hist, error_msg);
    s->ple_snapshots_ = Alloc<float>(a, spec * hist, error_msg);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg);
  if (model_->has_mtp()) {
    s->mtp_.k_cache = Alloc<__half>(
        a, static_cast<std::size_t>(max_context) * kv_row, error_msg);
    s->mtp_.v_cache = Alloc<__half>(
        a, static_cast<std::size_t>(max_context) * kv_row, error_msg);
    s->mtp_.h = Alloc<float>(a, c.HcDim(), error_msg);
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  return s;
}


/// Column-tile width for the routed expert GEMMs: the tile at or above twice
/// the mean bucket, so most experts fit one tile with little padding.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  if (const char* env = std::getenv("QFN_MOE_TILE")) {
    return std::atoi(env);
  }
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

bool Executor::Quantize(const float* x, std::uint32_t n_tokens,
                        std::uint32_t k, Q8Input* q,
                        std::string* error_msg) const {
  q->x = x;
  q->data = nullptr;
  q->n = n_tokens;
  q->k = k;
  if (n_tokens > kVecBatch) {
    return true;  // the tiled path quantizes per call
  }
  // Two slots alternate, so an input stays valid across one other
  // quantization; captured graphs replay the same alternation.
  void* slot = s_.x_q8[q8_slot_];
  q8_slot_ ^= 1u;
  if (qfn_mmq_quantize_q8_1(x, slot, static_cast<int>(n_tokens),
                            static_cast<int>(k), stream_) != 0) {
    AssignError(error_msg, "activation quantization failed");
    return false;
  }
  q->data = slot;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const Q8Input& q, float* out,
                     std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0 && q.data != nullptr) {
    if (w.cols != q.k) {
      AssignError(error_msg, "quantized input width mismatch");
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(w.data, nullptr, q.data, out,
                                    static_cast<int>(w.rows),
                                    static_cast<int>(q.n),
                                    static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  return Dense(w, q.x, out, q.n, error_msg);
}

bool Executor::GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                          const float* x, float* out, std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // One single-token launch computes both projections and the SwiGLU (the
  // tier fuses the gate for one column only).
  if (n_tokens == 1 && up.type == GgmlType::kQ8_0 &&
      gate.type == GgmlType::kQ8_0 && up.rows == gate.rows &&
      up.cols == gate.cols) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, up.cols, &xq, error_msg)) {
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(up.data, gate.data, xq.data, out,
                                    static_cast<int>(up.rows),
                                    static_cast<int>(n_tokens),
                                    static_cast<int>(up.cols), stream_) != 0) {
      AssignError(error_msg, "gated Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  // Swiglu is in place over its first operand.
  if (!Dense(gate, x, out, n_tokens, error_msg) ||
      !Dense(up, x, s_.shexp_gate, n_tokens, error_msg)) {
    return false;
  }
  Swiglu(out, s_.shexp_gate, static_cast<std::size_t>(n_tokens) * up.rows,
         stream_);
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0) {
    if (n_tokens <= kVecBatch) {
      Q8Input q;
      return Quantize(x, n_tokens, w.cols, &q, error_msg) &&
             Dense(w, q, out, error_msg);
    }
    if (qfn_mmq_q8_0_dense(w.data, x, out, static_cast<int>(w.rows),
                           static_cast<int>(n_tokens),
                           static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "Q8_0 GEMM failed");
      return false;
    }
    return true;
  }
  if (n_tokens <= kVecBatch) {
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  // Wide batches of the unquantized projections (router, alpha/beta,
  // indexer): out[t][m] = sum_k w[m][k] x[t][k].
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     static_cast<const float*>(w.data), k, x, k, &beta, out,
                     m) != HIPBLAS_STATUS_SUCCESS) {
      AssignError(error_msg, "hipBLAS GEMM failed");
      return false;
    }
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                    stream_);
  return blaslt_->Gemm({w.data, type, k, 0, false},
                       {s_.x_half, type, k, 0, false}, out, m, 0, m, n, k, 1,
                       error_msg);
}

void Executor::RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const {
  // The column grid is bounded by the largest expert bucket and the tile
  // width fitted to the whole distribution (see RouteHints); the fallback
  // bound is the token count with a tile near twice the mean bucket.
  if (routed_max_rows_ > 0) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    qfn_mmq_set_routed_tile_cols(routed_tile_cols_);
    return;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  qfn_mmq_set_routed_tile_cols(
      RoutedTileCols(n_tokens, config().num_experts_used, w.experts));
}

bool Executor::RouteHints(std::uint32_t n_tokens, std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the grid is cut to the real largest
  // bucket: the per-expert counts come back to the host (one short stall
  // per layer), which also lets the tile width follow the distribution.
  const Config& c = config();
  routed_max_rows_ = 0;
  if (n_tokens <= 4 * kVecBatch) {
    return true;
  }
  ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts,
               c.num_experts_used, stream_);
  if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                            c.num_experts * sizeof(std::uint32_t),
                            hipMemcpyDeviceToHost, stream_),
             "expert counts download", error_msg) ||
      !Check(hipStreamSynchronize(stream_), "expert counts", error_msg)) {
    return false;
  }
  std::uint32_t max_rows = 0;
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    max_rows = std::max(max_rows, counts_host_[e]);
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  if (const char* env = std::getenv("QFN_MOE_GRID_ROWS")) {
    routed_max_rows_ = static_cast<std::uint32_t>(std::atoi(env));
  }
  routed_tile_cols_ = qfn_mmq_routed_tile_cols_for_counts(
      counts_host_, static_cast<int>(c.num_experts));
  if (const char* env = std::getenv("QFN_MOE_TILE")) {
    routed_tile_cols_ = std::atoi(env);
  }
  return true;
}

bool Executor::Experts(const DeviceTensor& w, const float* x,
                       const std::int32_t* ids, float* out,
                       std::uint32_t n_rows, std::uint32_t n_used,
                       std::uint32_t n_tokens, std::string* error_msg) const {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(n_rows);
  const int U = static_cast<int>(n_used);
  // The vector entries loop over column chunks, so the decode-time down
  // projection (top-k rows, one expert each) stays on them as well.
  const bool tiled = n_rows > 4 * kVecBatch;
  if (tiled) {
    RoutedHints(w, n_tokens);
  }
  int rc = -1;
  switch (w.type) {
    case GgmlType::kQ4_K:
      rc = tiled ? qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, T, E, U,
                                        stream_)
                 : qfn_mmq_q4_K_moe_vec(w.data, x, ids, out, M, K, T, E, U,
                                        stream_);
      break;
    case GgmlType::kQ5_K:
      rc = tiled ? qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, M, K, T, E, U,
                                        stream_)
                 : qfn_mmq_q5_K_moe_vec(w.data, x, ids, out, M, K, T, E, U,
                                        stream_);
      break;
    case GgmlType::kQ5_1:
      rc = tiled ? qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, M, K, T, E, U,
                                        stream_)
                 : qfn_mmq_q5_1_moe_vec(w.data, x, ids, out, M, K, T, E, U,
                                        stream_);
      break;
    case GgmlType::kQ8_0:
      rc = tiled ? qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, M, K, T, E, U,
                                        stream_)
                 : qfn_mmq_q8_0_moe_vec(w.data, x, ids, out, M, K, T, E, U,
                                        stream_);
      break;
    default:
      break;
  }
  if (rc != 0) {
    AssignError(error_msg, "expert GEMM failed");
    return false;
  }
  return true;
}

bool Executor::ExpertPair(const DeviceTensor& a, const DeviceTensor& b,
                          const float* x, const std::int32_t* ids,
                          float* out_a, float* out_b, std::uint32_t n_tokens,
                          std::uint32_t n_used, std::string* error_msg) const {
  // One row gather and quantization feeds both projections; only the Q4_K
  // tile path has the paired entry.
  if (n_tokens > 4 * kVecBatch && a.type == GgmlType::kQ4_K &&
      b.type == GgmlType::kQ4_K && a.rows == b.rows && a.cols == b.cols) {
    RoutedHints(a, n_tokens);
    if (qfn_mmq_q4_K_moe_pair_unique(
            a.data, b.data, x, ids, out_a, out_b, static_cast<int>(a.rows),
            static_cast<int>(a.cols), static_cast<int>(n_tokens),
            static_cast<int>(a.experts), static_cast<int>(n_used),
            stream_) != 0) {
      AssignError(error_msg, "expert pair GEMM failed");
      return false;
    }
    return true;
  }
  return Experts(a, x, ids, out_a, n_tokens, n_used, n_tokens, error_msg) &&
         Experts(b, x, ids, out_b, n_tokens, n_used, n_tokens, error_msg);
}

bool Executor::HcMix(const DeviceMixer& m, const float* res, const float* xn,
                     float* mixed, float* inject, std::uint32_t n_tokens,
                     std::string* error_msg) const {
  const Config& c = config();
  // A null xn asks for this mixer's grouped norm of res; otherwise the
  // caller (the previous combine) already produced it.
  if (xn == nullptr) {
    RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
    xn = s_.xn;
  }
  if (!Dense(m.down, xn, s_.lo, n_tokens, error_msg)) {
    return false;
  }
  SiluScale(s_.lo, 1.0F / static_cast<float>(c.hc_count),
            static_cast<std::size_t>(n_tokens) * c.hc_low_rank, stream_);
  if (!Dense(m.up, s_.lo, s_.hc_gate, n_tokens, error_msg)) {
    return false;
  }
  const bool fused_inject =
      inject != nullptr && !m.inject.empty() && m.inject.type == GgmlType::kF32;
  HcMixEpilogue(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                mixed, inject, n_tokens, c.hidden_size, c.hc_count, stream_);
  inject_parts_ = fused_inject ? HcInjectParts(c.hidden_size) : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject &&
      !Dense(m.inject, xn, inject, n_tokens, error_msg)) {
    return false;
  }
  return true;
}

void Executor::PleFetch(Session& session, std::span<const std::int32_t> tokens,
                        bool speculative) const {
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  // A speculative batch keeps the hash window after every token.
  if (speculative) {
    for (std::uint32_t i = 0; i < n; ++i) {
      HashNgramRows(c, session.ngram_, tokens.subspan(i, 1),
                    std::span<std::uint32_t>(host_rows_.data() + i * c.ple_heads,
                                             c.ple_heads));
      session.ngram_snapshots_[i] = session.ngram_;
    }
  } else {
    HashNgramRows(c, session.ngram_, tokens,
                  std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
  }
  ple_read_ = std::async(std::launch::async, [this, n, &c] {
    return ngram_->Read(
        std::span<const std::uint32_t>(host_rows_.data(), n * c.ple_heads),
        std::span<float>(host_emb_,
                         static_cast<std::size_t>(n) * c.PleEmbeddingDim()));
  });
}

bool Executor::Ple(const DeviceLayer& l, Session& session, std::uint32_t n,
                   float* res, bool speculative, std::string* error_msg) const {
  const Config& c = config();
  const std::size_t emb_count = static_cast<std::size_t>(n) * c.PleEmbeddingDim();
  if (!ple_read_.valid() || !ple_read_.get()) {
    AssignError(error_msg, "n-gram table read failed");
    return false;
  }
  if (!Check(hipMemcpyAsync(s_.ple_emb, host_emb_, emb_count * sizeof(float),
                            hipMemcpyHostToDevice, stream_),
             "n-gram upload", error_msg)) {
    return false;
  }
  const std::uint32_t hc_dim = c.HcDim();
  // QFN_TRACE checksums of every PLE stage, after the per-layer ones.
  auto tr = [&](int slot, const float* x, std::size_t count) {
    if (trace_ != nullptr) {
      Checksum(x, count,
               trace_ + static_cast<std::size_t>(c.num_layers) * 8 + slot * 2,
               stream_);
    }
  };
  tr(0, s_.ple_emb, static_cast<std::size_t>(n) * c.PleEmbeddingDim());
  Q8Input emb;
  if (!Quantize(s_.ple_emb, n, c.PleEmbeddingDim(), &emb, error_msg) ||
      !Dense(l.ple_key, emb, s_.ple_key, error_msg) ||
      !Dense(l.ple_value, emb, s_.ple_value, error_msg)) {
    return false;
  }
  tr(1, s_.ple_key, static_cast<std::size_t>(n) * hc_dim);
  tr(2, s_.ple_value, static_cast<std::size_t>(n) * c.hidden_size);
  RmsNormRows(s_.ple_key, l.ple_norm_key.f32(), s_.ple_key, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(res, l.ple_norm_query.f32(), s_.ple_query, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  tr(3, s_.ple_key, static_cast<std::size_t>(n) * hc_dim);
  tr(4, s_.ple_query, static_cast<std::size_t>(n) * hc_dim);
  PleGate(s_.ple_key, s_.ple_query, s_.ple_value, s_.ple_gated, n,
          c.hidden_size, c.hc_count, stream_);
  tr(5, s_.ple_gated, static_cast<std::size_t>(n) * hc_dim);
  RmsNormRows(s_.ple_gated, l.ple_norm_conv.f32(), s_.ple_norm, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  PleConv(s_.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
          s_.ple_history_scratch, s_.ple_conv,
          speculative ? session.ple_snapshots_ : nullptr, n, hc_dim,
          c.ple_conv_kernel, c.ple_ngram_size, stream_);
  tr(6, s_.ple_conv, static_cast<std::size_t>(n) * hc_dim);
  PleInject(res, s_.ple_gated, s_.ple_conv,
            static_cast<std::size_t>(n) * hc_dim, stream_);
  tr(7, res, static_cast<std::size_t>(n) * hc_dim);
  return true;
}

bool Executor::LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                               const float* x, float* out,
                               std::uint32_t n_tokens, bool speculative,
                               std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t channels = c.SsmConvChannels();
  const float* qkv = s_.qkv;
  const float* z = s_.z;
  std::uint32_t qkv_stride = channels;
  std::uint32_t z_stride = c.SsmValueDim();
  if (!l.ssm_in.empty()) {
    // One GEMV yields [qkv | z] per row.
    if (!Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg)) {
      return false;
    }
    qkv = s_.qkvz;
    z = s_.qkvz + channels;
    qkv_stride = z_stride = l.ssm_in.rows;
  } else {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.ssm_qkv, xq, s_.qkv, error_msg) ||
        !Dense(l.ssm_gate, xq, s_.z, error_msg)) {
      return false;
    }
  }
  if (!Dense(l.ssm_alpha_beta, x, s_.alpha_beta, n_tokens, error_msg)) {
    return false;
  }
  GatedDeltaNet(qkv, qkv_stride, z, z_stride, s_.alpha_beta, l.ssm_conv1d.f32(),
                l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), s.conv_state,
                s_.conv_scratch, s_.qn, s_.kn, s_.gdn_raw, s.state, s_.gdn_out,
                speculative ? s.state_snapshots : nullptr,
                speculative ? s.conv_snapshots : nullptr, n_tokens,
                c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
                c.ssm_conv_kernel, c.rms_eps, stream_);
  return Dense(l.ssm_out, s_.gdn_out, out, n_tokens, error_msg);
}

bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         const std::uint32_t* first_block,
                         std::uint32_t start_pos, std::uint32_t pool_grid,
                         std::uint32_t max_context, bool sparse,
                         std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t kv_row = c.AttentionKvDim();
  if (!l.attn_qkv.empty()) {
    // One GEMV yields [q|gate ; k ; v] per row; the unpack splits it.
    if (!Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
      return false;
    }
    UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
  } else {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.attn_q, xq, s_.qg, error_msg) ||
        !Dense(l.attn_k, xq, s_.k, error_msg) ||
        !Dense(l.attn_v, xq, s_.v, error_msg)) {
      return false;
    }
    UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
  }
  RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
              c.head_dim, 1, c.rms_eps, stream_);
  RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
              c.head_dim, 1, c.rms_eps, stream_);
  Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
       c.rope_theta, stream_);
  Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
       c.rope_theta, stream_);
  StoreKv(s_.k, s.k_cache, n_tokens, kv_row, pos, stream_);
  StoreKv(s_.v, s.v_cache, n_tokens, kv_row, pos, stream_);

  // Raw indexer keys are always cached: a later chunk past the budget
  // pools them into block keys. The draft block keeps no indexer cache.
  if (s.index_k != nullptr) {
    if (!Dense(l.indexer_k, x, s_.ik, n_tokens, error_msg)) {
      return false;
    }
    StoreRows(s_.ik, s.index_k, n_tokens, c.indexer_head_dim, pos, stream_);
  }
  const std::uint32_t* mask = nullptr;
  if (sparse) {
    if (!Dense(l.indexer_q, x, s_.iq, n_tokens, error_msg)) {
      return false;
    }
    RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq,
                n_tokens * c.indexer_heads, c.indexer_head_dim, 1, c.rms_eps,
                stream_);
    Rope(s_.iq, n_tokens, c.indexer_heads, c.indexer_head_dim, c.rotary_dim,
         pos, c.rope_theta, stream_);
    PoolIndexerBlocks(s.index_k, l.indexer_k_norm.f32(), s.block_k,
                      first_block, pos, n_tokens, pool_grid,
                      c.compress_ratio, c.indexer_head_dim, c.rotary_dim,
                      c.rope_theta, c.rms_eps, stream_);
    const std::uint32_t max_blocks =
        (max_context + c.compress_ratio - 1) / c.compress_ratio;
    for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += select_chunk_) {
      const std::uint32_t n = std::min(select_chunk_, n_tokens - t0);
      SelectBlocks(s_.iq + static_cast<std::size_t>(t0) * c.indexer_heads *
                              c.indexer_head_dim,
                   s.block_k, s_.mask + static_cast<std::size_t>(t0) * mask_words_,
                   s_.scores, n, pos, t0, c.indexer_heads, c.indexer_head_dim,
                   c.compress_ratio, c.indexer_top_k / c.compress_ratio,
                   mask_words_, max_blocks, stream_);
    }
    mask = s_.mask;
  }
  // Wide batches score the dense window with GEMMs (never inside a graph:
  // the kv extent is a host value); the per-token kernel covers the rest.
  if (n_tokens > kVecBatch && start_pos + n_tokens <= s_.score_kv) {
    if (!BatchedAttention(s, mask, n_tokens, start_pos, error_msg)) {
      return false;
    }
  } else {
    rocm::Attention(s_.q, s.k_cache, s.v_cache, mask, mask_words_, s_.ctx,
                    n_tokens, pos, c.num_heads, c.num_kv_heads, c.head_dim,
                    c.compress_ratio, stream_);
  }
  SigmoidMul(s_.ctx, s_.attn_gate,
             static_cast<std::size_t>(n_tokens) * c.AttentionQDim(), stream_);
  return Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::BatchedAttention(const Session::AttentionState& s,
                                const std::uint32_t* mask,
                                std::uint32_t n_tokens, std::uint32_t start_pos,
                                std::string* error_msg) const {
  const Config& c = config();
  const int d = static_cast<int>(c.head_dim);
  const int n = static_cast<int>(n_tokens);
  const int n_kv = static_cast<int>(start_pos + n_tokens);
  const int group = static_cast<int>(c.num_heads / c.num_kv_heads);
  const int kv_stride = static_cast<int>(c.AttentionKvDim());
  const int q_stride = static_cast<int>(c.AttentionQDim());
  const long long scores_per_head = static_cast<long long>(n) * n_kv;
  NarrowActivations(s_.q, s_.q_half, false,
                    static_cast<std::size_t>(n) * q_stride, stream_);
  // Every query head of a kv group reads the same cache rows (batch stride
  // 0). Scores: S[t][j] = q_t . k_j per head.
  for (int g = 0; g < static_cast<int>(c.num_kv_heads); ++g) {
    if (!blaslt_->Gemm(
            {s.k_cache + g * d, HIP_R_16F, kv_stride, 0, false},
            {s_.q_half + static_cast<std::size_t>(g) * group * d, HIP_R_16F,
             q_stride, d, false},
            s_.attn_scores + static_cast<std::size_t>(g) * group * scores_per_head,
            n_kv, scores_per_head, n_kv, n, d, group, error_msg)) {
      return false;
    }
  }
  AttentionSoftmax(s_.attn_scores, mask, mask_words_, s_.probs, n_tokens, n_kv,
                   start_pos, c.num_heads, c.head_dim, c.compress_ratio,
                   stream_);
  // ctx[t][h*d + i] = sum_j P[t][j] v_j[i]; the cache is [j][i], so V
  // enters transposed.
  for (int g = 0; g < static_cast<int>(c.num_kv_heads); ++g) {
    if (!blaslt_->Gemm(
            {s.v_cache + g * d, HIP_R_16F, kv_stride, 0, true},
            {s_.probs + static_cast<std::size_t>(g) * group * scores_per_head,
             HIP_R_16F, n_kv, scores_per_head, false},
            s_.ctx + static_cast<std::size_t>(g) * group * d, q_stride, d, d,
            n, n_kv, group, error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Moe(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t slots = n_tokens * used;
  // Router logits and the shared-expert gate come out of one GEMM.
  if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  RouterTopK(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
             c.num_experts, used, stream_);
  if (!RouteHints(n_tokens, error_msg)) {
    return false;
  }
  if (!ExpertPair(l.ffn_gate_exps, l.ffn_up_exps, x, s_.ids, s_.gate_e,
                  s_.up_e, n_tokens, used, error_msg)) {
    return false;
  }
  Swiglu(s_.gate_e, s_.up_e, static_cast<std::size_t>(slots) * c.expert_ff,
         stream_);
  // The down projection sees one (token, slot) row per expert id.
  if (!Experts(l.ffn_down_exps, s_.gate_e, s_.ids, s_.down_e, slots, 1,
               n_tokens, error_msg)) {
    return false;
  }
  // Shared expert, gated by the last router row.
  if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                  error_msg) ||
      !Dense(l.shexp_down, s_.shexp_up, s_.shexp_out, n_tokens, error_msg)) {
    return false;
  }
  MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
              c.num_experts + 1, out, n_tokens, used, c.hidden_size, stream_);
  return true;
}

bool Executor::Head(const DeviceMixer& head, const float* res,
                    std::uint32_t n_rows, std::uint32_t vocab_rows,
                    std::string* error_msg) const {
  // The output matrix is [vocab][hidden], so its leading rows form a
  // contiguous smaller head.
  DeviceTensor output = model_->output();
  output.rows = std::min(output.rows, vocab_rows);
  if (!HcMix(head, res, nullptr, s_.mixed, nullptr, n_rows, error_msg) ||
      !Dense(output, s_.mixed, s_.logits, n_rows, error_msg)) {
    return false;
  }
  return Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_rows) * output.rows *
                                  sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg);
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body,
                   std::string* error_msg) const {
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    if (const auto it = session.graphs_.find(key); it != session.graphs_.end()) {
      exec = it->second;
    } else {
      hipGraph_t captured = nullptr;
      if (!Check(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal),
                 "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        return false;
      }
      session.graphs_.emplace(key, exec);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (std::getenv("QFN_GRAPH_DEBUG") != nullptr) {
      (void)hipStreamSynchronize(stream_);
      const auto t2 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "graph key %llx launch %.2f ms run %.2f ms\n",
                   static_cast<unsigned long long>(key),
                   std::chrono::duration<double, std::milli>(t1 - t0).count(),
                   std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return Check(hipStreamSynchronize(stream_), "forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, bool speculative,
                       std::string* error_msg) const {
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens, the control block (positions
  // the kernels read on the device) and the n-gram rows.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->hidden_row = -1;
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    PleFetch(session, tokens, speculative);
  }
  // Sparse selection only changes the result once a query can see more
  // than the token budget; every layer of this model shares one ratio.
  const bool sparse =
      c.compress_ratio > 0 && start_pos + n > c.indexer_top_k;
  const std::uint32_t complete =
      c.compress_ratio > 0 ? (start_pos + n) / c.compress_ratio : 0;
  const std::uint32_t pool_grid =
      sparse && complete > session.blocks_ ? complete - session.blocks_ : 0;
  // Decode-sized batches replay as graphs; a pooling backlog (the first
  // batch past the budget) needs the wider eager grid.
  const std::uint32_t graph_pool_grid = n / std::max(c.compress_ratio, 1u) + 1;
  const bool graph = graphs_enabled_ && trace_ == nullptr && n <= kVecBatch &&
                     pool_grid <= graph_pool_grid;
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (static_cast<std::uint64_t>(sparse) << 33);
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, speculative, sparse, start_pos,
                       graph ? graph_pool_grid : pool_grid, error_msg);
  };
  // A replayed graph copies the n-gram rows without passing through Ple,
  // so the read has to be complete before the launch.
  if (graph && session.graphs_.contains(key) && ple_read_.valid() &&
      !ple_read_.get()) {
    AssignError(error_msg, "n-gram table read failed");
    return false;
  }
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }
  if (trace_ != nullptr) {
    std::vector<float> h(static_cast<std::size_t>(c.num_layers) * 8 + 16);
    (void)hipMemcpy(h.data(), trace_, h.size() * 4, hipMemcpyDeviceToHost);
    const float* pl = h.data() + static_cast<std::size_t>(c.num_layers) * 8;
    std::fprintf(stderr, "trace ple emb %.5f %.5f key %.5f %.5f value %.5f %.5f keyn %.5f %.5f query %.5f %.5f gated %.5f %.5f conv %.5f %.5f res %.5f %.5f\n",
                 pl[0], pl[1], pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], pl[8], pl[9], pl[10], pl[11], pl[12], pl[13], pl[14], pl[15]);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      std::fprintf(stderr, "trace %u mixed %.5f %.5f attn %.5f %.5f ffn %.5f %.5f res %.5f %.5f\n", il,
                   h[il * 8], h[il * 8 + 1], h[il * 8 + 2], h[il * 8 + 3], h[il * 8 + 4], h[il * 8 + 5], h[il * 8 + 6], h[il * 8 + 7]);
    }
  }
  session.position_ += n;
  if (sparse) {
    session.blocks_ = complete;
  }
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool speculative,
                           bool sparse, std::uint32_t start_pos,
                           std::uint32_t pool_grid,
                           std::string* error_msg) const {
  const Config& c = config();
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
              s_.tokens, s_.res, n, c.hidden_size, c.hc_count, stream_);
  const auto& layers = model_->layers();
  bool normed = false;  ///< xn holds the next mixer's grouped norm of res
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& l = layers[il];
    if (c.IsPleLayer(il) &&
        !Ple(l, session, n, s_.res, speculative, error_msg)) {
      return false;
    }
    if (!HcMix(l.hc_attn, s_.res, normed ? s_.xn : nullptr, s_.mixed, s_.inject,
               n, error_msg)) {
      return false;
    }
    if (l.linear) {
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           speculative, error_msg)) {
        return false;
      }
    } else if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                          &session.control_->position,
                          &session.control_->blocks, start_pos, pool_grid,
                          session.max_context_, sparse, error_msg)) {
      return false;
    }
    if (trace_ != nullptr) {
      Checksum(s_.mixed, static_cast<std::size_t>(n) * c.hidden_size, trace_ + il * 8, stream_);
      Checksum(s_.block_out, static_cast<std::size_t>(n) * c.hidden_size, trace_ + il * 8 + 2, stream_);
    }
    // Each combine also norms the residual for the mixer that follows it,
    // unless PLE rewrites the residual first.
    HcCombine(s_.res, s_.block_out, s_.inject, inject_parts_,
              l.hc_ffn.norm.f32(), s_.xn, n,
              c.hidden_size, c.hc_count, c.rms_eps, stream_);
    if (!HcMix(l.hc_ffn, s_.res, s_.xn, s_.mixed, s_.inject, n, error_msg) ||
        !Moe(l, s_.mixed, s_.block_out, n, error_msg)) {
      return false;
    }
    const float* next_norm =
        il + 1 < c.num_layers
            ? (c.IsPleLayer(il + 1) ? nullptr : layers[il + 1].hc_attn.norm.f32())
            : model_->hc_head().norm.f32();
    HcCombine(s_.res, s_.block_out, s_.inject, inject_parts_, next_norm, s_.xn,
              n,
              c.hidden_size, c.hc_count, c.rms_eps, stream_);
    normed = next_norm != nullptr;
    if (trace_ != nullptr) {
      Checksum(s_.block_out, static_cast<std::size_t>(n) * c.hidden_size, trace_ + il * 8 + 4, stream_);
      Checksum(s_.res, static_cast<std::size_t>(n) * c.HcDim(), trace_ + il * 8 + 6, stream_);
    }
  }
  // Keep the wide residual of every row for the draft block.
  if (!Check(hipMemcpyAsync(s_.hc_keep, s_.res,
                            static_cast<std::size_t>(n) * c.HcDim() * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    // xn already holds the head mixer's norm of every row; take its tail.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    const DeviceMixer& head = model_->hc_head();
    if (!HcMix(head, nullptr, s_.xn + skip * c.HcDim(), s_.mixed, nullptr,
               n_logits, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg) ||
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) * c.vocab_size *
                                  sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  session.spec_tokens_ = 0;
  if (keep == n) {
    return true;
  }
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t slot = keep - 1;
  for (auto& l : session.linear_) {
    if (l.state == nullptr) {
      continue;
    }
    if (!Check(hipMemcpyAsync(l.state, l.state_snapshots + slot * state_elems,
                              state_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               "state rollback", error_msg) ||
        !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots + slot * conv_elems,
                              conv_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               "conv rollback", error_msg)) {
      return false;
    }
  }
  if (session.ple_history_ != nullptr) {
    const std::size_t hist = static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    if (!Check(hipMemcpyAsync(session.ple_history_,
                              session.ple_snapshots_ + slot * hist,
                              hist * sizeof(float), hipMemcpyDeviceToDevice,
                              stream_),
               "PLE rollback", error_msg)) {
      return false;
    }
    session.ngram_ = session.ngram_snapshots_[slot];
  }
  session.position_ = session.spec_base_ + keep;
  // Pooled block keys past the kept prefix are stale; they are rebuilt
  // from the raw keys when needed.
  session.blocks_ = std::min(session.blocks_, session.position_ / c.compress_ratio);
  return Check(hipStreamSynchronize(stream_), "rollback", error_msg);
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, float* logits,
                          std::string* error_msg) const {
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (!model_->has_mtp()) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }
  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n > options_.max_batch)) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  const std::uint32_t pos = session.mtp_.position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  const bool graph = graphs_enabled_ && trace_ == nullptr && n <= kVecBatch;
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
                            (std::uint64_t{1} << 40);
  const auto body = [&] { return MtpBody(session, n, pos, error_msg); };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  std::copy_n(logits_host_, options_.draft_rows, logits);
  session.mtp_.position = pos + n;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       std::string* error_msg) const {
  const Config& c = config();
  const DeviceLayer& l = model_->mtp();
  const std::uint32_t hc_dim = c.HcDim();
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
              s_.tokens, s_.mtp_embd, n, c.hidden_size, 1, stream_);
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  MtpHidden(s_.hc_keep, session.mtp_.h, &session.control_->hidden_row,
            s_.mtp_h, n, hc_dim, stream_);
  RmsNormRows(s_.mtp_h, l.nextn_hnorm.f32(), s_.mtp_h, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  MtpConcat(s_.mtp_embd, s_.mtp_h, s_.mtp_concat, n, c.hidden_size, c.hc_count,
            stream_);
  // eh_proj maps every stream's [embd ; hidden] pair to that stream's
  // residual: n * hc_count rows of 2*hidden through one [2*hidden -> hidden].
  if (!Dense(l.nextn_eh_proj, s_.mtp_concat, s_.mtp_res, n * c.hc_count,
             error_msg)) {
    return false;
  }
  Session::AttentionState attn;
  attn.k_cache = session.mtp_.k_cache;
  attn.v_cache = session.mtp_.v_cache;
  // The draft block's attention runs at its own position.
  if (!HcMix(l.hc_attn, s_.mtp_res, nullptr, s_.mixed, s_.inject, n, error_msg) ||
      !Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, nullptr, pos, 0,
                 session.max_context_, false, error_msg)) {
    return false;
  }
  HcCombine(s_.mtp_res, s_.block_out, s_.inject, inject_parts_,
            l.hc_ffn.norm.f32(), s_.xn, n,
            c.hidden_size, c.hc_count, c.rms_eps, stream_);
  if (!HcMix(l.hc_ffn, s_.mtp_res, s_.xn, s_.mixed, s_.inject, n, error_msg) ||
      !Moe(l, s_.mixed, s_.block_out, n, error_msg)) {
    return false;
  }
  HcCombine(s_.mtp_res, s_.block_out, s_.inject, inject_parts_, nullptr,
            nullptr, n,
            c.hidden_size, c.hc_count, c.rms_eps, stream_);
  const float* last = s_.mtp_res + static_cast<std::size_t>(n - 1) * hc_dim;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, hc_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return Head(l.nextn_head, last, 1, options_.draft_rows, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
