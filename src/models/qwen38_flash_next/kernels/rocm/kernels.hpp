#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
namespace gufo::models::qwen38_flash_next::rocm {

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ8_0 = 8,
  kBF16 = 30,
};

/// res[t][s][hidden] = table[tokens[t]] for every stream s (Q8_0 rows).
void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// Hyper-connection mix epilogue: mixed[t][i] = mean_s xn[t][s][i] *
/// sigmoid(gate[t][s][i]); with `inject_w` ([streams][hc_dim] f32) also
/// inject[t][s] = dot(inject_w[s], xn[t]).
/// Partial sums per inject logit the fused epilogue emits; `inject` holds
/// n_tokens * streams * HcInjectParts(hidden) floats.
std::uint32_t HcInjectParts(std::uint32_t hidden);
void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream);

/// res[t][s][i] += block_out[t][i] * 2*sigmoid(inject[t][s] / streams), and
/// when `gamma` is non-null also the next mixer's grouped RMSNorm of the
/// updated residual into `xn`.
/// `inject` is [t][s][inject_parts] partial sums (1 part for a plain GEMM).
void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream);

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);
/// dst[i] += src[i].
void AddInPlace(float* dst, const float* src, std::size_t count,
                hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, alpha/beta, indexer, inject) that the
/// quantized tier does not cover.
/// Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// PLE gate: for every stream s, sc = dot(key_n[t][s], query_n[t][s]) /
/// sqrt(hidden); gated[t][s][i] = value[t][i] * sigmoid(sgn(sc) *
/// sqrt(max(|sc|, 1e-6))).
void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream);

/// Dilated depthwise causal conv over tokens with SiLU:
/// out[t][c] = silu(sum_k w[c*kernel+k] * in[t - (kernel-1-k)*dilation][c]),
/// where tokens before the chunk come from `history` ([hist][channels],
/// oldest first). `history` is then advanced past the chunk.
/// `snapshots`, when non-null, receives the history as it stands after
/// every token: [n_tokens][hist][channels].
void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, float* snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream);

/// res[t][c] += gated[t][c] + conv[t][c] over hc_dim channels.
void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream);

/// Gated DeltaNet over a chunk of tokens for one layer. Runs the causal
/// conv (with rolling `conv_state`, [kernel-1][channels]) and the recurrence
/// on `state` ([v_heads][d][d]) sequentially over tokens, parallel over heads
/// and value dims. `alpha_beta` holds both projections per token:
/// [t][alpha(v_heads) | beta(v_heads)]. Outputs the normalized,
/// sigmoid-gated attention rows [tokens][v_heads*d] ready for the output
/// projection.
/// `state_snapshots` ([n_tokens][v_heads*d*d]) and `conv_snapshots`
/// ([n_tokens][(kernel-1)*channels]), when non-null, receive the recurrent
/// state as it stands after every token, so a speculative batch can be cut
/// back to any accepted prefix.
/// Scratch: `conv_scratch` ((n_tokens + kernel) * channels), `qn`/`kn`
/// (n_tokens * k_heads * d), `raw` (n_tokens * v_heads * d).
void GatedDeltaNet(const float* qkv, const float* z, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   float* state_snapshots, float* conv_snapshots,
                   std::uint32_t n_tokens, std::uint32_t k_heads,
                   std::uint32_t v_heads, std::uint32_t d, std::uint32_t kernel,
                   float eps, hipStream_t stream);

/// Splits the interleaved [q|gate] projection into q [t][heads][d] and
/// gate [t][heads*d].
void UnpackQGate(const float* qg, float* q, float* gate, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, hipStream_t stream);

/// NEOX partial rotary on x [t][heads][d] at positions start_pos + t.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim, std::uint32_t start_pos,
          float theta, hipStream_t stream);

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim].
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t start_pos,
             hipStream_t stream);

/// Pools raw indexer keys ([pos][dim] f32) of complete blocks
/// [first_block, first_block + n_blocks) into block keys: mean over `ratio`
/// positions, RMSNorm with gamma, rotary at the block start position.
void PoolIndexerBlocks(const float* raw_keys, const float* gamma, float* blocks,
                       std::uint32_t first_block, std::uint32_t n_blocks,
                       std::uint32_t ratio, std::uint32_t dim,
                       std::uint32_t rotary_dim, float theta, float eps,
                       hipStream_t stream);

/// Per query t (position start_pos + t): scores every complete block below
/// its own tail, keeps the `budget` highest, and writes a visibility bitmap
/// (`mask_words` uint32 per query, bit b = block b visible). Every block is
/// visible when the count fits the budget. `scores` is scratch of
/// n_tokens * max_blocks floats.
void SelectBlocks(const float* q, const float* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens, std::uint32_t start_pos,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream);

/// Causal GQA attention of `n_tokens` queries at start_pos + t against the
/// f16 caches, with block visibility from `mask` (null = dense) and the
/// incomplete tail always visible. out[t][heads*d].
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               std::uint32_t n_tokens, std::uint32_t start_pos,
               std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t ratio, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
/// Scaled, causal (and block-masked) softmax of raw scores [heads][n][n_kv]
/// into F16 probabilities; row (h, t) sees keys up to start_pos + t.
void AttentionSoftmax(const float* scores, const std::uint32_t* mask,
                      std::uint32_t mask_words, __half* probs,
                      std::uint32_t n_tokens, std::uint32_t n_kv,
                      std::uint32_t start_pos, std::uint32_t heads,
                      std::uint32_t d, std::uint32_t ratio, hipStream_t stream);

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);

/// MTP input: res[t][s][i] = eh_proj( [enorm(embd[t]) ; hnorm(h[t][s])] ) is
/// assembled here as concat[t][s][2*hidden] for the tier's GEMM.
void MtpConcat(const float* embd_n, const float* h_n, float* concat,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, hipStream_t stream);

/// Diagnostic: out[0] = sum of x[0..count), out[1] = sum of |x|.
void Checksum(const float* x, std::size_t count, float* out, hipStream_t stream);

/// argmax of logits[t][vocab] into out[t].
void Argmax(const float* logits, std::int32_t* out, std::uint32_t n_tokens,
            std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
