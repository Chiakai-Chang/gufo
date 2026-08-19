#ifndef STRIX_CORE_HIP_QWEN_GPU_EXECUTOR_HPP_
#define STRIX_CORE_HIP_QWEN_GPU_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen_forward.hpp"
#include "src/models/qwen_generator.hpp"
#include "src/models/qwen_state.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace strix::hip {

class HipblasLtGemm;

struct QwenGpuWeightRegion {
  const void* host_data{nullptr};
  void* device_data{nullptr};
  std::size_t size{0};
  bool owns_device_memory{false};
  bool host_registered{false};
};

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
class QwenGpuArena {
public:
  explicit QwenGpuArena(const core::ModelConfig& config,
                        std::uint32_t max_context = 4096);
  ~QwenGpuArena();

  QwenGpuArena(const QwenGpuArena&) = delete;
  QwenGpuArena& operator=(const QwenGpuArena&) = delete;
  QwenGpuArena(QwenGpuArena&&) noexcept;
  QwenGpuArena& operator=(QwenGpuArena&&) noexcept;

  void Reset() noexcept;

  float* d_hidden{nullptr};
  float* d_normed{nullptr};
  float* d_q{nullptr};
  float* d_k{nullptr};
  float* d_v{nullptr};
  float* d_attn_out{nullptr};
  float* d_ffn_gate{nullptr};
  float* d_ffn_up{nullptr};
  float* d_ffn_act{nullptr};
  float* d_ffn_out{nullptr};
  float* d_ssm_qkv{nullptr};
  float* d_conv_out{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_out{nullptr};
  float* d_alpha_buf{nullptr};
  float* d_beta_buf{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  float* d_kv_cache{nullptr};
  float* d_ssm_conv_state{nullptr};
  float* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};

  hipStream_t stream{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }

private:
  void FreeAll() noexcept;

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
};

/// End-to-end GPU model executor running directly on the gfx1151 RDNA 3.5 CUs.
class QwenGpuExecutor {
public:
  QwenGpuExecutor(models::QwenModelWeights weights,
                  std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
                  std::vector<QwenGpuWeightRegion> weight_regions,
                  std::uint32_t max_context = 4096);
  ~QwenGpuExecutor();

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Generates tokens auto-regressively on GPU with streaming callback.
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }

  /// Runs one single token forward step on GPU, returning next token ID.
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }

  /// Resets GPU cache and recurrent states in the arena.
  void Reset() noexcept { arena_.Reset(); }

private:
  models::QwenModelWeights weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
  QwenGpuArena arena_;
  std::vector<float> h_logits_;
};

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_CORE_HIP_QWEN_GPU_EXECUTOR_HPP_
