#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/qwen_gpu_executor.hpp"

#include <utility>

#include "src/core/hip/detail/qwen_gpu_weight_regions.hpp"
#include "src/core/hip/hip_utils.hpp"

namespace strix::hip {
using detail::ReleaseWeightRegions;

QwenGpuExecutor::QwenGpuExecutor(
    models::QwenModelWeights weights,
    std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions, std::uint32_t max_context)
    : weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      arena_(weights_.config, max_context),
      h_logits_(weights_.config.vocab_size, 0.0F) {}

QwenGpuExecutor::~QwenGpuExecutor() {
  (void)hipStreamSynchronize(arena_.stream);
  ReleaseWeightRegions(weight_regions_);
}

std::span<const float> QwenGpuExecutor::CopyLastLogits() {
  HIP_CHECK(hipMemcpyAsync(h_logits_.data(), arena_.d_logits,
                           h_logits_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

std::vector<tokenization::TokenId> QwenGpuExecutor::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }

  arena_.Reset();

  // 1. Batched GPU prompt prefill
  tokenization::TokenId next_token = ForwardPromptBatch(prompt_tokens);

  std::size_t cur_pos = prompt_tokens.size();
  const auto eos_id = tokenizer_->GetEosTokenId();

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == eos_id || next_token == 151643U ||
        next_token == 248044U || next_token == 248046U) {
      break;
    }

    output_tokens.push_back(next_token);
    if (on_token) {
      const auto piece = tokenizer_->DecodeToken(next_token);
      if (!on_token(next_token, piece)) {
        break;
      }
    }

    next_token = ForwardToken(next_token, static_cast<std::uint32_t>(cur_pos));
    ++cur_pos;
  }

  return output_tokens;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
