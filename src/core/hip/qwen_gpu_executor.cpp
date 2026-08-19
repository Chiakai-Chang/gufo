#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/qwen_gpu_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen_oracles.hpp"

namespace strix::hip {
namespace {

enum class WeightMappingMode {
  kAuto,
  kMapped,
  kCopy,
};

constexpr std::size_t kTiledAttentionBatch = 1024;

[[nodiscard]] WeightMappingMode GetWeightMappingMode() noexcept {
  const char* value = std::getenv("STRIX_GPU_WEIGHT_MODE");
  if (value == nullptr) {
    return WeightMappingMode::kAuto;
  }
  const std::string_view mode{value};
  if (mode == "mapped") {
    return WeightMappingMode::kMapped;
  }
  if (mode == "copy") {
    return WeightMappingMode::kCopy;
  }
  return WeightMappingMode::kAuto;
}

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_memory && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    }
    if (region.host_registered && region.host_data != nullptr) {
      (void)hipHostUnregister(const_cast<void*>(region.host_data));
    }
    region.device_data = nullptr;
    region.owns_device_memory = false;
    region.host_registered = false;
  }
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source,
    QwenGpuWeightRegion& destination) noexcept {
  const auto register_error =
      hipHostRegister(const_cast<void*>(source.data), source.size,
                      hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess) {
    return register_error;
  }

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, const_cast<void*>(source.data), 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(const_cast<void*>(source.data));
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .size = source.size,
                 .owns_device_memory = false,
                 .host_registered = true};
  return hipSuccess;
}

[[nodiscard]] hipError_t CopyRegionToDevice(
    const core::GgufMappedRegion& source,
    QwenGpuWeightRegion& destination) noexcept {
  void* device_data = nullptr;
  const auto malloc_error = hipMalloc(&device_data, source.size);
  if (malloc_error != hipSuccess) {
    return malloc_error;
  }
  const auto copy_error =
      hipMemcpy(device_data, source.data, source.size, hipMemcpyHostToDevice);
  if (copy_error != hipSuccess) {
    (void)hipFree(device_data);
    return copy_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .size = source.size,
                 .owns_device_memory = true,
                 .host_registered = false};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  hipDeviceProp_t properties{};
  const bool integrated =
      hipGetDeviceProperties(&properties, 0) == hipSuccess &&
      properties.integrated != 0;
  const auto mode = GetWeightMappingMode();
  const bool prefer_mapped = mode == WeightMappingMode::kMapped ||
                             (mode == WeightMappingMode::kAuto && integrated);

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t first_error = hipSuccess;
    hipError_t second_error = hipSuccess;

    if (prefer_mapped) {
      first_error = MapRegisteredRegion(source, destination);
      if (first_error != hipSuccess && mode == WeightMappingMode::kAuto) {
        second_error = CopyRegionToDevice(source, destination);
      }
    } else {
      first_error = CopyRegionToDevice(source, destination);
      if (first_error != hipSuccess && mode == WeightMappingMode::kAuto) {
        second_error = MapRegisteredRegion(source, destination);
      }
    }

    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        const auto final_error =
            second_error != hipSuccess ? second_error : first_error;
        *error_msg =
            "Failed to make GGUF shard " + std::to_string(i) +
            " GPU-visible in " +
            (prefer_mapped ? std::string("mapped") : std::string("copy")) +
            " mode: " + hipGetErrorString(final_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address &&
        tensor_address < region_address + region.size) {
      const auto offset = tensor_address - region_address;
      tensor.data =
          static_cast<const std::uint8_t*>(region.device_data) + offset;
      return true;
    }
  }
  return false;
}

}  // namespace

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context)
    : config_(config),
      max_context_(std::min(max_context, 4096U)),
      max_batch_(max_context_) {
  HIP_CHECK(hipStreamCreate(&stream));
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
  hipblaslt_gemm = std::make_unique<HipblasLtGemm>();

  const std::size_t hidden_size = config_.hidden_size;
  const std::size_t intermediate_size = config_.intermediate_size;
  const std::size_t vocab_size = config_.vocab_size;
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t batch = max_batch_;
  const std::size_t attention_size = config_.AttentionSize();
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config_.SsmQkvSize();
  const std::size_t ssm_inner_size = config_.ssm_inner_size;
  const std::size_t recurrent_width = std::max(attention_size, ssm_inner_size);
  const std::size_t projection_width =
      std::max(q_projection_size, ssm_qkv_size);
  const std::size_t time_step_rank = config_.ssm_time_step_rank;

  HIP_CHECK(hipMalloc(&d_hidden, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, batch * attention_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_attn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_act, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_qkv, batch * projection_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out, batch * ssm_qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_gate, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_out, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_prompt_tokens, batch * sizeof(std::uint32_t)));

  const std::size_t scratch_elements =
      batch * std::max<std::size_t>(
                  {intermediate_size, hidden_size, projection_width,
                   ssm_qkv_size + ssm_inner_size + (2 * time_step_rank)});
  HIP_CHECK(
      hipMalloc(&d_scratch_bf16, scratch_elements * sizeof(hip_bfloat16)));

  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim;
  HIP_CHECK(hipMalloc(&d_kv_cache, total_kv * sizeof(float) * 2));
  HIP_CHECK(
      hipMalloc(&d_attention_kv_f16, total_kv * sizeof(std::uint16_t) * 2));

  const std::size_t total_conv =
      num_layers * ssm_qkv_size * config_.ssm_conv_kernel;
  HIP_CHECK(hipMalloc(&d_ssm_conv_state, total_conv * sizeof(float)));

  const std::size_t total_deltanet = num_layers * time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();
  HIP_CHECK(hipMalloc(&d_ssm_deltanet_state, total_deltanet * sizeof(float)));

  Reset();
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_logits = other.d_logits;
  d_attention_kv_f16 = other.d_attention_kv_f16;
  d_kv_cache = other.d_kv_cache;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  stream = other.stream;
  hipblas_handle = other.hipblas_handle;
  hipblaslt_gemm = std::move(other.hipblaslt_gemm);
  d_scratch_bf16 = other.d_scratch_bf16;

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_logits = nullptr;
  other.d_attention_kv_f16 = nullptr;
  other.d_kv_cache = nullptr;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.stream = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_logits = other.d_logits;
    d_attention_kv_f16 = other.d_attention_kv_f16;
    d_kv_cache = other.d_kv_cache;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    stream = other.stream;
    hipblas_handle = other.hipblas_handle;
    hipblaslt_gemm = std::move(other.hipblaslt_gemm);
    d_scratch_bf16 = other.d_scratch_bf16;

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_logits = nullptr;
    other.d_attention_kv_f16 = nullptr;
    other.d_kv_cache = nullptr;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.stream = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
  }
  return *this;
}

void QwenGpuArena::Reset() noexcept {
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim * 2;
  const std::size_t total_conv =
      num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet = num_layers * config_.ssm_time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();

  if (d_kv_cache != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_kv_cache, 0, total_kv * sizeof(float), stream));
  }
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_deltanet_state, 0,
                             total_deltanet * sizeof(float), stream));
  }
}

void QwenGpuArena::FreeAll() noexcept {
  if (d_hidden != nullptr)
    HIP_CHECK(hipFree(d_hidden));
  if (d_normed != nullptr)
    HIP_CHECK(hipFree(d_normed));
  if (d_q != nullptr)
    HIP_CHECK(hipFree(d_q));
  if (d_k != nullptr)
    HIP_CHECK(hipFree(d_k));
  if (d_v != nullptr)
    HIP_CHECK(hipFree(d_v));
  if (d_attn_out != nullptr)
    HIP_CHECK(hipFree(d_attn_out));
  if (d_ffn_gate != nullptr)
    HIP_CHECK(hipFree(d_ffn_gate));
  if (d_ffn_up != nullptr)
    HIP_CHECK(hipFree(d_ffn_up));
  if (d_ffn_act != nullptr)
    HIP_CHECK(hipFree(d_ffn_act));
  if (d_ffn_out != nullptr)
    HIP_CHECK(hipFree(d_ffn_out));
  if (d_ssm_qkv != nullptr)
    HIP_CHECK(hipFree(d_ssm_qkv));
  if (d_conv_out != nullptr)
    HIP_CHECK(hipFree(d_conv_out));
  if (d_ssm_gate != nullptr)
    HIP_CHECK(hipFree(d_ssm_gate));
  if (d_ssm_out != nullptr)
    HIP_CHECK(hipFree(d_ssm_out));
  if (d_alpha_buf != nullptr)
    HIP_CHECK(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    HIP_CHECK(hipFree(d_beta_buf));
  if (d_logits != nullptr)
    HIP_CHECK(hipFree(d_logits));
  if (d_attention_kv_f16 != nullptr)
    HIP_CHECK(hipFree(d_attention_kv_f16));
  if (d_kv_cache != nullptr)
    HIP_CHECK(hipFree(d_kv_cache));
  if (d_ssm_conv_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_conv_state));
  if (d_ssm_deltanet_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_deltanet_state));
  if (d_prompt_tokens != nullptr)
    HIP_CHECK(hipFree(d_prompt_tokens));
  if (d_scratch_bf16 != nullptr)
    HIP_CHECK(hipFree(d_scratch_bf16));
  if (hipblas_handle != nullptr)
    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  hipblaslt_gemm.reset();
  if (stream != nullptr)
    HIP_CHECK(hipStreamDestroy(stream));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_logits = nullptr;
  d_attention_kv_f16 = nullptr;
  d_kv_cache = nullptr;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  hipblas_handle = nullptr;
  stream = nullptr;
}

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

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto weights_opt = models::QwenModelWeights::LoadFromGguf(reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(reader, error_msg);
  if (!tokenizer || tokenizer->GetVocabSize() <= 256) {
    auto bin_tok = tokenization::QwenTokenizer::CreateFromBinaryFile(
        "models/qwen_vocab.bin");
    if (bin_tok) {
      tokenizer = std::move(bin_tok);
    } else if (!tokenizer) {
      return nullptr;
    }
  }

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  const std::uint32_t context_len =
      std::min(weights_opt->config.context_length > 0
                   ? weights_opt->config.context_length
                   : 4096U,
               4096U);
  return std::make_unique<QwenGpuExecutor>(
      std::move(*weights_opt), std::move(tokenizer), std::move(weight_regions),
      context_len);
}

tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;

  // 1. Embedding lookup
  const bool embd_is_bf16 = weights_.token_embd.type == core::GgmlType::kBF16;
  LaunchEmbeddingLookup(weights_.token_embd.data, embd_is_bf16, token_id,
                        arena_.d_hidden, hidden_size, arena_.stream);

  // 2. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];

    // Pre-RMSNorm
    LaunchRMSNorm(arena_.d_hidden,
                  static_cast<const float*>(layer.attn_norm.data),
                  arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

    if (layer.is_full_attention) {
      // Full attention path
      const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
      const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
      const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
      const bool o_bf16 = layer.attn_output.type == core::GgmlType::kBF16;

      LaunchFusedQKVProjections(layer.attn_q.data, q_bf16, layer.attn_k.data,
                                k_bf16, layer.attn_v.data, v_bf16,
                                arena_.d_normed, arena_.d_ssm_qkv, arena_.d_k,
                                arena_.d_v, q_projection_size, kv_size,
                                hidden_size, arena_.stream);

      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                     config.num_attention_heads, config.head_dim,
                     arena_.stream);

      // QK-Norm
      if (!layer.attn_q_norm.empty()) {
        LaunchPerHeadRMSNorm(arena_.d_q,
                             static_cast<const float*>(layer.attn_q_norm.data),
                             arena_.d_q, config.num_attention_heads,
                             config.head_dim, 1e-6F, arena_.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchPerHeadRMSNorm(arena_.d_k,
                             static_cast<const float*>(layer.attn_k_norm.data),
                             arena_.d_k, config.num_key_value_heads,
                             config.head_dim, 1e-6F, arena_.stream);
      }

      // RoPE
      LaunchRoPE(arena_.d_q, arena_.d_k, config.num_attention_heads,
                 config.num_key_value_heads, config.head_dim, config.rotary_dim,
                 pos, config.rope_theta, arena_.stream);

      // Softmax Attention + Gating
      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena_.GetMaxContext() * config.head_dim;
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
      LaunchAttention(arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                      arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                      arena_.d_ssm_out, attn_layer_idx, pos,
                      arena_.GetMaxContext(), config.num_attention_heads,
                      config.num_key_value_heads, config.head_dim,
                      arena_.stream);

      // Output projection
      LaunchGEMV(layer.attn_output.data, o_bf16, arena_.d_ssm_out,
                 arena_.d_attn_out, hidden_size, attention_size, arena_.stream);
    } else {
      // SSM path
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;
      const bool out_bf16 = layer.ssm_out.type == core::GgmlType::kBF16;

      LaunchFusedSSMInputProjections(
          layer.attn_qkv.data, qkv_bf16, layer.attn_gate.data, gate_bf16,
          layer.ssm_alpha.data, alpha_bf16, layer.ssm_beta.data, beta_bf16,
          arena_.d_normed, arena_.d_ssm_qkv, arena_.d_ssm_gate,
          arena_.d_alpha_buf, arena_.d_beta_buf, hidden_size, ssm_qkv_size,
          ssm_inner_size, time_step_rank, arena_.stream);

      LaunchSSMConvRecurrence(
          arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, arena_.d_conv_out,
          arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
          arena_.d_ssm_out, l, ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena_.stream);

      LaunchGEMV(layer.ssm_out.data, out_bf16, arena_.d_ssm_out,
                 arena_.d_attn_out, hidden_size, ssm_inner_size, arena_.stream);
    }

    // Residual Add
    LaunchResidualAdd(arena_.d_hidden, arena_.d_attn_out, arena_.d_hidden,
                      hidden_size, arena_.stream);

    // FFN Pre-RMSNorm
    LaunchRMSNorm(arena_.d_hidden,
                  static_cast<const float*>(layer.ffn_norm.data),
                  arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

    // Fused SwiGLU FFN
    const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
    const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;
    const bool ffn_d_bf16 = layer.ffn_down.type == core::GgmlType::kBF16;

    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, ffn_g_bf16, layer.ffn_up.data,
                          ffn_u_bf16, arena_.d_normed, arena_.d_ffn_act,
                          intermediate_size, hidden_size, arena_.stream);

    LaunchGEMV(layer.ffn_down.data, ffn_d_bf16, arena_.d_ffn_act,
               arena_.d_ffn_out, hidden_size, intermediate_size, arena_.stream);

    // Residual Add
    LaunchResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                      hidden_size, arena_.stream);
  }

  if (!compute_logits) {
    return 0;
  }

  // 3. Final Output Norm
  LaunchRMSNorm(arena_.d_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden_size, 1e-6F, arena_.stream);

  // 4. LM Head Logits GEMV on final token
  const bool out_bf16 = weights_.output.type == core::GgmlType::kBF16;
  LaunchGEMV(weights_.output.data, out_bf16, arena_.d_normed, arena_.d_logits,
             vocab_size, hidden_size, arena_.stream);

  // 5. Parallel GPU Argmax and 4-byte host transfer
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);
  LaunchGPUArgmax(arena_.d_logits, d_out_token, vocab_size, arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }
  if (batch_size > arena_.GetMaxBatch()) {
    tokenization::TokenId next_token = 0;
    for (std::size_t p = 0; p < prompt_tokens.size(); ++p) {
      const bool is_last = (p + 1 == prompt_tokens.size());
      next_token = ForwardToken(prompt_tokens[p], static_cast<std::uint32_t>(p),
                                is_last);
    }
    return next_token;
  }

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  const bool embd_is_bf16 = weights_.token_embd.type == core::GgmlType::kBF16;
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data, embd_is_bf16,
                               arena_.d_prompt_tokens, arena_.d_hidden,
                               batch_size, hidden_size, arena_.stream);

  const bool do_profile = (std::getenv("STRIX_PROFILE") != nullptr);
  auto t_start = std::chrono::high_resolution_clock::now();
  double time_attn_proj = 0, time_ssm_recur = 0, time_ssm_out = 0;
  double time_ffn = 0, time_norm = 0;

  constexpr std::size_t hipblaslt_min_dimension = 1024;
  const auto launch_bf16_gemm = [&](const void* weights, const void* input,
                                    float* output, std::size_t m,
                                    std::size_t k) {
    const bool use_hipblaslt =
        m >= hipblaslt_min_dimension && k >= hipblaslt_min_dimension &&
        arena_.hipblaslt_gemm->RunBf16(weights, input, output, batch_size, m, k,
                                       arena_.stream);
    if (!use_hipblaslt) {
      LaunchHipblasGEMMBF16(arena_.hipblas_handle, weights, input, output,
                            batch_size, m, k, arena_.stream);
    }
  };

  // 3. Layer stack across all 32 layers
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_norm += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }

    if (layer.is_full_attention) {
      const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
      const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
      const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
      const bool o_bf16 = layer.attn_output.type == core::GgmlType::kBF16;

      if (q_bf16) {
        launch_bf16_gemm(layer.attn_q.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_qkv, q_projection_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_q.data, false,
                          arena_.d_normed, arena_.d_ssm_qkv, batch_size,
                          q_projection_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (k_bf16) {
        launch_bf16_gemm(layer.attn_k.data, arena_.d_scratch_bf16, arena_.d_k,
                         kv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_k.data, false,
                          arena_.d_normed, arena_.d_k, batch_size, kv_size,
                          hidden_size, arena_.d_scratch_bf16, arena_.stream);
      }

      if (v_bf16) {
        launch_bf16_gemm(layer.attn_v.data, arena_.d_scratch_bf16, arena_.d_v,
                         kv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_v.data, false,
                          arena_.d_normed, arena_.d_v, batch_size, kv_size,
                          hidden_size, arena_.d_scratch_bf16, arena_.stream);
      }

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      if (!layer.attn_q_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
            arena_.d_q, batch_size, config.num_attention_heads, config.head_dim,
            eps, arena_.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
            arena_.d_k, batch_size, config.num_key_value_heads, config.head_dim,
            eps, arena_.stream);
      }

      LaunchBatchedRoPE(arena_.d_q, arena_.d_k, batch_size,
                        config.num_attention_heads, config.num_key_value_heads,
                        config.head_dim, config.rotary_dim, 0,
                        config.rope_theta, arena_.stream);

      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena_.GetMaxContext() * config.head_dim;
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
      if (batch_size >= kTiledAttentionBatch) {
        bool launched = LaunchBatchedAttentionTile(
            arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
            arena_.d_kv_cache, arena_.d_kv_cache + total_k,
            arena_.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
            arena_.d_ssm_out, attn_layer_idx, 0, batch_size,
            arena_.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena_.stream);
        if (!launched) {
          launched = LaunchBatchedAttentionCk(
              arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
              arena_.d_kv_cache, arena_.d_kv_cache + total_k,
              arena_.d_attention_kv_f16,
              static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
              arena_.d_scratch_bf16, arena_.d_ssm_out, attn_layer_idx, 0,
              batch_size, arena_.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, arena_.stream);
        }
        if (!launched) {
          LaunchBatchedAttention(
              arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
              arena_.d_kv_cache, arena_.d_kv_cache + total_k, arena_.d_ssm_out,
              attn_layer_idx, 0, batch_size, arena_.GetMaxContext(),
              config.num_attention_heads, config.num_key_value_heads,
              config.head_dim, arena_.stream);
        }
      } else {
        LaunchBatchedAttention(
            arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
            arena_.d_kv_cache, arena_.d_kv_cache + total_k, arena_.d_ssm_out,
            attn_layer_idx, 0, batch_size, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena_.stream);
      }

      if (o_bf16) {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        launch_bf16_gemm(layer.attn_output.data, arena_.d_scratch_bf16,
                         arena_.d_attn_out, hidden_size, attention_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_output.data, false,
                          arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                          hidden_size, attention_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    } else {
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;
      const bool out_bf16 = layer.ssm_out.type == core::GgmlType::kBF16;

      if (qkv_bf16) {
        launch_bf16_gemm(layer.attn_qkv.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_qkv, ssm_qkv_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_qkv.data, false,
                          arena_.d_normed, arena_.d_ssm_qkv, batch_size,
                          ssm_qkv_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (gate_bf16) {
        launch_bf16_gemm(layer.attn_gate.data, arena_.d_scratch_bf16,
                         arena_.d_ssm_gate, ssm_inner_size, hidden_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.attn_gate.data, false,
                          arena_.d_normed, arena_.d_ssm_gate, batch_size,
                          ssm_inner_size, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (alpha_bf16) {
        LaunchHipblasGEMMBF16(arena_.hipblas_handle, layer.ssm_alpha.data,
                              arena_.d_scratch_bf16, arena_.d_alpha_buf,
                              batch_size, time_step_rank, hidden_size,
                              arena_.stream);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_alpha.data, false,
                          arena_.d_normed, arena_.d_alpha_buf, batch_size,
                          time_step_rank, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }

      if (beta_bf16) {
        LaunchHipblasGEMMBF16(arena_.hipblas_handle, layer.ssm_beta.data,
                              arena_.d_scratch_bf16, arena_.d_beta_buf,
                              batch_size, time_step_rank, hidden_size,
                              arena_.stream);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_beta.data, false,
                          arena_.d_normed, arena_.d_beta_buf, batch_size,
                          time_step_rank, hidden_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      LaunchBatchedSSMConvRecurrence(
          arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, arena_.d_conv_out,
          arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
          arena_.d_ssm_out, l, batch_size, ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena_.stream);

      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_recur +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      if (out_bf16) {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        launch_bf16_gemm(layer.ssm_out.data, arena_.d_scratch_bf16,
                         arena_.d_attn_out, hidden_size, ssm_inner_size);
      } else {
        LaunchHipblasGEMM(arena_.hipblas_handle, layer.ssm_out.data, false,
                          arena_.d_ssm_out, arena_.d_attn_out, batch_size,
                          hidden_size, ssm_inner_size, arena_.d_scratch_bf16,
                          arena_.stream);
      }
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_out +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    }

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch_size, hidden_size,
                             arena_.stream);

    // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);

    const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
    const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;
    const bool ffn_d_bf16 = layer.ffn_down.type == core::GgmlType::kBF16;

    if (ffn_g_bf16) {
      launch_bf16_gemm(layer.ffn_gate.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_gate, intermediate_size, hidden_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_gate.data, false,
                        arena_.d_normed, arena_.d_ffn_gate, batch_size,
                        intermediate_size, hidden_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    if (ffn_u_bf16) {
      launch_bf16_gemm(layer.ffn_up.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_up, intermediate_size, hidden_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_up.data, false,
                        arena_.d_normed, arena_.d_ffn_up, batch_size,
                        intermediate_size, hidden_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    LaunchBatchedSwiGLUActivation(
        arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
        arena_.d_scratch_bf16, batch_size * intermediate_size, arena_.stream);

    if (ffn_d_bf16) {
      launch_bf16_gemm(layer.ffn_down.data, arena_.d_scratch_bf16,
                       arena_.d_ffn_out, hidden_size, intermediate_size);
    } else {
      LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_down.data, false,
                        arena_.d_ffn_act, arena_.d_ffn_out, batch_size,
                        hidden_size, intermediate_size, arena_.d_scratch_bf16,
                        arena_.stream);
    }

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                             batch_size, hidden_size, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_ffn += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }
  }

  if (do_profile) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\n[STRIX_PROFILE B=" << batch_size << "] Total: " << total_ms
              << " ms (" << (batch_size / (total_ms / 1000.0)) << " tok/s)\n"
              << "  - Norms:      " << time_norm << " ms\n"
              << "  - Input Proj: " << time_attn_proj << " ms\n"
              << "  - SSM Recur:  " << time_ssm_recur << " ms\n"
              << "  - SSM Out:    " << time_ssm_out << " ms\n"
              << "  - FFN (3 GEMM): " << time_ffn << " ms\n";
  }

  // 4. Output Norm for final token
  const float* final_hidden =
      arena_.d_hidden + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  const bool out_bf16 = weights_.output.type == core::GgmlType::kBF16;
  LaunchGEMV(weights_.output.data, out_bf16, arena_.d_normed, arena_.d_logits,
             vocab_size, hidden_size, arena_.stream);

  // 6. GPU Argmax
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);
  LaunchGPUArgmax(arena_.d_logits, d_out_token, vocab_size, arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
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
