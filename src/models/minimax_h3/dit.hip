#if defined(ENGINE_ENABLE_HIP)

#include "src/models/minimax_h3/dit.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/dit_ops.cuh"

namespace strix::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::size_t kMaximumAttentionRows = 16'384;
// Pinned by byte-repeat validation on gfx1151 with the repository's ROCm
// 7.2.3 toolchain. A changed rocBLAS database fails session construction.
constexpr rocblas_int kGfx1151RocblasBf16Solution = -1162085109;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool ValidateTarget(int* device, std::string* error) {
  hipError_t status = hipGetDevice(device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDevice", status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, *device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDeviceProperties", status));
    return false;
  }
  if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 DiT supports only integrated gfx1151 Strix Halo; "
             "detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  if (properties.sharedMemPerBlock <
      kMaximumAttentionRows * sizeof(float)) {
    SetError(error, "gfx1151 reports insufficient shared memory for H3 SDPA");
    return false;
  }
  return true;
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      SetError(error, HipError("hipMalloc", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
    bytes_ = 0;
  }

  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

bool HasShape(const TensorDescriptor& descriptor,
              std::initializer_list<std::uint64_t> shape) {
  return descriptor.dtype == DType::kBFloat16 &&
         descriptor.shape ==
             std::vector<std::uint64_t>(shape.begin(), shape.end());
}

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error, "MiniMax H3 DiT tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (descriptor->phase != Phase::kDitCore ||
      !HasShape(*descriptor, shape)) {
    SetError(error, "MiniMax H3 DiT tensor contract differs: " +
                        std::string(name));
    return nullptr;
  }
  return descriptor;
}

struct LoadedWeight {
  const TensorDescriptor* descriptor{nullptr};
  DeviceBuffer storage;
};

bool LoadWeight(const ModelInventory& inventory,
                const TensorDescriptor& descriptor, int device,
                hipStream_t stream, void* staging, std::size_t staging_bytes,
                const CancellationToken* cancellation, LoadedWeight* output,
                std::string* error) {
  if (descriptor.payload_bytes > std::numeric_limits<std::size_t>::max()) {
    SetError(error, "MiniMax H3 DiT weight exceeds addressable memory");
    return false;
  }
  const auto bytes = static_cast<std::size_t>(descriptor.payload_bytes);
  if (!output->storage.Allocate(bytes, error)) {
    return false;
  }
  const std::filesystem::path path =
      inventory.model_root() / descriptor.shard;
  const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0) {
    SetError(error, "cannot open MiniMax H3 DiT shard " + path.string());
    return false;
  }
  std::size_t copied = 0;
  bool success = true;
  while (copied < bytes) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 DiT weight load cancelled");
      success = false;
      break;
    }
    const std::size_t chunk = std::min(staging_bytes, bytes - copied);
    std::size_t read_bytes = 0;
    while (read_bytes < chunk) {
      const ssize_t result =
          pread(file, static_cast<std::byte*>(staging) + read_bytes,
                chunk - read_bytes,
                static_cast<off_t>(descriptor.file_offset + copied +
                                   read_bytes));
      if (result <= 0) {
        SetError(error, "short read loading MiniMax H3 DiT tensor " +
                            descriptor.name);
        success = false;
        break;
      }
      read_bytes += static_cast<std::size_t>(result);
    }
    if (!success) {
      break;
    }
    hipError_t status = hipSetDevice(device);
    if (status == hipSuccess) {
      status = hipMemcpyAsync(
          static_cast<std::byte*>(output->storage.data()) + copied, staging,
          chunk, hipMemcpyHostToDevice, stream);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("copy MiniMax H3 DiT weight", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  (void)close(file);
  if (!success) {
    output->storage.Reset();
    return false;
  }
  output->descriptor = &descriptor;
  return true;
}

class Bf16Gemm {
private:
  struct Plan {
    std::size_t batch;
    std::size_t rows;
    std::size_t columns;
    rocblas_int solution_index;
    bool selected;
  };

public:
  Bf16Gemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    plans_.reserve(4);
  }
  ~Bf16Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  [[nodiscard]] std::size_t workspace_bytes() const noexcept { return 0; }

  bool Prepare(std::size_t batch, std::size_t rows, std::size_t columns,
               std::string* error) {
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        batch > static_cast<std::size_t>(
                    std::numeric_limits<rocblas_int>::max()) ||
        rows >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        columns > static_cast<std::size_t>(
                      std::numeric_limits<rocblas_int>::max())) {
      SetError(error, "invalid rocBLAS BF16 DiT matmul dimensions");
      return false;
    }
    if (FindPlan(batch, rows, columns) == nullptr) {
      plans_.push_back(
          {batch, rows, columns, kGfx1151RocblasBf16Solution, false});
    }
    return true;
  }

  bool SelectDeterministic(
      const std::uint16_t* weight, const std::uint16_t* input,
      std::uint16_t* output, std::uint16_t* reference,
      unsigned long long* first_mismatch, std::size_t batch, std::size_t rows,
      std::size_t columns, hipStream_t stream, std::string* error) {
    if (!Prepare(batch, rows, columns, error) ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "cannot prepare deterministic rocBLAS DiT solution");
      return false;
    }
    Plan* plan = FindPlan(batch, rows, columns);
    if (plan == nullptr) {
      SetError(error, "rocBLAS BF16 DiT plan lookup failed");
      return false;
    }
    const std::size_t output_elements = batch * rows;
    if (Execute(weight, input, output, batch, rows, columns,
                plan->solution_index) != rocblas_status_success) {
      SetError(error,
               "pinned gfx1151 rocBLAS BF16 DiT solution is unavailable");
      return false;
    }
    hipError_t hip_status =
        hipMemcpyAsync(reference, output,
                       output_elements * sizeof(std::uint16_t),
                       hipMemcpyDeviceToDevice, stream);
    if (hip_status == hipSuccess &&
        Execute(weight, input, output, batch, rows, columns,
                plan->solution_index) != rocblas_status_success) {
      hip_status = hipErrorUnknown;
    }
    if (hip_status == hipSuccess) {
      hip_status = hipMemsetAsync(first_mismatch, 0xFF,
                                  sizeof(unsigned long long), stream);
    }
    if (hip_status == hipSuccess) {
      dit_ops::LaunchFirstMismatch(reference, output, output_elements,
                                   first_mismatch, stream);
      hip_status = hipPeekAtLastError();
    }
    unsigned long long host_mismatch = 0;
    if (hip_status == hipSuccess) {
      hip_status =
          hipMemcpyAsync(&host_mismatch, first_mismatch, sizeof(host_mismatch),
                         hipMemcpyDeviceToHost, stream);
    }
    if (hip_status == hipSuccess) {
      hip_status = hipStreamSynchronize(stream);
    }
    if (hip_status != hipSuccess) {
      SetError(error, HipError("validate pinned rocBLAS DiT solution",
                               hip_status));
      return false;
    }
    if (host_mismatch != std::numeric_limits<unsigned long long>::max()) {
      SetError(error,
               "pinned gfx1151 rocBLAS BF16 DiT solution is not deterministic");
      return false;
    }
    plan->selected = true;
    return true;
  }

  bool Run(const std::uint16_t* weight, const std::uint16_t* input,
           std::uint16_t* output, std::size_t batch, std::size_t rows,
           std::size_t columns, hipStream_t stream, std::string* error) {
    Plan* plan = FindPlan(batch, rows, columns);
    if (plan == nullptr || !plan->selected) {
      SetError(error, "rocBLAS BF16 DiT solution was not pinned");
      return false;
    }
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "cannot set rocBLAS DiT stream");
      return false;
    }
    const rocblas_status status =
        Execute(weight, input, output, batch, rows, columns,
                plan->solution_index);
    if (status != rocblas_status_success) {
      SetError(error, "rocBLAS BF16 DiT matmul dispatch failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  Plan* FindPlan(std::size_t batch, std::size_t rows, std::size_t columns) {
    for (Plan& plan : plans_) {
      if (plan.batch == batch && plan.rows == rows &&
          plan.columns == columns) {
        return &plan;
      }
    }
    return nullptr;
  }

  rocblas_status Execute(const std::uint16_t* weight,
                         const std::uint16_t* input, std::uint16_t* output,
                         std::size_t batch, std::size_t rows,
                         std::size_t columns,
                         rocblas_int solution_index) const {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(columns), &alpha, weight,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(columns), input,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(columns), &beta,
        output, rocblas_datatype_bf16_r, static_cast<rocblas_int>(rows),
        output, rocblas_datatype_bf16_r, static_cast<rocblas_int>(rows),
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index,
        solution_index, 0);
  }

  rocblas_handle handle_{nullptr};
  std::vector<Plan> plans_;
};

bool CheckLaunch(std::string_view label, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(label, status));
  return false;
}

std::string FindFirstNonFinite(std::string_view name,
                               std::span<const std::uint16_t> values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::uint32_t bits = static_cast<std::uint32_t>(values[index]) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      return std::string(name) + "[" + std::to_string(index) + "]=" +
             std::to_string(value);
    }
  }
  return {};
}

}  // namespace

struct DitBlockSession::Impl {
  std::size_t block_index{0};
  std::size_t rows{0};
  std::size_t modulation_rows{0};
  int device{0};
  hipStream_t stream{nullptr};
  hipEvent_t start_event{nullptr};
  hipEvent_t stop_event{nullptr};
  void* staging{nullptr};
  Bf16Gemm gemm;

  LoadedWeight norm1;
  LoadedWeight norm2;
  LoadedWeight qkv_weight;
  LoadedWeight query_norm;
  LoadedWeight key_norm;
  LoadedWeight output_weight;
  LoadedWeight fc1_weight;
  LoadedWeight fc2_weight;

  DeviceBuffer hidden;
  DeviceBuffer modulation;
  DeviceBuffer row_map;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer modulation_attention;
  DeviceBuffer qkv;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer attention_heads;
  DeviceBuffer attention_output;
  DeviceBuffer gated_attention;
  DeviceBuffer modulation_mlp;
  DeviceBuffer fc1;
  DeviceBuffer activated;
  DeviceBuffer mlp_output;
  DeviceBuffer block_output;
  DeviceBuffer gemm_reference;
  DeviceBuffer gemm_first_mismatch;

  std::uint64_t weight_bytes{0};
  std::uint64_t activation_bytes{0};
  double load_ms{0.0};

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (start_event != nullptr) {
      (void)hipEventDestroy(start_event);
    }
    if (stop_event != nullptr) {
      (void)hipEventDestroy(stop_event);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }
};

DitBlockSession::DitBlockSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DitBlockSession::~DitBlockSession() = default;
DitBlockSession::DitBlockSession(DitBlockSession&&) noexcept = default;
DitBlockSession& DitBlockSession::operator=(DitBlockSession&&) noexcept =
    default;

std::unique_ptr<DitBlockSession> DitBlockSession::Create(
    const ModelInventory& inventory, const DitBlockOptions& options,
    const CancellationToken* cancellation, std::string* error) {
  if (options.block_index >= kDitBlocks || options.rows == 0 ||
      options.rows > kMaximumAttentionRows ||
      options.rows > std::numeric_limits<std::uint32_t>::max() ||
      options.modulation_rows == 0 ||
      options.modulation_rows > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "invalid MiniMax H3 DiT block options");
    return nullptr;
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT session creation cancelled");
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->block_index = options.block_index;
  impl->rows = options.rows;
  impl->modulation_rows = options.modulation_rows;
  if (!ValidateTarget(&impl->device, error)) {
    return nullptr;
  }
  hipError_t status =
      hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  if (status == hipSuccess) {
    status = hipEventCreate(&impl->start_event);
  }
  if (status == hipSuccess) {
    status = hipEventCreate(&impl->stop_event);
  }
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("create MiniMax H3 DiT resources", status));
    return nullptr;
  }

  const std::string prefix =
      "blocks." + std::to_string(options.block_index) + ".";
  const TensorDescriptor* norm1 =
      RequireTensor(inventory, prefix + "norm1.weight", {kDitHiddenSize},
                    error);
  const TensorDescriptor* norm2 =
      RequireTensor(inventory, prefix + "norm2.weight", {kDitHiddenSize},
                    error);
  const TensorDescriptor* qkv_weight = RequireTensor(
      inventory, prefix + "attn.qkv_proj.weight",
      {kDitAttentionWidth * 3U, kDitHiddenSize}, error);
  const TensorDescriptor* query_norm =
      RequireTensor(inventory, prefix + "attn.q_norm.weight",
                    {kDitHeadDimension}, error);
  const TensorDescriptor* key_norm =
      RequireTensor(inventory, prefix + "attn.k_norm.weight",
                    {kDitHeadDimension}, error);
  const TensorDescriptor* output_weight = RequireTensor(
      inventory, prefix + "attn.out_proj.weight",
      {kDitHiddenSize, kDitAttentionWidth}, error);
  const TensorDescriptor* fc1_weight = RequireTensor(
      inventory, prefix + "mlp.fc1.weight",
      {kDitFeedForwardSize * 2U, kDitHiddenSize}, error);
  const TensorDescriptor* fc2_weight = RequireTensor(
      inventory, prefix + "mlp.fc2.weight",
      {kDitHiddenSize, kDitFeedForwardSize}, error);
  if (norm1 == nullptr || norm2 == nullptr || qkv_weight == nullptr ||
      query_norm == nullptr || key_norm == nullptr ||
      output_weight == nullptr || fc1_weight == nullptr ||
      fc2_weight == nullptr) {
    return nullptr;
  }

  const auto load_begin = Clock::now();
  const std::array<std::pair<const TensorDescriptor*, LoadedWeight*>, 8>
      weights = {{
          {norm1, &impl->norm1},
          {norm2, &impl->norm2},
          {qkv_weight, &impl->qkv_weight},
          {query_norm, &impl->query_norm},
          {key_norm, &impl->key_norm},
          {output_weight, &impl->output_weight},
          {fc1_weight, &impl->fc1_weight},
          {fc2_weight, &impl->fc2_weight},
      }};
  for (const auto& [descriptor, weight] : weights) {
    if (!LoadWeight(inventory, *descriptor, impl->device, impl->stream,
                    impl->staging, kWeightStagingBytes, cancellation, weight,
                    error)) {
      return nullptr;
    }
    impl->weight_bytes += descriptor->payload_bytes;
  }
  impl->load_ms = Milliseconds(load_begin, Clock::now());

  const auto bf16_bytes = [](std::size_t elements) {
    return elements * sizeof(std::uint16_t);
  };
  const std::size_t hidden_elements = options.rows * kDitHiddenSize;
  const std::size_t attention_elements =
      options.rows * kDitAttentionWidth;
  const std::size_t qkv_elements = attention_elements * 3U;
  const std::size_t fc1_elements =
      options.rows * kDitFeedForwardSize * 2U;
  const std::size_t activated_elements =
      options.rows * kDitFeedForwardSize;
  const std::size_t modulation_elements =
      options.modulation_rows * kDitModulationSlots * kDitHiddenSize;
  const std::size_t rope_elements = options.rows * kDitRopeHalf;
  const auto allocate = [&](DeviceBuffer* buffer, std::size_t bytes) {
    if (!buffer->Allocate(bytes, error)) {
      return false;
    }
    impl->activation_bytes += bytes;
    return true;
  };
  if (!allocate(&impl->hidden, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->modulation, bf16_bytes(modulation_elements)) ||
      !allocate(&impl->row_map,
                options.rows * sizeof(std::uint32_t)) ||
      !allocate(&impl->rope_cos, bf16_bytes(rope_elements)) ||
      !allocate(&impl->rope_sin, bf16_bytes(rope_elements)) ||
      !allocate(&impl->modulation_attention, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->qkv, bf16_bytes(qkv_elements)) ||
      !allocate(&impl->query, bf16_bytes(attention_elements)) ||
      !allocate(&impl->key, bf16_bytes(attention_elements)) ||
      !allocate(&impl->value, bf16_bytes(attention_elements)) ||
      !allocate(&impl->attention_heads, bf16_bytes(attention_elements)) ||
      !allocate(&impl->attention_output, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->gated_attention, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->modulation_mlp, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->fc1, bf16_bytes(fc1_elements)) ||
      !allocate(&impl->activated, bf16_bytes(activated_elements)) ||
      !allocate(&impl->mlp_output, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->block_output, bf16_bytes(hidden_elements)) ||
      !allocate(&impl->gemm_reference, bf16_bytes(fc1_elements)) ||
      !allocate(&impl->gemm_first_mismatch,
                sizeof(unsigned long long))) {
    return nullptr;
  }
  if (!impl->gemm.Prepare(options.rows, kDitAttentionWidth * 3U,
                          kDitHiddenSize, error) ||
      !impl->gemm.Prepare(options.rows, kDitHiddenSize, kDitAttentionWidth,
                          error) ||
      !impl->gemm.Prepare(options.rows, kDitFeedForwardSize * 2U,
                          kDitHiddenSize, error) ||
      !impl->gemm.Prepare(options.rows, kDitHiddenSize,
                          kDitFeedForwardSize, error)) {
    return nullptr;
  }
  // Select a concrete byte-deterministic rocBLAS solution for every fixed
  // orientation. This also performs all library initialization before Run.
  const auto fill = [&](DeviceBuffer* buffer, std::uint32_t seed) {
    dit_ops::LaunchFillBf16Pattern(
        buffer->As<std::uint16_t>(),
        buffer->bytes() / sizeof(std::uint16_t), seed, impl->stream);
    return hipPeekAtLastError() == hipSuccess;
  };
  if (!fill(&impl->hidden, 0x169001U) ||
      !fill(&impl->attention_heads, 0x169002U) ||
      !fill(&impl->modulation_mlp, 0x169003U) ||
      !fill(&impl->activated, 0x169004U) ||
      !impl->gemm.SelectDeterministic(
          impl->qkv_weight.storage.As<std::uint16_t>(),
          impl->hidden.As<std::uint16_t>(), impl->qkv.As<std::uint16_t>(),
          impl->gemm_reference.As<std::uint16_t>(),
          impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
          kDitAttentionWidth * 3U, kDitHiddenSize, impl->stream, error) ||
      !impl->gemm.SelectDeterministic(
          impl->output_weight.storage.As<std::uint16_t>(),
          impl->attention_heads.As<std::uint16_t>(),
          impl->attention_output.As<std::uint16_t>(),
          impl->gemm_reference.As<std::uint16_t>(),
          impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
          kDitHiddenSize, kDitAttentionWidth, impl->stream, error) ||
      !impl->gemm.SelectDeterministic(
          impl->fc1_weight.storage.As<std::uint16_t>(),
          impl->modulation_mlp.As<std::uint16_t>(),
          impl->fc1.As<std::uint16_t>(),
          impl->gemm_reference.As<std::uint16_t>(),
          impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
          kDitFeedForwardSize * 2U, kDitHiddenSize, impl->stream, error) ||
      !impl->gemm.SelectDeterministic(
          impl->fc2_weight.storage.As<std::uint16_t>(),
          impl->activated.As<std::uint16_t>(),
          impl->mlp_output.As<std::uint16_t>(),
          impl->gemm_reference.As<std::uint16_t>(),
          impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
          kDitHiddenSize, kDitFeedForwardSize, impl->stream, error)) {
    return nullptr;
  }
  status = hipStreamSynchronize(impl->stream);
  if (status != hipSuccess) {
    SetError(error, HipError("warm MiniMax H3 DiT GEMMs", status));
    return nullptr;
  }
  return std::unique_ptr<DitBlockSession>(
      new DitBlockSession(std::move(impl)));
}

bool DitBlockSession::Run(const DitBlockInput& input,
                          const CancellationToken* cancellation,
                          DitBlockRetained* retained,
                          DitBlockTelemetry* telemetry, std::string* error) {
  if (impl_ == nullptr || retained == nullptr || telemetry == nullptr) {
    SetError(error, "invalid MiniMax H3 DiT run arguments");
    return false;
  }
  const std::size_t hidden_elements = impl_->rows * kDitHiddenSize;
  const std::size_t modulation_elements =
      impl_->modulation_rows * kDitModulationSlots * kDitHiddenSize;
  const std::size_t rope_elements = impl_->rows * kDitRopeHalf;
  if (input.hidden.size() != hidden_elements ||
      input.modulation.size() != modulation_elements ||
      input.row_map.size() != impl_->rows ||
      input.rope_cos.size() != rope_elements ||
      input.rope_sin.size() != rope_elements) {
    SetError(error, "MiniMax H3 DiT input tensor size differs from session");
    return false;
  }
  for (const std::uint32_t row : input.row_map) {
    if (row >= impl_->modulation_rows) {
      SetError(error, "MiniMax H3 DiT row map exceeds modulation rows");
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT block cancelled before submission");
    return false;
  }

  retained->rows = impl_->rows;
  retained->width = kDitHiddenSize;
  retained->modulation_attention.resize(hidden_elements);
  retained->attention_output.resize(hidden_elements);
  retained->modulation_mlp.resize(hidden_elements);
  retained->block_output.resize(hidden_elements);
  *telemetry = {};
  telemetry->weight_bytes = impl_->weight_bytes;
  telemetry->activation_bytes = impl_->activation_bytes;
  telemetry->gemm_workspace_bytes = impl_->gemm.workspace_bytes();
  telemetry->load_ms = impl_->load_ms;
  telemetry->scratch_address =
      reinterpret_cast<std::uintptr_t>(impl_->qkv.data());

  const auto copy = [&](void* target, const void* source, std::size_t bytes,
                        hipMemcpyKind kind, std::string_view label) {
    const hipError_t status =
        hipMemcpyAsync(target, source, bytes, kind, impl_->stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  };
  if (!copy(impl_->hidden.data(), input.hidden.data(), input.hidden.size_bytes(),
            hipMemcpyHostToDevice, "copy H3 DiT hidden") ||
      !copy(impl_->modulation.data(), input.modulation.data(),
            input.modulation.size_bytes(), hipMemcpyHostToDevice,
            "copy H3 DiT modulation") ||
      !copy(impl_->row_map.data(), input.row_map.data(),
            input.row_map.size_bytes(), hipMemcpyHostToDevice,
            "copy H3 DiT row map") ||
      !copy(impl_->rope_cos.data(), input.rope_cos.data(),
            input.rope_cos.size_bytes(), hipMemcpyHostToDevice,
            "copy H3 DiT rope cosine") ||
      !copy(impl_->rope_sin.data(), input.rope_sin.data(),
            input.rope_sin.size_bytes(), hipMemcpyHostToDevice,
            "copy H3 DiT rope sine")) {
    return false;
  }

  const auto rows = static_cast<std::uint32_t>(impl_->rows);
  if (hipEventRecord(impl_->start_event, impl_->stream) != hipSuccess) {
    SetError(error, "cannot record MiniMax H3 DiT start event");
    return false;
  }
  dit_ops::LaunchAdaLn(
      impl_->hidden.As<std::uint16_t>(), impl_->norm1.storage.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(),
      impl_->row_map.As<std::uint32_t>(),
      impl_->modulation_attention.As<std::uint16_t>(), rows, kDitHiddenSize,
      kDitModulationSlots, 0, 1, kDitNormEpsilon, impl_->stream);
  if (!CheckLaunch("H3 DiT attention AdaLN", error) ||
      !impl_->gemm.Run(impl_->qkv_weight.storage.As<std::uint16_t>(),
                       impl_->modulation_attention.As<std::uint16_t>(),
                       impl_->qkv.As<std::uint16_t>(), impl_->rows,
                       kDitAttentionWidth * 3U, kDitHiddenSize, impl_->stream,
                       error)) {
    return false;
  }
  dit_ops::LaunchGroupedQkvNormRope(
      impl_->qkv.As<std::uint16_t>(),
      impl_->query_norm.storage.As<std::uint16_t>(),
      impl_->key_norm.storage.As<std::uint16_t>(),
      impl_->rope_cos.As<std::uint16_t>(), impl_->rope_sin.As<std::uint16_t>(),
      impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
      impl_->value.As<std::uint16_t>(), rows, kDitAttentionHeads,
      kDitHeadDimension, kDitRopeHalf, kDitNormEpsilon, impl_->stream);
  if (!CheckLaunch("H3 DiT grouped QKV norm/RoPE", error)) {
    return false;
  }
  dit_ops::LaunchFullAttention(
      impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
      impl_->value.As<std::uint16_t>(),
      impl_->attention_heads.As<std::uint16_t>(), rows, kDitAttentionHeads,
      kDitHeadDimension,
      1.0F / std::sqrt(static_cast<float>(kDitHeadDimension)), impl_->stream);
  if (!CheckLaunch("H3 DiT full SDPA", error) ||
      !impl_->gemm.Run(impl_->output_weight.storage.As<std::uint16_t>(),
                       impl_->attention_heads.As<std::uint16_t>(),
                       impl_->attention_output.As<std::uint16_t>(), impl_->rows,
                       kDitHiddenSize, kDitAttentionWidth, impl_->stream,
                       error)) {
    return false;
  }
  dit_ops::LaunchGate(
      impl_->hidden.As<std::uint16_t>(),
      impl_->attention_output.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(),
      impl_->row_map.As<std::uint32_t>(),
      impl_->gated_attention.As<std::uint16_t>(), rows, kDitHiddenSize,
      kDitModulationSlots, 2, impl_->stream);
  if (!CheckLaunch("H3 DiT attention gated residual", error)) {
    return false;
  }
  dit_ops::LaunchAdaLn(
      impl_->gated_attention.As<std::uint16_t>(),
      impl_->norm2.storage.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(),
      impl_->row_map.As<std::uint32_t>(),
      impl_->modulation_mlp.As<std::uint16_t>(), rows, kDitHiddenSize,
      kDitModulationSlots, 3, 4, kDitNormEpsilon, impl_->stream);
  if (!CheckLaunch("H3 DiT MLP AdaLN", error) ||
      !impl_->gemm.Run(impl_->fc1_weight.storage.As<std::uint16_t>(),
                       impl_->modulation_mlp.As<std::uint16_t>(),
                       impl_->fc1.As<std::uint16_t>(), impl_->rows,
                       kDitFeedForwardSize * 2U, kDitHiddenSize, impl_->stream,
                       error)) {
    return false;
  }
  dit_ops::LaunchSwiGlu(impl_->fc1.As<std::uint16_t>(),
                        impl_->activated.As<std::uint16_t>(), rows,
                        kDitFeedForwardSize, impl_->stream);
  if (!CheckLaunch("H3 DiT SwiGLU", error) ||
      !impl_->gemm.Run(impl_->fc2_weight.storage.As<std::uint16_t>(),
                       impl_->activated.As<std::uint16_t>(),
                       impl_->mlp_output.As<std::uint16_t>(), impl_->rows,
                       kDitHiddenSize, kDitFeedForwardSize, impl_->stream,
                       error)) {
    return false;
  }
  dit_ops::LaunchGate(
      impl_->gated_attention.As<std::uint16_t>(),
      impl_->mlp_output.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(),
      impl_->row_map.As<std::uint32_t>(),
      impl_->block_output.As<std::uint16_t>(), rows, kDitHiddenSize,
      kDitModulationSlots, 5, impl_->stream);
  if (!CheckLaunch("H3 DiT MLP gated residual", error) ||
      hipEventRecord(impl_->stop_event, impl_->stream) != hipSuccess) {
    if (error != nullptr && error->empty()) {
      *error = "cannot record MiniMax H3 DiT stop event";
    }
    return false;
  }

  if (!copy(retained->modulation_attention.data(),
            impl_->modulation_attention.data(),
            retained->modulation_attention.size() * sizeof(std::uint16_t),
            hipMemcpyDeviceToHost, "read H3 DiT attention modulation") ||
      !copy(retained->attention_output.data(), impl_->attention_output.data(),
            retained->attention_output.size() * sizeof(std::uint16_t),
            hipMemcpyDeviceToHost, "read H3 DiT attention output") ||
      !copy(retained->modulation_mlp.data(), impl_->modulation_mlp.data(),
            retained->modulation_mlp.size() * sizeof(std::uint16_t),
            hipMemcpyDeviceToHost, "read H3 DiT MLP modulation") ||
      !copy(retained->block_output.data(), impl_->block_output.data(),
            retained->block_output.size() * sizeof(std::uint16_t),
            hipMemcpyDeviceToHost, "read H3 DiT block output")) {
    return false;
  }
  const hipError_t synchronize = hipStreamSynchronize(impl_->stream);
  if (synchronize != hipSuccess) {
    SetError(error, HipError("synchronize MiniMax H3 DiT block", synchronize));
    return false;
  }
  float elapsed = 0.0F;
  if (hipEventElapsedTime(&elapsed, impl_->start_event, impl_->stop_event) !=
      hipSuccess) {
    SetError(error, "cannot read MiniMax H3 DiT elapsed time");
    return false;
  }
  telemetry->gpu_ms = elapsed;
  telemetry->dispatches = 11;

  const std::array<std::pair<std::string_view,
                             const std::vector<std::uint16_t>*>,
                   4>
      retained_tensors = {{
          {"modulation_attention", &retained->modulation_attention},
          {"attention_output", &retained->attention_output},
          {"modulation_mlp", &retained->modulation_mlp},
          {"block_output", &retained->block_output},
      }};
  for (const auto& [name, values] : retained_tensors) {
    telemetry->first_non_finite = FindFirstNonFinite(name, *values);
    if (!telemetry->first_non_finite.empty()) {
      SetError(error, "MiniMax H3 DiT produced non-finite value at " +
                          telemetry->first_non_finite);
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT block cancelled after completion");
    return false;
  }
  return true;
}

std::size_t DitBlockSession::rows() const noexcept {
  return impl_ == nullptr ? 0 : impl_->rows;
}

std::size_t DitBlockSession::block_index() const noexcept {
  return impl_ == nullptr ? 0 : impl_->block_index;
}

std::uintptr_t DitBlockSession::scratch_address() const noexcept {
  return impl_ == nullptr
             ? 0
             : reinterpret_cast<std::uintptr_t>(impl_->qkv.data());
}

}  // namespace strix::minimax_h3

#endif
