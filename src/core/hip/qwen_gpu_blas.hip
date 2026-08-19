#if defined(ENGINE_ENABLE_HIP)
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstddef>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

namespace {

constexpr std::size_t kHipblasLtMaxAlgorithms = 16;
constexpr std::size_t kHipblasLtWideFfnThreshold = 3;
constexpr std::size_t kHipblasLtLargeBatch = 256;
constexpr std::size_t kHipblasLtLargeWideRank = 4;
constexpr std::size_t kHipblasLtTallRank = 5;

struct HipblasLtPlan {
  std::size_t batch_size{0};
  std::size_t m{0};
  std::size_t k{0};
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t a_layout{nullptr};
  hipblasLtMatrixLayout_t x_layout{nullptr};
  hipblasLtMatrixLayout_t y_layout{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};

  HipblasLtPlan() = default;
  HipblasLtPlan(const HipblasLtPlan&) = delete;
  HipblasLtPlan& operator=(const HipblasLtPlan&) = delete;

  ~HipblasLtPlan() {
    if (a_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    }
    if (x_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(x_layout);
    }
    if (y_layout != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(y_layout);
    }
    if (operation != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation);
    }
  }
};

[[nodiscard]] std::size_t SelectHipblasLtHeuristicRank(
    std::size_t batch_size, std::size_t m, std::size_t k,
    std::size_t result_count) noexcept {
  if (batch_size >= kHipblasLtLargeBatch) {
    if (m <= 1024) {
      return kHipblasLtLargeWideRank;
    }
    return k >= kHipblasLtWideFfnThreshold * m ? kHipblasLtTallRank
                                               : kHipblasLtLargeWideRank;
  }
  const std::size_t tall_m_threshold = batch_size >= 128 ? k : 2 * k;
  if (m > tall_m_threshold || k >= kHipblasLtWideFfnThreshold * m) {
    return kHipblasLtTallRank;
  }
  return result_count - 1;
}

[[nodiscard]] bool IsUsable(
    const hipblasLtMatmulHeuristicResult_t& result) noexcept {
  return result.state == HIPBLAS_STATUS_SUCCESS && result.workspaceSize == 0;
}

[[nodiscard]] std::unique_ptr<HipblasLtPlan> CreateHipblasLtPlan(
    hipblasLtHandle_t handle, std::size_t batch_size, std::size_t m,
    std::size_t k) {
  auto plan = std::make_unique<HipblasLtPlan>();
  plan->batch_size = batch_size;
  plan->m = m;
  plan->k = k;

  if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  if (hipblasLtMatmulDescSetAttribute(
          plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  if (hipblasLtMatrixLayoutCreate(&plan->a_layout, HIP_R_16BF, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->x_layout, HIP_R_16BF, k, batch_size,
                                  k) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan->y_layout, HIP_R_32F, m, batch_size,
                                  m) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }

  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  constexpr std::size_t max_workspace_bytes = 0;
  const auto preference_status = hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_bytes, sizeof(max_workspace_bytes));

  std::vector<hipblasLtMatmulHeuristicResult_t> results(
      kHipblasLtMaxAlgorithms);
  int result_count = 0;
  const auto heuristic_status =
      preference_status == HIPBLAS_STATUS_SUCCESS
          ? hipblasLtMatmulAlgoGetHeuristic(
                handle, plan->operation, plan->a_layout, plan->x_layout,
                plan->y_layout, plan->y_layout, preference,
                static_cast<int>(results.size()), results.data(), &result_count)
          : preference_status;
  (void)hipblasLtMatmulPreferenceDestroy(preference);

  if (heuristic_status != HIPBLAS_STATUS_SUCCESS || result_count <= 0) {
    return nullptr;
  }

  const auto result_span =
      std::span(results.data(), static_cast<std::size_t>(result_count));
  const auto preferred_rank =
      SelectHipblasLtHeuristicRank(batch_size, m, k, result_span.size());
  if (preferred_rank < result_span.size() &&
      IsUsable(result_span[preferred_rank])) {
    plan->algorithm = result_span[preferred_rank].algo;
    return plan;
  }

  const auto fallback = std::ranges::find_if(
      result_span, [](const auto& result) { return IsUsable(result); });
  if (fallback == result_span.end()) {
    return nullptr;
  }
  plan->algorithm = fallback->algo;
  return plan;
}

}  // namespace

struct HipblasLtGemm::Impl {
  struct PlanLookup {
    HipblasLtPlan* plan{nullptr};
    bool cache_hit{false};
  };

  hipblasLtHandle_t handle{nullptr};
  std::vector<std::unique_ptr<HipblasLtPlan>> plans;

  Impl() {
    if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
      handle = nullptr;
    }
  }

  ~Impl() {
    plans.clear();
    if (handle != nullptr) {
      (void)hipblasLtDestroy(handle);
    }
  }

  [[nodiscard]] PlanLookup GetPlan(std::size_t batch_size, std::size_t m,
                                   std::size_t k) {
    const auto existing = std::ranges::find_if(plans, [&](const auto& plan) {
      return plan->batch_size == batch_size && plan->m == m && plan->k == k;
    });
    if (existing != plans.end()) {
      return {.plan = existing->get(), .cache_hit = true};
    }
    auto plan = CreateHipblasLtPlan(handle, batch_size, m, k);
    if (plan == nullptr) {
      return {};
    }
    auto* result = plan.get();
    plans.push_back(std::move(plan));
    return {.plan = result, .cache_hit = false};
  }
};

HipblasLtGemm::HipblasLtGemm() : impl_(std::make_unique<Impl>()) {}

HipblasLtGemm::~HipblasLtGemm() = default;

HipblasLtGemm::HipblasLtGemm(HipblasLtGemm&&) noexcept = default;

HipblasLtGemm& HipblasLtGemm::operator=(HipblasLtGemm&&) noexcept = default;

bool HipblasLtGemm::RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                            std::size_t batch_size, std::size_t m,
                            std::size_t k, hipStream_t stream,
                            HipblasLtDispatchInfo* dispatch_info) {
  if (impl_ == nullptr || impl_->handle == nullptr || batch_size == 0 ||
      m == 0 || k == 0) {
    return false;
  }
  const auto lookup = impl_->GetPlan(batch_size, m, k);
  if (lookup.plan == nullptr) {
    return false;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const bool succeeded =
      hipblasLtMatmul(impl_->handle, lookup.plan->operation, &alpha, a_bf16,
                      lookup.plan->a_layout, x_bf16, lookup.plan->x_layout,
                      &beta, y, lookup.plan->y_layout, y, lookup.plan->y_layout,
                      &lookup.plan->algorithm, nullptr, 0,
                      stream) == HIPBLAS_STATUS_SUCCESS;
  if (!succeeded) {
    return false;
  }

  if (dispatch_info != nullptr || detail::DispatchTelemetryEnabled()) {
    const int algorithm_id =
        hipblaslt_ext::getIndexFromAlgo(lookup.plan->algorithm);
    const std::string solution_name = hipblaslt_ext::getSolutionNameFromAlgo(
        impl_->handle, lookup.plan->algorithm);
    const std::string kernel_name = hipblaslt_ext::getKernelNameFromAlgo(
        impl_->handle, lookup.plan->algorithm);
    if (dispatch_info != nullptr) {
      *dispatch_info = {
          .algorithm_id = algorithm_id,
          .solution_name = solution_name,
          .kernel_name = kernel_name,
          .plan_cache_hit = lookup.cache_hit,
      };
    }
    detail::EmitHipblasLtDispatch(batch_size, m, k, algorithm_id, solution_name,
                                  kernel_name, lookup.cache_hit);
  }
  return true;
}

void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  if (is_bf16) {
    LaunchFloatToBfloat16(X, d_x_bf16_buf, batch_size * K, stream);
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A,
        HIP_R_16BF, static_cast<int>(K), d_x_bf16_buf, HIP_R_16BF,
        static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  } else {
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A, HIP_R_32F,
        static_cast<int>(K), X, HIP_R_32F, static_cast<int>(K), &beta, Y,
        HIP_R_32F, static_cast<int>(M), HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT));
  }
}

void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  HIPBLAS_CHECK(hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(batch_size), static_cast<int>(K), &alpha, A_bf16,
      HIP_R_16BF, static_cast<int>(K), d_x_bf16, HIP_R_16BF,
      static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
