#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

#include <algorithm>
#include <array>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <sstream>

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;
constexpr int kCandidates = 16;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

}  // namespace

struct BlasLt::Problem {
  Operand a;
  Operand b;
  float* out;
  int ldc;
  long long stride_c;
  int m, n, k, batch;

  /// Token counts round up to a power of two: one plan per size class.
  [[nodiscard]] int ClassN() const {
    int c = 16;
    while (c < n) {
      c *= 2;
    }
    return c;
  }
  [[nodiscard]] std::string Key() const {
    std::ostringstream key;
    key << a.type << ':' << a.transpose << ':' << b.type << ':' << b.transpose
        << ':' << m << ':' << k << ':' << batch << ':' << ClassN();
    return key.str();
  }
};

BlasLt::~BlasLt() {
  if (workspace_ != nullptr) {
    (void)hipFree(workspace_);
  }
  if (handle_ != nullptr) {
    (void)hipblasLtDestroy(handle_);
  }
}

std::unique_ptr<BlasLt> BlasLt::Create(hipStream_t stream,
                                       std::uint32_t tuning_n,
                                       std::string* error_msg) {
  std::unique_ptr<BlasLt> b(new BlasLt());
  b->stream_ = stream;
  b->tuning_n_ = std::max<std::uint32_t>(1, tuning_n);
  if (hipblasLtCreate(&b->handle_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasLtCreate failed");
    return nullptr;
  }
  if (hipMalloc(&b->workspace_, kWorkspaceBytes) != hipSuccess) {
    AssignError(error_msg, "hipBLASLt workspace allocation failed");
    return nullptr;
  }
  b->workspace_bytes_ = kWorkspaceBytes;
  return b;
}

bool BlasLt::Describe(const Problem& p, hipblasLtMatmulDesc_t* desc,
                      hipblasLtMatrixLayout_t* la, hipblasLtMatrixLayout_t* lb,
                      hipblasLtMatrixLayout_t* lc) const {
  // Column-major view: C(m x n) = opA(A) * opB(B). A row-major [m][k]
  // operand is a column-major (k x m) matrix, so it enters transposed; a
  // transposed operand ([k][m] in memory) enters as is.
  const hipblasOperation_t op_a = p.a.transpose ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  const hipblasOperation_t op_b = p.b.transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  if (hipblasLtMatmulDescCreate(desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(*desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &op_a,
                                      sizeof(op_a)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(*desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &op_b,
                                      sizeof(op_b)) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  const auto layout = [&](hipblasLtMatrixLayout_t* l, hipDataType type,
                          std::uint64_t rows, std::uint64_t cols, int ld,
                          long long stride) {
    if (hipblasLtMatrixLayoutCreate(l, type, rows, cols, ld) !=
        HIPBLAS_STATUS_SUCCESS) {
      return false;
    }
    const std::int32_t count = p.batch;
    return hipblasLtMatrixLayoutSetAttribute(
               *l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
               sizeof(count)) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutSetAttribute(
               *l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
               sizeof(stride)) == HIPBLAS_STATUS_SUCCESS;
  };
  // Layout dimensions describe the stored (untransposed) matrix.
  const std::uint64_t m = p.m;
  const std::uint64_t n = p.n;
  const std::uint64_t k = p.k;
  return layout(la, p.a.type, p.a.transpose ? m : k, p.a.transpose ? k : m,
                p.a.ld, p.a.stride) &&
         layout(lb, p.b.type, p.b.transpose ? n : k, p.b.transpose ? k : n,
                p.b.ld, p.b.stride) &&
         layout(lc, HIP_R_32F, m, n, p.ldc, p.stride_c);
}

bool BlasLt::Tune(const Problem& p, float* out, Plan* plan,
                  std::string* error_msg) {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t la = nullptr;
  hipblasLtMatrixLayout_t lb = nullptr;
  hipblasLtMatrixLayout_t lc = nullptr;
  if (!Describe(p, &desc, &la, &lb, &lc)) {
    AssignError(error_msg, "hipBLASLt descriptor creation failed");
    return false;
  }
  hipblasLtMatmulPreference_t pref = nullptr;
  std::array<hipblasLtMatmulHeuristicResult_t, kCandidates> results{};
  int found = 0;
  const bool listed =
      hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
          sizeof(workspace_bytes_)) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmulAlgoGetHeuristic(handle_, desc, la, lb, lc, lc, pref,
                                      kCandidates, results.data(),
                                      &found) == HIPBLAS_STATUS_SUCCESS &&
      found > 0;
  bool ok = listed;
  if (listed) {
    const float one = 1.0F;
    const float zero = 0.0F;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    float best_ms = 0.0F;
    bool have_best = false;
    for (int i = 0; i < found; ++i) {
      // Workspace-backed kernels split K and reduce afterwards; the plain
      // ones keep one summation order per output, which the repeatability
      // checks rely on.
      if (results[i].state != HIPBLAS_STATUS_SUCCESS ||
          results[i].workspaceSize != 0) {
        continue;
      }
      const auto run = [&] {
        return hipblasLtMatmul(handle_, desc, &one, p.a.data, la, p.b.data, lb,
                               &zero, out, lc, out, lc, &results[i].algo,
                               workspace_, workspace_bytes_,
                               stream_) == HIPBLAS_STATUS_SUCCESS;
      };
      if (!run()) {
        continue;
      }
      (void)hipEventRecord(start, stream_);
      constexpr int kReps = 3;
      bool ran = true;
      for (int r = 0; r < kReps && ran; ++r) {
        ran = run();
      }
      (void)hipEventRecord(stop, stream_);
      (void)hipEventSynchronize(stop);
      float ms = 0.0F;
      (void)hipEventElapsedTime(&ms, start, stop);
      if (ran && (!have_best || ms < best_ms)) {
        best_ms = ms;
        have_best = true;
        plan->algo = results[i].algo;
      }
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    ok = have_best;
    plan->tuned = have_best;
  }
  if (pref != nullptr) {
    (void)hipblasLtMatmulPreferenceDestroy(pref);
  }
  (void)hipblasLtMatrixLayoutDestroy(lc);
  (void)hipblasLtMatrixLayoutDestroy(lb);
  (void)hipblasLtMatrixLayoutDestroy(la);
  (void)hipblasLtMatmulDescDestroy(desc);
  if (!ok) {
    AssignError(error_msg, "no hipBLASLt kernel for the GEMM shape");
  }
  return ok;
}

bool BlasLt::Gemm(const Operand& a, const Operand& b, float* out, int ldc,
                  long long stride_c, int m, int n, int k, int batch,
                  std::string* error_msg) {
  const Problem p{a, b, out, ldc, stride_c, m, n, k, batch};
  Plan& plan = plans_[p.Key()];
  if (!plan.tuned) {
    // Time the class at the top of its size bucket (never past the buffers
    // sized for tuning_n); the layouts below carry the real n.
    Problem widest = p;
    widest.n = std::min<int>(p.ClassN(), static_cast<int>(tuning_n_));
    widest.n = std::max<int>(widest.n, n);
    if (!Tune(widest, out, &plan, error_msg)) {
      return false;
    }
  }
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t la = nullptr;
  hipblasLtMatrixLayout_t lb = nullptr;
  hipblasLtMatrixLayout_t lc = nullptr;
  if (!Describe(p, &desc, &la, &lb, &lc)) {
    AssignError(error_msg, "hipBLASLt descriptor creation failed");
    return false;
  }
  const float one = 1.0F;
  const float zero = 0.0F;
  // The kernel was picked at the tuning width; make sure it also takes
  // this one.
  std::size_t needed = 0;
  hipblasStatus_t status = hipblaslt_ext::matmulIsAlgoSupported(
      handle_, desc, &one, la, lb, &zero, lc, lc, plan.algo, needed);
  if (status == HIPBLAS_STATUS_SUCCESS && needed > workspace_bytes_) {
    status = HIPBLAS_STATUS_INVALID_VALUE;
  }
  if (status == HIPBLAS_STATUS_SUCCESS) {
    status = hipblasLtMatmul(handle_, desc, &one, a.data, la, b.data, lb, &zero,
                             out, lc, out, lc, &plan.algo, workspace_,
                             workspace_bytes_, stream_);
  }
  (void)hipblasLtMatrixLayoutDestroy(lc);
  (void)hipblasLtMatrixLayoutDestroy(lb);
  (void)hipblasLtMatrixLayoutDestroy(la);
  (void)hipblasLtMatmulDescDestroy(desc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipBLASLt GEMM failed");
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
