#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BLASLT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BLASLT_HPP_

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace gufo::models::qwen38_flash_next::rocm {

/// 16-bit hipBLASLt GEMMs with F32 accumulation and output. hipBLASLt's
/// heuristic order is unreliable on gfx1151 (its first candidate runs up to
/// eight times slower than its fifth), so every new shape class is timed
/// once and the fastest workspace-free kernel is kept.
class BlasLt {
public:
  /// A 16-bit matrix. Row-major [rows][cols] with `ld` elements between
  /// rows; `transpose` presents it as its transpose to the product.
  struct Operand {
    const void* data;
    hipDataType type;  ///< HIP_R_16F or HIP_R_16BF
    int ld;
    long long stride;  ///< elements between batch members
    bool transpose;
  };

  ~BlasLt();
  BlasLt(const BlasLt&) = delete;
  BlasLt& operator=(const BlasLt&) = delete;

  [[nodiscard]] static std::unique_ptr<BlasLt> Create(
      hipStream_t stream, std::uint32_t tuning_n, std::string* error_msg);

  /// out[t][i] = sum_k A(i, k) * B(t, k) for i < m, t < n, with `out` row
  /// stride `ldc` and batch stride `stride_c`: `a` is (m x k) after its
  /// transpose flag, `b` (n x k) after its flag. Untransposed `a` is a
  /// weight matrix [m][k]; a transposed one is stored [k][m].
  [[nodiscard]] bool Gemm(const Operand& a, const Operand& b, float* out,
                          int ldc, long long stride_c, int m, int n, int k,
                          int batch, std::string* error_msg);

private:
  BlasLt() = default;

  struct Plan {
    hipblasLtMatmulAlgo_t algo{};
    bool tuned{false};
  };
  struct Problem;
  bool Describe(const Problem& p, hipblasLtMatmulDesc_t* desc,
                hipblasLtMatrixLayout_t* la, hipblasLtMatrixLayout_t* lb,
                hipblasLtMatrixLayout_t* lc) const;
  bool Tune(const Problem& p, float* out, Plan* plan, std::string* error_msg);

  hipblasLtHandle_t handle_{nullptr};
  hipStream_t stream_{nullptr};
  std::uint32_t tuning_n_{512};
  void* workspace_{nullptr};
  std::size_t workspace_bytes_{0};
  std::unordered_map<std::string, Plan> plans_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BLASLT_HPP_
