#ifndef STRIX_MODELS_MINIMAX_H3_DIT_HPP_
#define STRIX_MODELS_MINIMAX_H3_DIT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace strix::minimax_h3 {

inline constexpr std::size_t kDitBlocks = 50;
inline constexpr std::size_t kDitHiddenSize = 5376;
inline constexpr std::size_t kDitAttentionHeads = 56;
inline constexpr std::size_t kDitHeadDimension = 128;
inline constexpr std::size_t kDitAttentionWidth =
    kDitAttentionHeads * kDitHeadDimension;
inline constexpr std::size_t kDitFeedForwardSize = 14336;
inline constexpr std::size_t kDitRopeHalf = 48;
inline constexpr std::size_t kDitModulationSlots = 6;
inline constexpr float kDitNormEpsilon = 1.0e-5F;

struct DitBlockOptions {
  std::size_t block_index{0};
  std::size_t rows{0};
  std::size_t modulation_rows{0};
  bool row_parallel_attention{true};
  void* attention_score_workspace{nullptr};
  std::size_t attention_score_workspace_bytes{0};
  void* attention_probability_workspace{nullptr};
  std::size_t attention_probability_workspace_bytes{0};
};

struct DitBlockInput {
  std::span<const std::uint16_t> hidden;
  std::span<const std::uint16_t> modulation;
  std::span<const std::uint32_t> row_map;
  std::span<const std::uint16_t> rope_cos;
  std::span<const std::uint16_t> rope_sin;
};

struct DitPreparedInput {
  std::span<const std::uint16_t> hidden;
  std::span<const std::uint32_t> row_map;
};

struct DitBlockRetained {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<std::uint16_t> modulation_attention;
  std::vector<std::uint16_t> attention_output;
  std::vector<std::uint16_t> modulation_mlp;
  std::vector<std::uint16_t> block_output;
};

struct DitBlockTelemetry {
  std::uint64_t weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t gemm_workspace_bytes{0};
  std::uint64_t dispatches{0};
  double load_ms{0.0};
  double gpu_ms{0.0};
  std::uintptr_t scratch_address{0};
  std::string first_non_finite;
};

class DitBlockSession {
public:
  ~DitBlockSession();

  DitBlockSession(const DitBlockSession&) = delete;
  DitBlockSession& operator=(const DitBlockSession&) = delete;
  DitBlockSession(DitBlockSession&&) noexcept;
  DitBlockSession& operator=(DitBlockSession&&) noexcept;

  [[nodiscard]] static std::unique_ptr<DitBlockSession> Create(
      const ModelInventory& inventory, const DitBlockOptions& options,
      const CancellationToken* cancellation, std::string* error = nullptr);

  [[nodiscard]] bool PrepareConstants(std::span<const std::uint16_t> modulation,
                                      std::span<const std::uint16_t> rope_cos,
                                      std::span<const std::uint16_t> rope_sin,
                                      const CancellationToken* cancellation,
                                      std::string* error = nullptr);

  [[nodiscard]] bool RunPrepared(const DitPreparedInput& input,
                                 const CancellationToken* cancellation,
                                 std::span<std::uint16_t> output,
                                 DitBlockTelemetry* telemetry,
                                 std::string* error = nullptr);

  [[nodiscard]] bool Run(const DitBlockInput& input,
                         const CancellationToken* cancellation,
                         DitBlockRetained* retained,
                         DitBlockTelemetry* telemetry,
                         std::string* error = nullptr);

  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t block_index() const noexcept;
  [[nodiscard]] std::uintptr_t scratch_address() const noexcept;
  [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
  [[nodiscard]] std::uint64_t activation_bytes() const noexcept;
  [[nodiscard]] double load_ms() const noexcept;

private:
  struct Impl;
  explicit DitBlockSession(std::unique_ptr<Impl> impl);
  [[nodiscard]] bool RunInternal(const DitPreparedInput& input,
                                 const CancellationToken* cancellation,
                                 std::span<std::uint16_t> modulation_attention,
                                 std::span<std::uint16_t> attention_output,
                                 std::span<std::uint16_t> modulation_mlp,
                                 std::span<std::uint16_t> block_output,
                                 DitBlockTelemetry* telemetry,
                                 std::string* error);
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::minimax_h3

#endif  // STRIX_MODELS_MINIMAX_H3_DIT_HPP_
