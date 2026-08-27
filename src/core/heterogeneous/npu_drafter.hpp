#ifndef GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_
#define GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"

#ifdef ENGINE_ENABLE_XRT
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#endif

namespace gufo::heterogeneous {

struct NpuDrafterConfig {
  std::string mtp_model_path{
      "models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"};
  std::uint32_t device_index{0};
  std::uint32_t max_draft_tokens{4};
  std::uint32_t vocab_size{152064};
  bool enable_xrt{true};
  std::size_t shared_buffer_size{64 * 1024};  // 64KB unified DMA buffer
};

/// Heterogeneous XDNA2 NPU Draft Backend via XRT unified memory
class NpuDraftBackend : public speculative::IDraftBackend {
public:
  explicit NpuDraftBackend(NpuDrafterConfig config = {});
  ~NpuDraftBackend() override;

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "NpuXdna2DraftBackend";
  }

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

  [[nodiscard]] bool IsNpuActive() const noexcept { return npu_available_; }
  [[nodiscard]] bool HasMtpModel() const noexcept { return has_mtp_model_; }

  [[nodiscard]] const std::string& GetStatusMessage() const noexcept {
    return status_message_;
  }

private:
  void InitializeXrt();
  void LoadMtpGguf();

  NpuDrafterConfig config_;
  bool npu_available_{false};
  bool has_mtp_model_{false};
  std::string status_message_{"Uninitialized"};
  std::unique_ptr<core::GgufReader> mtp_reader_;

#ifdef ENGINE_ENABLE_XRT
  std::unique_ptr<xrt::device> device_;
  std::unique_ptr<xrt::bo> shared_bo_;
#endif
  std::vector<tokenization::TokenId> history_;
};

}  // namespace gufo::heterogeneous

#endif  // GUFO_CORE_HETEROGENEOUS_NPU_DRAFTER_HPP_
