#ifndef STRIX_CORE_HIP_DETAIL_HIP_GRAPH_DECODE_EXECUTOR_HPP_
#define STRIX_CORE_HIP_DETAIL_HIP_GRAPH_DECODE_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "src/core/hip/detail/dispatch_telemetry.hpp"

namespace strix::hip::detail {

class HipGraphDecodeExecutor {
public:
  HipGraphDecodeExecutor() {
    const char* env = std::getenv("STRIX_ENABLE_HIP_GRAPH");
    if (env != nullptr) {
      const std::string_view val(env);
      if (val == "0" || val == "false" || val == "OFF" || val == "off") {
        is_enabled_ = false;
      }
    }
  }

  ~HipGraphDecodeExecutor() { Reset(); }

  void Reset() noexcept {
    if (instance_ != nullptr) {
      (void)hipGraphExecDestroy(instance_);
      instance_ = nullptr;
    }
    if (graph_ != nullptr) {
      (void)hipGraphDestroy(graph_);
      graph_ = nullptr;
    }
    is_captured_ = false;
    capture_attempted_ = false;
  }

  [[nodiscard]] bool IsEnabled() const noexcept { return is_enabled_; }
  [[nodiscard]] bool IsCaptured() const noexcept { return is_captured_; }

  template<typename CaptureFn>
  bool TryCapture(hipStream_t stream, CaptureFn&& capture_fn) {
    if (!is_enabled_ || capture_attempted_) {
      EmitGraphDispatch(is_enabled_ ? "miss_already_attempted" : "disabled");
      return false;
    }
    capture_attempted_ = true;

    hipError_t err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
      is_enabled_ = false;
      EmitGraphDispatch("miss_begin_failed");
      return false;
    }

    capture_fn();

    err = hipStreamEndCapture(stream, &graph_);
    if (err != hipSuccess || graph_ == nullptr) {
      if (graph_ != nullptr) {
        (void)hipGraphDestroy(graph_);
        graph_ = nullptr;
      }
      is_enabled_ = false;
      EmitGraphDispatch("miss_end_failed");
      return false;
    }

    err = hipGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    if (err != hipSuccess || instance_ == nullptr) {
      if (instance_ != nullptr) {
        (void)hipGraphExecDestroy(instance_);
        instance_ = nullptr;
      }
      if (graph_ != nullptr) {
        (void)hipGraphDestroy(graph_);
        graph_ = nullptr;
      }
      is_enabled_ = false;
      EmitGraphDispatch("miss_instantiate_failed");
      return false;
    }

    is_captured_ = true;
    EmitGraphDispatch("miss_captured");
    return true;
  }

  bool Launch(hipStream_t stream) {
    if (!is_captured_ || instance_ == nullptr) {
      EmitGraphDispatch("launch_without_capture");
      return false;
    }
    hipError_t err = hipGraphLaunch(instance_, stream);
    EmitGraphDispatch(err == hipSuccess ? "hit" : "launch_failed");
    return err == hipSuccess;
  }

private:
  hipGraph_t graph_{nullptr};
  hipGraphExec_t instance_{nullptr};
  bool is_captured_{false};
  bool capture_attempted_{false};
  bool is_enabled_{true};
};

}  // namespace strix::hip::detail

#endif  // STRIX_CORE_HIP_DETAIL_HIP_GRAPH_DECODE_EXECUTOR_HPP_
