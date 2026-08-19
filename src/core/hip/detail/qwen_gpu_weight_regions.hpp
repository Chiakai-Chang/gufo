#ifndef STRIX_CORE_HIP_DETAIL_QWEN_GPU_WEIGHT_REGIONS_HPP_
#define STRIX_CORE_HIP_DETAIL_QWEN_GPU_WEIGHT_REGIONS_HPP_

#include "src/core/hip/qwen_gpu_executor.hpp"

#if defined(ENGINE_ENABLE_HIP)
namespace strix::hip::detail {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept;

}  // namespace strix::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_CORE_HIP_DETAIL_QWEN_GPU_WEIGHT_REGIONS_HPP_
