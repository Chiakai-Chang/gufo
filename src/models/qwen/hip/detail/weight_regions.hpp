#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_

#include "src/models/qwen/hip/executor.hpp"

#if defined(ENGINE_ENABLE_HIP)
namespace gufo::hip::detail {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept;

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
