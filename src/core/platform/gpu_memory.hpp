// Free/total device memory for capacity planning.
// On Windows, hipMemGetInfo reports free memory of the dedicated segment only,
// so it drops to zero once weights spill into shared memory even though more
// allocations still succeed. DXGI budgets cover both segments.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

#ifdef _WIN32
extern "C" int gufo_dxgi_free_bytes(unsigned long long* free_bytes);
#endif

namespace gufo::platform {

inline hipError_t DeviceMemoryInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
  const hipError_t error = hipMemGetInfo(free_bytes, total_bytes);
#ifdef _WIN32
  unsigned long long dxgi_free = 0;
  if (error == hipSuccess && gufo_dxgi_free_bytes(&dxgi_free) == 0 && dxgi_free > *free_bytes)
    *free_bytes = static_cast<std::size_t>(dxgi_free);
#endif
  return error;
}

}  // namespace gufo::platform
