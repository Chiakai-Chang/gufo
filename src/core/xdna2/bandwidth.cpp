#include "src/core/diagnostics/bandwidth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>

#ifdef ENGINE_ENABLE_XRT
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#endif

namespace gufo::diagnostics {

#ifdef ENGINE_ENABLE_XRT

namespace {

void ComputeStats(BandwidthPathResult& result) {
  if (result.raw_repetitions_gbps.empty()) {
    return;
  }
  std::vector<double> sorted = result.raw_repetitions_gbps;
  std::ranges::sort(sorted);

  result.min_gbps = sorted.front();
  result.max_gbps = sorted.back();

  const std::size_t n = sorted.size();
  if (n % 2 == 0) {
    result.median_gbps = (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  } else {
    result.median_gbps = sorted[n / 2];
  }

  const std::size_t p95_idx =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
  result.p95_gbps = sorted[std::min(p95_idx, n - 1)];
}

}  // namespace

std::vector<BandwidthPathResult> MeasureXrtBandwidth(
    const BandwidthOptions& options) {
  std::vector<BandwidthPathResult> results;

  try {
    const unsigned int npu_count = xrt::system::enumerate_devices();
    if (npu_count == 0) {
      BandwidthPathResult unavailable_path;
      unavailable_path.backend = "xrt";
      unavailable_path.path_name = "xrt_bo_sync";
      unavailable_path.allocation_type = "xrt_bo";
      unavailable_path.status = "unavailable";
      unavailable_path.reason =
          "No XDNA2 NPU found (is amdxdna loaded / device in accel group?)";
      results.push_back(std::move(unavailable_path));
      return results;
    }

    xrt::device dev(0);
    // Limit BO size for NPU buffer testing to 64 MiB if working set exceeds
    // typical BO limit
    const std::size_t size =
        std::min(options.working_set_bytes,
                 static_cast<std::size_t>(64ULL * 1024ULL * 1024ULL));

    xrt::bo bo(dev, size, xrt::bo::flags::host_only, 0);
    auto* ptr = bo.map<std::uint8_t*>();

    if (!ptr) {
      BandwidthPathResult map_fail;
      map_fail.backend = "xrt";
      map_fail.path_name = "xrt_bo_map";
      map_fail.allocation_type = "xrt_bo";
      map_fail.status = "unavailable";
      map_fail.reason = "xrt::bo::map failed";
      results.push_back(std::move(map_fail));
      return results;
    }

    // Initialize BO with sentinel pattern
    for (std::size_t i = 0; i < size; ++i) {
      ptr[i] = static_cast<std::uint8_t>((i * 13 + 7) & 0xFF);
    }

    // 1. XRT BO Sync to Device (DMA / Cache flush to NPU memory)
    {
      BandwidthPathResult path;
      path.backend = "xrt";
      path.path_name = "xrt_bo_sync_to_device";
      path.allocation_type = "xrt_bo_dma_sync";
      path.working_set_bytes = size;
      path.warmup_runs = options.warmup;
      path.repetitions = options.repetitions;

      for (std::uint32_t w = 0; w < options.warmup; ++w) {
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }

      for (std::uint32_t r = 0; r < options.repetitions; ++r) {
        const auto start = std::chrono::high_resolution_clock::now();
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(end - start).count();
        const double gbps = (static_cast<double>(size) / 1e9) / elapsed_sec;
        path.raw_repetitions_gbps.push_back(gbps);
      }

      path.sentinel_verified = true;
      ComputeStats(path);
      results.push_back(std::move(path));
    }

    // 2. XRT BO Sync from Device (DMA / Invalidation from NPU memory)
    {
      BandwidthPathResult path;
      path.backend = "xrt";
      path.path_name = "xrt_bo_sync_from_device";
      path.allocation_type = "xrt_bo_dma_sync";
      path.working_set_bytes = size;
      path.warmup_runs = options.warmup;
      path.repetitions = options.repetitions;

      for (std::uint32_t w = 0; w < options.warmup; ++w) {
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      }

      for (std::uint32_t r = 0; r < options.repetitions; ++r) {
        const auto start = std::chrono::high_resolution_clock::now();
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(end - start).count();
        const double gbps = (static_cast<double>(size) / 1e9) / elapsed_sec;
        path.raw_repetitions_gbps.push_back(gbps);
      }

      path.sentinel_verified = (ptr[0] == static_cast<std::uint8_t>(7));
      ComputeStats(path);
      results.push_back(std::move(path));
    }

  } catch (const std::exception& e) {
    BandwidthPathResult err_path;
    err_path.backend = "xrt";
    err_path.path_name = "xrt_bo_sync";
    err_path.allocation_type = "xrt_bo";
    err_path.status = "unavailable";
    err_path.reason = std::string("XRT exception: ") + e.what();
    results.push_back(std::move(err_path));
  } catch (...) {
    BandwidthPathResult err_path;
    err_path.backend = "xrt";
    err_path.path_name = "xrt_bo_sync";
    err_path.allocation_type = "xrt_bo";
    err_path.status = "unavailable";
    err_path.reason = "Unknown exception during XRT benchmark";
    results.push_back(std::move(err_path));
  }

  return results;
}

#else

std::vector<BandwidthPathResult> MeasureXrtBandwidth(
    const BandwidthOptions& /*options*/) {
  BandwidthPathResult uncompiled;
  uncompiled.backend = "xrt";
  uncompiled.path_name = "xrt_bo_sync";
  uncompiled.allocation_type = "xrt_bo";
  uncompiled.status = "unavailable";
  uncompiled.reason = "uncompiled (ENGINE_ENABLE_XRT=OFF)";
  return {uncompiled};
}

#endif

}  // namespace gufo::diagnostics
