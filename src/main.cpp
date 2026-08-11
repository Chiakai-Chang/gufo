// strix.cpp — hello world / ROCm + NPU smoke test.
//
// Prints build info, verifies the ROCm/HIP GPU toolchain (when built with
// ENGINE_ENABLE_HIP=1) and the XRT NPU shim (when ENGINE_ENABLE_XRT=1):
// ROCm version, GPU device count/arch/unified memory, and XRT NPU discovery.

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef ENGINE_ENABLE_HIP
#include <hip/hip_runtime.h>

static const char *hipErrStr(hipError_t e) { return hipGetErrorName(e); }

static void checkHip(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error on %s: %s\n", what, hipErrStr(e));
    std::exit(1);
  }
}

static void checkGpu() {
  std::printf("ROCm version: %d.%d.%d\n",
              HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);

  int devCount = 0;
  checkHip(hipGetDeviceCount(&devCount), "hipGetDeviceCount");
  std::printf("HIP device count: %d\n", devCount);
  if (devCount == 0) {
    std::printf("No HIP device found — is kfd loaded / device in render group?\n");
    return;
  }

  for (int i = 0; i < devCount; ++i) {
    hipDeviceProp_t prop{};
    checkHip(hipGetDeviceProperties(&prop, i), "hipGetDeviceProperties");
    std::printf("device[%d]: %s\n", i, prop.name);
    std::printf("  arch:        %s\n", prop.gcnArchName);
    std::printf("  computeCap:  %d.%d\n", prop.major, prop.minor);

    // Unified memory (LPDDR5X) reported as the device global-memory pool.
    std::printf("  totalMem:    %.2f GiB (unified)\n",
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Live available-memory snapshot (GTT + VRAM combined on Strix Halo).
    size_t free = 0, total = 0;
    checkHip(hipMemGetInfo(&free, &total), "hipMemGetInfo");
    std::printf("  hipMemGetInfo free=%.2f GiB total=%.2f GiB\n",
                (double)free / (1024.0 * 1024.0 * 1024.0),
                (double)total / (1024.0 * 1024.0 * 1024.0));

    // Quick sanity: a tiny unified-memory allocation + memset + free.
    char *buf = nullptr;
    checkHip(hipMalloc(&buf, 4096), "hipMalloc");
    checkHip(hipMemset(buf, 0xAB, 4096), "hipMemset");
    checkHip(hipFree(buf), "hipFree");
    std::printf("  unified-mem alloc/memset/free OK\n");
  }
}
#endif

#ifdef ENGINE_ENABLE_XRT
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>

static void checkNpu() {
  std::printf("XRT NPU shim enabled\n");
  unsigned int npuCount = xrt::system::enumerate_devices();
  std::printf("XRT device count (NPU): %u\n", npuCount);
  for (unsigned int i = 0; i < npuCount; ++i) {
    xrt::device dev(i);
    std::string name = dev.get_info<xrt::info::device::name>();
    std::printf("  npu[%u]: %s\n", i, name.c_str());
  }
  if (npuCount == 0) {
    std::printf("No NPU found — is amdxdna loaded? ls /sys/class/accel\n");
  }
}
#endif

int main() {
  std::printf("Hello, Strix Halo!\n");

#ifdef ENGINE_ENABLE_HIP
  std::printf("Build: CMake, C++20, ROCm/HIP enabled\n");
  checkGpu();
#else
  std::printf("Build: CMake, C++20, CPU-only (no ROCm)\n");
#endif

#ifdef ENGINE_ENABLE_XRT
  checkNpu();
#endif

  std::printf("OK\n");
  return 0;
}
