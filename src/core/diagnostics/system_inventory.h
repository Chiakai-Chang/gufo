#ifndef STRIX_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_
#define STRIX_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include "src/core/diagnostics/linux_sysfs.h"

namespace strix::diagnostics {

struct CpuInventory {
  std::string model_name;
  std::string vendor;
  std::string architecture{"x86_64"};
  std::uint32_t logical_cores{0};
  std::uint32_t physical_cores{0};
  std::uint32_t numa_nodes{1};
  std::string source{"/proc/cpuinfo"};
};

struct MemoryInventory {
  std::uint64_t total_bytes{0};
  std::uint64_t available_bytes{0};
  std::uint64_t free_bytes{0};
  std::string memory_type{"LPDDR5X (Unified)"};
  bool is_unified{true};
  std::string source{"/proc/meminfo"};
};

struct GpuInventory {
  std::string name{"AMD Radeon 8060S Graphics"};
  std::string architecture{"gfx1151"};
  std::string raw_arch_name;
  std::uint32_t compute_units{40};
  std::uint64_t total_memory_bytes{0};
  std::string driver_name{"amdgpu"};
  std::string pci_id{"1002:1586"};
  std::string pci_slot;
  std::string power_mode{"auto"};
  std::string clock_state{"unavailable (sysfs unreadable or power-managed)"};
  std::map<std::string, std::string> temperatures;
  std::string availability{"available"};
  std::string source{"HIP + /sys/class/drm"};
};

struct NpuInventory {
  std::string identity{"AMD XDNA2 NPU"};
  std::string architecture{"XDNA2"};
  std::string pci_device_id{"1022:17f0"};
  std::string driver_name{"amdxdna"};
  std::string pci_slot;
  std::string firmware_version{"npu.sbin (1.1.2.64/65)"};
  std::uint32_t aie_tiles_spatial{32};  // 4x8 spatial AIE2P tile array
  std::string availability{"available"};
  std::string source{"XRT + /sys/class/accel"};
};

struct DriverToolchainInventory {
  std::string kernel_release;
  std::string amdgpu_status{"loaded"};
  std::string amdxdna_status{"loaded"};
  std::string rocm_version{"7.2.3"};
  std::string xrt_commit{"8661761775a266b11992a3bd6eb08209d88aa845"};
  std::string xrt_plugin_version{"2.21.0"};
  std::string cxx_compiler{"GCC 15.3.0"};
  std::string cpp_standard{"C++20"};
  std::string source{"Nix Toolchain Pins + Linux UAPI"};
};

struct SystemInventory {
  CpuInventory cpu;
  MemoryInventory memory;
  GpuInventory gpu;
  NpuInventory npu;
  DriverToolchainInventory toolchain;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] SystemInventory CollectSystemInventory(
    const LinuxSysfs& sysfs = LinuxSysfs());

}  // namespace strix::diagnostics

#endif  // STRIX_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_
