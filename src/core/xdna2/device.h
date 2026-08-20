#ifndef STRIX_CORE_XDNA2_DEVICE_H_
#define STRIX_CORE_XDNA2_DEVICE_H_

#include <cstdint>
#include <string>

#include "src/core/diagnostics/system_inventory.h"

namespace strix::xdna2 {

struct XrtDeviceInfo {
  bool available{false};
  std::uint32_t device_count{0};
  std::uint32_t device_index{0};
  std::string name;
  std::string architecture;
  std::string driver;
  std::string firmware;
  std::string error_category;
  std::string detected;
  std::string required;
  std::string remediation;
};

[[nodiscard]] XrtDeviceInfo DiscoverXrtDevice(
    std::uint32_t device_index, const diagnostics::SystemInventory& inventory);

}  // namespace strix::xdna2

#endif  // STRIX_CORE_XDNA2_DEVICE_H_
