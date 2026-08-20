#include "src/core/xdna2/device.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <string_view>

#ifdef ENGINE_ENABLE_XRT
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>
#endif

namespace strix::xdna2 {

namespace {

bool ContainsCaseInsensitive(std::string value, std::string needle) {
  std::ranges::transform(value, value.begin(),
                         [](unsigned char ch) { return std::tolower(ch); });
  std::ranges::transform(needle, needle.begin(),
                         [](unsigned char ch) { return std::tolower(ch); });
  return value.find(needle) != std::string::npos;
}

void SetDiscoveryFailure(XrtDeviceInfo& info, std::string message) {
  info.detected = std::move(message);
  info.required = "accessible XDNA2/AIE2P device through the pinned XRT plugin";
  if (ContainsCaseInsensitive(info.detected, "permission denied") ||
      ContainsCaseInsensitive(info.detected, "operation not permitted")) {
    info.error_category = "permission_denied";
    info.remediation =
        "Grant the process access to the accel render device and retry";
  } else if (ContainsCaseInsensitive(info.detected, "plugin") ||
             ContainsCaseInsensitive(info.detected, "driver")) {
    info.error_category = "plugin_or_driver_unavailable";
    info.remediation =
        "Load amdxdna and use the Nix-packaged XRT amdxdna plugin";
  } else {
    info.error_category = "discovery_error";
    info.remediation =
        "Run strix-server diagnose --section npu and inspect XRT diagnostics";
  }
}

void SetCompatibilityFailure(XrtDeviceInfo& info, std::string category,
                             std::string detected, std::string required,
                             std::string remediation) {
  info.error_category = std::move(category);
  info.detected = std::move(detected);
  info.required = std::move(required);
  info.remediation = std::move(remediation);
}

bool IsSupportedFirmware(std::string_view firmware) {
  return firmware.find("1.1.2.64") != std::string_view::npos ||
         firmware.find("1.1.2.65") != std::string_view::npos;
}

}  // namespace

XrtDeviceInfo DiscoverXrtDevice(std::uint32_t device_index,
                                const diagnostics::SystemInventory& inventory) {
  XrtDeviceInfo info;
  info.device_index = device_index;
  info.architecture = inventory.npu.architecture;
  info.driver = inventory.npu.driver_name;
  info.firmware = inventory.npu.firmware_version;

  if (info.architecture != "XDNA2" && info.architecture != "AIE2P") {
    SetCompatibilityFailure(
        info, "architecture_mismatch",
        info.architecture.empty() ? "unknown" : info.architecture,
        "XDNA2/AIE2P", "Run only on the supported AMD Strix Halo XDNA2 target");
    return info;
  }
  if (info.driver != "amdxdna" ||
      inventory.toolchain.amdxdna_status != "loaded") {
    SetCompatibilityFailure(
        info, "driver_unsupported",
        info.driver + " (" + inventory.toolchain.amdxdna_status + ")",
        "amdxdna loaded", "Load the supported amdxdna driver and retry");
    return info;
  }
  if (!IsSupportedFirmware(info.firmware)) {
    SetCompatibilityFailure(
        info, "firmware_unsupported",
        info.firmware.empty() ? "unknown" : info.firmware,
        "XDNA2 NPU firmware 1.1.2.64 or 1.1.2.65",
        "Install the supported NPU firmware package and reboot");
    return info;
  }

#ifdef ENGINE_ENABLE_XRT
  try {
    info.device_count = xrt::system::enumerate_devices();
    if (info.device_count == 0) {
      SetDiscoveryFailure(info, "XRT enumerated zero devices");
      info.error_category = "device_unavailable";
      return info;
    }
    if (device_index >= info.device_count) {
      SetDiscoveryFailure(
          info, "requested device index " + std::to_string(device_index) +
                    " but XRT enumerated " + std::to_string(info.device_count) +
                    " device(s)");
      info.error_category = "device_index_out_of_range";
      return info;
    }
    xrt::device device(device_index);
    info.name = device.get_info<xrt::info::device::name>();
    info.available = true;
    return info;
  } catch (const std::exception& error) {
    SetDiscoveryFailure(info, error.what());
    return info;
  }
#else
  SetDiscoveryFailure(info, "binary was built without ENGINE_ENABLE_XRT");
  info.error_category = "xrt_uncompiled";
  info.remediation = "Build the supported Nix package with XRT enabled";
  return info;
#endif
}

}  // namespace strix::xdna2
