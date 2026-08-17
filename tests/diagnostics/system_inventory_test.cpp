#include "src/core/diagnostics/system_inventory.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/linux_sysfs.h"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

class TempMockSysfs {
public:
  explicit TempMockSysfs(std::string_view name) {
    const auto timestamp =
        std::chrono::system_clock::now().time_since_epoch().count();
    root_path_ = std::filesystem::temp_directory_path() /
                 ("strix_test_mock_" + std::string(name) + "_" +
                  std::to_string(timestamp));
    std::filesystem::create_directories(root_path_ / "sys");
    std::filesystem::create_directories(root_path_ / "proc");
  }

  ~TempMockSysfs() {
    std::error_code ec;
    std::filesystem::remove_all(root_path_, ec);
  }

  TempMockSysfs(const TempMockSysfs&) = delete;
  TempMockSysfs& operator=(const TempMockSysfs&) = delete;
  TempMockSysfs(TempMockSysfs&&) = delete;
  TempMockSysfs& operator=(TempMockSysfs&&) = delete;

  void WriteFile(const std::filesystem::path& rel_path,
                 std::string_view content) {
    const auto full_path = root_path_ / rel_path;
    std::filesystem::create_directories(full_path.parent_path());
    std::ofstream out(full_path);
    out << content;
  }

  [[nodiscard]] strix::diagnostics::LinuxSysfs GetSysfs() const {
    return strix::diagnostics::LinuxSysfs(root_path_ / "sys",
                                          root_path_ / "proc");
  }

private:
  std::filesystem::path root_path_;
};

void TestSupportedStrixHaloFixture() {
  TempMockSysfs mock("supported");
  mock.WriteFile("proc/cpuinfo",
                 "processor\t: 0\n"
                 "model name\t: Generic AMD Strix Halo APU\n"
                 "siblings\t: 32\n"
                 "cpu cores\t: 16\n");
  mock.WriteFile("proc/meminfo",
                 "MemTotal:       131072000 kB\n"
                 "MemFree:         65536000 kB\n"
                 "MemAvailable:    98304000 kB\n");
  mock.WriteFile("sys/class/drm/card0/device/device", "0x1586\n");
  mock.WriteFile("sys/class/drm/card0/device/vendor", "0x1002\n");
  mock.WriteFile("sys/class/drm/card0/device/uevent",
                 "DRIVER=amdgpu\nPCI_ID=1002:1586\n");
  mock.WriteFile("sys/class/accel/accel0/device/device", "0x17f0\n");
  mock.WriteFile("sys/class/accel/accel0/device/vendor", "0x1022\n");
  mock.WriteFile("sys/class/accel/accel0/device/uevent",
                 "DRIVER=amdxdna\nPCI_ID=1022:17F0\n");
  mock.WriteFile("sys/module/amdgpu/version", "amdgpu-strix\n");
  mock.WriteFile("sys/module/amdxdna/version", "amdxdna-xdna2\n");

  const auto sysfs = mock.GetSysfs();
  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);

  Expect(inv.cpu.logical_cores == 32, "Logical cores == 32");
  Expect(inv.cpu.physical_cores == 16, "Physical cores == 16");
  Expect(inv.gpu.driver_name == "amdgpu", "GPU driver is amdgpu");
  Expect(inv.gpu.pci_id == "1002:1586", "GPU PCI ID is 1002:1586");
  Expect(inv.npu.driver_name == "amdxdna", "NPU driver is amdxdna");
  Expect(inv.npu.architecture == "XDNA2", "NPU architecture is XDNA2");

  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);
  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kSupported,
         "Supported Strix Halo mock passes compatibility");
}

void TestWrongGpuFixture() {
  TempMockSysfs mock("wrong_gpu");
  mock.WriteFile("proc/cpuinfo",
                 "processor\t: 0\n"
                 "model name\t: Generic CPU\n"
                 "siblings\t: 16\n"
                 "cpu cores\t: 8\n");
  mock.WriteFile("sys/class/drm/card0/device/device", "0x744c\n");
  mock.WriteFile("sys/class/drm/card0/device/vendor", "0x1002\n");
  mock.WriteFile("sys/class/drm/card0/device/uevent",
                 "DRIVER=amdgpu\nPCI_ID=1002:744C\n");
  mock.WriteFile("sys/module/amdgpu/version", "amdgpu\n");

  const auto sysfs = mock.GetSysfs();
  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);
  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);

  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kUnsupported,
         "Wrong GPU mock fails compatibility");
}

void TestAbsentAmdxdnaFixture() {
  TempMockSysfs mock("absent_npu");
  mock.WriteFile("proc/cpuinfo",
                 "processor\t: 0\n"
                 "model name\t: Generic CPU\n"
                 "siblings\t: 16\n"
                 "cpu cores\t: 8\n");
  mock.WriteFile("sys/class/drm/card0/device/device", "0x1586\n");
  mock.WriteFile("sys/class/drm/card0/device/vendor", "0x1002\n");
  mock.WriteFile("sys/class/drm/card0/device/uevent",
                 "DRIVER=amdgpu\nPCI_ID=1002:1586\n");

  const auto sysfs = mock.GetSysfs();
  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);
  Expect(inv.toolchain.amdxdna_status != "loaded", "amdxdna is not loaded");

  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);
  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kUnsupported,
         "Absent amdxdna fails compatibility");
}

}  // namespace

int main() {
  std::cout << "Running system inventory & compatibility test suite...\n";

  TestSupportedStrixHaloFixture();
  TestWrongGpuFixture();
  TestAbsentAmdxdnaFixture();

  std::cout
      << "All system inventory & compatibility tests passed successfully.\n";
  return 0;
}
