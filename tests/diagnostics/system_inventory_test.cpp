#include "src/core/diagnostics/system_inventory.h"

#include <array>
#include <cstdlib>
#include <filesystem>
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

std::filesystem::path FindFixturesRoot() {
#ifdef TEST_FIXTURES_DIR
  if (std::filesystem::exists(TEST_FIXTURES_DIR)) {
    return TEST_FIXTURES_DIR;
  }
#endif
  const std::array<std::filesystem::path, 3> paths = {
      "tests/fixtures/diagnostics/sysfs",
      "../tests/fixtures/diagnostics/sysfs",
      "../../tests/fixtures/diagnostics/sysfs",
  };
  for (const auto& path : paths) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return "tests/fixtures/diagnostics/sysfs";
}

void TestSupportedStrixHaloFixture() {
  const auto fixture_base = FindFixturesRoot() / "supported_strix_halo";
  const strix::diagnostics::LinuxSysfs sysfs(fixture_base / "sys",
                                             fixture_base / "proc");

  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);
  Expect(inv.cpu.model_name.find("AMD RYZEN AI MAX+ 395") != std::string::npos,
         "CPU model matches Ryzen AI Max+ 395");
  Expect(inv.cpu.logical_cores == 32, "Logical cores == 32");
  Expect(inv.memory.total_bytes > 100ULL * 1024 * 1024 * 1024,
         "Memory > 100 GiB");
  Expect(inv.gpu.driver_name == "amdgpu", "GPU driver is amdgpu");
  Expect(inv.gpu.pci_id == "1002:1586", "GPU PCI ID is 1002:1586");
  Expect(inv.npu.driver_name == "amdxdna", "NPU driver is amdxdna");
  Expect(inv.npu.architecture == "XDNA2", "NPU architecture is XDNA2");

  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);
  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kSupported,
         "Supported Strix Halo fixture passes compatibility");

  const std::string json = inv.ToJson();
  Expect(json.find("\"modelName\":") != std::string::npos,
         "JSON contains modelName");
  Expect(json.find("\"XDNA2\"") != std::string::npos, "JSON contains XDNA2");
}

void TestWrongGpuFixture() {
  const auto fixture_base = FindFixturesRoot() / "wrong_gpu";
  const strix::diagnostics::LinuxSysfs sysfs(fixture_base / "sys",
                                             fixture_base / "proc");

  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);
  Expect(inv.gpu.pci_id == "1002:744C", "Wrong GPU PCI ID 1002:744C");

  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);
  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kUnsupported,
         "Wrong GPU fixture fails compatibility");

  bool found_gpu_error = false;
  for (const auto& item : comp.items) {
    if (item.component == "gpu" &&
        item.verdict ==
            strix::diagnostics::CompatibilityVerdict::kUnsupported) {
      found_gpu_error = true;
      Expect(!item.remediation_hint.empty(),
             "Remediation hint is provided for wrong GPU");
    }
  }
  Expect(found_gpu_error, "GPU component is marked unsupported");
}

void TestAbsentAmdxdnaFixture() {
  const auto fixture_base = FindFixturesRoot() / "absent_amdxdna";
  const strix::diagnostics::LinuxSysfs sysfs(fixture_base / "sys",
                                             fixture_base / "proc");

  const auto inv = strix::diagnostics::CollectSystemInventory(sysfs);
  Expect(inv.toolchain.amdxdna_status != "loaded", "amdxdna is not loaded");

  const auto comp = strix::diagnostics::EvaluateCompatibility(inv);
  Expect(comp.overall_verdict ==
             strix::diagnostics::CompatibilityVerdict::kUnsupported,
         "Absent amdxdna fails compatibility");

  bool found_driver_error = false;
  for (const auto& item : comp.items) {
    if (item.component == "kernel_driver" &&
        item.verdict ==
            strix::diagnostics::CompatibilityVerdict::kUnsupported) {
      found_driver_error = true;
      Expect(item.remediation_hint.find("amdxdna") != std::string::npos,
             "Remediation hint mentions amdxdna");
    }
  }
  Expect(found_driver_error, "Driver component is marked unsupported");
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
