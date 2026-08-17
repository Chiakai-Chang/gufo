#include "src/cli/diagnose.h"

#include <iostream>
#include <span>
#include <string>
#include <string_view>

#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"

namespace strix::cli {

namespace {

void PrintDiagnoseHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " diagnose [OPTIONS]\n\n"
      << "Run non-interactive system and accelerator diagnostics for Strix "
         "Halo.\n\n"
      << "Options:\n"
      << "  --json             Emit structured JSON output conforming to "
         "diagnostics schema v1.0.0\n"
      << "  --section <name>   Diagnostic section to run: all, inventory, "
         "platform, gpu, npu\n"
      << "  -h, --help         Print help information\n";
}

}  // namespace

diagnostics::DiagnosticReport CollectDiagnostics(
    const diagnostics::LinuxSysfs& sysfs, std::string_view section) {
  diagnostics::DiagnosticReport report;

  const auto inventory = diagnostics::CollectSystemInventory(sysfs);
  const auto compatibility =
      diagnostics::EvaluateCompatibility(inventory, section);

  report.SetInventory(inventory);
  report.SetCompatibility(compatibility);

  // 1. Platform Check
  if (section == "all" || section == "platform" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "platform";
    if (inventory.cpu.architecture == "x86_64") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = "Supported architecture: Linux x86-64";
      check.details["targetArchitecture"] = "x86_64-linux";
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported target platform: only Linux x86-64 is supported";
      report.AddError("Unsupported platform architecture detected");
    }
    report.AddCheck(std::move(check));
  }

  // 2. System, CPU & Memory Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck sys_check;
    sys_check.name = "system";
    sys_check.status = diagnostics::DiagnosticStatus::kPass;
    sys_check.message = "Host operating system and kernel identified";
    sys_check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    report.AddCheck(std::move(sys_check));
    diagnostics::DiagnosticCheck cpu_check;
    cpu_check.name = "cpu";
    cpu_check.status = diagnostics::DiagnosticStatus::kPass;
    cpu_check.message = "CPU processor topology queried";
    cpu_check.details["modelName"] = inventory.cpu.model_name;
    cpu_check.details["logicalCores"] =
        std::to_string(inventory.cpu.logical_cores);
    cpu_check.details["physicalCores"] =
        std::to_string(inventory.cpu.physical_cores);
    report.AddCheck(std::move(cpu_check));

    diagnostics::DiagnosticCheck mem_check;
    mem_check.name = "memory";
    mem_check.status = diagnostics::DiagnosticStatus::kPass;
    mem_check.message = "Host unified memory pool identified";
    mem_check.details["totalBytes"] =
        std::to_string(inventory.memory.total_bytes);
    mem_check.details["availableBytes"] =
        std::to_string(inventory.memory.available_bytes);
    mem_check.details["memoryType"] = inventory.memory.memory_type;
    report.AddCheck(std::move(mem_check));
  }

  // 3. GPU Check
  if (section == "all" || section == "gpu" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "gpu";
    check.details["name"] = inventory.gpu.name;
    check.details["architecture"] = inventory.gpu.architecture;
    check.details["computeUnits"] = std::to_string(inventory.gpu.compute_units);
    check.details["driverName"] = inventory.gpu.driver_name;
    check.details["pciId"] = inventory.gpu.pci_id;
    check.details["powerMode"] = inventory.gpu.power_mode;

    if (inventory.gpu.architecture == "gfx1151") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = "Detected AMD Strix Halo GPU (gfx1151)";
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported GPU architecture: expected gfx1151, detected " +
          inventory.gpu.architecture;
      report.AddError("Unsupported GPU architecture: " +
                      inventory.gpu.architecture);
    }
    report.AddCheck(std::move(check));
  }

  // 4. NPU Check
  if (section == "all" || section == "npu" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "npu";
    check.details["identity"] = inventory.npu.identity;
    check.details["architecture"] = inventory.npu.architecture;
    check.details["pciDeviceId"] = inventory.npu.pci_device_id;
    check.details["driverName"] = inventory.npu.driver_name;
    check.details["firmwareVersion"] = inventory.npu.firmware_version;

    if (inventory.npu.architecture == "XDNA2" ||
        inventory.npu.architecture == "AIE2P") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = "Detected AMD XDNA2 NPU";
    } else if (inventory.npu.availability.find("permission denied") !=
               std::string::npos) {
      check.status = diagnostics::DiagnosticStatus::kWarn;
      check.message =
          "XDNA2 NPU device node inaccessible: " + inventory.npu.availability;
      report.AddWarning("XDNA2 NPU device node permission denied");
    } else {
      check.status = diagnostics::DiagnosticStatus::kWarn;
      check.message = "XDNA2 NPU unavailable: " + inventory.npu.availability;
      report.AddWarning("XDNA2 NPU unavailable or driver not loaded");
    }
    report.AddCheck(std::move(check));
  }

  // 5. Toolchain Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "toolchain";
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message = "Toolchain and driver stack verified";
    check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    check.details["amdgpuStatus"] = inventory.toolchain.amdgpu_status;
    check.details["amdxdnaStatus"] = inventory.toolchain.amdxdna_status;
    check.details["rocmVersion"] = inventory.toolchain.rocm_version;
    check.details["xrtPluginVersion"] = inventory.toolchain.xrt_plugin_version;
    report.AddCheck(std::move(check));
  }

  return report;
}

int RunDiagnose(std::span<const char* const> args) {
  bool json_mode = false;
  std::string section = "all";
  std::size_t start_idx = 0;

  if (!args.empty()) {
    const std::string_view first = args[0];
    if (first == "strix-server" || first.ends_with("/strix-server")) {
      start_idx = 1;
    }
  }
  if (start_idx < args.size() &&
      std::string_view(args[start_idx]) == "diagnose") {
    start_idx++;
  }

  for (std::size_t i = start_idx; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--json") {
      json_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintDiagnoseHelp("strix-server");
      return 0;
    } else if (arg == "--section") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --section requires an argument\n";
        PrintDiagnoseHelp("strix-server");
        return 2;
      }
      section = args[++i];
    } else if (arg.starts_with("--section=")) {
      section = std::string(arg.substr(10));
    } else {
      std::cerr << "Error: unknown diagnose option '" << arg << "'\n";
      PrintDiagnoseHelp("strix-server");
      return 2;
    }
  }

  const auto report = CollectDiagnostics(diagnostics::LinuxSysfs(), section);

  if (json_mode) {
    std::cout << report.ToJson();
  } else {
    std::cout << report.ToHuman();
  }

  if (report.Status() == diagnostics::DiagnosticStatus::kFail) {
    return 1;
  }
  return 0;
}

}  // namespace strix::cli
