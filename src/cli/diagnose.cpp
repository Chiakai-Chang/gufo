#include "src/cli/diagnose.h"

#include <sys/utsname.h>

#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(ENGINE_ENABLE_XRT)
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>
#endif

namespace strix::cli {

namespace {

void CollectPlatform(diagnostics::DiagnosticReport& report) {
  diagnostics::DiagnosticCheck check;
  check.name = "platform";

#if defined(__linux__) && defined(__x86_64__)
  check.status = diagnostics::DiagnosticStatus::kPass;
  check.message = "Supported architecture: Linux x86-64";
  check.details["targetArchitecture"] = "x86_64-linux";
#else
  check.status = diagnostics::DiagnosticStatus::kFail;
  check.message = "Unsupported target platform: only Linux x86-64 is supported";
  report.AddError("Unsupported platform architecture detected");
#endif

  report.AddCheck(std::move(check));
}

void CollectSystem(diagnostics::DiagnosticReport& report) {
  diagnostics::DiagnosticCheck check;
  check.name = "system";

  struct utsname uts{};
  if (uname(&uts) == 0) {
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message = "Host operating system and kernel identified";
    check.details["sysname"] = uts.sysname;
    check.details["nodename"] = uts.nodename;
    check.details["release"] = uts.release;
    check.details["version"] = uts.version;
    check.details["machine"] = uts.machine;
  } else {
    check.status = diagnostics::DiagnosticStatus::kWarn;
    check.message = "Failed to query host uname information";
    report.AddWarning("Could not query host uname");
  }

  report.AddCheck(std::move(check));
}

void CollectToolchain(diagnostics::DiagnosticReport& report) {
  diagnostics::DiagnosticCheck check;
  check.name = "toolchain";
  check.status = diagnostics::DiagnosticStatus::kPass;
  check.message = "C++20 runtime environment configured";

  check.details["cppStandard"] = "C++20";
#if defined(__clang__)
  check.details["compiler"] = "Clang " + std::to_string(__clang_major__) + "." +
                              std::to_string(__clang_minor__) + "." +
                              std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  check.details["compiler"] = "GCC " + std::to_string(__GNUC__) + "." +
                              std::to_string(__GNUC_MINOR__) + "." +
                              std::to_string(__GNUC_PATCHLEVEL__);
#else
  check.details["compiler"] = "Unknown";
#endif

#if defined(ENGINE_ENABLE_HIP)
  check.details["rocmHipBackend"] = "enabled";
#else
  check.details["rocmHipBackend"] = "disabled";
#endif

#if defined(ENGINE_ENABLE_XRT)
  check.details["xrtNpuBackend"] = "enabled";
#else
  check.details["xrtNpuBackend"] = "disabled";
#endif

  report.AddCheck(std::move(check));
}

void CollectGpu(diagnostics::DiagnosticReport& report) {
  diagnostics::DiagnosticCheck check;
  check.name = "gpu";

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  const hipError_t err = hipGetDeviceCount(&dev_count);
  if (err != hipSuccess) {
    check.status = diagnostics::DiagnosticStatus::kWarn;
    check.message = "HIP runtime initialized, but no GPU devices accessible";
    check.details["hipError"] = hipGetErrorName(err);
    report.AddWarning("HIP device enumeration failed or no device available");
  } else if (dev_count == 0) {
    check.status = diagnostics::DiagnosticStatus::kWarn;
    check.message = "No HIP devices detected on host system";
    report.AddWarning("Zero HIP GPU devices found");
  } else {
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message =
        "Detected " + std::to_string(dev_count) + " accessible HIP device(s)";
    check.details["deviceCount"] = std::to_string(dev_count);

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
      check.details["device0_name"] = prop.name;
      check.details["device0_gcnArchName"] = prop.gcnArchName;
      check.details["device0_totalGlobalMem"] =
          std::to_string(prop.totalGlobalMem);
      check.details["device0_multiProcessorCount"] =
          std::to_string(prop.multiProcessorCount);
    }
  }
#else
  check.status = diagnostics::DiagnosticStatus::kPass;
  check.message = "ROCm/HIP backend was not compiled into this binary";
  check.details["hipSupport"] = "uncompiled";
#endif

  report.AddCheck(std::move(check));
}

void CollectNpu(diagnostics::DiagnosticReport& report) {
  diagnostics::DiagnosticCheck check;
  check.name = "npu";

#if defined(ENGINE_ENABLE_XRT)
  try {
    const unsigned int npu_count = xrt::system::enumerate_devices();
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message =
        "Detected " + std::to_string(npu_count) + " accessible XRT device(s)";
    check.details["xrtDeviceCount"] = std::to_string(npu_count);

    for (unsigned int i = 0; i < npu_count; ++i) {
      auto dev = xrt::device(i);
      std::string dev_name = dev.get_info<xrt::info::device::name>();
      check.details["npu" + std::to_string(i) + "_name"] = dev_name;
    }
  } catch (const std::exception& e) {
    check.status = diagnostics::DiagnosticStatus::kWarn;
    check.message = "XRT NPU shim enabled, but device query failed: " +
                    std::string(e.what());
    check.details["xrtError"] = e.what();
    report.AddWarning("XRT NPU device unavailable or inaccessible");
  }
#else
  check.status = diagnostics::DiagnosticStatus::kPass;
  check.message = "XRT NPU backend was not compiled into this binary";
  check.details["xrtSupport"] = "uncompiled";
#endif

  report.AddCheck(std::move(check));
}

void PrintDiagnoseHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " diagnose [OPTIONS]\n\n"
      << "Run non-interactive system and accelerator diagnostics for Strix "
         "Halo.\n\n"
      << "Options:\n"
      << "  --json         Emit structured JSON output conforming to "
         "diagnostics schema v1.0.0\n"
      << "  -h, --help     Print help information\n";
}

}  // namespace

diagnostics::DiagnosticReport CollectDiagnostics() {
  diagnostics::DiagnosticReport report;
  CollectPlatform(report);
  CollectSystem(report);
  CollectToolchain(report);
  CollectGpu(report);
  CollectNpu(report);
  return report;
}

int RunDiagnose(std::span<const char* const> args) {
  bool json_mode = false;
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
    } else {
      std::cerr << "Error: unknown diagnose option '" << arg << "'\n";
      PrintDiagnoseHelp("strix-server");
      return 2;
    }
  }

  const auto report = CollectDiagnostics();

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
