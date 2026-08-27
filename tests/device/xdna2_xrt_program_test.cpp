#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/xdna2/smoke.h"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const auto inventory = gufo::diagnostics::CollectSystemInventory(
      gufo::diagnostics::LinuxSysfs());
  const auto fingerprint =
      gufo::diagnostics::GenerateMachineFingerprint(inventory);

  auto unsupported_driver_inventory = inventory;
  unsupported_driver_inventory.npu.driver_name = "unsupported";
  const auto unsupported_driver =
      gufo::xdna2::DiscoverXrtDevice(0, unsupported_driver_inventory);
  Expect(unsupported_driver.error_category == "driver_unsupported",
         "unsupported drivers are distinguished");

  auto unsupported_firmware_inventory = inventory;
  unsupported_firmware_inventory.npu.firmware_version = "unsupported";
  const auto unsupported_firmware =
      gufo::xdna2::DiscoverXrtDevice(0, unsupported_firmware_inventory);
  Expect(unsupported_firmware.error_category == "firmware_unsupported",
         "unsupported firmware is distinguished");

  gufo::xdna2::XrtSmokeOptions options;
  options.iterations = 2;
  options.timeout_ms = 30000;
#ifdef GUFO_AIE_SMOKE_PROGRAM_DIR
  options.program_dir = GUFO_AIE_SMOKE_PROGRAM_DIR;
#endif

  auto invalid_options = options;
  invalid_options.iterations = 0;
  const auto invalid_report =
      gufo::xdna2::RunXrtSmoke(invalid_options, inventory, fingerprint);
  Expect(invalid_report.failure.category == "invalid_options",
         "zero iterations fail with invalid_options");

  auto missing_program = options;
  missing_program.iterations = 1;
  missing_program.program_dir =
      std::filesystem::temp_directory_path() /
      ("gufo-xdna2-program-missing-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto missing_report =
      gufo::xdna2::RunXrtSmoke(missing_program, inventory, fingerprint);
  Expect(missing_report.failure.category == "program_missing",
         "missing artifacts fail with program_missing");

  const auto report = gufo::xdna2::RunXrtSmoke(options, inventory, fingerprint);
  if (!report.Success()) {
    std::cerr << report.ToHuman();
    return 1;
  }

  Expect(report.completed_iterations == options.iterations,
         "all requested iterations completed");
  Expect(report.bo_allocations == 2, "exactly two BOs were allocated");
  Expect(report.buffers_reused, "BOs were reused");
  Expect(report.partition_columns > 0,
         "context uses the program-declared partition");
  Expect(report.context_mode == "shared", "context uses shared access mode");
  Expect(!report.quarantined, "context was not quarantined");
  Expect(!report.first_mismatch.has_value(), "output matched exactly");

  const auto validation =
      gufo::diagnostics::ValidateArtifactContent(report.ToJson());
  Expect(validation.is_valid, "XDNA2 program JSON validates");

  std::cout << report.ToHuman();
  return 0;
}
