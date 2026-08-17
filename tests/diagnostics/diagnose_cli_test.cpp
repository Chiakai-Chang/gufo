#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "src/cli/diagnose.h"
#include "src/core/diagnostics/report.h"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void TestReportSchemaAndDefaults() {
  const strix::diagnostics::DiagnosticReport report;
  Expect(report.SchemaVersion() == "1.0.0", "SchemaVersion == 1.0.0");
  Expect(report.Status() == strix::diagnostics::DiagnosticStatus::kPass,
         "Status == kPass");
  Expect(!report.Timestamp().empty(), "Timestamp not empty");

  const std::string json = report.ToJson();
  Expect(json.find("\"schemaVersion\": \"1.0.0\"") != std::string::npos,
         "schemaVersion in json");
  Expect(json.find("\"status\": \"PASS\"") != std::string::npos,
         "status in json");
  Expect(json.find("\"checks\": [") != std::string::npos, "checks in json");
  Expect(json.find("\"warnings\": [") != std::string::npos, "warnings in json");
  Expect(json.find("\"errors\": [") != std::string::npos, "errors in json");

  const std::string human = report.ToHuman();
  Expect(
      human.find("=== Strix Halo Engine Diagnostics ===") != std::string::npos,
      "title in human output");
  Expect(human.find("Schema Version  : 1.0.0") != std::string::npos,
         "schema in human output");
}

void TestReportStatusTransitions() {
  strix::diagnostics::DiagnosticReport report;
  Expect(report.Status() == strix::diagnostics::DiagnosticStatus::kPass,
         "initial kPass");

  report.AddWarning("Non-fatal test warning");
  Expect(report.Status() == strix::diagnostics::DiagnosticStatus::kWarn,
         "transition to kWarn");
  Expect(report.Warnings().size() == 1, "warning count == 1");

  report.AddError("Fatal test error");
  Expect(report.Status() == strix::diagnostics::DiagnosticStatus::kFail,
         "transition to kFail");
  Expect(report.Errors().size() == 1, "error count == 1");

  const std::string json = report.ToJson();
  Expect(json.find("\"status\": \"FAIL\"") != std::string::npos,
         "FAIL in json status");
  Expect(json.find("Non-fatal test warning") != std::string::npos,
         "warning in json");
  Expect(json.find("Fatal test error") != std::string::npos, "error in json");
}

void TestCollectDiagnostics() {
  const auto report = strix::cli::CollectDiagnostics();
  Expect(!report.Checks().empty(), "Checks not empty");

  bool has_platform = false;
  bool has_system = false;
  bool has_toolchain = false;

  for (const auto& check : report.Checks()) {
    if (check.name == "platform") {
      has_platform = true;
      Expect(check.status == strix::diagnostics::DiagnosticStatus::kPass,
             "platform check pass");
    } else if (check.name == "system") {
      has_system = true;
    } else if (check.name == "toolchain") {
      has_toolchain = true;
      Expect(check.status == strix::diagnostics::DiagnosticStatus::kPass,
             "toolchain check pass");
    }
  }

  Expect(has_platform, "has platform check");
  Expect(has_system, "has system check");
  Expect(has_toolchain, "has toolchain check");
}

void TestCliHelpAndInvalidOptions() {
  // Test --help
  const std::array<const char*, 2> help_args{"diagnose", "--help"};
  const int help_rc = strix::cli::RunDiagnose(help_args);
  Expect(help_rc == 0, "--help rc == 0");

  // Test -h
  const std::array<const char*, 2> h_args{"diagnose", "-h"};
  const int h_rc = strix::cli::RunDiagnose(h_args);
  Expect(h_rc == 0, "-h rc == 0");

  // Test invalid option
  const std::array<const char*, 2> invalid_args{"diagnose", "--unknown-flag"};
  const int invalid_rc = strix::cli::RunDiagnose(invalid_args);
  Expect(invalid_rc == 2, "invalid flag rc == 2");
}

void TestCliExecution() {
  // Test human output
  const std::array<const char*, 1> default_args{"diagnose"};
  const int def_rc = strix::cli::RunDiagnose(default_args);
  Expect(def_rc == 0 || def_rc == 1, "default diagnose execution");

  // Test JSON output
  const std::array<const char*, 2> json_args{"diagnose", "--json"};
  const int json_rc = strix::cli::RunDiagnose(json_args);
  Expect(json_rc == 0 || json_rc == 1, "json diagnose execution");
}

}  // namespace

int main() {
  std::cout << "Running diagnose CLI & report test suite...\n";

  TestReportSchemaAndDefaults();
  TestReportStatusTransitions();
  TestCollectDiagnostics();
  TestCliHelpAndInvalidOptions();
  TestCliExecution();

  std::cout << "All diagnose CLI tests passed successfully.\n";
  return 0;
}
