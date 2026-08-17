#include "src/core/diagnostics/bandwidth.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"

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
      "tests/fixtures/diagnostics",
      "../tests/fixtures/diagnostics",
      "../../tests/fixtures/diagnostics",
  };
  for (const auto& path : paths) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return "tests/fixtures/diagnostics";
}

void TestCpuBandwidthCorrectness() {
  strix::diagnostics::BandwidthOptions opts;
  opts.backends = {"cpu"};
  opts.warmup = 1;
  opts.repetitions = 3;
  opts.working_set_bytes = 16ULL * 1024ULL * 1024ULL;  // 16 MiB for fast test

  const auto results = strix::diagnostics::MeasureCpuBandwidth(opts);
  Expect(results.size() == 3,
         "Expected 3 CPU bandwidth paths (copy, read, write)");

  for (const auto& r : results) {
    Expect(r.backend == "cpu", "Backend is cpu");
    Expect(r.sentinel_verified, "Sentinel bytes verified");
    Expect(r.status == "completed", "Status is completed");
    Expect(r.raw_repetitions_gbps.size() == 3, "Recorded 3 repetitions");
    Expect(r.median_gbps > 0.0, "Median GB/s > 0");
    Expect(r.min_gbps <= r.median_gbps, "min <= median");
    Expect(r.median_gbps <= r.max_gbps, "median <= max");
    Expect(r.p95_gbps >= r.median_gbps, "p95 >= median");
  }
}

void TestBandwidthReportJsonStructure() {
  const auto inv = strix::diagnostics::CollectSystemInventory();
  const auto fp = strix::diagnostics::GenerateMachineFingerprint(inv);

  strix::diagnostics::BandwidthOptions opts;
  opts.backends = {"cpu"};
  opts.warmup = 1;
  opts.repetitions = 2;
  opts.working_set_bytes = 8ULL * 1024ULL * 1024ULL;

  const auto report = strix::diagnostics::RunBandwidthBenchmark(opts, fp);
  const std::string json = report.ToJson();

  Expect(json.find("\"schemaVersion\": \"1.0.0\"") != std::string::npos,
         "Has schemaVersion");
  Expect(json.find("\"fingerprintId\": \"" + fp.fingerprint_id + "\"") !=
             std::string::npos,
         "Has matching fingerprintId");
  Expect(json.find("\"paths\":") != std::string::npos, "Has paths array");
  Expect(json.find("\"cpu_copy\"") != std::string::npos, "Has cpu_copy path");
  Expect(json.find("\"rawRepetitionsGbps\":") != std::string::npos,
         "Has rawRepetitionsGbps");

  // Validate artifact
  const auto val_res = strix::diagnostics::ValidateArtifactContent(json);
  Expect(val_res.is_valid, "BandwidthReport JSON is valid artifact");
}

void TestSchemaFileExists() {
  const auto schema_path = FindFixturesRoot() / "bandwidth_schema.json";
  Expect(std::filesystem::exists(schema_path),
         "bandwidth_schema.json fixture exists");
  std::ifstream f(schema_path);
  Expect(f.is_open(), "Can open bandwidth_schema.json");
  std::string line;
  std::getline(f, line);
  Expect(!line.empty(), "bandwidth_schema.json is non-empty");
}

}  // namespace

int main() {
  std::cout << "Running memory bandwidth benchmark test suite...\n";

  TestCpuBandwidthCorrectness();
  TestBandwidthReportJsonStructure();
  TestSchemaFileExists();

  std::cout << "All memory bandwidth tests passed successfully.\n";
  return 0;
}
