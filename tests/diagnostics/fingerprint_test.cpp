#include "src/core/diagnostics/fingerprint.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/diagnostics/artifact_validator.h"
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

void TestSha256Implementation() {
  const std::string input = "hello world";
  const std::string hash = strix::diagnostics::ComputeSha256Hex(input);
  Expect(hash ==
             "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
         "SHA-256 standard test vector matches");
}

void TestCanonicalJsonAndStability() {
  strix::diagnostics::CanonicalFingerprint fp1;
  fp1.cpu_model = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S";
  fp1.cpu_logical_cores = 32;
  fp1.cpu_physical_cores = 16;
  fp1.gpu_name = "AMD Radeon 8060S Graphics";
  fp1.gpu_architecture = "gfx1151";
  fp1.gpu_compute_units = 40;
  fp1.npu_identity = "AMD XDNA2 NPU";
  fp1.npu_architecture = "XDNA2";
  fp1.kernel_release = "7.1.8";

  const std::string id1 = fp1.ComputeFingerprintId();
  Expect(id1.size() == 64, "Fingerprint ID is 64 hex characters");

  strix::diagnostics::CanonicalFingerprint fp2 = fp1;
  const std::string id2 = fp2.ComputeFingerprintId();
  Expect(id1 == id2, "Identical canonical fingerprints produce identical IDs");

  // Modify one field
  fp2.cpu_logical_cores = 16;
  const std::string id3 = fp2.ComputeFingerprintId();
  Expect(id1 != id3, "Field mutation changes fingerprint ID");
}

void TestPrivacyRedaction() {
  const auto inv = strix::diagnostics::CollectSystemInventory();
  const auto fp = strix::diagnostics::GenerateMachineFingerprint(inv);
  const std::string json = fp.ToJson();
  const std::string human = fp.ToHuman();

  Expect(json.find("\"hostname\":") == std::string::npos,
         "No hostname in JSON");
  Expect(json.find("\"username\":") == std::string::npos,
         "No username in JSON");
  Expect(json.find("\"secret\":") == std::string::npos, "No secret in JSON");
  Expect(json.find("/home/") == std::string::npos, "No /home/ paths in JSON");

  Expect(human.find("password") == std::string::npos,
         "No sensitive data in human text");
}

void TestArtifactValidatorGolden() {
  const auto golden_path = FindFixturesRoot() / "fingerprint_v1.json";
  const auto result = strix::diagnostics::ValidateArtifactFile(golden_path);
  if (!result.is_valid) {
    for (const auto& err : result.errors) {
      std::cerr << "Golden validation error: " << err << "\n";
    }
  }
  Expect(result.is_valid, "Golden fingerprint_v1.json passes validation");
  Expect(result.errors.empty(), "Zero errors on golden fixture");
  Expect(result.schema_version == "1.0.0", "Schema version is 1.0.0");
  Expect(!result.fingerprint_id.empty(), "Fingerprint ID is present");
}

void TestArtifactValidatorRejections() {
  // Test empty artifact
  const auto empty_res = strix::diagnostics::ValidateArtifactContent("");
  Expect(!empty_res.is_valid, "Empty artifact is rejected");

  // Test missing fingerprintId
  const std::string missing_fp =
      "{\n  \"schemaVersion\": \"1.0.0\",\n  \"status\": \"PASS\"\n}";
  const auto missing_fp_res =
      strix::diagnostics::ValidateArtifactContent(missing_fp);
  Expect(!missing_fp_res.is_valid, "Missing fingerprintId is rejected");

  // Test hash mismatch
  const std::string hash_mismatch =
      "{\n"
      "  \"schemaVersion\": \"1.0.0\",\n"
      "  \"fingerprintId\": "
      "\"0000000000000000000000000000000000000000000000000000000000000000\",\n"
      "  \"canonical\": {\n"
      "    \"cppStandard\": \"C++20\",\n"
      "    \"cpuArchitecture\": \"x86_64\",\n"
      "    \"cpuLogicalCores\": 32,\n"
      "    \"cpuModel\": \"AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\",\n"
      "    \"cpuPhysicalCores\": 16,\n"
      "    \"cxxCompiler\": \"GCC 15.3.0\",\n"
      "    \"gpuArchitecture\": \"gfx1151\",\n"
      "    \"gpuComputeUnits\": 40,\n"
      "    \"gpuDriver\": \"amdgpu\",\n"
      "    \"gpuName\": \"AMD Radeon 8060S Graphics\",\n"
      "    \"gpuPciId\": \"1002:1586\",\n"
      "    \"kernelRelease\": \"7.1.8\",\n"
      "    \"memoryTotalBytes\": 134309888000,\n"
      "    \"memoryType\": \"LPDDR5X (Unified)\",\n"
      "    \"npuArchitecture\": \"XDNA2\",\n"
      "    \"npuDriver\": \"amdxdna\",\n"
      "    \"npuFirmwareVersion\": \"npu.sbin (1.1.2.64/65)\",\n"
      "    \"npuIdentity\": \"AMD XDNA2 NPU\",\n"
      "    \"npuPciId\": \"1022:17f0\",\n"
      "    \"rocmVersion\": \"7.2.3\",\n"
      "    \"schemaVersion\": \"1.0.0\",\n"
      "    \"xrtCommit\": \"8661761775a266b11992a3bd6eb08209d88aa845\"\n"
      "  }\n"
      "}";
  const auto mismatch_res =
      strix::diagnostics::ValidateArtifactContent(hash_mismatch);
  Expect(!mismatch_res.is_valid, "Hash mismatch is rejected");

  // Test unsupported GPU architecture
  const std::string wrong_arch =
      "{\n"
      "  \"schemaVersion\": \"1.0.0\",\n"
      "  \"fingerprintId\": \"dummy\",\n"
      "  \"gpuArchitecture\": \"gfx1100\"\n"
      "}";
  const auto arch_res = strix::diagnostics::ValidateArtifactContent(wrong_arch);
  Expect(!arch_res.is_valid, "Incompatible GPU architecture is rejected");
}

}  // namespace

int main() {
  std::cout
      << "Running machine fingerprint & artifact validator test suite...\n";

  TestSha256Implementation();
  TestCanonicalJsonAndStability();
  TestPrivacyRedaction();
  TestArtifactValidatorGolden();
  TestArtifactValidatorRejections();

  std::cout << "All machine fingerprint & artifact validator tests passed "
               "successfully.\n";
  return 0;
}
