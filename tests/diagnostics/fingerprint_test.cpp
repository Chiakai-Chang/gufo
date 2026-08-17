#include "src/core/diagnostics/fingerprint.h"

#include <cstdlib>
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

void TestSha256Implementation() {
  const std::string input = "hello world";
  const std::string hash = strix::diagnostics::ComputeSha256Hex(input);
  Expect(hash ==
             "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
         "SHA-256 standard test vector matches");
}

void TestCanonicalJsonAndStability() {
  strix::diagnostics::CanonicalFingerprint fp1;
  fp1.cpu_model = "Generic AMD Strix Halo APU";
  fp1.cpu_logical_cores = 32;
  fp1.cpu_physical_cores = 16;
  fp1.gpu_name = "Generic Radeon Graphics";
  fp1.gpu_architecture = "gfx1151";
  fp1.gpu_compute_units = 40;
  fp1.npu_identity = "Generic XDNA2 NPU";
  fp1.npu_architecture = "XDNA2";
  fp1.kernel_release = "6.14.0";

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

void TestArtifactValidatorWithValidFingerprint() {
  strix::diagnostics::SystemInventory mock_inv;
  mock_inv.cpu.architecture = "x86_64";
  mock_inv.cpu.logical_cores = 32;
  mock_inv.cpu.physical_cores = 16;
  mock_inv.gpu.architecture = "gfx1151";
  mock_inv.gpu.compute_units = 40;
  mock_inv.npu.architecture = "XDNA2";

  const auto fp = strix::diagnostics::GenerateMachineFingerprint(mock_inv);
  const std::string json = fp.ToJson();

  const auto result = strix::diagnostics::ValidateArtifactContent(json);
  Expect(result.is_valid, "Valid generated fingerprint passes validation");
  Expect(result.errors.empty(), "Zero errors on valid artifact");
  Expect(result.schema_version == "1.0.0", "Schema version is 1.0.0");
  Expect(result.fingerprint_id == fp.fingerprint_id, "Fingerprint ID matches");
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
      "    \"cpuModel\": \"Generic Strix Halo CPU\",\n"
      "    \"cpuPhysicalCores\": 16,\n"
      "    \"cxxCompiler\": \"GCC 15.0.0\",\n"
      "    \"gpuArchitecture\": \"gfx1151\",\n"
      "    \"gpuComputeUnits\": 40,\n"
      "    \"gpuDriver\": \"amdgpu\",\n"
      "    \"gpuName\": \"Generic GPU\",\n"
      "    \"gpuPciId\": \"1002:1586\",\n"
      "    \"kernelRelease\": \"6.14.0\",\n"
      "    \"memoryTotalBytes\": 134309888000,\n"
      "    \"memoryType\": \"LPDDR5X (Unified)\",\n"
      "    \"npuArchitecture\": \"XDNA2\",\n"
      "    \"npuDriver\": \"amdxdna\",\n"
      "    \"npuFirmwareVersion\": \"npu.sbin\",\n"
      "    \"npuIdentity\": \"AMD XDNA2 NPU\",\n"
      "    \"npuPciId\": \"1022:17f0\",\n"
      "    \"rocmVersion\": \"7.0.0\",\n"
      "    \"schemaVersion\": \"1.0.0\",\n"
      "    \"xrtCommit\": \"abcdef0123456789\"\n"
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
  TestArtifactValidatorWithValidFingerprint();
  TestArtifactValidatorRejections();

  std::cout << "All machine fingerprint & artifact validator tests passed "
               "successfully.\n";
  return 0;
}
