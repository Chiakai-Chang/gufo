#ifndef GUFO_CORE_XDNA2_SMOKE_H_
#define GUFO_CORE_XDNA2_SMOKE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/xdna2/device.h"

namespace gufo::xdna2 {

struct XrtSmokeOptions {
  std::uint32_t iterations{100};
  std::uint32_t timeout_ms{30000};
  std::uint32_t device_index{0};
  std::filesystem::path program_dir;
};

struct XrtSmokeFailure {
  std::string category;
  std::string detected;
  std::string required;
  std::string remediation;
};

struct XrtSmokeReport {
  std::string schema_version{"1.0.0"};
  std::string artifact_type{"xrtSmoke"};
  std::string fingerprint_id;
  diagnostics::CanonicalFingerprint canonical;
  std::string timestamp;
  std::string target{"npu2"};
  std::string abi{"xrt-elf-v1"};
  std::string program_sha256;
  std::string xclbin_sha256;
  std::string elf_sha256;
  XrtDeviceInfo device;
  std::string xclbin_uuid;
  std::string kernel_name;
  std::string context_mode;
  std::uint32_t partition_columns{0};
  std::string resource_evidence;
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
  std::uint32_t iterations{0};
  std::uint32_t completed_iterations{0};
  std::uint32_t timeout_ms{0};
  std::uint32_t bo_allocations{0};
  bool buffers_reused{false};
  bool quarantined{false};
  std::string status{"failed"};
  std::string completion_status{"not_started"};
  std::optional<std::size_t> first_mismatch;
  double setup_ms{0.0};
  double median_command_us{0.0};
  double p95_command_us{0.0};
  std::vector<double> command_us;
  XrtSmokeFailure failure;

  [[nodiscard]] bool Success() const;
  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] XrtSmokeReport RunXrtSmoke(
    const XrtSmokeOptions& options,
    const diagnostics::SystemInventory& inventory,
    const diagnostics::MachineFingerprint& fingerprint);

}  // namespace gufo::xdna2

#endif  // GUFO_CORE_XDNA2_SMOKE_H_
