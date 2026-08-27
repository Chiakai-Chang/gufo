#include "src/core/xdna2/smoke.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef ENGINE_ENABLE_XRT
#include <gufo/aie_smoke_manifest.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#endif

namespace gufo::xdna2 {

namespace {

constexpr std::size_t kElementCount = 1024;

std::string EscapeJson(std::string_view value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        output << character;
        break;
    }
  }
  return output.str();
}

std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

#ifdef ENGINE_ENABLE_XRT

std::string ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

#endif

void SetFailure(XrtSmokeReport& report, std::string category,
                std::string detected, std::string required,
                std::string remediation) {
  report.status = "failed";
  report.completion_status = category;
  report.failure.category = std::move(category);
  report.failure.detected = std::move(detected);
  report.failure.required = std::move(required);
  report.failure.remediation = std::move(remediation);
}

#ifdef ENGINE_ENABLE_XRT

void ComputeCommandStatistics(XrtSmokeReport& report) {
  if (report.command_us.empty()) {
    return;
  }
  auto sorted = report.command_us;
  std::ranges::sort(sorted);
  const auto size = sorted.size();
  if (size % 2 == 0) {
    report.median_command_us =
        (sorted[(size / 2) - 1] + sorted[size / 2]) / 2.0;
  } else {
    report.median_command_us = sorted[size / 2];
  }
  const auto p95_index = std::min(
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(size))) - 1,
      size - 1);
  report.p95_command_us = sorted[p95_index];
}

std::filesystem::path DefaultProgramDir() {
#ifdef GUFO_AIE_SMOKE_PROGRAM_DIR
  return GUFO_AIE_SMOKE_PROGRAM_DIR;
#else
  return {};
#endif
}

bool ValidateProgram(const std::filesystem::path& program_dir,
                     XrtSmokeReport& report) {
  const auto xclbin_path = program_dir / "smoke.xclbin";
  const auto elf_path = program_dir / "smoke.insts.elf";
  const std::string xclbin = ReadBinaryFile(xclbin_path);
  const std::string elf = ReadBinaryFile(elf_path);
  if (xclbin.empty() || elf.empty()) {
    SetFailure(report, "program_missing",
               "required XCLBIN or instruction ELF is absent",
               "Nix-packaged smoke.xclbin and smoke.insts.elf",
               "Rebuild the default Nix package to install reviewed artifacts");
    return false;
  }

  report.program_sha256 =
      diagnostics::ComputeSha256Hex(std::string(xclbin).append(elf));
  report.xclbin_sha256 = diagnostics::ComputeSha256Hex(xclbin);
  report.elf_sha256 = diagnostics::ComputeSha256Hex(elf);

  const bool target_ok = generated::kTarget == report.target;
  const bool abi_ok = generated::kAbi == report.abi;
  const bool hashes_ok = generated::kProgramSha256 == report.program_sha256 &&
                         generated::kXclbinSha256 == report.xclbin_sha256 &&
                         generated::kElfSha256 == report.elf_sha256;
  if (!target_ok || !abi_ok || !hashes_ok) {
    SetFailure(
        report, "program_incompatible",
        "artifact target, ABI, or content hash does not match manifest",
        "npu2 / xrt-elf-v1 with pinned SHA-256 values",
        "Discard the artifact and rebuild it from the checked-in source");
    return false;
  }
  return true;
}

std::string SelectKernelName(const xrt::xclbin& xclbin) {
  const auto kernels = xclbin.get_kernels();
  const auto match =
      std::ranges::find_if(kernels, [](const xrt::xclbin::kernel& kernel) {
        return kernel.get_name().starts_with("MLIR_AIE");
      });
  if (match != kernels.end()) {
    return match->get_name();
  }
  if (kernels.size() == 1) {
    return kernels.front().get_name();
  }
  return {};
}

#endif

}  // namespace

bool XrtSmokeReport::Success() const {
  return status == "completed" && completed_iterations == iterations &&
         !quarantined && !first_mismatch.has_value();
}

std::string XrtSmokeReport::ToJson() const {
  std::ostringstream output;
  output << "{\n";
  output << "  \"schemaVersion\": \"" << EscapeJson(schema_version) << "\",\n";
  output << "  \"artifactType\": \"" << EscapeJson(artifact_type) << "\",\n";
  output << "  \"fingerprintId\": \"" << EscapeJson(fingerprint_id) << "\",\n";
  output << "  \"canonical\": " << canonical.CanonicalJson() << ",\n";
  output << "  \"timestamp\": \"" << EscapeJson(timestamp) << "\",\n";
  output << "  \"program\": {\n";
  output << "    \"target\": \"" << EscapeJson(target) << "\",\n";
  output << "    \"abi\": \"" << EscapeJson(abi) << "\",\n";
  output << "    \"programSha256\": \"" << EscapeJson(program_sha256)
         << "\",\n";
  output << "    \"xclbinSha256\": \"" << EscapeJson(xclbin_sha256) << "\",\n";
  output << "    \"elfSha256\": \"" << EscapeJson(elf_sha256) << "\"\n";
  output << "  },\n";
  output << "  \"context\": {\n";
  output << "    \"deviceCount\": " << device.device_count << ",\n";
  output << "    \"deviceIndex\": " << device.device_index << ",\n";
  output << "    \"deviceName\": \"" << EscapeJson(device.name) << "\",\n";
  output << "    \"npuArchitecture\": \"" << EscapeJson(device.architecture)
         << "\",\n";
  output << "    \"driver\": \"" << EscapeJson(device.driver) << "\",\n";
  output << "    \"firmware\": \"" << EscapeJson(device.firmware) << "\",\n";
  output << "    \"xclbinUuid\": \"" << EscapeJson(xclbin_uuid) << "\",\n";
  output << "    \"kernelName\": \"" << EscapeJson(kernel_name) << "\",\n";
  output << "    \"contextMode\": \"" << EscapeJson(context_mode) << "\",\n";
  output << "    \"partitionColumns\": " << partition_columns << ",\n";
  output << "    \"resourceEvidence\": \"" << EscapeJson(resource_evidence)
         << "\",\n";
  output << "    \"inputBytes\": " << input_bytes << ",\n";
  output << "    \"outputBytes\": " << output_bytes << ",\n";
  output << "    \"boAllocations\": " << bo_allocations << ",\n";
  output << "    \"buffersReused\": " << (buffers_reused ? "true" : "false")
         << "\n";
  output << "  },\n";
  output << "  \"result\": {\n";
  output << "    \"status\": \"" << EscapeJson(status) << "\",\n";
  output << "    \"completionStatus\": \"" << EscapeJson(completion_status)
         << "\",\n";
  output << "    \"iterations\": " << iterations << ",\n";
  output << "    \"completedIterations\": " << completed_iterations << ",\n";
  output << "    \"timeoutMs\": " << timeout_ms << ",\n";
  output << "    \"quarantined\": " << (quarantined ? "true" : "false")
         << ",\n";
  output << "    \"firstMismatch\": ";
  if (first_mismatch.has_value()) {
    output << *first_mismatch;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "    \"setupMs\": " << std::fixed << std::setprecision(3)
         << setup_ms << ",\n";
  output << "    \"medianCommandUs\": " << median_command_us << ",\n";
  output << "    \"p95CommandUs\": " << p95_command_us << "\n";
  output << "  },\n";
  output << "  \"failure\": {\n";
  output << "    \"category\": \"" << EscapeJson(failure.category) << "\",\n";
  output << "    \"detected\": \"" << EscapeJson(failure.detected) << "\",\n";
  output << "    \"required\": \"" << EscapeJson(failure.required) << "\",\n";
  output << "    \"remediation\": \"" << EscapeJson(failure.remediation)
         << "\"\n";
  output << "  }\n";
  output << "}\n";
  return output.str();
}

std::string XrtSmokeReport::ToHuman() const {
  std::ostringstream output;
  output << "=== XDNA2 XRT Deterministic Smoke ===\n";
  output << "Status          : " << status << "\n";
  output << "Device          : " << device.name << " [" << device.architecture
         << "]\n";
  output << "Program SHA-256 : " << program_sha256 << "\n";
  output << "Kernel          : " << kernel_name << "\n";
  output << "Iterations      : " << completed_iterations << "/" << iterations
         << "\n";
  output << "Median command  : " << std::fixed << std::setprecision(2)
         << median_command_us << " us\n";
  if (!failure.category.empty()) {
    output << "Failure          : " << failure.category << " ("
           << failure.detected << ")\n";
    output << "Remediation      : " << failure.remediation << "\n";
  }
  return output.str();
}

XrtSmokeReport RunXrtSmoke(const XrtSmokeOptions& options,
                           const diagnostics::SystemInventory& inventory,
                           const diagnostics::MachineFingerprint& fingerprint) {
  XrtSmokeReport report;
  report.fingerprint_id = fingerprint.fingerprint_id;
  report.canonical = fingerprint.canonical;
  report.timestamp = CurrentIso8601Utc();
  report.iterations = options.iterations;
  report.timeout_ms = options.timeout_ms;
  report.input_bytes = kElementCount * sizeof(std::int32_t);
  report.output_bytes = kElementCount * sizeof(std::int32_t);
  report.device = DiscoverXrtDevice(options.device_index, inventory);

  if (options.iterations == 0 || options.timeout_ms == 0) {
    SetFailure(report, "invalid_options", "zero iterations or timeout",
               "iterations and timeoutMs greater than zero",
               "Provide bounded positive values");
    return report;
  }
  if (!report.device.available) {
    SetFailure(report, report.device.error_category, report.device.detected,
               report.device.required, report.device.remediation);
    return report;
  }

#ifdef ENGINE_ENABLE_XRT
  const auto program_dir =
      options.program_dir.empty() ? DefaultProgramDir() : options.program_dir;
  if (!ValidateProgram(program_dir, report)) {
    return report;
  }

  const auto setup_start = std::chrono::steady_clock::now();
  try {
    const auto xclbin_path = program_dir / "smoke.xclbin";
    const auto elf_path = program_dir / "smoke.insts.elf";

    xrt::device device(options.device_index);
    xrt::xclbin xclbin(xclbin_path.string());
    report.kernel_name = SelectKernelName(xclbin);
    if (report.kernel_name.empty()) {
      SetFailure(report, "program_incompatible",
                 "XCLBIN has no unique MLIR_AIE kernel",
                 "one reviewed MLIR_AIE kernel entry",
                 "Rebuild the smoke artifact with the pinned MLIR-AIE package");
      return report;
    }

    device.register_xclbin(xclbin);
    report.xclbin_uuid = xclbin.get_uuid().to_string();
    xrt::hw_context context(device, xclbin.get_uuid());
    report.partition_columns = generated::kPartitionColumns;
    report.resource_evidence =
        "generated from XCLBIN AIE_PARTITION.column_width";
    if (report.partition_columns == 0) {
      SetFailure(report, "program_incompatible",
                 "XCLBIN manifest declares a zero-column partition",
                 "positive XDNA2 partition size",
                 "Rebuild the smoke artifact with the pinned MLIR-AIE package");
      return report;
    }
    report.context_mode =
        context.get_mode() == xrt::hw_context::access_mode::shared
            ? "shared"
            : "exclusive";
    xrt::elf elf(elf_path.string());
    xrt::module module(elf);
    xrt::ext::kernel kernel(context, module, report.kernel_name);

    xrt::bo input_bo = xrt::ext::bo(device, report.input_bytes);
    xrt::bo output_bo = xrt::ext::bo(device, report.output_bytes);
    report.bo_allocations = 2;
    report.buffers_reused = true;

    auto* input = input_bo.map<std::int32_t*>();
    auto* output = output_bo.map<std::int32_t*>();
    if (input == nullptr || output == nullptr) {
      SetFailure(report, "buffer_failure", "XRT returned a null BO mapping",
                 "two mapped host-visible BOs",
                 "Inspect XRT BO allocation and device memory availability");
      return report;
    }

    report.setup_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - setup_start)
                          .count();

    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      const std::int32_t base =
          static_cast<std::int32_t>((iteration + 1) * 100000);
      for (std::size_t index = 0; index < kElementCount; ++index) {
        input[index] = base + static_cast<std::int32_t>(index);
        output[index] = std::numeric_limits<std::int32_t>::min() +
                        static_cast<std::int32_t>(index);
      }
      input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      const auto command_start = std::chrono::steady_clock::now();
      auto run = kernel(3U, 0U, 0U, input_bo, output_bo);
      const auto wait_status =
          run.wait2(std::chrono::milliseconds(options.timeout_ms));
      if (wait_status == std::cv_status::timeout) {
        run.abort();
        report.quarantined = true;
        SetFailure(report, "timeout",
                   "command exceeded " + std::to_string(options.timeout_ms) +
                       " ms and the context was quarantined",
                   "finite successful completion",
                   "Reset the NPU context before running further commands");
        return report;
      }
      report.command_us.push_back(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - command_start)
              .count());

      output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      for (std::size_t index = 0; index < kElementCount; ++index) {
        const auto expected = base + static_cast<std::int32_t>(index) + 1;
        if (output[index] != expected) {
          report.first_mismatch = index;
          SetFailure(report, "output_mismatch",
                     "iteration " + std::to_string(iteration) + ", index " +
                         std::to_string(index) + ", detected " +
                         std::to_string(output[index]),
                     std::to_string(expected),
                     "Quarantine the artifact and inspect NPU program/runtime "
                     "compatibility");
          return report;
        }
      }
      ++report.completed_iterations;
    }

    ComputeCommandStatistics(report);
    report.status = "completed";
    report.completion_status = "completed";
    return report;
  } catch (const xrt::run::command_error& error) {
    SetFailure(report, "command_error", error.what(), "ERT_CMD_STATE_COMPLETED",
               "Inspect firmware logs and do not reuse the failed context");
    return report;
  } catch (const std::exception& error) {
    SetFailure(report, "xrt_error", error.what(),
               "successful XRT context, BO, and command lifecycle",
               "Run NPU diagnostics and verify the pinned driver/plugin stack");
    return report;
  }
#else
  SetFailure(report, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT",
             "Nix package with XRT enabled",
             "Build the supported default Nix package");
  return report;
#endif
}

}  // namespace gufo::xdna2
