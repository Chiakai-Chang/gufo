#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/core/hip/detail/hipblaslt_plan_database.hpp"

namespace {

using gufo::hip::detail::HipblasLtPlanDatabase;
using gufo::hip::detail::HipblasLtPlanDatabaseKey;
using gufo::hip::detail::HipblasLtPlanDatabaseLoadStatus;
using gufo::hip::detail::HipblasLtPlanDataType;
using gufo::hip::detail::HipblasLtPlanRecord;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

HipblasLtPlanDatabase MakeDatabase() {
  return {
      .key =
          {
              .hardware_fingerprint = "gfx1151-test",
              .hip_runtime_version = 70203000,
              .hipblaslt_version = 10202,
          },
      .records =
          {
              {
                  .batch_size = 128,
                  .m = 17408,
                  .k = 5120,
                  .data_type = HipblasLtPlanDataType::kBfloat16,
                  .algorithm_id = 42,
                  .algorithm_blob = {1, 2, 3, 4},
                  .workspace_bytes = 4 * 1024 * 1024,
                  .median_nanoseconds = 125000,
                  .solution_name = "solution-42",
                  .kernel_name = "kernel-42",
              },
          },
  };
}

}  // namespace

int main() {
  const auto temporary_directory =
      std::filesystem::temp_directory_path() / "gufo-hipblaslt-cache-test";
  std::filesystem::create_directories(temporary_directory);
  const auto database_path = temporary_directory / "plans.bin";
  const auto truncated_path = temporary_directory / "truncated.bin";

  const auto expected = MakeDatabase();
  std::string error;
  Expect(gufo::hip::detail::SaveHipblasLtPlanDatabase(database_path, expected,
                                                      &error),
         "save database: " + error);

  const auto loaded =
      gufo::hip::detail::LoadHipblasLtPlanDatabase(database_path, expected.key);
  Expect(loaded.status == HipblasLtPlanDatabaseLoadStatus::kLoaded,
         "load compatible database");
  Expect(loaded.database.key == expected.key, "round-trip database key");
  Expect(loaded.database.records == expected.records,
         "round-trip database records");

  const auto* record = gufo::hip::detail::FindHipblasLtPlanRecord(
      loaded.database, 128, 17408, 5120, HipblasLtPlanDataType::kBfloat16);
  Expect(record != nullptr && record->algorithm_id == 42,
         "lookup exact GEMM shape");
  Expect(gufo::hip::detail::FindHipblasLtPlanRecord(
             loaded.database, 256, 17408, 5120,
             HipblasLtPlanDataType::kBfloat16) == nullptr,
         "reject unrecorded GEMM shape");

  auto incompatible_key = expected.key;
  ++incompatible_key.hip_runtime_version;
  const auto incompatible = gufo::hip::detail::LoadHipblasLtPlanDatabase(
      database_path, incompatible_key);
  Expect(incompatible.status == HipblasLtPlanDatabaseLoadStatus::kIncompatible,
         "detect ROCm runtime change");

  {
    std::ofstream truncated(truncated_path, std::ios::binary);
    truncated.write("STRIXLT", 7);
  }
  const auto invalid = gufo::hip::detail::LoadHipblasLtPlanDatabase(
      truncated_path, expected.key);
  Expect(invalid.status == HipblasLtPlanDatabaseLoadStatus::kInvalid,
         "reject truncated database");

  auto duplicate = expected;
  duplicate.records.push_back(duplicate.records.front());
  Expect(!gufo::hip::detail::SaveHipblasLtPlanDatabase(
             temporary_directory / "duplicate.bin", duplicate, &error),
         "reject duplicate shape records");

  std::filesystem::remove_all(temporary_directory);
  std::cout << "PASS: hipBLASLt plan database tests\n";
  return 0;
}
