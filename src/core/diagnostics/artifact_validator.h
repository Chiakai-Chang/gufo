#ifndef STRIX_CORE_DIAGNOSTICS_ARTIFACT_VALIDATOR_H_
#define STRIX_CORE_DIAGNOSTICS_ARTIFACT_VALIDATOR_H_

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace strix::diagnostics {

struct ValidationResult {
  bool is_valid{true};
  std::string fingerprint_id;
  std::string schema_version{"1.0.0"};
  std::vector<std::string> errors;
  std::vector<std::string> warnings;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] ValidationResult ValidateArtifactContent(
    std::string_view content);

[[nodiscard]] ValidationResult ValidateArtifactFile(
    const std::filesystem::path& file_path);

}  // namespace strix::diagnostics

#endif  // STRIX_CORE_DIAGNOSTICS_ARTIFACT_VALIDATOR_H_
