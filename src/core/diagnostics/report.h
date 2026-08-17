#ifndef STRIX_CORE_DIAGNOSTICS_REPORT_H_
#define STRIX_CORE_DIAGNOSTICS_REPORT_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace strix::diagnostics {

enum class DiagnosticStatus : std::uint8_t {
  kPass,
  kWarn,
  kFail,
};

[[nodiscard]] std::string_view ToString(DiagnosticStatus status);
[[nodiscard]] DiagnosticStatus FromString(std::string_view str);

struct DiagnosticCheck {
  std::string name;
  DiagnosticStatus status{DiagnosticStatus::kPass};
  std::string message;
  std::map<std::string, std::string> details;
};

class DiagnosticReport {
public:
  DiagnosticReport();

  void SetEngineRevision(std::string_view revision);
  void SetTimestamp(std::string_view iso8601_utc);
  void AddCheck(DiagnosticCheck check);
  void AddWarning(std::string_view warning);
  void AddError(std::string_view error);

  [[nodiscard]] std::string_view SchemaVersion() const {
    return schema_version_;
  }
  [[nodiscard]] std::string_view EngineRevision() const {
    return engine_revision_;
  }
  [[nodiscard]] std::string_view Timestamp() const { return timestamp_; }
  [[nodiscard]] DiagnosticStatus Status() const { return overall_status_; }
  [[nodiscard]] const std::vector<DiagnosticCheck>& Checks() const {
    return checks_;
  }
  [[nodiscard]] const std::vector<std::string>& Warnings() const {
    return warnings_;
  }
  [[nodiscard]] const std::vector<std::string>& Errors() const {
    return errors_;
  }

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;

private:
  void RecomputeStatus();

  std::string schema_version_{"1.0.0"};
  std::string engine_revision_{"0.1.0"};
  std::string timestamp_;
  DiagnosticStatus overall_status_{DiagnosticStatus::kPass};
  std::vector<DiagnosticCheck> checks_;
  std::vector<std::string> warnings_;
  std::vector<std::string> errors_;
};

}  // namespace strix::diagnostics

#endif  // STRIX_CORE_DIAGNOSTICS_REPORT_H_
