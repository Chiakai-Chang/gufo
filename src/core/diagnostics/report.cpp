#include "src/core/diagnostics/report.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace strix::diagnostics {

namespace {

std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_utc{};
  gmtime_r(&time_t_now, &tm_utc);

  std::ostringstream oss;
  oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

std::string EscapeJson(std::string_view str) {
  std::ostringstream oss;
  for (const char ch : str) {
    switch (ch) {
      case '"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
        } else {
          oss << ch;
        }
        break;
    }
  }
  return oss.str();
}

}  // namespace

std::string_view ToString(DiagnosticStatus status) {
  switch (status) {
    case DiagnosticStatus::kPass:
      return "PASS";
    case DiagnosticStatus::kWarn:
      return "WARN";
    case DiagnosticStatus::kFail:
      return "FAIL";
  }
  return "UNKNOWN";
}

DiagnosticStatus FromString(std::string_view str) {
  if (str == "PASS" || str == "pass") {
    return DiagnosticStatus::kPass;
  }
  if (str == "WARN" || str == "warn") {
    return DiagnosticStatus::kWarn;
  }
  return DiagnosticStatus::kFail;
}

DiagnosticReport::DiagnosticReport() : timestamp_(CurrentIso8601Utc()) {}

void DiagnosticReport::SetEngineRevision(std::string_view revision) {
  engine_revision_ = std::string(revision);
}

void DiagnosticReport::SetTimestamp(std::string_view iso8601_utc) {
  timestamp_ = std::string(iso8601_utc);
}

void DiagnosticReport::AddCheck(DiagnosticCheck check) {
  checks_.push_back(std::move(check));
  RecomputeStatus();
}

void DiagnosticReport::AddWarning(std::string_view warning) {
  warnings_.emplace_back(warning);
  if (overall_status_ == DiagnosticStatus::kPass) {
    overall_status_ = DiagnosticStatus::kWarn;
  }
}

void DiagnosticReport::AddError(std::string_view error) {
  errors_.emplace_back(error);
  overall_status_ = DiagnosticStatus::kFail;
}

void DiagnosticReport::RecomputeStatus() {
  DiagnosticStatus st = DiagnosticStatus::kPass;
  for (const auto& check : checks_) {
    if (check.status == DiagnosticStatus::kFail) {
      st = DiagnosticStatus::kFail;
      break;
    }
    if (check.status == DiagnosticStatus::kWarn) {
      st = DiagnosticStatus::kWarn;
    }
  }
  if (!errors_.empty()) {
    st = DiagnosticStatus::kFail;
  } else if (!warnings_.empty() && st == DiagnosticStatus::kPass) {
    st = DiagnosticStatus::kWarn;
  }
  overall_status_ = st;
}

std::string DiagnosticReport::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"schemaVersion\": \"" << EscapeJson(schema_version_) << "\",\n";
  oss << "  \"engineRevision\": \"" << EscapeJson(engine_revision_) << "\",\n";
  oss << "  \"timestamp\": \"" << EscapeJson(timestamp_) << "\",\n";
  oss << "  \"status\": \"" << ToString(overall_status_) << "\",\n";

  // checks array
  oss << "  \"checks\": [\n";
  for (std::size_t i = 0; i < checks_.size(); ++i) {
    const auto& c = checks_[i];
    oss << "    {\n";
    oss << "      \"name\": \"" << EscapeJson(c.name) << "\",\n";
    oss << "      \"status\": \"" << ToString(c.status) << "\",\n";
    oss << "      \"message\": \"" << EscapeJson(c.message) << "\",\n";
    oss << "      \"details\": {";
    if (!c.details.empty()) {
      oss << "\n";
      std::size_t detail_idx = 0;
      for (const auto& [k, v] : c.details) {
        oss << "        \"" << EscapeJson(k) << "\": \"" << EscapeJson(v)
            << "\"";
        if (++detail_idx < c.details.size()) {
          oss << ",";
        }
        oss << "\n";
      }
      oss << "      }\n";
    } else {
      oss << "}\n";
    }
    oss << "    }";
    if (i + 1 < checks_.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ],\n";

  // warnings array
  oss << "  \"warnings\": [";
  if (!warnings_.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < warnings_.size(); ++i) {
      oss << "    \"" << EscapeJson(warnings_[i]) << "\"";
      if (i + 1 < warnings_.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ],\n";
  } else {
    oss << "],\n";
  }

  // errors array
  oss << "  \"errors\": [";
  if (!errors_.empty()) {
    oss << "\n";
    for (std::size_t i = 0; i < errors_.size(); ++i) {
      oss << "    \"" << EscapeJson(errors_[i]) << "\"";
      if (i + 1 < errors_.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]\n";
  } else {
    oss << "]\n";
  }

  oss << "}\n";
  return oss.str();
}

std::string DiagnosticReport::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Strix Halo Engine Diagnostics ===\n";
  oss << "Engine Revision : " << engine_revision_ << "\n";
  oss << "Schema Version  : " << schema_version_ << "\n";
  oss << "Timestamp       : " << timestamp_ << "\n";
  oss << "Overall Status  : [" << ToString(overall_status_) << "]\n\n";

  oss << "--- Checks ---\n";
  for (const auto& check : checks_) {
    oss << "  [" << ToString(check.status) << "] " << check.name << ": "
        << check.message << "\n";
    for (const auto& [k, v] : check.details) {
      oss << "      - " << k << ": " << v << "\n";
    }
  }

  if (!warnings_.empty()) {
    oss << "\n--- Warnings ---\n";
    for (const auto& w : warnings_) {
      oss << "  [WARN] " << w << "\n";
    }
  }

  if (!errors_.empty()) {
    oss << "\n--- Errors ---\n";
    for (const auto& e : errors_) {
      oss << "  [ERROR] " << e << "\n";
    }
  }

  return oss.str();
}

}  // namespace strix::diagnostics
