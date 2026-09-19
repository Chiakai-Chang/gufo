#ifndef GUFO_SERVER_LOGGING_HPP_
#define GUFO_SERVER_LOGGING_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace gufo::server {

enum class LogLevel : std::uint8_t {
  kInfo,
  kWarn,
  kError,
};

class Logger {
public:
  /// Logs a general server or engine message:
  /// YYYY-MM-DD HH:MM:SS [INFO] [component] message
  static void Log(LogLevel level, std::string_view component,
                  std::string_view message);

  /// Logs completion after the response body, including streamed generation.
  static void LogRequest(std::string_view id, std::string_view method,
                         std::string_view path, int status_code,
                         double duration_ms, std::string_view details = "",
                         std::string_view outcome = "completed");

  /// Process RSS and host available memory; these are not additive GPU totals.
  static std::string MemoryStatus();

  /// Convenience helpers
  static void Info(std::string_view component, std::string_view message) {
    Log(LogLevel::kInfo, component, message);
  }

  static void Warn(std::string_view component, std::string_view message) {
    Log(LogLevel::kWarn, component, message);
  }

  static void Error(std::string_view component, std::string_view message) {
    Log(LogLevel::kError, component, message);
  }
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_LOGGING_HPP_
