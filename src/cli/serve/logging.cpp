#include "src/cli/serve/logging.hpp"

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace gufo::server {

namespace {

std::mutex& LogMutex() {
  static std::mutex mutex;
  return mutex;
}

bool ShouldUseColor() {
  static const bool enabled = [] {
    if (std::getenv("NO_COLOR") != nullptr) {
      return false;
    }
    return ::isatty(fileno(stdout)) != 0;
  }();
  return enabled;
}

std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#if defined(_POSIX_THREAD_SAFE_FUNCTIONS)
  ::localtime_r(&time_t_now, &tm_buf);
#else
  tm_buf = *std::localtime(&time_t_now);
#endif
  std::array<char, 32> buffer{};
  const std::size_t written =
      std::strftime(buffer.data(), buffer.size(), "%Y-%m-%d %H:%M:%S", &tm_buf);
  return {buffer.data(), written};
}

const char* LevelTag(LogLevel level, bool color) {
  if (!color) {
    switch (level) {
      case LogLevel::kInfo:
        return "[INFO]";
      case LogLevel::kWarn:
        return "[WARN]";
      case LogLevel::kError:
        return "[ERROR]";
    }
    return "[INFO]";
  }

  switch (level) {
    case LogLevel::kInfo:
      return "\033[32m[INFO]\033[0m";  // Green
    case LogLevel::kWarn:
      return "\033[33m[WARN]\033[0m";  // Yellow
    case LogLevel::kError:
      return "\033[31m[ERROR]\033[0m";  // Red
  }
  return "\033[32m[INFO]\033[0m";
}

}  // namespace

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view message) {
  const bool color = ShouldUseColor();
  const std::string timestamp = CurrentTimestamp();

  std::ostringstream out;
  if (color) {
    out << "\033[90m" << timestamp << "\033[0m " << LevelTag(level, color)
        << " \033[36m[" << component << "]\033[0m " << message;
  } else {
    out << timestamp << " " << LevelTag(level, color) << " [" << component
        << "] " << message;
  }

  const std::lock_guard<std::mutex> lock(LogMutex());
  std::cout << out.str() << "\n" << std::flush;
}

void Logger::LogRequest(std::string_view method, std::string_view path,
                        int status_code, std::string_view reason,
                        double duration_ms, std::string_view details) {
  const bool color = ShouldUseColor();
  const std::string timestamp = CurrentTimestamp();

  LogLevel level = LogLevel::kInfo;
  if (status_code >= 500) {
    level = LogLevel::kError;
  } else if (status_code >= 400) {
    level = LogLevel::kWarn;
  }

  std::ostringstream out;
  if (color) {
    out << "\033[90m" << timestamp << "\033[0m " << LevelTag(level, color)
        << " \033[35m[http]\033[0m \033[1m" << method << "\033[0m " << path
        << " ";

    // Color status code
    if (status_code < 300) {
      out << "\033[32m" << status_code << " " << reason << "\033[0m";
    } else if (status_code < 500) {
      out << "\033[33m" << status_code << " " << reason << "\033[0m";
    } else {
      out << "\033[31m" << status_code << " " << reason << "\033[0m";
    }

    out << " in \033[32m" << std::fixed << std::setprecision(1) << duration_ms
        << "ms\033[0m";
    if (!details.empty()) {
      out << " \033[90m|\033[0m " << details;
    }
  } else {
    out << timestamp << " " << LevelTag(level, color) << " [http] " << method
        << " " << path << " " << status_code << " " << reason << " in "
        << std::fixed << std::setprecision(1) << duration_ms << "ms";
    if (!details.empty()) {
      out << " | " << details;
    }
  }

  const std::lock_guard<std::mutex> lock(LogMutex());
  std::cout << out.str() << "\n" << std::flush;
}

}  // namespace gufo::server
