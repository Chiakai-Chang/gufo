#ifndef STRIX_SERVER_HTTP_SERVER_HPP_
#define STRIX_SERVER_HTTP_SERVER_HPP_

#include <atomic>
#include <cctype>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"

namespace strix::server {

class VideoJobService;
class TtsService;

struct HttpRequest {
  std::string method;  // "GET" / "POST" / ...
  std::string path;    // "/v1/chat/completions" (no query)
  std::string query;   // raw query string (no leading '?')
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
  TextGenerationBackend::CancellationCheck is_cancelled;

  /// URL-decoded value of a query param, or "" if absent.
  std::string query_param(const std::string& key) const;
  /// Case-insensitive request-header lookup, or "" if absent.
  std::string header(std::string_view name) const {
    const auto equal_case_insensitive = [](std::string_view lhs,
                                           std::string_view rhs) {
      if (lhs.size() != rhs.size()) {
        return false;
      }
      for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(lhs[index])) !=
            std::tolower(static_cast<unsigned char>(rhs[index]))) {
          return false;
        }
      }
      return true;
    };
    for (const auto& [header_name, value] : headers) {
      if (equal_case_insensitive(header_name, name)) {
        return value;
      }
    }
    return "";
  }
};

struct HttpResponse {
  using BodyWriter = std::function<bool(std::string_view)>;
  using StreamingBody = std::function<void(const BodyWriter&)>;

  int status = 200;
  std::string reason = "OK";
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
  StreamingBody streaming_body;
};

using Handler =
    std::function<HttpResponse(const HttpRequest&, TextGenerationBackend&)>;

struct HttpServerLimits {
  std::size_t max_request_body_bytes{static_cast<std::size_t>(8) * 1024 * 1024};
  std::size_t max_connections{16};
};

/// Minimal bounded HTTP/1.1 server for trusted-LAN model serving.
class HttpServer {
public:
  HttpServer(std::string host, int port,
             std::shared_ptr<TextGenerationBackend> backend,
             std::shared_ptr<VideoJobService> video_jobs = nullptr,
             std::shared_ptr<TtsService> tts = nullptr,
             HttpServerLimits limits = {});
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  void add(const std::string& method, const std::string& path, Handler handler);

  /// Bind + listen. Returns false and sets *error on failure.
  bool start(std::string* error);

  /// Blocking accept loop.
  void run();

  void stop();

  [[nodiscard]] int port() const noexcept { return port_; }

private:
  struct ConnectionWorker;

  HttpResponse handle_request(const HttpRequest& req);
  void handle_connection(int client_fd);
  void reap_workers();
  void register_routes();

  std::string host_;
  int port_;
  std::shared_ptr<TextGenerationBackend> backend_;
  std::shared_ptr<VideoJobService> video_jobs_;
  std::shared_ptr<TtsService> tts_;
  HttpServerLimits limits_;
  int listen_fd_ = -1;
  std::atomic<bool> stopped_{false};
  std::mutex workers_mutex_;
  std::vector<std::unique_ptr<ConnectionWorker>> workers_;
  std::vector<std::pair<std::pair<std::string, std::string>, Handler>> routes_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_HTTP_SERVER_HPP_
