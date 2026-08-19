#ifndef STRIX_SERVER_HTTP_SERVER_HPP_
#define STRIX_SERVER_HTTP_SERVER_HPP_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/server/inference_backend.hpp"

namespace strix::server {

struct HttpRequest {
  std::string method;  // "GET" / "POST" / ...
  std::string path;    // "/v1/chat/completions" (no query)
  std::string query;   // raw query string (no leading '?')
  std::string body;
  InferenceBackend::CancellationCheck is_cancelled;

  /// URL-decoded value of a query param, or "" if absent.
  std::string query_param(const std::string& key) const;
};

struct HttpResponse {
  int status = 200;
  std::string reason = "OK";
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

using Handler =
    std::function<HttpResponse(const HttpRequest&, InferenceBackend&)>;

/// Minimal single-process HTTP/1.1 server. One detached worker thread is
/// spawned per accepted connection.
class HttpServer {
public:
  HttpServer(std::string host, int port,
             std::shared_ptr<InferenceBackend> backend);

  void add(const std::string& method, const std::string& path, Handler handler);

  /// Bind + listen. Returns false and sets *error on failure.
  bool start(std::string* error);

  /// Blocking accept loop.
  void run();

  void stop();

private:
  HttpResponse handle_request(const HttpRequest& req);
  void handle_connection(int client_fd);
  void register_routes();

  std::string host_;
  int port_;
  std::shared_ptr<InferenceBackend> backend_;
  int listen_fd_ = -1;
  std::atomic<bool> stopped_{false};
  std::vector<std::pair<std::pair<std::string, std::string>, Handler>> routes_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_HTTP_SERVER_HPP_
