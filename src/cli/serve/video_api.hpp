#ifndef GUFO_SERVER_VIDEO_API_HPP_
#define GUFO_SERVER_VIDEO_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/video_jobs.hpp"

namespace gufo::server {

inline constexpr std::string_view kVideoApiSchema = "gufo.video-api.v1";

[[nodiscard]] bool IsVideoApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleVideoApiRequest(const HttpRequest& request,
                                                 VideoJobService& service);

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_API_HPP_
