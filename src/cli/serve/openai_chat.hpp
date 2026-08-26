#ifndef STRIX_SERVER_OPENAI_CHAT_HPP_
#define STRIX_SERVER_OPENAI_CHAT_HPP_

#include "src/cli/serve/http_server.hpp"

namespace strix::server {

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// use HttpResponse::streaming_body and apply socket backpressure directly to
/// model generation.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

}  // namespace strix::server

#endif  // STRIX_SERVER_OPENAI_CHAT_HPP_
