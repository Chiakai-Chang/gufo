#include "src/cli/serve/openai_chat.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

class FakeBackend final : public strix::server::TextGenerationBackend {
public:
  [[nodiscard]] std::string model_id() const override { return "test-model"; }
  [[nodiscard]] bool ready() const override { return true; }
  [[nodiscard]] SamplingDefaults sampling_defaults() const override {
    return defaults;
  }

  Result complete(std::string_view, std::size_t, float,
                  const CancellationCheck&, const TokenCallback&) override {
    return {};
  }

  Result chat(const strix::server::ChatRequest& request, std::size_t max_tokens,
              float temperature, const CancellationCheck& is_cancelled,
              const TokenCallback& on_token) override {
    ++chat_calls;
    last_request = request;
    last_max_tokens = max_tokens;
    last_temperature = temperature;

    Result result;
    result.prompt_tokens = 7;
    result.cached_prompt_tokens = 5;
    result.cache_hit = true;
    for (const std::string& piece : pieces) {
      if ((is_cancelled && is_cancelled()) || (on_token && !on_token(piece))) {
        result.cancelled = true;
        result.finish_reason = FinishReason::kCancelled;
        return result;
      }
      result.text += piece;
      result.tokens.push_back(
          static_cast<strix::tokenization::TokenId>(result.tokens.size()));
      if (block_after_first_piece &&
          result.tokens.size() == static_cast<std::size_t>(1)) {
        {
          const std::lock_guard<std::mutex> lock(mutex);
          first_piece_emitted = true;
        }
        condition.notify_all();
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return released; });
      }
    }
    result.completion_tokens = result.tokens.size();
    result.finish_reason = finish_reason;
    completed.store(true);
    return result;
  }

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }

  bool WaitForFirstPiece() {
    std::unique_lock<std::mutex> lock(mutex);
    return condition.wait_for(lock, 2s, [&] { return first_piece_emitted; });
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    condition.notify_all();
  }

  std::vector<std::string> pieces;
  FinishReason finish_reason{FinishReason::kStop};
  bool block_after_first_piece{false};
  std::atomic<bool> completed{false};
  std::atomic<int> chat_calls{0};
  strix::server::ChatRequest last_request;
  SamplingDefaults defaults;
  std::size_t last_max_tokens{0};
  float last_temperature{0.0F};

private:
  std::mutex mutex;
  std::condition_variable condition;
  bool first_piece_emitted{false};
  bool released{false};
};

strix::server::HttpRequest Request(std::string body) {
  return {
      .method = "POST",
      .path = "/v1/chat/completions",
      .query = {},
      .body = std::move(body),
      .headers = {},
      .is_cancelled = {},
  };
}

void TestStreamingIsLive() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.finish_reason =
      strix::server::TextGenerationBackend::FinishReason::kLength;
  backend.block_after_first_piece = true;

  auto response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":2,
        "stream":true,
        "stream_options":{"include_usage":true}
      })"),
                                                  backend);
  Expect(response.status == 200, "Streaming request is accepted");
  Expect(static_cast<bool>(response.streaming_body),
         "Streaming request returns a streaming body");

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::string output;
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view chunk) {
      {
        const std::lock_guard<std::mutex> lock(output_mutex);
        output.append(chunk);
      }
      output_condition.notify_all();
      return true;
    });
  });

  Expect(backend.WaitForFirstPiece(), "Backend emits the first token");
  {
    std::unique_lock<std::mutex> lock(output_mutex);
    Expect(output_condition.wait_for(
               lock, 2s,
               [&] {
                 return output.find(R"("content":"Hel")") != std::string::npos;
               }),
           "First content delta is written promptly");
    Expect(!backend.completed.load(),
           "First content delta arrives before generation completes");
  }

  backend.Release();
  writer.join();

  Expect(output.find(R"("finish_reason":"length")") != std::string::npos,
         "Token limit is reported as finish_reason length");
  Expect(output.find(R"("prompt_tokens":7)") != std::string::npos,
         "Usage is emitted when requested");
  Expect(output.find(R"("cached_tokens":5)") != std::string::npos,
         "Usage reports transparently reused prompt tokens");
  Expect(output.ends_with("data: [DONE]\n\n"),
         "Stream terminates with the OpenAI DONE sentinel");
}

void TestToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>",
  };

  const auto response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"weather in Rome"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"get_weather",
            "description":"Get weather",
            "parameters":{
              "type":"object",
              "properties":{"city":{"type":"string"}},
              "required":["city"]
            }
          }
        }],
        "tool_choice":"required",
        "stream":false
      })"),
                                                        backend);

  Expect(response.status == 200, "Tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Tool generation reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"get_weather")") != std::string::npos,
         "Tool name is translated to OpenAI format");
  Expect(response.body.find(R"(\"city\":\"Rome\")") != std::string::npos,
         "Tool arguments are translated to JSON");
  Expect(backend.last_request.tools.size() == 1,
         "Tool schema reaches the model backend");
  Expect(backend.last_request.tool_choice ==
             strix::server::ChatRequest::ToolChoice::kRequired,
         "Required tool choice reaches the model backend");
}

void TestDeepSeekToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<｜DSML｜tool_calls｜>\n"
      "<｜DS｜invoke name=\"read\">\n"
      "<｜DS｜parameter name=\"path\" string=\"true\">"
      "/etc/hostname</｜DS｜parameter>\n"
      "</｜DS｜invoke>\n"
      "</｜DSML｜tool_calls｜>",
  };

  const auto response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"read the hostname"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"read",
            "description":"Read a file",
            "parameters":{
              "type":"object",
              "properties":{"path":{"type":"string"}},
              "required":["path"]
            }
          }
        }],
        "stream":false
      })"),
                                                        backend);

  Expect(response.status == 200, "DeepSeek tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Hybrid DeepSeek syntax reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"read")") != std::string::npos,
         "Hybrid DeepSeek tool name is translated");
  Expect(
      response.body.find(R"(\"path\":\"/etc/hostname\")") != std::string::npos,
      "Hybrid DeepSeek tool arguments are translated");
}

void TestBackendSamplingDefaults() {
  FakeBackend backend;
  backend.defaults = {
      .max_tokens = 37,
      .temperature = 0.25F,
  };

  const auto default_response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                                backend);
  Expect(default_response.status == 200, "Defaulted request is accepted");
  Expect(backend.last_max_tokens == 37,
         "Backend max-token default reaches generation");
  Expect(backend.last_temperature > 0.24F && backend.last_temperature < 0.26F,
         "Backend temperature default reaches generation");

  const auto override_response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":11,
        "temperature":0
      })"),
                                                                 backend);
  Expect(override_response.status == 200, "Sampling override is accepted");
  Expect(backend.last_max_tokens == 11,
         "Explicit max tokens override the backend default");
  Expect(backend.last_temperature == 0.0F,
         "Explicit temperature overrides the backend default");
}

void TestWrongModelIsRejected() {
  FakeBackend backend;
  const auto response = strix::server::HandleOpenAiChat(Request(R"({
        "model":"wrong-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                        backend);

  Expect(response.status == 404, "Unknown model is rejected");
  Expect(response.body.find("model_not_found") != std::string::npos,
         "Unknown model returns a stable error code");
  Expect(backend.chat_calls.load() == 0,
         "Rejected request never reaches the backend");
}

}  // namespace

int main() {
  TestStreamingIsLive();
  TestBackendSamplingDefaults();
  TestToolCallsAreStructured();
  TestDeepSeekToolCallsAreStructured();
  TestWrongModelIsRejected();
  std::cout << "All OpenAI chat protocol tests passed\n";
  return 0;
}
