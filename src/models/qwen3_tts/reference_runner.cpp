#include "src/models/qwen3_tts/reference_runner.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::models::qwen3_tts {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool ContainsNull(std::string_view value) {
  return value.find('\0') != std::string_view::npos;
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    const std::string prefix = "/tmp/gufo-qwen3-tts-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    if (const char* created = mkdtemp(pattern.data()); created != nullptr) {
      path_ = created;
    }
  }

  ~TemporaryDirectory() {
    if (!path_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  TemporaryDirectory(TemporaryDirectory&&) = delete;
  TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::string ReadTextFile(const std::filesystem::path& path,
                         std::size_t maximum_bytes) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > maximum_bytes) {
    return {};
  }
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

template<typename Element>
bool ReadBinaryFile(const std::filesystem::path& path,
                    std::size_t element_count, std::vector<Element>* output) {
  if (output == nullptr ||
      element_count >
          std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
    return false;
  }
  const std::size_t bytes = element_count * sizeof(Element);
  std::error_code error;
  if (std::filesystem::file_size(path, error) != bytes || error) {
    return false;
  }
  output->resize(element_count);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(bytes));
  return input.good() || input.eof();
}

std::vector<std::string> BuildArguments(
    const OfficialReferenceOptions& options, const SynthesisRequest& request,
    const std::filesystem::path& output_directory) {
  std::vector<std::string> arguments{
      options.python_executable.string(),
      options.runner_script.string(),
      "--reference-root",
      options.reference_root.string(),
      "--model",
      options.model_root.string(),
      "--out",
      output_directory.string(),
      "--text",
      request.text,
      "--speaker",
      request.speaker,
      "--language",
      request.language,
      "--max-new-tokens",
      std::to_string(request.max_new_tokens),
      "--seed",
      std::to_string(request.seed),
  };
  if (!request.instruct.empty()) {
    arguments.emplace_back("--instruct");
    arguments.push_back(request.instruct);
  }
  if (request.greedy) {
    arguments.emplace_back("--greedy");
  }
  return arguments;
}

int WaitForChild(pid_t child,
                 const OfficialReferenceRunner::CancellationCheck& cancelled) {
  while (true) {
    int status = 0;
    const pid_t waited = waitpid(child, &status, WNOHANG);
    if (waited == child) {
      return status;
    }
    if (waited < 0 && errno != EINTR) {
      return -1;
    }
    if (cancelled && cancelled()) {
      (void)kill(child, SIGTERM);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < deadline) {
        const pid_t terminated = waitpid(child, &status, WNOHANG);
        if (terminated == child) {
          return status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      (void)kill(child, SIGKILL);
      (void)waitpid(child, &status, 0);
      return status;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

bool ChildSucceeded(int status) {
  return status >= 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace

OfficialReferenceRunner::OfficialReferenceRunner(
    OfficialReferenceOptions options)
    : options_(std::move(options)) {}

bool OfficialReferenceRunner::Validate(std::string* error) const {
  if (!std::filesystem::is_directory(options_.model_root)) {
    SetError(error, "Qwen3-TTS model directory does not exist");
    return false;
  }
  if (!std::filesystem::is_regular_file(options_.model_root /
                                        "model.safetensors") ||
      !std::filesystem::is_regular_file(
          options_.model_root / "speech_tokenizer" / "model.safetensors")) {
    SetError(error, "Qwen3-TTS model weights are incomplete");
    return false;
  }
  if (!std::filesystem::is_directory(options_.reference_root / "qwen_tts")) {
    SetError(error, "official Qwen3-TTS source directory is invalid");
    return false;
  }
  if (!std::filesystem::is_regular_file(options_.python_executable)) {
    SetError(error, "official Qwen3-TTS Python executable does not exist");
    return false;
  }
  if (!std::filesystem::is_regular_file(options_.runner_script)) {
    SetError(error, "Qwen3-TTS reference runner script does not exist");
    return false;
  }
  return true;
}

bool OfficialReferenceRunner::Generate(const SynthesisRequest& request,
                                       const CancellationCheck& is_cancelled,
                                       SynthesisResult* result,
                                       std::string* error) const {
  if (result == nullptr) {
    SetError(error, "Qwen3-TTS reference result must not be null");
    return false;
  }
  *result = {};
  if (request.text.empty() || request.speaker.empty() ||
      request.language.empty() || request.max_new_tokens == 0) {
    SetError(error, "Qwen3-TTS reference request is incomplete");
    return false;
  }
  if (ContainsNull(request.text) || ContainsNull(request.speaker) ||
      ContainsNull(request.language) || ContainsNull(request.instruct)) {
    SetError(error, "Qwen3-TTS reference arguments must not contain NUL");
    return false;
  }
  if (!Validate(error)) {
    return false;
  }
  if (is_cancelled && is_cancelled()) {
    SetError(error, "Qwen3-TTS reference generation cancelled");
    return false;
  }

  const TemporaryDirectory temporary;
  if (temporary.path().empty()) {
    SetError(error, "cannot create Qwen3-TTS reference output directory");
    return false;
  }
  const std::filesystem::path log_path = temporary.path() / "reference.log";
  const int log = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (log < 0) {
    SetError(error, "cannot create Qwen3-TTS reference log");
    return false;
  }

  std::vector<std::string> arguments =
      BuildArguments(options_, request, temporary.path());
  std::vector<char*> raw_arguments;
  raw_arguments.reserve(arguments.size() + 1);
  for (std::string& argument : arguments) {
    raw_arguments.push_back(argument.data());
  }
  raw_arguments.push_back(nullptr);

  const pid_t child = fork();
  if (child == 0) {
    (void)dup2(log, STDOUT_FILENO);
    (void)dup2(log, STDERR_FILENO);
    close(log);
    execv(raw_arguments.front(), raw_arguments.data());
    _exit(127);
  }
  close(log);
  if (child < 0) {
    SetError(error, "cannot start official Qwen3-TTS reference process");
    return false;
  }

  const int status = WaitForChild(child, is_cancelled);
  if (!ChildSucceeded(status)) {
    std::string message;
    if (is_cancelled && is_cancelled()) {
      message = "Qwen3-TTS reference generation cancelled";
    } else {
      message = ReadTextFile(log_path, 64U << 10U);
    }
    if (message.empty()) {
      message = "official Qwen3-TTS reference process failed";
    }
    SetError(error, std::move(message));
    return false;
  }

  const std::string metadata_text =
      ReadTextFile(temporary.path() / "meta.json", 1U << 20U);
  try {
    const json::Value metadata = json::parse(metadata_text);
    const std::size_t sample_count = metadata.member_size("samples");
    const std::size_t code_steps = metadata.member_size("code_steps");
    const std::size_t code_groups = metadata.member_size("code_groups");
    const std::size_t sample_rate = metadata.member_size("sample_rate");
    if (sample_count == 0 || code_steps == 0 || code_groups == 0 ||
        sample_rate == 0 ||
        sample_rate > std::numeric_limits<std::uint32_t>::max() ||
        code_groups > std::numeric_limits<std::uint32_t>::max() ||
        code_steps > std::numeric_limits<std::size_t>::max() / code_groups) {
      throw std::runtime_error("invalid reference output dimensions");
    }
    SynthesisResult generated;
    generated.sample_rate = static_cast<std::uint32_t>(sample_rate);
    generated.code_groups = static_cast<std::uint32_t>(code_groups);
    if (!ReadBinaryFile(temporary.path() / "waveform.f32", sample_count,
                        &generated.samples) ||
        !ReadBinaryFile(temporary.path() / "codes.i32",
                        code_steps * code_groups, &generated.codes)) {
      throw std::runtime_error("invalid reference output payload");
    }
    *result = std::move(generated);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, std::string("cannot read Qwen3-TTS reference output: ") +
                        exception.what());
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts
