#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "src/models/deepseek_v4_flash/engine.hpp"

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

}  // namespace

int main() {
  const char* model_path = std::getenv("STRIX_DEEPSEEK_V4_FLASH_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') {
    std::cout << "SKIP: STRIX_DEEPSEEK_V4_FLASH_MODEL is not set\n";
    return 77;
  }

  using strix::models::deepseek_v4_flash::Model;
  using strix::models::deepseek_v4_flash::ModelOptions;

  std::string error;
  const auto model = Model::Load(model_path,
                                 ModelOptions{
                                     .max_context = 4096,
                                     .prefill_chunk = 2048,
                                     .power_percent = 100,
                                 },
                                 &error);
  Expect(model != nullptr, error.c_str());
  Expect(model->VocabSize() > 0, "vocabulary size");
  Expect(!model->ModelName().empty(), "model name");

  const auto prompt =
      model->EncodeChat("You are a concise assistant.", "Reply with one word.");
  Expect(!prompt.empty(), "chat prompt tokenization");

  auto session = model->CreateSession(4096, &error);
  Expect(session != nullptr, error.c_str());
  Expect(session->Sync(prompt, &error), error.c_str());
  Expect(session->Position() == static_cast<int>(prompt.size()),
         "prefill position");

  const auto logits = session->CopyLogits(&error);
  Expect(!logits.empty(), error.c_str());
  Expect(std::all_of(logits.begin(), logits.end(),
                     [](float value) { return std::isfinite(value); }),
         "finite logits");

  const int checkpoint_position = session->Position();
  const int first_token = session->SelectNext(0.0F, nullptr);
  Expect(first_token >= 0, "first greedy token");
  Expect(!model->DecodeToken(first_token).empty(), "first token text");
  auto snapshot = session->SaveSnapshot(&error);
  Expect(snapshot != nullptr, error.c_str());
  Expect(snapshot->SizeBytes() > 0, "snapshot payload");
  Expect(session->Evaluate(first_token, &error), error.c_str());

  const int second_token = session->SelectNext(0.0F, nullptr);
  Expect(second_token >= 0, "second greedy token");
  Expect(session->Evaluate(second_token, &error), error.c_str());
  Expect(session->Position() == static_cast<int>(prompt.size()) + 2,
         "decode position");

  Expect(session->RestoreSnapshot(*snapshot, &error), error.c_str());
  Expect(session->Position() == checkpoint_position, "restored position");
  Expect(session->SelectNext(0.0F, nullptr) == first_token,
         "restored greedy token");
  Expect(session->CopyLogits(&error) == logits, "restored logits");

  std::cout << "DeepSeek V4 Flash model/session test passed\n";
  return 0;
}
