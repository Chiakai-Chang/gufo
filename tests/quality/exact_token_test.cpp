#include <cassert>
#include <iostream>
#include <vector>

#include "src/models/qwen_forward.hpp"
#include "src/models/qwen_generator.hpp"

void TestDeterministicArgmax() {
  std::vector<float> logits = {0.1F, 5.2F, -1.0F, 5.19F, 2.0F};
  const auto tok = strix::models::GreedyArgmax(logits);
  assert(tok == 1);

  // Exact determinism: 1000 repetitions must yield identical result
  for (int i = 0; i < 1000; ++i) {
    assert(strix::models::GreedyArgmax(logits) == 1);
  }
}

void TestDeterministicArgmaxTies() {
  std::vector<float> logits = {1.0F, 5.0F, 5.0F, 2.0F};
  // First occurrence of maximum value
  const auto tok = strix::models::GreedyArgmax(logits);
  assert(tok == 1);
}

int main() {
  TestDeterministicArgmax();
  TestDeterministicArgmaxTies();
  std::cout << "All exact token determinism tests passed.\n";
  return 0;
}
