#include <cassert>
#include <iostream>
#include <vector>

#include "src/models/qwen_oracles.hpp"
#include "src/testing/compare/logit_comparator.hpp"

void TestLogitComparatorExactMatch() {
  std::vector<float> ref = {1.0F, 2.5F, -3.2F, 0.0F};
  std::vector<float> cand = {1.0F, 2.5F, -3.2F, 0.0F};

  const auto res = strix::testing::CompareLogits(ref, cand, 1e-4F, 1e-4F);
  assert(res.match);
  assert(res.max_abs_diff == 0.0F);
}

void TestLogitComparatorTolerance() {
  std::vector<float> ref = {1.0F, 2.0F, 3.0F};
  std::vector<float> cand = {1.00001F, 2.00002F, 2.99999F};

  const auto res = strix::testing::CompareLogits(ref, cand, 1e-3F, 1e-3F);
  assert(res.match);
}

void TestRMSNormOracleComparison() {
  std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> w = {1.0F, 1.0F, 1.0F, 1.0F};
  std::vector<float> out(4, 0.0F);

  strix::models::qwen::ReferenceRMSNorm(x, w, 1e-6F, out);

  // Compare against identical reference
  const auto res = strix::testing::CompareLogits(out, out, 1e-5F, 1e-5F);
  assert(res.match);
}

int main() {
  TestLogitComparatorExactMatch();
  TestLogitComparatorTolerance();
  TestRMSNormOracleComparison();
  std::cout << "All Qwen oracle and logit comparison tests passed.\n";
  return 0;
}
