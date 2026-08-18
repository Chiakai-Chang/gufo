#ifndef STRIX_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_
#define STRIX_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_

#include <cstddef>
#include <span>
#include <string>

namespace strix::testing {

struct LogitCompareResult {
  bool match{true};
  float max_abs_diff{0.0F};
  float max_rel_diff{0.0F};
  std::size_t first_mismatch_idx{0};
  std::string details;
};

/// Compares candidate logits against reference logits with specified atol and
/// rtol tolerances.
LogitCompareResult CompareLogits(std::span<const float> reference,
                                 std::span<const float> candidate,
                                 float atol = 1e-3F, float rtol = 1e-3F);

}  // namespace strix::testing

#endif  // STRIX_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_
