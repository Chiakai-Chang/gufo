#include "src/testing/compare/logit_comparator.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace strix::testing {

LogitCompareResult CompareLogits(std::span<const float> reference,
                                 std::span<const float> candidate,
                                 float atol, float rtol) {
  LogitCompareResult res;
  if (reference.size() != candidate.size()) {
    res.match = false;
    res.details = "Size mismatch: ref=" + std::to_string(reference.size()) +
                  ", cand=" + std::to_string(candidate.size());
    return res;
  }

  for (std::size_t i = 0; i < reference.size(); ++i) {
    const float ref = reference[i];
    const float cand = candidate[i];
    const float abs_diff = std::abs(ref - cand);
    const float denom = std::max(std::abs(ref), std::abs(cand));
    const float rel_diff = (denom > 1e-7F) ? (abs_diff / denom) : abs_diff;

    if (abs_diff > res.max_abs_diff) res.max_abs_diff = abs_diff;
    if (rel_diff > res.max_rel_diff) res.max_rel_diff = rel_diff;

    const float tol = atol + rtol * std::abs(ref);
    if (abs_diff > tol && res.match) {
      res.match = false;
      res.first_mismatch_idx = i;
      std::ostringstream ss;
      ss << "Mismatch at index " << i << ": ref=" << ref << ", cand=" << cand
         << ", abs_diff=" << abs_diff << " (tol=" << tol << ")";
      res.details = ss.str();
    }
  }

  return res;
}

}  // namespace strix::testing
