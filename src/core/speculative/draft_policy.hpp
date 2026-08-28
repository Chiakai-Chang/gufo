#ifndef GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_
#define GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_

#include <string_view>

namespace gufo::speculative {

/// Resolves the `auto` draft-policy default for a backend.
///
/// The adaptive controllers exist because a verification batch used to get
/// steeply more expensive as it got wider, so trimming the width bought back
/// time. For DFlash-2 that premise no longer holds: a batch now costs about
/// what a single decode step costs at every width in 2..8, so trimming only
/// emits fewer tokens for the same cost. Measured on both pinned suites, fixed
/// width 7 beats rolling and accepted-EMA while staying greedy-exact, and
/// acceptance *rate* is inversely correlated with throughput there. Backends
/// whose verifier cost still grows with width keep the rolling controller.
[[nodiscard]] inline std::string_view ResolveDraftPolicy(
    std::string_view requested, bool block_diffusion_draft) noexcept {
  if (requested != "auto") {
    return requested;
  }
  return block_diffusion_draft ? "fixed" : "rolling";
}

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_
