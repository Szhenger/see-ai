#ifndef SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_

#include <cstddef>
#include <cstdio>
#include <expected>
#include <source_location>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// passing/ — errors formed while orchestrating SIR passes in the analysis
// subsystem, outside the analytic methods themselves (compiler/analysis/):
// the pass manager's invariant gate.
// Failure discipline: a pass's own error is propagated verbatim (tests match
// it exactly); corruption found *after* a pass is attributed to that pass by
// name so the offender is never ambiguous.
// =============================================================================

namespace seeml::diag::passing {

inline constexpr std::string_view kPassManager = "PassManager";

/// Post-pass verification failure, attributed to the offending pass:
/// "PassManager: SIR invariants violated after pass '<pass>': <why>".
[[nodiscard]] inline std::unexpected<std::string> InvariantsViolated(
    std::string_view pass, std::string_view why) {
  std::string m;
  m.reserve(pass.size() + why.size() + 48);
  m.append("SIR invariants violated after pass '")
      .append(pass)
      .append("': ")
      .append(why);
  return Fail(kPassManager, m);
}

/// Progress note after a pass runs clean:
/// "PassManager: pass '<pass>' ok (<ops> ops)".
inline void PassNote(std::string_view pass, size_t num_ops, double ms,
                     const std::source_location loc = std::source_location::current()) {
  std::string m;
  m.reserve(pass.size() + 40);
  char elapsed[32];
  std::snprintf(elapsed, sizeof(elapsed), "%.3f ms", ms);
  m.append("pass '").append(pass).append("' ok (")
      .append(std::to_string(num_ops)).append(" ops, ").append(elapsed)
      .append(")");
  Note(kPassManager, m, loc);
}

}  // namespace seeml::diag::passing

#endif  // SEEML_COMPILER_DIAGNOSTICS_PASSING_ERROR_H_
