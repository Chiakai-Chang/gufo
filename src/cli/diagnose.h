#ifndef STRIX_CLI_DIAGNOSE_H_
#define STRIX_CLI_DIAGNOSE_H_

#include <span>
#include <string_view>

#include "src/core/diagnostics/report.h"

namespace strix::cli {

[[nodiscard]] diagnostics::DiagnosticReport CollectDiagnostics();
int RunDiagnose(std::span<const char* const> args);

}  // namespace strix::cli

#endif  // STRIX_CLI_DIAGNOSE_H_
