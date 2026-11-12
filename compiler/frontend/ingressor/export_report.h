#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_

#include <expected>
#include <string>

#include "source/language/model_format.h"
#include "source/plan/config.h"

// =============================================================================
// Exporter verification (S4 #150): the exporter's output is checked, not
// trusted. tool/export_model.py writes `<model>.smf.export.json` beside an
// imported checkpoint — the SMF's ContentHash64, its tokenizer section's
// hash, and what parity the export proved (the ids the tokenizer-parity
// harness compared against the reference tokenizer; the max logit delta
// against the reference forward). The ingressor reads it strictly, holds
// both hashes to the model it loaded, and hands the figures to the driver,
// which records them in the plan's metadata.
// =============================================================================

namespace seeml::update {

inline constexpr const char* kExportReportSuffix = ".export.json";

/// Whether `model` needs a report: an imported checkpoint — a model loaded
/// from a file that carries a tokenizer section. Exporter-native demo
/// models have no reference to be in parity with.
inline bool NeedsExportReport(const SmfModel& model) {
  return model.content_hash != 0 && !model.tokenizer.empty();
}

/// Reads `smf_path` + ".export.json" and verifies it against `model`. An
/// absent file returns a report with `present == false` unless the model
/// needs one, in which case it is refused; a file that is there is always
/// held to the model (a stale or foreign report is refused, never ignored).
[[nodiscard]] std::expected<ExportReport, std::string> ReadExportReport(
    const std::string& smf_path, const SmfModel& model);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_
