#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_

#include <cmath>
#include <expected>
#include <optional>
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

/// What a report's figures hold whoever filled them in — the reader below,
/// or a config built in code, which the driver refuses when it fails: a
/// figure without its check and a check without a figure say nothing, and
/// what the plan's metadata records must be what the loader accepts and a
/// committed model's report can state again. The first violated rule, or
/// nothing; an absent report holds them all.
inline std::optional<std::string> ExportReportViolation(const ExportReport& r) {
  if (!r.present) return std::nullopt;
  if (r.report_hash == 0) return std::string("the report carries no hash");
  if (!std::isfinite(r.logits_parity_max_delta) ||
      r.logits_parity_max_delta < 0.0f)
    return std::string(
        "logits_parity_max_delta is not a finite non-negative number");
  if (!r.tokenizer_parity && r.tokenizer_parity_ids != 0)
    return std::string(
        "tokenizer_parity_ids is set but the parity was not checked");
  if (r.tokenizer_parity && r.tokenizer_parity_ids == 0)
    return std::string("the tokenizer parity is marked checked over zero ids");
  if (!r.logits_parity && r.logits_parity_max_delta != 0.0f)
    return std::string(
        "logits_parity_max_delta is set but the parity was not checked");
  return std::nullopt;
}

/// Reads `smf_path` + ".export.json" and verifies it against `model`. An
/// absent file returns a report with `present == false` unless the model
/// needs one, in which case it is refused; a file that is there is always
/// held to the model (a stale, foreign or unreadable report is refused,
/// never ignored).
[[nodiscard]] std::expected<ExportReport, std::string> ReadExportReport(
    const std::string& smf_path, const SmfModel& model);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_EXPORT_REPORT_H_
