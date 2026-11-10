#ifndef SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_
#define SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "compiler/driver/statistics.h"
#include "compiler/frontend/topology/sema.h"
#include "source/contract/plan_contract.h"
#include "source/language/model_format.h"
#include "source/plan/config.h"

// =============================================================================
// The compile-time driver of the data path (S2 #148, this directory's
// README): `--data` names raw text (`.txt` — one document; `.jsonl` — one
// {"text": ...} per line) or a pre-tokenized SDS, and this runs the vendored
// stages over it with the model's own tokenizer section, holds the result to
// the model (the tokenizer identity, the geometry, the id bound, the labels
// the loss stages), and fills the accountant's corpus facts. The artifact is
// the canonical SDS the plan is pinned to by hash, and for raw text the
// manifest of what was dropped.
// =============================================================================

namespace seeml::update::tokenization {

struct CorpusIngress {
  std::vector<uint8_t> sds;     // the canonical corpus tokenized here from
                                // raw text; empty for a given SDS, which is
                                // the corpus as is (nothing to write)
  bool from_sds = false;        // `--data` named an SDS
  std::string manifest_json;    // raw text only
  // The corpus's side of the feeder contract (source/contract/
  // plan_contract.h): what the device will hold the plan's header to, so
  // the driver can run the device's own rule on the header it emits —
  // every id checked, the split identity the records will carry once the
  // device partitions them. corpus_hash is ContentHash64 of the SDS bytes.
  CorpusShape shape;
  CorpusFacts facts;
};

/// What the plan will stage for the loss, decided by the driver's loss
/// choice before the corpus is held to it.
struct PlanLabels {
  uint32_t kind = 0;    // 0 none, 1 class index (i32), 2 dense f32[width]
  uint64_t width = 0;   // the output width (the classes, or the dense width)
};

/// The labels `loss` stages over an output `width` wide: class ids under
/// the cross-entropy (alone or beside the distillation term), dense targets
/// under MSE, none under distillation alone.
PlanLabels LabelsOf(LossKind loss, uint64_t width);

/// `rows_per_step` is the compiled batch times the accumulation (rows:
/// tokens for a token model), and `steps` the default step budget, for the
/// epochs the plan covers. `labels` is what the plan stages (LabelsOf).
std::expected<CorpusIngress, std::string> IngestCorpus(
    const std::string& path, const SmfModel& model,
    const sema::Topology& topology, PlanLabels labels, uint64_t rows_per_step,
    uint64_t steps);

}  // namespace seeml::update::tokenization

#endif  // SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_
