#ifndef SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_
#define SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "compiler/driver/statistics.h"
#include "compiler/frontend/topology/sema.h"
#include "source/language/model_format.h"

// =============================================================================
// The compile-time driver of the data path (S2 #148, this directory's
// README): `--data` names raw text (`.txt` — one document; `.jsonl` — one
// {"text": ...} per line) or a pre-tokenized SDS, and this runs the vendored
// stages over it with the model's own tokenizer section, holds the result to
// the model (the tokenizer identity, the geometry, the id bound), and fills
// the accountant's corpus facts. The artifact is the canonical SDS the plan
// is pinned to by hash, and for raw text the manifest of what was dropped.
// =============================================================================

namespace seeml::update::tokenization {

struct CorpusIngress {
  std::vector<uint8_t> sds;     // the canonical corpus (raw text), or the
                                // SDS file's bytes as given
  bool from_sds = false;        // `--data` named an SDS: nothing to write
  std::string manifest_json;    // raw text only
  uint64_t corpus_hash = 0;     // ContentHash64 of `sds`
  uint64_t tokenizer_hash = 0;  // the section's; 0 = none
  bool masked = false;          // the corpus carries a loss mask
  CorpusFacts facts;
};

/// `rows_per_step` is the compiled batch times the accumulation (rows:
/// tokens for a token model), and `steps` the default step budget, for the
/// epochs the plan covers.
std::expected<CorpusIngress, std::string> IngestCorpus(
    const std::string& path, const SmfModel& model,
    const sema::Topology& topology, uint64_t rows_per_step, uint64_t steps);

}  // namespace seeml::update::tokenization

#endif  // SEEML_COMPILER_FRONTEND_TOKENIZATION_CORPUS_INGRESS_H_
