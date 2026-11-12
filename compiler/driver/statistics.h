#ifndef SEEML_COMPILER_DRIVER_STATISTICS_H_
#define SEEML_COMPILER_DRIVER_STATISTICS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/ingressor/model_reader.h"

// =============================================================================
// The compiler's statistics screens (S0 #146). The driver records, at the
// sites that compute them, the numbers its passes return and its notes used
// to print once (CompileFacts); FillCompileScreens turns those and the
// compiled plan into the frontend, analysis and backend screens and the
// runtime screen's planned rows (source/statistics/screens.h).
// =============================================================================

namespace seeml::update {

/// The frontend's statistics, one struct per stage (S1 #147, seeai.md §3):
/// each stage fills its own as it runs, and the frontend screen is a
/// rendering of the three.
struct IngressorFacts {
  // The model as read.
  uint64_t model_ops = 0;
  uint64_t model_tensors = 0;
  uint64_t weight_tensors = 0;
  uint64_t weight_bytes = 0;
  bool token_input = false;
  uint64_t input_width = 0;  // features per row (a feature model)
  uint64_t seq_len = 0;      // tokens per sequence (a token model)
  uint64_t content_hash = 0;  // the file's identity; 0 = built in memory
  // Every constant tensor's value range, from the ingress sweep.
  std::vector<WeightRange> ranges;
};

struct TopologyFacts {
  // What the verification walk learned about the graph.
  uint64_t vocabulary = 0;  // rows of the embedding table; 0 = none
  uint64_t activations = 0;  // forward values computation emits
  uint64_t activation_bytes = 0;  // their f32 bytes at the compiled batch
  uint64_t embedding_ops = 0;
  uint64_t attention_ops = 0;
  bool has_teacher = false;
  uint64_t teacher_ops = 0;
  // The adapter census: the weights LoRA will adapt, by the grafter's rule.
  uint64_t adapted_tensors = 0;
  uint64_t adapted_bytes = 0;
};

struct AccountantFacts {
  // The step-0 footprint, as the early gate judged it.
  uint64_t footprint_weights = 0;
  uint64_t footprint_activations = 0;
  uint64_t footprint_deltas = 0;
  uint64_t footprint_caches = 0;
  uint64_t footprint_total = 0;
  uint64_t memory_budget = 0;  // configured, else the host's memory
  // Adapter-only: the frozen weights are read from the model file, so the
  // bound counts none of them.
  bool weights_mapped = false;
};

/// The corpus the compile took (S2 #148, `--data`): what came in, what was
/// kept and why not, and what the plan's step budget covers of it.
struct CorpusFacts {
  std::string kind;  // "text", "jsonl" or "sds"
  // Raw text only: the documents and the sanitizer's counts.
  uint64_t documents_in = 0;
  uint64_t documents_kept = 0;
  uint64_t dropped_empty = 0;
  uint64_t dropped_duplicate = 0;
  uint64_t dropped_degenerate = 0;
  uint64_t invalid_utf8_bytes = 0;
  uint64_t control_chars = 0;
  uint64_t bidi_controls = 0;
  uint64_t remainder_tokens = 0;    // ids the last partial window dropped
  uint64_t straddling_windows = 0;  // windows holding a document boundary
  // Every corpus.
  uint64_t records = 0;        // samples (windows of S + 1 ids, or rows)
  uint64_t record_width = 0;   // ids per record (S + 1), or features
  uint64_t tokens = 0;         // ids the records hold (token corpora)
  uint64_t target_tokens = 0;  // positions the loss counts (all, unmasked)
  uint64_t total_tokens = 0;   // positions per record × records (S each)
  uint64_t distinct_ids = 0;   // distinct token ids (token corpora)
  uint64_t vocabulary = 0;     // the model's, for the ratio
  double batches_per_epoch = 0.0;  // records ÷ sequences (rows) per step
  double epochs_covered = 0.0;     // the step budget × that ÷ records
  std::optional<double> label_entropy_nats;  // the unigram loss floor
  uint64_t corpus_hash = 0;
  uint64_t tokenizer_hash = 0;
  bool masked = false;
  // The compiled split (S3 #149): the held-out records, the seed they were
  // drawn with and the indices' hash the plan carries.
  bool split = false;
  uint64_t val_records = 0;
  uint64_t split_seed = 0;
  uint64_t val_split_hash = 0;
};

struct FrontendStats {
  IngressorFacts ingressor;
  TopologyFacts topology;
  AccountantFacts accountant;
  std::optional<CorpusFacts> corpus;  // set when the compile took --data
};

/// One arena tensor, for the backend screen's table.
struct TensorFact {
  std::string name;
  uint64_t bytes = 0;
  std::string segment;  // "persistent" | "io" | "transient"
};

/// What the driver learned while compiling that CompiledUpdate does not
/// otherwise carry. Filled where each number is computed.
struct CompileFacts {
  FrontendStats frontend;
  // Analysis: what the passes returned.
  uint64_t ops_in = 0;
  uint64_t ops_out = 0;
  uint64_t gradients = 0;
  uint64_t rope_tables = 0;
  uint64_t epilogue_fusions = 0;
  uint64_t addend_fusions = 0;
  uint64_t chain_fusions = 0;
  uint64_t dce_removed = 0;
  uint64_t attention_ops = 0;
  bool remat_budget_configured = false;
  uint64_t int8_weights = 0;
  uint64_t bf16_weights = 0;
  // What the calculus added and the fusions removed (S5 #151).
  uint64_t adjoint_ops = 0;             // ops autodiff emitted
  uint64_t saved_activation_bytes = 0;  // forward values the backward reads
  uint64_t epilogue_bytes_removed = 0;  // intermediates no longer written
  uint64_t addend_bytes_removed = 0;
  uint64_t chain_bytes_removed = 0;
  // Backend (S6 #152): the five largest values the arena holds.
  std::vector<TensorFact> largest_tensors;
};

/// Fills `result.screens` from `facts`, the config and the compiled plan:
/// frontend, analysis and backend in full, the runtime screen's planned
/// rows (the device fills the rest).
void FillCompileScreens(const CompileFacts& facts, const UpdateConfig& config,
                        CompiledUpdate& result);

/// The console's four screens, in order; `verbose` adds the verbose rows
/// and the detail tables.
std::string RenderCompileScreens(const CompileScreens& screens, bool verbose);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_DRIVER_STATISTICS_H_
