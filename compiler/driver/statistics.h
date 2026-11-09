#ifndef SEEML_COMPILER_DRIVER_STATISTICS_H_
#define SEEML_COMPILER_DRIVER_STATISTICS_H_

#include <cstdint>
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

struct FrontendStats {
  IngressorFacts ingressor;
  TopologyFacts topology;
  AccountantFacts accountant;
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
