#ifndef SEEML_COMPILER_ANALYSIS_STATISTICS_PRECISION_H_
#define SEEML_COMPILER_ANALYSIS_STATISTICS_PRECISION_H_

#include <cstdint>
#include <vector>

#include "source/plan/update_types.h"

// =============================================================================
// Precision assignment (plan v18, F2 #130 / F4 #132; v20, F3 #131; moved out
// of the driver by S5 #151): which instructions of the train stream may run
// on a certified relaxed kernel.
//
// Under --precision certified-bf16 every frozen-weight GEMM of the train
// stream (and so of the step program split from it) — a product whose B
// operand is rodata: the base and teacher weights in f32, int8 or bf16, NN
// forward and NT dX — carries kFlagRelaxed. The adapter GEMMs (A and B live
// in the arena) and everything else stay exact, as does the eval program:
// the gate scores what ships in reference arithmetic. The bit is a
// permission the backend may decline; what it permits is bounded by the
// certificate the package must carry. Plan v20: the tiled attention
// instructions carry it too — the backend may run their blocks' products on
// matrix tiles in another f32 summation order; the certificate prices their
// attn.*.tiled sites.
// =============================================================================

namespace seeml::update {

struct RelaxedCounts {
  uint64_t gemms = 0;      // frozen-weight GEMMs flagged
  uint64_t attention = 0;  // tiled attention instructions flagged
};

/// Sets kFlagRelaxed on every instruction of `train` the rule above admits
/// and returns how many of each kind it flagged.
RelaxedCounts AssignRelaxedPrecision(std::vector<UpdateInstruction>& train);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_STATISTICS_PRECISION_H_
