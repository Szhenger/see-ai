#ifndef SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_
#define SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "source/plan/update_types.h"

namespace seeml::update_rt {

/// The step split (SEEML_STEP_TIMING builds; docs/benchmarks.md, Tier A):
/// seconds in the forward, backward and optimizer phases of the train
/// program, over `steps` optimizer steps. In every other build all fields
/// stay zero and no clock is read inside the step, so the shipped runtime
/// is untouched. The fields are unconditional so the header is identical
/// under both builds (no ODR hazard between differently-configured
/// objects).
struct StepTimings {
  uint64_t steps = 0;         // optimizer steps accumulated (G grad
                              // executions each under accumulation)
  double fwd_seconds = 0.0;
  double bwd_seconds = 0.0;
  double opt_seconds = 0.0;
};

/// Where the train program's phases begin: the forward pass ends after the
/// LAST loss-forward instruction (kXEntPlusKL emits two, adjacent), and the
/// optimizer phase starts at the FIRST clip / step instruction — everything
/// between is the backward sweep. A missing family degrades to an empty
/// range, never a bad split.
struct PhaseBoundaries {
  size_t bwd_begin = 0;
  size_t opt_begin = 0;
};
inline PhaseBoundaries ScanPhases(
    const std::vector<seeml::update::UpdateInstruction>& train) {
  namespace up = seeml::update;
  PhaseBoundaries p{train.size(), train.size()};
  for (size_t i = 0; i < train.size(); ++i) {
    const auto op = static_cast<up::OpCode>(train[i].opcode);
    if (op == up::OpCode::kSoftmaxXEntFwd ||
        op == up::OpCode::kSoftmaxXEntMaskedFwd || op == up::OpCode::kMseFwd ||
        op == up::OpCode::kKLDistillFwd)
      p.bwd_begin = i + 1;
    if ((op == up::OpCode::kClipNorm || op == up::OpCode::kSqNormAccum ||
         op == up::OpCode::kClipGlobal || op == up::OpCode::kSgdStep ||
         op == up::OpCode::kAdamWStep) &&
        p.opt_begin == train.size())
      p.opt_begin = i;
  }
  if (p.opt_begin < p.bwd_begin) p.opt_begin = p.bwd_begin;
  return p;
}

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_
