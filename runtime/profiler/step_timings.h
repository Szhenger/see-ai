#ifndef SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_
#define SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>
#ifdef SEEML_STEP_TIMING
#include <chrono>
#endif

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

/// Every SEEML_STEP_TIMING site, in one place (#202 item 3): the dispatcher
/// calls these hooks and holds no clock of its own. Built without
/// SEEML_STEP_TIMING, each hook is the bare call it wraps — TrainProgram
/// runs the one range [0, n), StepProgram runs the step program, CountStep
/// does nothing — so the shipped runtime reads no clock inside the step.
/// The macro-dependent bodies are instantiated only in the dispatcher's
/// translation unit, which each binary builds one way; the data layout is
/// unconditional, as StepTimings' is.
class StepProfiler {
 public:
  /// The phase boundaries of a newly bound train program, and a fresh split.
  void Bind([[maybe_unused]] const std::vector<seeml::update::UpdateInstruction>& train) {
#ifdef SEEML_STEP_TIMING
    phases_ = ScanPhases(train);
    timings_ = {};
#endif
  }

  /// Runs the train program of `n` instructions through `run(begin, end)`:
  /// split at the phase boundaries and clocked per phase when timing.
  template <class RunRange>
  std::expected<void, std::string> TrainProgram(size_t n, RunRange&& run) {
#ifdef SEEML_STEP_TIMING
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    if (auto r = run(size_t{0}, phases_.bwd_begin); !r) return r;
    const auto t1 = clock::now();
    if (auto r = run(phases_.bwd_begin, phases_.opt_begin); !r) return r;
    const auto t2 = clock::now();
    if (auto r = run(phases_.opt_begin, n); !r) return r;
    const auto t3 = clock::now();
    timings_.fwd_seconds += std::chrono::duration<double>(t1 - t0).count();
    timings_.bwd_seconds += std::chrono::duration<double>(t2 - t1).count();
    timings_.opt_seconds += std::chrono::duration<double>(t3 - t2).count();
    return {};
#else
    return run(size_t{0}, n);
#endif
  }

  /// Runs the optimizer program (gradient accumulation) through `run()`,
  /// charged to the optimizer phase when timing.
  template <class Run>
  std::expected<void, std::string> StepProgram(Run&& run) {
#ifdef SEEML_STEP_TIMING
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto r = run();
    timings_.opt_seconds +=
        std::chrono::duration<double>(clock::now() - t0).count();
    return r;
#else
    return run();
#endif
  }

  /// One optimizer step (G grad executions plus the step, under
  /// accumulation) into the split's denominator.
  void CountStep() {
#ifdef SEEML_STEP_TIMING
    ++timings_.steps;
#endif
  }

  const StepTimings& timings() const { return timings_; }
  void Reset() { timings_ = {}; }

 private:
  StepTimings timings_{};
  PhaseBoundaries phases_{};
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PROFILER_STEP_TIMINGS_H_
