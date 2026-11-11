#ifndef SEEML_SOURCE_RECIPE_RUN_FLAGS_H_
#define SEEML_SOURCE_RECIPE_RUN_FLAGS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "source/plan/config.h"  // LrSchedule
#include "source/plan/schema.h"

// =============================================================================
// The runner's flags, held to the plan they run (S4 #150): the generated
// runner validates its command line with the recipe's own rules, so a flag
// that cannot apply is refused on the device as it is on the build host.
// Pure and header-only; vendored with the runner.
// =============================================================================

namespace seeml::update::recipe {

/// The runner flags the rules read, as parsed (ParseU64Strict for the
/// integers: no sign, no whitespace).
struct RunFlags {
  bool steps_given = false;
  uint64_t steps = 0;             // 0 = the plan's compiled default
  // A checkpoint is there to resume from, and it is this plan's — not the
  // bare --resume flag: with no checkpoint, or one bound to another plan,
  // the engine starts a fresh run, and the fresh run's rule applies to it.
  bool resume = false;
  bool has_checkpoint_path = false;
  bool checkpoint_every_given = false;
  uint64_t checkpoint_every = 0;
  bool eval_every_given = false;  // an explicit count, not "auto"
  uint64_t eval_every = 0;
  bool has_validation = false;    // a held-out split exists for this run
};

/// The first rule the flags break against `header`, or nothing.
inline std::optional<std::string> RunFlagsViolation(const PlanHeader& header,
                                                    const RunFlags& f) {
  // Review N20: a fresh run's --steps is the schedule's horizon; inside
  // the compiled warmup the cosine phase has no steps at all and the run
  // ends mid-ramp. This is the early usage error for a fresh run, from
  // the flags and the header alone. The rule itself is the engine's
  // (UpdateEngine::Train refuses any horizon inside the warmup): only it
  // knows a resume's step, so a resume whose step + steps still ends on the
  // ramp, and a run whose checkpoint did not load, are refused there.
  if (f.steps_given && !f.resume && f.steps != 0 &&
      header.lr_schedule == static_cast<uint32_t>(LrSchedule::kCosineWithWarmup) &&
      f.steps <= header.warmup_steps)
    return "--steps " + std::to_string(f.steps) +
           " must exceed the plan's compiled warmup (" +
           std::to_string(header.warmup_steps) +
           " steps): the run would end inside the ramp";
  // Review N24: two flags that were silently ignored without their partner.
  if (f.checkpoint_every_given && f.checkpoint_every != 0 &&
      !f.has_checkpoint_path)
    return std::string("--checkpoint-every needs --checkpoint: there is no "
                       "file to write the checkpoints to");
  if (f.eval_every_given && f.eval_every != 0 && !f.has_validation)
    return std::string("--eval-every needs a validation split (a plan that "
                       "compiled one, or --val-frac > 0): there is nothing "
                       "to evaluate on");
  return std::nullopt;
}

}  // namespace seeml::update::recipe

#endif  // SEEML_SOURCE_RECIPE_RUN_FLAGS_H_
