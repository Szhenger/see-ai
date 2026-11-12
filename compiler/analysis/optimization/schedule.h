#ifndef SEEML_COMPILER_ANALYSIS_OPTIMIZATION_SCHEDULE_H_
#define SEEML_COMPILER_ANALYSIS_OPTIMIZATION_SCHEDULE_H_

#include <cstdint>
#include <expected>
#include <string>

#include "source/plan/config.h"

// =============================================================================
// The learning-rate schedule and the accumulation seed (E8 #91, roadmap 2a;
// moved out of the driver by S5 #151): minimization's rules, beside the
// optimizer synthesis they parameterize.
// =============================================================================

namespace seeml::update {

/// The schedule triple (E8, #91). Each of these used to compile and then
/// misbehave silently at run time: a zero budget is a plan with nothing to
/// default to; a warmup that swallows the budget leaves no horizon, so
/// every later step ran at the floor; warmup under a constant schedule was
/// ignored; and a zero floor trains the schedule's final step — and any
/// step past a horizon — at LR 0, moving nothing but AdamW's moments.
[[nodiscard]] std::expected<void, std::string> VerifySchedule(
    const OptimizerSpec& optimizer, uint64_t default_steps);

/// The schedule as the plan header records it. The floor and warmup are the
/// cosine schedule's; a constant-schedule plan records neither (VerifySchedule
/// proved it carries no warmup), so such plans are byte-identical to the
/// ones compiled before the floor's default moved from 0 to 0.1.
struct ScheduleWords {
  uint32_t lr_schedule = 0;
  uint64_t warmup_steps = 0;
  float min_lr_factor = 0.0f;
};
ScheduleWords ScheduleWordsOf(const OptimizerSpec& optimizer);

/// The seed dL/dL handed into autodiff under gradient accumulation: each
/// micro-batch's gradient is a mean over its own rows, so 1/G makes G
/// folded gradients the mean over the effective batch of batch x G rows.
inline float AccumulationSeed(uint32_t accum) {
  return 1.0f / static_cast<float>(accum);
}

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_OPTIMIZATION_SCHEDULE_H_
