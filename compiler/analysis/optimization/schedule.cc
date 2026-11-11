#include "compiler/analysis/optimization/schedule.h"

#include "compiler/diagnostics/generating/error.h"

namespace seeml::update {

namespace generating = seeml::diag::generating;

// The diagnostics keep the driver's unit: they are the compile's refusal of
// its own configuration, worded as they were before the rule moved here.
std::expected<void, std::string> VerifySchedule(const OptimizerSpec& o,
                                                uint64_t default_steps) {
  const bool cosine = o.lr_schedule == LrSchedule::kCosineWithWarmup;
  if (default_steps == 0)
    return generating::Error(generating::kDriver,
                             "the step budget must be positive");
  if (!cosine && o.warmup_steps != 0)
    return generating::Error(
        generating::kDriver,
        "warmup applies to the cosine schedule only; under the constant "
        "schedule it would be silently ignored");
  if (cosine && o.warmup_steps >= default_steps)
    return generating::Error(
        generating::kDriver,
        "warmup (" + std::to_string(o.warmup_steps) +
            " steps) must be shorter than the step budget (" +
            std::to_string(default_steps) + ")");
  if (cosine && !(o.min_lr_factor >= 0.0f && o.min_lr_factor <= 1.0f))
    return generating::Error(generating::kDriver,
                             "min_lr_factor must lie in [0, 1]");
  if (cosine && o.min_lr_factor == 0.0f && !o.allow_zero_lr)
    return generating::Error(
        generating::kDriver,
        "a cosine floor of 0 trains the schedule's last step at a learning "
        "rate of zero; set min_lr_factor > 0 (the default is 0.1) or "
        "allow_zero_lr to ask for it deliberately");
  return {};
}

ScheduleWords ScheduleWordsOf(const OptimizerSpec& o) {
  const bool cosine = o.lr_schedule == LrSchedule::kCosineWithWarmup;
  return {.lr_schedule = static_cast<uint32_t>(o.lr_schedule),
          .warmup_steps = cosine ? o.warmup_steps : 0,
          .min_lr_factor = cosine ? o.min_lr_factor : 0.0f};
}

}  // namespace seeml::update
