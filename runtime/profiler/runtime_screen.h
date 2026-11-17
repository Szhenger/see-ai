#ifndef SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_
#define SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_

#include <optional>
#include <string_view>

#include "runtime/dispatcher/update_engine.h"
#include "runtime/profiler/energy_meter.h"
#include "source/statistics/screen.h"

namespace seeml::update_rt {

/// The runtime screen (S0 #146; the profiler's, S7 #153): the plan's own
/// numbers — the identity manifest's hashes, the memory the device holds,
/// the work of a step and the step `peak_gflops` predicts — beside what the
/// run reported: the measured step, achieved rate and utilization, peak RSS
/// against the plan, checkpoint cost, the losses, the input-blind loss
/// (`label_entropy`, for a plan that trains on class labels), the accuracy
/// and the gate's `verdict`. With an `energy` reading of the training run
/// (#202), the energy rows: the domain that is this run's compute — the
/// GPU's on Metal, the CPU's on cpu — or why there is none. model_update
/// prints it; seeml-bench carries it in each row.
void FillRuntimeScreen(seeml::update::Screen& runtime_screen,
                       const UpdateEngine& engine, const TrainReport& report,
                       double peak_gflops, std::string_view verdict,
                       std::optional<double> label_entropy,
                       const EnergyReading* energy = nullptr);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_
