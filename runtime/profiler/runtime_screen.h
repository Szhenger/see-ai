#ifndef SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_
#define SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_

#include <optional>
#include <string_view>

#include "runtime/dispatcher/update_engine.h"
#include "source/statistics/screen.h"

namespace seeml::update_rt {

// runtime/profiler/energy_meter.h; declared here so an includer of the
// screen does not take the meter's CoreFoundation headers with it.
struct EnergyReading;

/// The runtime screen (S0 #146; the profiler's, S7 #153): the plan's own
/// numbers — the identity manifest's hashes, the memory the device holds,
/// the work of a step and the step `peak_gflops` predicts — beside what the
/// run reported: the measured step, achieved rate and utilization, peak RSS
/// against the plan, checkpoint cost, the losses, the input-blind loss
/// (`label_entropy`, for a plan that trains on class labels), the accuracy
/// and the gate's `verdict`. With an `energy` reading of the training run
/// (#202), the energy rows (FillEnergyRows). model_update prints it;
/// seeml-bench carries it in each row.
void FillRuntimeScreen(seeml::update::Screen& runtime_screen,
                       const UpdateEngine& engine, const TrainReport& report,
                       double peak_gflops, std::string_view verdict,
                       std::optional<double> label_entropy,
                       const EnergyReading* energy = nullptr);

/// The energy rows (#202 item 3): on a GPU run, the GPU domain's joules over
/// the training run and per token (`tokens` = steps x rows per optimizer
/// step); otherwise, and whenever that domain read nothing, energy_meter
/// says why and no number is shown. A cpu run never gets a number — the
/// seeml-bench rule: the counters are whole-SoC (the GPU's moves with the
/// display, not the update) and an unprivileged reader sees the CPU's in
/// bursts, a backlog landing at once (docs/benchmarks.md), so an interval's
/// CPU joules are not the run's.
void FillEnergyRows(seeml::update::Screen& runtime_screen, bool gpu_backend,
                    uint64_t tokens, const EnergyReading& energy);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PROFILER_RUNTIME_SCREEN_H_
