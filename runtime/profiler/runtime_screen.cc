#include "runtime/profiler/runtime_screen.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include "runtime/profiler/energy_meter.h"
#include "source/parallel/parallel_for.h"
#include "source/statistics/plan_work.h"
#include "source/platform/memory.h"
#include "source/statistics/screens.h"

namespace seeml::update_rt {

void FillRuntimeScreen(seeml::update::Screen& runtime_screen,
                       const UpdateEngine& engine, const TrainReport& report,
                       double peak_gflops, std::string_view verdict,
                       std::optional<double> label_entropy,
                       const EnergyReading* energy) {
  const uint64_t planned = engine.header().arena_size + engine.plan_size();
  const seeml::update::PlanWork work = seeml::update::MeasurePlanWork(
      {engine.plan_data(), engine.plan_size()});
  const double flops = static_cast<double>(work.FlopsPerStep());
  // The identity manifest (S4 #150): the four inputs' hashes, as the plan
  // carries them — the model's was verified against the file at commit
  // binding, the corpus's and the tokenizer's by the feeder contract.
  {
    auto hex = [](uint64_t v) {
      char buf[24];
      std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
      return std::string(buf);
    };
    const seeml::update::PlanHeader& h = engine.header();
    if (h.source_model_hash)
      runtime_screen.Set("model_hash", hex(h.source_model_hash));
    if (h.tokenizer_hash)
      runtime_screen.Set("tokenizer_hash", hex(h.tokenizer_hash));
    if (h.data_flags & seeml::update::kDataFlagCorpusPinned)
      runtime_screen.Set("corpus_hash", hex(h.corpus_hash));
    if (engine.metadata().present)
      runtime_screen.Set("recipe_hash",
                         hex(engine.metadata().meta.recipe_hash));
  }
  runtime_screen.Set("device_bytes", planned);
  runtime_screen.Set("flops_per_step", work.FlopsPerStep());
  if (peak_gflops > 0.0)
    runtime_screen.Set("predicted_step_ms", flops / (peak_gflops * 1e6));
  runtime_screen.Set("backend", std::string(engine.backend_name()));
  runtime_screen.Set("steps", report.steps);
  if (report.steps > 0 && report.step_seconds > 0.0) {
    const double step_ms = report.step_seconds * 1e3 /
                           static_cast<double>(report.steps);
    runtime_screen.Set("measured_step_ms", step_ms);
    const double achieved = flops / (step_ms * 1e6);  // GFLOP/s
    runtime_screen.Set("achieved_gflops", achieved);
    if (peak_gflops > 0.0)
      runtime_screen.Set("utilization", 100.0 * achieved / peak_gflops);
  }
  if (const uint64_t rss = seeml::update::PeakRssBytes(); rss != 0) {
    runtime_screen.Set("peak_rss_bytes", rss);
    if (planned != 0)
      runtime_screen.Set("rss_over_planned", static_cast<double>(rss) /
                                                   static_cast<double>(planned));
  }
  // The storage role (CR4 #145): what a checkpoint cost on disk.
  if (report.checkpoints > 0) {
    runtime_screen.Set("checkpoints", report.checkpoints);
    runtime_screen.Set("checkpoint_bytes",
                       report.checkpoint_bytes / report.checkpoints);
    runtime_screen.Set("checkpoint_fsyncs",
                       static_cast<double>(report.checkpoint_fsyncs) /
                           static_cast<double>(report.checkpoints));
  }
  // A run that trained no step (a resume with nothing left) measured no
  // training loss: its zero placeholders are not losses.
  if (report.steps > 0) {
    runtime_screen.Set("train_loss_first",
                       static_cast<double>(report.initial_avg_loss));
    runtime_screen.Set("train_loss_last",
                       static_cast<double>(report.final_avg_loss));
  }
  if (report.has_validation) {
    runtime_screen.Set("validation_loss_first",
                       static_cast<double>(report.val_initial_loss));
    // A stopped run (#202) never scored its end state.
    if (!report.stopped_early)
      runtime_screen.Set("validation_loss_last",
                         static_cast<double>(report.val_final_loss));
  }
  // The input-blind reference is a cross-entropy over the labels the
  // plan trains on: a plan that stages none (distillation, whose loss is a
  // KL) has no such level — the caller passes none.
  if (label_entropy) runtime_screen.Set("label_entropy", *label_entropy);
  if (report.has_val_accuracy && !report.stopped_early)
    runtime_screen.Set("validation_accuracy",
                       static_cast<double>(report.val_final_accuracy));
  runtime_screen.Set("verdict", std::string(verdict));
  if (energy) {
    // Only the domain that is this run's own compute: the counters are
    // whole-SoC, so on a CPU run the GPU's counter moves with the display,
    // not with the update. seeml-bench's energy_j_per_step is stricter: it
    // reads the GPU's alone and gives a CPU run no number, because an
    // unprivileged reader on macOS 26 sees the CPU's counter in bursts (a
    // backlog landing at once; docs/benchmarks.md). A CPU number here is
    // whole-SoC and may carry such a burst.
    const bool gpu =
        std::string_view(engine.backend_name()).starts_with("metal");
    const std::string domain = gpu ? "GPU Energy" : "CPU Energy";
    const double joules = EnergyMeter::Domain(*energy, domain);
    const uint64_t tokens =
        report.steps * engine.header().batch *
        std::max<uint64_t>(1, engine.header().grad_accum_steps);
    if (joules > 0.0) {
      runtime_screen.Set("energy_meter", "IOReport " + domain);
      runtime_screen.Set("energy_joules", joules);
      if (tokens > 0)
        runtime_screen.Set("energy_per_token",
                           joules / static_cast<double>(tokens));
    } else {
      runtime_screen.Set(
          "energy_meter",
          "unavailable: " +
              (!energy->available
                   ? energy->reason
                   : "the " + domain + " counter did not move for this "
                     "reader" +
                         (gpu ? std::string()
                              : std::string(" (an unprivileged reader sees "
                                            "the CPU's in bursts, or not at "
                                            "all, on macOS 26)"))));
    }
  }
  // The pool width this run used (#202 item 1): a throughput knob, never a
  // numerics one, recorded so a measured step is read at its width.
  runtime_screen.Set("threads",
                     static_cast<uint64_t>(seeml::update::ParallelThreadCount()));
}

}  // namespace seeml::update_rt
