# runtime/profiler — measured beside predicted

- `step_timings.h` (S7 #153): `StepTimings` holds the step split, the seconds spent in the forward, backward and optimizer phases. It is filled in `SEEML_STEP_TIMING` builds. `ScanPhases` finds where those phases begin in the train program.
- `energy_meter.h` (F6 #134; here since #202): `EnergyMeter` reads the SoC's energy counters through IOReport, which it opens at run time. It is real under Apple clang, which needs CoreFoundation and links it in packages built there. Everywhere else it is a stub that reports itself unavailable.
- `runtime_screen.{h,cc}`: `FillRuntimeScreen` fills the runtime screen. On one side are the plan's own numbers: the identity manifest's hashes, the memory the device holds, the work of a step, and the step time the stated peak predicts. Beside them is what the run measured: step time, achieved rate and utilization, peak RSS against the plan, checkpoint cost, the losses, the input-blind loss, the accuracy and the gate's verdict.
  - With an energy reading of the training run (#202), it adds the energy rows (`FillEnergyRows`): `energy_meter`, `energy_joules` and `energy_per_token` (joules ÷ the tokens trained, `TokensTrained`: steps × rows per optimizer step). A GPU run reads the GPU domain. A cpu run never gets a number, which is seeml-bench's rule: the counters are whole-SoC, so the GPU's moves with the display rather than the update, and an unprivileged reader sees the CPU's in bursts, a backlog landing at once. Whenever no number is shown, `energy_meter` says why, the most basic reason first: a host with no meter to read (off Apple, no library; the reading's `metered` is false) gets the meter's own reason on either backend, then a cpu run the rule, then a GPU run whose domain read nothing the reading's reason.
  - `model_update` prints this screen and writes it to `--report`.
  - `seeml-bench` carries the same struct in every thread row, under `"runtime"` (`test/tool/bench_runtime_test.py`).

The always-on step and evaluation clocks are still read in the engine's train loop, as `TrainReport::step_seconds` and `eval_seconds`. The screen reports them.

Every instrumentation site goes through the profiler (#202 item 3):

- **`StepProfiler`** (`step_timings.h`) owns the `SEEML_STEP_TIMING` phase scan, clocks and step count. The dispatcher calls its hooks and keeps no clock of its own. In every other build, each hook is the bare call it wraps.
- **`KernelProfile`** and **`SpanLog`** (`kernel_profile.h`) hold and print the Metal backend's `SEEML_METAL_PROFILE` per-kernel table and its `SEEML_METAL_SPANS` command-buffer rows. They are plain C++. The backend keeps only the decisions it has to make: serializing the stream per instruction for the profile, and reading each command buffer's GPU timestamps.
