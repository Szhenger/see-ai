# runtime/profiler — measured beside predicted

- `step_timings.h` (S7 #153): `StepTimings` holds the step split, the seconds spent in the forward, backward and optimizer phases. It is filled in `SEEML_STEP_TIMING` builds. `ScanPhases` finds where those phases begin in the train program.
- `energy_meter.h` (F6 #134; here since #202): `EnergyMeter` reads the SoC's energy counters through IOReport, which it opens at run time. It is real under Apple clang, which needs CoreFoundation and links it in packages built there. Everywhere else it is a stub that reports itself unavailable.
- `runtime_screen.{h,cc}`: `FillRuntimeScreen` fills the runtime screen. On one side are the plan's own numbers: the identity manifest's hashes, the memory the device holds, the work of a step, and the step time the stated peak predicts. Beside them is what the run measured: step time, achieved rate and utilization, peak RSS against the plan, checkpoint cost, the losses, the input-blind loss, the accuracy and the gate's verdict.
  - With an energy reading of the training run (#202), it adds the energy rows: `energy_meter`, `energy_joules` and `energy_per_token` (joules ÷ steps × rows per optimizer step). They read only the domain that is the run's own compute: the GPU's on Metal, the CPU's on cpu. The counters are whole-SoC, so on a CPU run the GPU's counter moves with everything else. When that domain read nothing, `energy_meter` says why, and no number is shown.
  - `model_update` prints this screen and writes it to `--report`.
  - `seeml-bench` carries the same struct in every thread row, under `"runtime"` (`test/tool/bench_runtime_test.py`).

The always-on step and evaluation clocks are still read in the engine's train loop, as `TrainReport::step_seconds` and `eval_seconds`. The screen reports them.

Not gathered here yet: the `SEEML_STEP_TIMING` sites that fill `StepTimings` (the phase clocks in `ExecuteTrainProgram` and `ExecuteStepProgram`, the step counts in `ExecuteTrainOnce` and the train loop, all in `dispatcher/update_engine.cc`) and the `SEEML_METAL_PROFILE` per-instruction spans in `executor/metal_backend.mm`.
