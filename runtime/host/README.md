# runtime/host — the package entry

`update_main.cc` (S7 #153) is the generated main every package carries. The
emitter (`compiler/backend/packaging/native_emitter.cc`) reads it from the
repository root and writes it into the package as `update_main.cc`. On the
way it substitutes two markers: the compile's stated peaks,
`0.0 /*seeml:peak_gflops*/` and `0.0 /*seeml:gpu_peak_gflops*/`.

It holds the host role:

- **Flags:** strict parsing, refused by name, and held to the plan with
  `recipe::RunFlagsViolation`.
- **Startup:** backend selection, the plan load (embedded or beside the
  binary, #158), the corpus and the validation split.
- **The run and the verdict:** exit 0 accepted, 3 rejected.
- **Output:** the runtime screen (the profiler's `FillRuntimeScreen`) and
  the staged `--report`.
- **Commit:** merge and the durable commit, with the committed model's
  exporter report. Exit 4 when a requested output failed after the commit
  landed.

Exit codes:

- 0 committed;
- 1 a runtime or input failure;
- 2 a usage error;
- 3 rejected by the gate;
- 4 committed with a failed output;
- 5 stopped by SIGINT or SIGTERM. The model is untouched, and with
  `--checkpoint` the stop is checkpointed, so `--resume` finishes the run.

**Cancellation (#202 item 1):**

- **The signal:** a SIGINT or SIGTERM sets an atomic flag, which the engine
  polls once per step through `TrainOptions::should_stop`. The handlers are
  installed just before training, so a signal during load keeps its default
  action, since there is nothing to checkpoint yet. A second signal kills
  (`SA_RESETHAND`).
- **The stop:** the run stops at the step boundary and writes a checkpoint
  there before anything else touches the arena. It skips the closing
  evaluation, and the gate never sees the partial update: `--force`
  cannot commit it.
- **The resume:** resuming from that checkpoint commits the bytes the
  uninterrupted run commits (`test/tool/runner_stop_test.py`).

**Threads:** `--threads N` sets the worker pool, over `$SEEML_THREADS` and
the hardware's count, before the plan loads. The runtime screen records the
width in its `threads` row. Every width computes the same bits.

Not here yet: a QoS or affinity policy for the workers. It would change
every absolute timing in the shared pool, so it waits for a host that can be
measured cleanly (#202).

It is not compiled into any library of the tree (it defines `main`). Every
package build compiles it, and so do the emitter tests and the end-to-end
CI jobs.
