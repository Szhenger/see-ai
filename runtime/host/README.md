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

Exit codes: 0 committed, 1 a runtime or input failure, 2 a usage error,
3 rejected by the gate, 4 committed with a failed output.

Not here yet:

- cooperative cancellation (`TrainOptions::should_stop` exists, but nothing
  sets it);
- a thread and affinity policy (`SEEML_THREADS` is read by the pool).

It is not compiled into any library of the tree (it defines `main`). Every
package build compiles it, and so do the emitter tests and the end-to-end
CI jobs.
