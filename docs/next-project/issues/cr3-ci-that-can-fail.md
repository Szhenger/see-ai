---
title: "SeeAI CR3: CI that can fail — the seam job's lost exit status first, then the fuzzer that never runs an accepted plan, the instrumentation that only nightly builds, and the tests whose assertions cannot fire"
number: 144
labels: bug,testing,correctness
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The review's one high finding in CI: the seam job pipes `unittest` into
`tee` without `pipefail`, so a failed Python suite merges green (F00). With
it: the `! grep` doctrine check is not enforced under errexit (F01); the
plan fuzzer never executes an accepted plan (F22) and the fuzz target
instruments only the harness file (N41); the HF-import compile check
hard-codes a build directory, skips silently and runs before any build
(F36); `seeml-bench` and `SEEML_STEP_TIMING` code compile only in the
nightly job (N44) and the NumPy- and tier-2-gated Python tests run in no job
(N45); the `SEEML_ACCELERATE` path is never compiled or tested (N43); the
overflow tests do not overflow (F30, F31), the addend rejection fires on the
wrong check (F32), the non-const-weight test tests a missing tensor (F26),
the concurrent-writer test passes when every write fails (F27), the accuracy
test cannot detect the leak it guards (F28); `build.sh` treats
`SEEML_ACCELERATE=off` as on (F23). F9 (#140) owns the measurement-code
items (N43, N44, N45, F36) and is referenced, not duplicated. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

F00 ships first as its own one-line PR. Then, in one PR: `set -o pipefail`
and explicit `if grep` in every workflow step; the fuzzer runs one bounded
step under ASan per accepted plan with the parsers instrumented; each named
test rewritten so removing the guard it protects fails it; the falsy-value
check shared by every build-script switch.

## Status 2026-11-05

F00 landed first, on its own (the seam job's pipe status checked behind the
`tee`, the skipped-test check an explicit `if grep`). Then, in one commit:
both workflows name their shell (`defaults: run: shell: bash`, which is
`bash -eo pipefail` for every step), every `! grep` is an explicit if that
exits 1, the one pipe into `grep -q` is a file and a grep (SIGPIPE under
pipefail); `build.sh`'s three switches and the emitted build script's Metal
switch share one falsy rule — `SEEML_NO_METAL=0` now leaves Metal built,
the one semantic change (F23); the six named tests rewritten and proven by
mutation — each fails with the guard it protects removed (the table is in
the PR): F30 targets the byte count, F31 a count that wraps small and
nonzero, F32 asserts the flags guard by message with a rodata B, F26
presents a non-constant weight (the resolver tells a missing tensor from a
non-constant one), F27 asserts both writers' results, F28 uses non-uniform
labels. F36 was already fixed (the HF-import check builds first and fails
on a skip). F22 / N41 followed: the fuzzer executes an accepted plan with
the libraries instrumented and the job gates on coverage — and caught its
first bug within the day: a header asking for an arena of exabytes passed
every contract (the arena size was bounded against overflow alone) and
reached `aligned_alloc`; the engine refuses an arena beyond the host's
memory before the allocation now, by name, and the harness bounds the
arena before the load as it bounds the step. N43 / N44 / N45
are F9's (#140).

The first run that could fail did, twice, on checks that had never run:
the seam job's Python suite had died on import since it was written (the
path form `-m unittest test/tool/formats_test.py` imports `test.tool`,
which the hosted Python's standard-library `test` package shadows) and its
`! grep skipped` turned the ImportError into a pass — it runs by discovery
now, and its twelve tests ran in CI for the first time; and the e2e job's
"nothing Python ships in the package" grep, dead under `set -e` as a
`!`-inverted non-final line, matched the word in a comment of the vendored
`dataset.h` — it asserts a dependency's shapes now (an interpreter invoked
by a packaged script, an embedding API in a packaged source, the word in
the binary), not prose.

## Status 2026-11-20 — the F22 / N41 claim corrected (#208, #207)

The paragraph above says the fuzzer executes an accepted plan and the job
gates on coverage. In CI neither was true (the 2026-11-17 review; #208):
the pull-request job started from an empty corpus and in ninety seconds
never built a plan that passes the seal and the contract — its own log
lists `ExecuteOnce`, `UpdateEngine::Train` and every function of the
validator with zero hits — and its gate, `grep "COVERED_FUNC.*runtime/"`,
is satisfied by any `UNCOVERED_FUNC` line under `runtime/`. Two more
things were found by running the harness on a valid plan: its one
training step was refused for every plan compiled under the default
schedule (a one-step horizon ends inside the compiled warmup), and its
work bound did not cover `grad_accum_steps` — a one-bit mutation of an
accumulating plan runs a "single step" of a billion micro-batches, a unit
that prints nothing and does not return (a candidate for the nightly's
three cancelled runs, #207; the nightly's own input is lost, so that is
not shown).

What each place does, as of this change:

- **Every suite-running job** (the three build-and-test hosts, the
  determinism matrix, ASan+UBSan per pull request, TSan nightly):
  `seeml_fuzz_replay_test` runs the checked-in seed corpus
  (`test/fuzz/seeds/`, one valid input per arm, one sealed plan per fixture
  family) through the harness's entry point and asserts, per seed, what it
  reached — a plan accepted and executed with its family's opcodes, a
  compile completed and its plan run, a tokenizer section loaded, a window
  ingested, a recipe applied, a report read — then refuses a battery of
  malformed inputs and runs 128 structured mutations of every seed. This is
  where "the harness executes an accepted plan" is held, and it needs no
  workflow.
- **The pull-request fuzz job** (`fuzz-smoke`, 90 s) and **the nightly**
  (`fuzz-extended`, 15 min, cached corpus) build the same harness with
  libFuzzer. Until the workflow change that accompanies #208 / #207 is
  applied they still start from an empty (or cached) corpus with the old
  gate and no `-timeout`: they exercise the refusal paths, and a finding
  made on a cancelled nightly is not uploaded. With it applied: both start
  from `test/fuzz/seeds`; the pull-request job fails unless libFuzzer
  reports non-zero hits for `ExecuteOnce`, the validator, `UpdateEngine::
  Train`, the instruction dispatch, a completed compile and each
  text-facing parser; both pass `-timeout=25 -rss_limit_mb=2048`; the
  nightly's fuzz step has its own time limit below the job's, uploads every
  finding prefix and the log's tail on `always()`, and saves its corpus
  with an explicit step on `always()`.

## Acceptance

- A deliberately failing Python test turns the seam job red.
- Each listed test fails when its guard is removed (mutation check in the
  PR description).
- The fuzz corpus reports edges inside `runtime/` and `compiler/`, not only
  the harness.
