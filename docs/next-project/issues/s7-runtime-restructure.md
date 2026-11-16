---
title: "SeeAI S7: runtime restructure — loader, gating, profiler and host split out of the engine; verifier, pipeline, storage renamed; the runtime as the fourth reporter"
number: 153
labels: enhancement,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

`runtime/engine/update_engine.cc` is 1,214 lines holding four roles: plan
loading with the seal and source-model hash checks, the dispatch loop, the
gate with best-state tracking and patience, and the instrumentation behind
two environment variables spread over fifteen sites. The review found what
that hides: the step program left out of the loader's softmax and vocabulary
scans (F07, a heap write from a plan that passed every contract); header
refs that may carry the source bit and bypass the feeder's validated labels
(N04); the provenance proof assuming the program never writes the IO slots
(N03); rodata alignment never proven (N05); a patience stop not honored on
resume (N12). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §6.

## Design

1. Nine roles in lifecycle order: `host/` (the generated main and flag
   validation from S4; cancellation; thread and affinity policy), `loader/`
   (map, seal, header, source hash, resource contract, the one arena
   allocation; manifest, tokenizer hash and ledger verified — the security
   boundary; F07/N03/N04/N05 fixed here as the coverage table of cr2
   demands), `verifier/` (was validator), `pipeline/` (was feeder; runs the
   vendored contract check, sanitizer and tokenizer on device data),
   `dispatcher/`, `executor/`, `gating/` (evaluate, best state, patience —
   N12 fixed here), `storage/` (was custodian), `profiler/` (the fifteen
   sites in one place; the runtime screen: measured step split beside the
   backend's prediction, RSS beside arena, loss beside the loss floor,
   energy per token where the platform reports it), with `diagnostics/`
   cross-cutting.
2. `source/platform/` seam: file mapping, durable write and replace, memory
   query, host query — POSIX now, Win32 later (the storage commit proof is
   re-derived per OS and is a follow-up, §9).

## Acceptance

- Every emitted package is byte-identical in behavior: same committed model
  hashes on the fixtures and on SmolLM-135M before and after the move.
- F07, N03, N04, N05, N12 reproductions refused or corrected, with tests.
- `model_update` prints the runtime screen by default; `seeml-bench` rows
  carry the same struct.

## Status (2026-11-16, PR #200)

Acceptance met. Moves only; each was verified on its own:

- the nine committed-model hashes (fixtures, checkpoint/resume, a split
  patience run, int8, global clip, Metal, SmolLM-135M on Metal) are the
  pre-move baseline's;
- 33 C++ suites, the Python tool suites, plan identity.

| Move | Where | What it holds |
|---|---|---|
| 1 loader | `runtime/loader/` | `DecodePlan` (header, metadata, programs, the eval tail, the scans F07 fixed), `AllocateArena`, the plan contract (moved from `dispatcher/`) |
| 2 gating | `runtime/gating/` | `GateState`: the source model's score, the best state, patience |
| 3 profiler | `runtime/profiler/` | `StepTimings`, `ScanPhases`, `FillRuntimeScreen`; `seeml-bench` thread rows carry the same screen under `"runtime"` |
| 4 host | `runtime/host/update_main.cc` | the generated main, out of the compiler's string literal; the emitter reads and vendors it |
| 5 platform | `source/platform/` | `file_map.h`, `memory.h`, `durable.h`: header-only, POSIX |
| 6 N12 | `dispatcher/update_engine.cc` | a resume whose checkpoint spent the patience stops where the uninterrupted run did |

The six moves cover the acceptance items as follows:

- **Identical behaviour:** the committed hashes above.
- **F07, N03, N04, N05:** corrected with tests under #143 (CR residuals,
  #183).
- **N12:** `UpdateEngineBest.AResumeWithSpentPatienceStopsWhereTheRunStopped`
  fails without the fix (the resume trains one more step) and passes with
  it.
- **The shared screen:** `model_update` prints the runtime screen, and
  `seeml-bench` rows carry the same struct (`test/tool/bench_runtime_test.py`).

Design text not delivered here, for a follow-up issue:

- **host:** cooperative cancellation (`TrainOptions::should_stop` exists,
  but nothing sets it from a signal); a thread and affinity policy.
- **pipeline:** the device raw-text feeder that calls the vendored
  tokenizer and ingest.
- **profiler:** energy per token; the `SEEML_STEP_TIMING` and
  `SEEML_METAL_PROFILE` sites still sit in the dispatcher and the Metal
  backend.
- **platform:** the storage writer's sidecar writes, renames and lock still
  call the OS directly; the Win32 commit proof (§9).
