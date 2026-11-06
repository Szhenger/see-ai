---
title: "SeeAI S0: no leaky abstractions — every subsystem prints its first-principles statistics screen, one struct rendered to console and report, tested on the fixtures"
number: 146
labels: enhancement,doctrine,design,docs
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P0
---
## Why

The compiler computes almost every number an engineer would want and prints
eight prose notes and one summary line. The compile report on disk carries
17 passes with op counts and milliseconds, the resource analyzer's byte
split, 211 adapter geometries with their scales, and the three programs'
instruction counts; the console shows none of it. The runtime knows the
per-phase step split, the per-opcode Metal profile and the arena, and prints
"resource contract" and "accepted". This issue heads the milestone: the
doctrine that each subsystem explains itself quantitatively. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §1–§2.

## Design

1. One statistics struct per subsystem (frontend, analysis, backend,
   runtime); the console screen and the compile report's JSON are two
   renderings of that struct — one source of truth, the P6 seam rule.
2. One screen per subsystem by default, per-tensor and per-op detail behind
   `--verbose`. Every number carries its unit and its derivation clause.
3. The backend screen ends with a predicted step time (plan FLOPs ÷ the
   host peak the user passes, the `--peak-gflops` convention of
   `seeml-bench`); the runtime's screen prints the measured step beside it,
   so the ratio is the utilization and the gap is where the next issue
   comes from. RSS beside arena; loss beside the accountant's floor.
4. Screens print aggregates, hashes and losses, never record contents.
5. Golden statistic blocks for the standard fixtures in the test suite, so a
   pass that silently changes what it does changes a number the suite sees.
6. `docs/usage.md` gains "What the compiler tells you": the real
   SmolLM-135M screens, annotated.

## Status 2026-11-06

Landed on `claude/v100b-s0` (#181) in five steps. (1) `source/statistics/`: a screen is one static row table — key, label, kind, unit, the one-clause derivation, flags — rendered four ways (console, report JSON, golden block, the schema `seeml-abi` publishes), with `Problems()` naming every departure from its spec; `plan_work.h` counts a plan's work from its own instructions (seeml-bench's GEMM unit unchanged — five fixtures before and after — and the attention model FLOPs beside it); `process.h` reads peak RSS in bytes on every platform. (2) The compiler's four screens — frontend, analysis, backend and the runtime as planned — filled from the numbers each site computes, printed by default in place of the summary line and the pass notes, `--verbose` for the detail, `--peak-gflops` for the predicted step, carried by the compile report's `"screens"` (additive, schema 1) and held to the manifest by the seam suite. (3) Golden blocks for two MLP fixtures and the `dec_wide` shape; folding one GEMM fewer fails them. (4) The runtime screen on the device: the plan's own rows beside the measured step (the steps alone), achieved GFLOP/s, utilization, peak RSS beside the plan, the losses beside the training labels' entropy, the verdict — printed after the gate and reported; seeml-bench rows gain the attention work and the predicted step. (5) `docs/usage.md` "What the Compiler Tells You": the real SmolLM-135M screens, annotated — 46 lines by default.

Not here: the screens carry the corpus-derived input-blind loss rather than an accountant's "loss floor" (the accountant derives memory, not loss); the console has no `--quiet`; the frontend's sub-subsystems (§3) and the runtime's roles (§6) report as the four screens rather than one screen each — S1 and S7 restructure them.

## Acceptance

- SmolLM-135M compile prints four screens totalling ≤ 60 lines by default;
  `--report` JSON carries the same structs; `seeml-abi` publishes their
  schema.
- Golden blocks for `dec_wide` and the MLP fixtures pass under all
  toolchains; a deliberate one-op change in a pass fails them.
- The runtime's measured-vs-predicted line appears in every `model_update`
  run and in `seeml-bench` rows.
