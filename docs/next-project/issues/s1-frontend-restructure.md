---
title: "SeeAI S1: frontend restructure — ingressor, accountant, topology, computation; shape checks lifted out of the build loop; the cross-input and non-finite checks the review found missing"
number: 147
labels: enhancement,correctness,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The frontend's stages are interleaved: whole-graph checks run before
construction, but the per-op shape checks run inside `BuildForward` one op
at a time, so "verify, then build" does not exist as a boundary and the
cross-input checks have no home. The 2026-09-24 review found the holes that
follow from that: const payload placement never validated (F04: alignment,
header overlap, payload overlap); student names starting `t::` not reserved
(F06); output row count never checked against the compiled batch (F68, a
heap write downstream); teacher `seq_len` never checked against the student
(F02, high — a wrong distillation target with no diagnostic); `--loss mse`
on a token model compiles and every dataset is rejected (N19); the "lower
bound" footprint over-estimates and refuses compiles that fit (N29) and
ignores `--bf16-base` (F03); a NaN weight passes ingest and is later cast to
int8 (F12/F34). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3.
N29 and F03 landed ahead of the restructure, on 2026-11-03 under F5
(#133): the accountant's bound counts the forward values the backward
reads (the VJP rules decide which, the adapted targets decide which
products) and the base at its stored width, drops rematerialization's
share where `--remat` may run, and the compile report states the
estimate, the exact footprint and their ratio (`docs/compiler.md`, "The
step-0 bound counts what the backward reads"). What stays for S1 is the
home: the accountant's screen and the ratio printed on it.

## Design

1. Directories `frontend/ingressor`, `frontend/accountant`,
   `frontend/topology`, `frontend/computation`; `representation/` and
   `operator/` stay as shared infrastructure; `model_writer` moves out as
   egress.
2. Topology runs to completion over the decoded model before computation
   starts: order, names, output, per-op shapes (from the tensor table), and
   the cross-input checks — student↔teacher (seq_len, output rows vs batch),
   loss↔input kind, adapter↔weight, tokenizer vocab == embedding rows ==
   head width. Computation becomes a mechanical translation with no failure
   modes of its own.
3. The ingressor refuses non-finite weights and validates payload placement
   (64-byte alignment, after the metadata, non-overlapping sorted ranges).
4. The accountant's estimate is a **tested lower bound**: estimate ≤ the
   backend's exact arena on every fixture, bf16 counted at 2 bytes; the
   screen prints it as an estimate with the honesty ratio.
5. Each stage fills its statistics struct (S0).

## Status 2026-11-09

Landed on `claude/v100b-s1s2s3`, the first step of the S1/S2/S3 plan.
The boundary exists: `sema::VerifyModel` (`frontend/topology/sema.cc`)
runs to completion over the decoded model before anything else — the
whole-graph checks, every op's arity, operand existence and constness,
and one shape rule per op kind applied to the tensor table and the batch
— and returns the shape of every value; `BuildForward`
(`frontend/computation/parser.cc`) reads those shapes and recomputes none,
so computation has no shape rule of its own. `sema::VerifyCrossInputs`
holds what no single model can know: the student's output rows are the
batch (F68), a token model takes no MSE loss (N19), every `--targets`
filter names a tensor, no student name starts with `t::` (F06), and a
teacher shares the input kind and width, the seq_len of a sequential
student (F02) and the output shape. The ingressor proves payload
placement on load — 64-byte aligned, after the metadata, sorted and
non-overlapping (F04) — and owns the value rule: `SweepWeights` refuses
the first non-finite element by role, tensor and element (F12 / F34,
moved here from the driver) and records each tensor's range in the same
pass. The accountant's `AdaptedWeights` rule is topology's one reading
of it, and under `--adapter-only` the bound counts no weight bytes (the
plan carries none; the 2026-11-09 review's B1, where the bound read
1.31× the exact footprint and refused a budget the plan fit). The driver
runs `ingress` and `topology` as phases of their own before the
accountant (`finite-sweep` is gone), fills `FrontendStats{ingressor,
topology, accountant}`, and the frontend screen renders the three:
adapted weights, where the frozen weights live, `bound ÷ exact`, and
under `--verbose` every frozen tensor with its payload and value range.
Verified: all 29 C++ suites and the 183 Python tests; the plan content
is byte-identical to main a72a6ec on 14 fixture compiles (the demo MLP
under xent / int8 + grad-accum / mse / kl / xent+kl, the demo decoder
under xent / int8 / bf16 / tiled / remat / adapter-only / relaxed /
targets, and `dec_wide`) — the harness compares `seeml-seeu-dump --json`
with the header's `version` and `plan_hash` masked, and in this step
those two were unchanged as well, so every plan hash is main's. The
goldens and the ABI manifest carry the new rows; one row moved: `bound_ratio`
is a frontend row now (it was the backend's), so a reader of `--report`
finds it under `screens.frontend` — the manifest records the move. Not
here: the tokenizer-vocab == embedding-rows == head-width rule waits on
the SMF tokenizer section (S2). A layering note for S5: `sema.h` takes
`LossKind` from `source/plan/config.h` for the loss↔input rule, so the
frontend's topology now names a plan-layer enum.

## Acceptance

- [x] Every F04/F06/F68/F02/N19/N29/F03/F12 reproduction from the review is
  a compile-time refusal with a diagnostic naming the stage — F04
  (`Smf.LoadProvesPayloadPlacement`), F06 / F68 / N19 / F02
  (`UpdateCompiler.TopologyRefusesWhatTheReviewFoundMissing`), F12
  (`Smf.TheSweepRefusesNonFiniteWeightsAndReportsRanges`,
  `UpdateCompiler.ANonFiniteWeightIsRefusedBeforeAnyPass`), N29 / F03
  (landed 2026-11-03 under F5; the lower-bound test now also covers bf16,
  a teacher, adapter-only and the MLP fixture).
- [x] Bit-identical plans for every fixture against main (a restructure,
  not a semantic change) — 14 fixture compiles identical to a72a6ec,
  2026-11-09.
- [x] The frontend screen prints the parameter census by role, per-tensor
  ranges, the estimate and its ratio — `adapted weights`, `frozen weights
  live`, `step-0 bound`, `bound ÷ exact`, and the `--verbose` `tensors`
  table (name, bytes, min, max).
