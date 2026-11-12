---
title: "SeeAI S5: analysis restructure — calculus, algebra, statistics, topology, optimization; the reduction-order ledger; the seed handed into autodiff; the analysis screen"
number: 151
labels: enhancement,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

`compiler/analysis/` is `algebra`, `calculus`, `reviewer`, `updater` — two
names that say what the mathematics is and two that do not. Optimizer
synthesis sits in calculus though it is minimization, not differentiation;
the schedule lives in the driver; the grad-accumulation `1/G` seed is a
coupling hidden inside autodiff; precision assignment for `kFlagRelaxed` is
decided in the driver; and nothing records, per compiled plan, which passes
changed reduction order and under which certificate (E10, F7's fourth step,
the relaxed family all do). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §4.

## Design

1. Five directories: `calculus/` (autodiff, seed as a parameter, the masked
   VJP from S3, remat machinery for F5), `algebra/` (LoRA graft, merge, the
   three fusers, RoPE table, the attention segment mask), `statistics/`
   (quantization review, attention tiling, precision assignment, adapter
   policy), `topology/` (DCE, verify-after-every-pass, the ledger),
   `optimization/` (SGD/AdamW, schedule and horizon, clipping, the seed,
   8-bit moments). Pass manager at the root; `updater/` and `reviewer/`
   gone; conv lowering deleted (LM scope).
2. The **reduction-order ledger**: one line per pass — bit-identical to its
   input or not, and why — stored in the plan header (S6) and printed.
3. The **analysis screen**: trainable parameters and fraction, adjoints
   emitted, activation bytes saved for the backward, FLOPs
   forward/backward/optimizer, bytes of arena traffic removed per fusion,
   dead ops removed, the ledger.
4. Best-practice defaults from §4, changed in the recipe (S4) under the
   three rules: clipping 1.0, learning rate 1e-4 with cosine and warmup,
   α/√r scaling — each with a frontier row (F1) before it flips.

The LoRA lowering itself is F7 (#138) and is not restated here; F7 lands in
`algebra/`.

## Status 2026-11-11

The moves of design item 1 are complete on `claude/v100b-s4s5s6` (step 2 of
the S4/S5/S6 plan), every plan bit-identical. The five directories already
existed; what was left moved: conv lowering is deleted with its pass, its
diagnostics unit and its tests (the frontend's conv2d / im2col operator
builders stay — frontend scope — and nothing lowers them now); the
rematerializer is in `calculus/`; the `kFlagRelaxed` assignment is
`statistics/precision` (`AssignRelaxedPrecision`), out of the driver; the
schedule triple, the header's schedule words and the accumulation seed are
`optimization/schedule` (`VerifySchedule`, `ScheduleWordsOf`,
`AccumulationSeed` — the 1/G handed into autodiff as a parameter). The moved
diagnostics keep their wording and their `UpdateCompiler` unit. Verified:
plan content identical to main 523e4fc on 17 fixtures (the 14 of S1 plus the
pinned, split and masked `--data` variants); the only golden change is the
pass list losing `conv-lowering`. Open: the reduction-order ledger (item 2 —
it is stored in the plan, so it lands with the v23 metadata section), the
analysis screen's new rows (item 3), the attention segment mask, 8-bit
moments, and the defaults flip (item 4: its own PR, with F1 rows measured on
AC power).

## Status 2026-11-12

On `claude/v100b-s4s5s6-close`: **the ledger** is in the plan (v23 metadata
section) and on the analysis screen — one entry per pass that ran, each
bit-identical, and `precision` the one reordering entry, whose instruction
count the loader holds equal to the plan's `kFlagRelaxed` instructions
(acceptance 2: zero reordering entries on an exact package; on a relaxed one
`precision` accounts for every relaxed GEMM and tiled-attention instruction —
they are flagged by the precision assignment and by nothing else). **The
screen** states adjoint ops, the activations saved for the backward, forward
/ backward / optimizer work, what each fusion stopped writing. Acceptance 3
as it can be held: the work rows are `GemmFlops + AttentionFlops` of the
plan's own eval, train and step programs (`source/statistics/plan_work.h`,
the unit of `seeml-bench` and the runtime screen; forward + backward equals
the runtime's work per step) — `seeml-plan-probe --profile` reports time per
opcode and no FLOP figure, so there is nothing of the probe's to match; byte
rows use the arena binder's size function. **Open: the defaults flip**
(clip 1.0, lr 1e-4 cosine + warmup, alpha / sqrt(r)) and its F1 rows — the
acceptance asks for its own PR with measurements, and the host has been on
battery for this whole branch, so no F1 row was taken. Also not done:
the attention segment operand / mask, 8-bit moments, the MSE opcode drop.

## Acceptance

- Every fixture compiles to a bit-identical plan before and after the move
  (directory change only), then the defaults flip in their own PR with the
  F1 rows attached.
- The ledger names every pass that changes arithmetic on the F2 relaxed
  package and none on an exact package.
- The analysis screen's numbers match `seeml-plan-probe --profile` for
  FLOPs and the arena binder for bytes.
