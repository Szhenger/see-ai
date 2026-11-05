---
title: "SeeAI CR2: one contract function in source/ — the compiler's topology, the loader and verifier, the plan probe and the Python oracle all call it, and a program × check coverage table proves nothing is skipped"
number: 143
labels: bug,correctness,core-plane,python-plane
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The contract checks exist in several copies and each copy skips something.
The step program is left out of the softmax class-width and vocabulary scans
(F07: a heap write from a plan that passed every contract); the provenance
proof assumes the program never writes the staged IO slots (N03); header
input and label refs may carry the source bit and bypass the feeder's
validated labels (N04); rodata alignment is never proven (N05); the plan
probe skips `VerifyPlanContract`, `VerifyExecutorContract` and the feeder
checks (F58); the Python oracle's `Corpus.check` skips most of the feeder
contract (F47) and its `diff` never checks bytes outside declared write
extents, so an out-of-bounds kernel write still agrees (N33). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3
(doctrine: shared functions in `source/`), §8.

## Design

1. Move the geometry contract out of `runtime/engine/contract.cc` into
   `source/contract/` as pure functions over the plan header and the
   corpus header; the compiler's topology (S1), the loader (S7), the probe
   and, through the ABI manifest, the Python oracle call the same code.
2. A **coverage table** in the test suite: every program (train, step, eval,
   merge) × every check (class width, vocab bound, IO-slot write
   disjointness, source-bit refs, rodata alignment, label kind, label width,
   token bound); a check missing from a program is a failing test, not a
   review finding.
3. The oracle's `diff` compares the whole arena, not declared extents.

## Status 2026-11-05

The C++ plane: F07 — the softmax width-agreement check and both width
scans walk all four programs (the step program was outside all three: a
step-program softmax of a narrower width passed every contract and its
rows were indexed past by labels another program's width admitted);
N03 — no program writes the staged input or label slot, refused by the
executor contract from the validator's own extents; N05 — the rodata
section's offset is proven 16-aligned (every rodata ref is relative to
it); N04 already held (the header's slot refs refuse the rodata and
source bits). The coverage table is
`test/runtime/dispatcher/engine_test.cc` (`CoverageTable.*`): a compiled
plan opened for surgery, instruction 0 of a chosen program replaced and
the self-hash re-sealed, on two fixtures that accumulate gradients so the
step program exists — class width, vocabulary, input- and label-slot
writes and a write through a source ref, each × every program; the
header's slot bits; the section alignment; the feeder's label kind,
input kind and token bound. A check missing from a program is a failing
row: with the old program sets the step rows fail. Design item 1 — the
contract as pure functions in `source/contract/`, shared with the
compiler's topology, the probe and the Python oracle — and F58 / F47 /
N33 (the probe calling no contract; the oracle's `Corpus.check` and its
`diff` over declared extents only) stay open: the Python plane's
remainder.

## Acceptance

- F07, N03, N04, N05, F58, F47, N33 reproductions refused by both planes.
- The coverage table is complete and is what the review's "three-way check"
  would have found in CI.
