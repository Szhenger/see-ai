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
row: with the old program sets the step rows fail.

The Python plane and the probe followed. F58: `seeml-plan-probe` runs
`VerifyPlanContract` and `VerifyExecutorContract` exactly as the engine
does before it prices or traces a plan (the Python plane cannot re-seal a
plan, so the refusal has no Python test; the engine's tests prove the
contracts, and the differential suite runs the probe on every matrix
plan). F47: the oracle's `Corpus.check` mirrors `VerifyFeederContract`
rule for rule — the geometry, the label kind and width, and every staged
value's bound (class labels below the narrowest softmax width, token ids
below the narrowest table) — and its `label_classes` / `vocab` scan all
four programs as the engine does since F07. N33: the diff compares the
whole C++ arena image against the one it sent and names the first byte
changed outside every declared write extent — a kernel writing where the
bounds proof says it does not, which the per-extent comparison could never
see. Tests: four refused corpora and two accepted ones against the matrix's
plans; a crafted image with one stray byte.

## Status 2026-11-06 — design item 1

The geometry contract is one pure statement in `source/contract/`:
`PlanHeaderViolation(header, plan_size)` — the section table, the version
gates, the arena relations, the I/O slots, the hyperparameters — and
`CorpusGeometryViolation(header, CorpusShape)` — the feeder's geometry
half — each returning the violated rule's reason, with the bounds helpers
beside them (`source/contract/bounds.h`). The runtime's loader and feeder
wrap them under the engine's unit (the feeder keeps the value half, which
reads the records); the compiler runs the same header function on the plan
it has just emitted, so a header the device would refuse is a backend
contract violation at compile time; the probe reaches it through the
runtime's wrapper. The Python oracle cannot call it, so it is held to the
same rules by its own tests and, for the constants, by the ABI manifest:
the reader refuses CR1's immediate floats and the header's hyperparameters
as the device does, and `seeml-abi` publishes the temperature bounds as
f32 bits. (The packer's `check_plan` stays a magic sniff on purpose: it
never interprets instructions, and the packaged device validates in full.)

## Acceptance

- F07, N03, N04, N05, F58, F47, N33 reproductions refused by both planes.
- The coverage table is complete and is what the review's "three-way check"
  would have found in CI.
