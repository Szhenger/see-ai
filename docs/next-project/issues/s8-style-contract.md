---
title: "SeeAI S8: the style contract — write the house style down and enforce it: the two-dialect decision, .clang-format, the packed-word helpers, the four oversized functions, tool/ idiom parity, the test-framework tightening, and the doc-drift list"
number: 154
labels: enhancement,docs,design,testing
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P2
---
## Root cause

The 2026-09-01 Style Audit graded the tree A− and found the discipline lives
entirely in people's heads: two undocumented dialects (Google-style in
`seeml::update`, MLIR-flavored camelCase and 4-space indent in
`seeml::sir`, ~24 files on the second side, a few straddling); no
`.clang-format`, no STYLE.md, no CONTRIBUTING.md; one bit-unpacking
expression hand-inlined ~47 times across engine, validator and the dumper
while `UnitOf` already exists; four functions of 277–560 lines
(`CompileImpl`, `ValidateInstruction`, `LowerOps`, `seeml-bench`'s
`main`); `source/` and `tool/` at 0 `[[nodiscard]]` against 93 elsewhere;
three tools with three argument parsers; `seeml-update-compile` exiting 1
where the docs promise 2 (also found by the v1.2.4 Field Report's ledger and
the STAR document); `using namespace` in 20 of 25 test suites; no
`SKIP()`, so hardware-gated tests report green for zero work. None of the
seven recommendations was ever filed. The Two-Plane Port Review's four
doc-drift items are folded in here. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

1. Decide the dialect question (document per-subsystem, or normalize), then
   commit `.clang-format` (Google base, 80 columns) and a short STYLE.md:
   the diagnostic-unit contract, the assert-vs-expected boundary, the
   exit-code table, ownership conventions. Add `-Wold-style-cast` (zero
   violations today).
2. `HiWord`/`LoWord` helpers beside `MakeArenaRef` in
   `source/plan/instruction.h`; delete the ~47 copies.
   Superseded 2026-11-18 in the names only, the clause as first written:
   *`HiWord`/`LoWord`.* The helpers landed as `HighWord` / `LowWord`
   (with `PackWords`): Carbon's `ToolUtils.h`, which Accelerate pulls into
   every translation unit that includes it, defines `HiWord` and `LoWord`
   as macros.
3. Decompose the four functions (S6 takes `CompileImpl`; this issue takes
   the other three).
4. `source/` and `tool/` to the tree's idiom: `[[nodiscard]]`, `std::span`
   for (ptr, size) pairs, one expected-based argument parser shared by the
   CLIs, exit codes per the documented contract, the dumper's bounds-checked
   wire-read rule in bench.
5. Test framework: per-symbol `using`, four duplicated helpers into
   `test/support/`, the two gtest-colliding macros renamed, `SKIP()`,
   operands printed by `EXPECT_TRUE` and haystacks by
   `EXPECT_STR_CONTAINS`.
6. Doc drift: `docs/formats.md` SMF version heading, `SPECIFICATION.md`'s
   Python-surface sentence, `tool/README.md`'s doctrine line, and the stale
   test count.

## Status 2026-11-24

Acceptance 3 is met and held by a test. The three dispatch paths the
2026-11-19 status left over the limit — `MetalBackend::Encode` (635
lines), `CpuBackend::Execute` (398), `UpdateEngine::TrainImpl` (377) — and
the two the stack had since pushed or left over it,
`VerifyExecutorContract` (228) and the emitter's `BuildScript` (245), are
decomposed with every case body and message moved verbatim: the two
backends hand an instruction to one function a kernel family, the train
loop runs five named phases, the executor contract is one function a rule.
The committed model of a SmolLM-135M package is the same before and after
on both backends and the step time does not move
(`docs/benchmarks.md`, "The v1.0.0.B close"; measured on battery).
`test/tool/function_length_test.py` holds the limit over `compiler/`,
`runtime/` and `tool/`; the scanner it shares with `driver_shape_test.py`
no longer reads a preprocessor line as a function head, which had hidden
every function of four files, `Execute` among them.

Acceptance 1's CI half: `ci.yml` runs `clang-format-19 --dry-run -Werror`
on the clang leg and the exit-code test with no skip allowed. Design
item 6's drift list grew by three and they are corrected:
`runtime/README.md` and `docs/runtime.md` said "until S7" after S7,
`source/README.md` listed four of nine directories, `seeai.md` §3 called
the device's raw-text re-run open after #202 shipped it.

Still as the tree states it, not as the issue first asked: about thirty
`(pointer, size)` parameter pairs and the suites' `using namespace`
(STYLE.md sections 5 and 6 name both as the direction for new code).

## Acceptance

- `clang-format --dry-run -Werror` clean in CI; STYLE.md linked from the
  README.
- `git grep -c '>> 32'` in engine/validator/dumper drops to the helpers'
  definitions.
- No function over 200 lines in `compiler/`, `runtime/`, `tool/`.
- All three CLIs exit 2 on a bad command line; a test pins it.
