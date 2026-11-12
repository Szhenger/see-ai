---
title: "SeeAI S4: the identity manifest and the recipe file — every input hashed into the plan, a declarative recipe with a schema and bounds, the exporter's output verified rather than trusted"
number: 150
labels: enhancement,correctness,python-plane,doctrine
plane: Python plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

A fine-tune is four inputs — model, tokenizer, corpus, recipe — and the plan
records the hash of one. The recipe is about forty flags with no schema, and
the review found what that costs: `ParseU64` accepts `" -1"` in the CLI and
the generated runner (F56, F63); `--teacher` / `--distill-weight` /
`--temperature` silently ignored unless the loss uses them (N15);
`--lora-rank` unbounded (N18, overflow emits an invalid plan);
`--temperature` unbounded (N26); an empty `--targets` adapts every weight
(N27); a runner `--steps` inside the compiled warmup (N20); the runner
ignoring `--checkpoint-every` without `--checkpoint` and `--eval-every`
without a split (N24). The exporter's output is trusted as-is. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3
(doctrine lines: recipe, exporter verification).

## Design

1. **Recipe file** with a JSON schema in `source/` (published by
   `seeml-abi`), every numeric field bounded, cross-field rules (warmup <
   steps, teacher fields require a distillation loss, targets non-empty);
   flags override fields; the resolved recipe is hashed. The generated
   runner validates its flags with the same rules.
2. **Identity manifest** in the plan header: model hash, tokenizer hash,
   corpus hashes (train, held-out), resolved recipe hash, split seed, mask
   policy, the exporter's parity figures. The loader (S7) verifies it; the
   accountant prints it.
3. **Exporter verification**: the frontend checks the exporter's output —
   file hash, tokenizer parity report, logits-parity figure — and records
   them; a missing report is a refusal, not a warning.

## Status 2026-11-11

Design item 1 landed on `claude/v100b-s4s5s6` (step 3 of the S4/S5/S6 plan);
items 2 and 3 wait on the one format bump the plan gives all three issues.
**The recipe** is one table, `source/recipe/recipe.h`: thirty fields, each
with its file key, its flag, a kind and — for every numeric one — a bound.
`recipe::Apply` is the only place a value is parsed: completely
(`ParseU64Strict` takes digits alone, so `" -1"` is refused where `strtoull`
skipped the space and negated the rest — F56), against its bound, then
against the other fields. Every refusal names the flag, the field and the
bound: `--lora-rank (recipe field lora_rank) must be an integer in [1, 4096],
got '5000'` (N18); `--temperature` / `--distill-weight` / `--teacher` without
a distillation loss (N15); the temperature's f32 range (N26); `--targets`
naming nothing (N27); the cosine schedule's knobs under `const`, warmup not
shorter than steps, a split without the corpus. `seeml-update-compile` no
longer parses a recipe flag by hand: it loops over the table, and
`--recipe recipe.json` supplies the same fields as one flat JSON object
(`source/recipe/recipe_file.h`, a strict reader: unknown, duplicate,
mistyped or nested keys refused by name) which flags override. The resolved
recipe — every field, defaults included, floats as their f32 bits — hashes
to `recipe_hash` in `--report`; a recipe file and its flags compile the same
plan byte for byte with the same hash (`test/tool/recipe_test.py`). A config
built in code is held to the bounds whose violation emitted an invalid plan
(`recipe::ConfigViolation`, in the driver). `seeml-abi` publishes the table
as `abi.json`'s `recipe` block. **The runner** validates its flags with the
same rules (`source/recipe/run_flags.h`, vendored): its integers are digits
alone (F63), a fresh run's `--steps` inside the compiled warmup is refused
(N20), as are `--checkpoint-every` without `--checkpoint` and `--eval-every`
without a validation split (N24) — each exercised through an emitted
runner. After the reviews of #185 the warmup rule itself is the engine's:
`UpdateEngine::Train` refuses any horizon inside the compiled warmup, so a
resume whose step + steps still ends on the ramp and a run whose checkpoint
did not load (and so started fresh) are refused where only the engine can
see them; the runner's check stays as the early usage error for a fresh
run, and `frontier_exec run` mirrors the engine. Every boolean recipe flag
has a `--no-` form, so a flag overrides a recipe file's boolean either way
(both forms at once is refused). Acceptance 1 holds on both planes. Not here: the recipe hash, the
held-out corpus hash and the exporter's parity figures are not in the plan
yet (item 2: the header is full at 320 bytes, so they land with the v23
metadata section), the runtime screen does not print the four hashes, and
the exporter's output is still trusted (item 3). Acceptance 3 is read, per
the plan, as "the recipe-file compile is byte-identical to the flags
compile": the 2026-09-22 plan predates v22 and its bytes cannot be
reproduced. The debug toggles (`--no-fuse-*`, `--no-rope-table`), the kernel
tiles and the paths (`--data`, `--teacher`) are not recipe fields.

## Status 2026-11-12

Design items 2 and 3 landed on `claude/v100b-s4s5s6-close`. **The manifest**
is the plan's v23 metadata section (`source/plan/metadata.h`; the header
stays 320 bytes, the section sits in the padding after the emit table under
`data_flags` bit 3): the resolved recipe's hash, the exporter report's hash
and its parity figures, beside the header's model, tokenizer, corpus and
held-out-split hashes, split seed and mask flag. The loader proves the
section before reading it (`PlanMetadataViolation`, mutation-tested); the
frontend screen, `--report` (`identity`) and the runtime screen print the
four hashes (acceptance 2: SmolLM-135M, 2026-11-12, model `4448e52d513b5011`,
tokenizer `bc20bbcbeb260e1f`, corpus `8671aa8ec28e361f`, recipe
`124555bc49b44c47`, the same four on the compile and the Metal run).
**Exporter verification**: `tool/export_model.py` writes
`<model>.smf.export.json` (file hash, section hash, tokenizer-parity ids,
logits-parity delta, each marked run or not); `ReadExportReport` holds it to
the loaded model; an imported checkpoint without one is refused, on the CLI
and in the library. The device's committed model gets its own report from
the engine, so it compiles again. Not here: the report records the parity
figures and sets no threshold on them, and a report with no parity run is
accepted (the item asks that they be checked against the file and recorded);
the hash is of the resolved recipe, so `--remat auto` and `--remat on` that
resolve alike share it.

## Acceptance

- Every N15/N18/N20/N24/N26/N27/F56/F63 input is refused with a diagnostic
  naming the field and its bound, on both the compiler and the runner.
- A package states its four input hashes in `--report` and in the runtime
  screen; the loader refuses a source model or corpus whose hash disagrees.
- One recipe file reproduces the 2026-09-22 frontier row byte-for-byte.
