---
title: "SeeAI S3: the validation subsystem — the held-out split decided at compile time, and the loss mask for instruction data with a masked cross-entropy opcode"
number: 149
labels: enhancement,core-plane,doctrine
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

Two decisions define the objective the gate scores and neither is compiled.
The held-out set is a runtime `--val-frac` tail slice of whatever corpus the
device has, which the same data source controls — the last defense against
poisoned data is chosen by the data. And every position in a batch is a loss
target: there is no mask in the ISA, so instruction data (prompt + response)
trains the model to predict the prompt. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3 (validation row).

## Design

1. **Split.** With `--val-data`, validation tokenizes it with the same
   tokenizer (S2) and records both hashes. Without one, a seeded partition
   of the training corpus after deduplication (no held-out record has a twin
   in training); seed and held-out indices go into the plan so the device
   scores the same set. The feeder checks all of it at load from the shared
   contract function.
2. **Mask.** For instruction data the chat template from the SMF tokenizer
   section marks response positions as targets. The SDS record gains a
   per-position mask (part of S2's version bump); the ISA gains
   `kSoftmaxXEntMaskedFwd/Bwd` with the mean taken over target tokens; the
   VJP rule lands in `analysis/calculus`. Plain text records carry an
   all-ones mask and the existing opcode, bit-identical to today.
3. The accountant reports target tokens vs total tokens per record.
4. Attention across packed documents: the attention op gains a segment
   operand (block-diagonal within a window) or the screen prints how many
   windows straddle a document boundary — never silence (§4 algebra).

## Status 2026-11-09

Landed on `claude/v100b-s1s2s3` as step 4 of the S1/S2/S3 plan, on the
formats of step 2 (SDS v3's per-position mask, SEEU v22's `split_seed` /
`val_frac_bits` / `val_split_hash` / `data_flags`) and the data path of
step 3. **The split (item 1).** `source/contract/split.h` is the function:
`HeldOutIndices(records, seed, frac)` — a Fisher–Yates permutation of the
record indices from splitmix64 seeded in the split's own domain, the first
`HeldOutCount` positions (the positional split's rule: at least one, at
most n − 1), sorted — and `SplitHash`, `ContentHash64` over them.
`seeml-update-compile --val-frac F [--split-seed S]` beside `--data` draws
it after ingest (so a raw-text corpus is deduplicated first and no held-out
document has a twin in training), writes the three words and
`kDataFlagSplit`, prints the `validation` row and the report's `data`
object; two compiles of the same corpus and seed produce identical plans.
`Dataset::SplitHeldOut` moves the named records on the device, the feeder
contract refuses a corpus whose split hash is not the plan's ("the corpus
was not split as the plan compiled it"), the emitted runner recomputes the
function from the corpus it was given, checks the hash, and refuses
`--val-frac` as not its decision; `frontier_exec run` mirrors all of it,
and the two planes are pinned to one literal fixture (n = 10, seed = 1,
30 % → records 1, 5, 7, hash e731eed086e2a95d). `--val-data` (a separate
validation file) is not implemented: the seeded partition is the one
mechanism, said here. **The mask (item 2).** Plan v22 carries the masked
cross-entropy pair `kSoftmaxXEntMaskedFwd` / `Bwd` (51 / 52): the
operands of 11 / 12 with −1 marking an ignored position, the mean over
the target rows alone (0 for a batch with none), every row's probabilities
written, zero gradient on ignored rows — CPU (`loss.cc`), Metal
(`k_xent_masked_*`, held to the CPU by the twin test at three shapes, a
shape with no target included), the validator (shapes, the v22 gate), the
lowering of `sc_high.softmax_xent_masked` / `sc_low.softmax_xent_masked_grad`,
the VJP rule in `analysis/calculus`, the oracle's kernels, which PyTorch's
`cross_entropy(ignore_index=-1)` and its autograd hold to 1e-5
(`test/tool/masked_loss_test.py`, tier 2). The executor contract binds the
flag and the opcodes three ways (a masked opcode without the flag, a plain
one under it, the flag with none); the engine's eval tail, ragged-batch
mean and accuracy run over target rows alone. A `.jsonl` record may be
`{"prompt", "response"}`: rendered through ChatML (the section must name
`<|im_start|>` / `<|im_end|>` as special tokens, which SmolLM's does), the
two sides encoded separately (the GPT-2 pattern splits at the template's
`\n`, so the ids equal the joint encoding), the response side the
targets, the joining EOS a target only after a plain document, windows
with no target dropped and counted, a masked SDS v3 out; the compile then
carries the masked objective (`--masked-loss` compiles it for a plan fed a
masked corpus later; the cross-entropy alone — MSE / KL are refused). The
chat template is not read from the section's `chat_template` bytes: the
core renders ChatML itself (a Jinja renderer is out of scope), which is
the template SmolLM2-Instruct and Qwen2 use; a checkpoint with another
template trains with ChatML markup regardless, said here. **Items 3 / 4**
landed in step 3: the frontend screen's `loss targets` row (target of all
positions) and `straddling windows` (PLAN D8; no segment operand).
**Acceptance:** two compiles of the same corpus and seed are byte-identical
and the device refuses a disagreeing split (`UpdateCompiler
.TheDataPathPinsThePlanToTheCanonicalCorpus`, the contract tests, the
runner strings); the masked loss matches the PyTorch reference to 1e-5
(the oracle, tier 2) and the C++ kernels match the oracle instruction by
instruction on a masked plan in the frontier matrix (`dec_masked`), while
plain-text plans are bit-identical to main on the 14 identity fixtures;
the gate's set is named in the report (`data.val_split_hash`,
`data.val_frac`, `data.split_seed`) and on the frontend screen. The
whole-batch eval mean stays the mean of per-batch masked means (each over
its own target count); the ragged final batch is scored over its real
target rows. Not here: the F1 frontier row re-run from raw text through
the compiled split (S2 acceptance 3), since `tool/frontier_run.py` still
compiles without `--data` and passes `--val-frac` to the runner; a
compiled-split plan would refuse that flag — the harness is to switch to
`--data` + `--val-frac` at compile time when the row is next re-run.
**Run end to end (2026-11-09):** the demo decoder compiled with `--data
decoder_corpus.sds --val-frac 0.25 --split-seed 3 --build` (192 records,
48 held out, split 6317ea27c92d5e82) trains 2 steps through the emitted
runner and reports the held-out loss and accuracy; the same runner refuses
`--val-frac 0.1` ("--val-frac is not a runner decision for this plan"), a
corpus one record shorter ("does not yield the plan's compiled split"), and
a masked twin of the corpus ("pinned to another corpus"); the same decoder
compiled with `--data masked.sds --val-frac 0.25 --build` (861 of 1536
positions targets) trains 2 steps on the masked cross-entropy with a
periodic validation, the best state and accuracy judged on target rows.

## Acceptance

- Two compiles of the same corpus and seed produce byte-identical split
  indices; the device refuses a plan whose split hashes disagree with its
  corpus.
- Masked loss: on a prompt/response fixture the response-only loss matches a
  PyTorch reference to 1e-5; plain-text plans are bit-identical to main.
- The gate's validation set is named in the compile report with its hash.
