# Using SeeAI

## The Shape of the Workflow

SeeAI compiles an on-device model update *ahead of time*: LoRA adapters are grafted onto a frozen model, the backward pass and optimizer are synthesized as a fixed instruction stream bound to a pre-planned arena, and the result is executed on-device by a zero-dependency VM. One `.seeu` plan = one complete, gated, resumable, atomically-committed update.

The workflow has three steps on two machines:

```
build host:  1. export   PyTorch model  ──▶  model.smf (+ corpus.sds)
             2. compile  model.smf      ──▶  pkg/  (plan + vendored runtime + build.sh)
device:      3. update   model.smf + corpus.sds  ──▶  updated.smf, or no change at all
```

Notice what does *not* travel to the device: PyTorch, Python, this repository. The device receives a folder that builds with any C++23 compiler. If you want to understand what each step does internally, [compiler.md](compiler.md) and [runtime.md](runtime.md) go deep; the binary files exchanged between the steps are specified in [formats.md](formats.md). This document just gets you running.

## Step 1: Export the Model (build host, PyTorch)

The exporter needs Python 3 with PyTorch and NumPy on the build host. The stack it is developed and tested on is CPython 3.14.7 with PyTorch 2.14 and NumPy 2.5 (`pip install -r tool/requirements-pinned.txt`); it also runs on anything down to Python 3.9, NumPy 1.17 and torch 1.7 (`tool/requirements.txt` states those floors), and the compiler reads every SMF/SDS file any version has written. Export streams the model straight from the framework's arrays to disk, so a 500 MB model needs about 500 MB, not several times that. Nothing on the device side touches Python.

The quickest start — a demo model, teacher, and synthetic corpus in one command:

```bash
python3 tool/export_model.py --demo out/
```

This writes `model.smf` (a small `Linear(16,32) → ReLU → Linear(32,4)` classifier), `teacher.smf` (a wider sibling, for distillation experiments), and `corpus.sds` (2,048 labeled samples). Everything downstream can be tried against these three files.

Every demo dimension is a flag, so randomized experiments need no bespoke script: `--width`, `--depth`, `--samples`, `--seed`, and `--corpus-kind class|dense|none` (`none` writes the unlabeled corpus that `--loss kl` distillation wants) for `--demo`; add `--vocab`, `--heads`, `--seq-len`, `--blocks`, `--ffn`, and `--rope-base` (the rotary θ — 10000 by default, 500000 for Llama 3, 1000000 for Qwen; written per Rope op as SMF v5 `attr1` and lowered into the plan verbatim) for `--demo-decoder`. Defaults reproduce the classic demos byte-for-byte, and — as with `seeml-update-compile` — a flag that cannot apply to the requested mode is a hard error (exit 2), never silently ignored. There is also `--corpus data.npz out.sds`, which converts saved NumPy arrays (`records` for token corpora; `inputs` + optional `labels` for feature corpora) into an SDS file without writing any Python.

For your own model, use the two functions the script exports:

```python
from export_model import export_smf, export_sds
export_smf(sequential_model, "model.smf")   # Linear/ReLU/GELU/SiLU/LayerNorm
export_sds(inputs, labels, "corpus.sds")    # labels: int32 classes, dense f32, or None
```

The exporter accepts an `nn.Sequential` of `Linear`, `ReLU`, `GELU`, `SiLU`, and `LayerNorm` modules — anything else is a loud `ValueError`, not a silent skip. The exporter also refuses a model the device would compute differently (P7, #96): the device's GELU is the tanh approximation, so a plain `nn.GELU()` — PyTorch's default, the erf form — is an error that names the fix (`nn.GELU(approximate='tanh')`, or `export_smf(..., gelu_tanh_ok=True)` to accept a drift of up to 4.7e-4 per activation); and a `LayerNorm`'s own `eps` is written into the file (SMF v6) whenever it is not 1e-5, as is a Hugging Face checkpoint's `rms_norm_eps` under `--hf`. One detail worth knowing so the format makes sense later: PyTorch stores a `Linear`'s weight as `[out, in]`, but SMF's `MatMul(x, W)` wants `[in, out]`, so the exporter transposes on the way out. Labels: pass int32 class indices for cross-entropy, dense float vectors for MSE, or `None` for a distillation corpus (the teacher provides the targets).

**Token-native decoders.** For a decoder transformer that consumes raw token ids (SMF v4+; the writer emits v5, which adds the per-op RoPE base), export the frozen embedding table alongside the blocks, and a corpus of plain ids — no labels, no embedded vectors:

```python
from export_model import export_token_decoder_smf, export_token_sds
export_token_decoder_smf(embedding, blocks, head, "decoder.smf",
                         seq_len=S, num_heads=H)   # embedding: [V, D] f32
export_token_sds(records, "decoder_corpus.sds")    # records: [N, S+1] i32
```

**Importing a Hugging Face decoder.** A local Llama-class checkpoint directory (`llama`, `qwen2`, SmolLM2 — `config.json` plus `model.safetensors` or its shard index) becomes a token-native SMF with NumPy alone:

```bash
huggingface-cli download HuggingFaceTB/SmolLM-135M            # once; any local dir works
python3 tool/export_model.py --hf <model_dir> smollm.smf --seq-len 128 \
  --text-corpus docs.txt docs.sds --hf-parity
```

The walk transposes every Linear to `MatMul(x, W)` layout, repeats grouped-query k/v heads to one per query head (the format carries no KV heads yet), permutes q/k features within each head from Hugging Face's rotate-half RoPE pairs `(c, c + d/2)` to SeeAI's interleaved `(2c, 2c+1)` (the scores are a dot product over `d`, so a permutation applied to both sides is exact), carries `rope_theta` per Rope op, adds Qwen2's q/k/v biases as `AddBias` ops, and ties `w_head` to the embedding when the checkpoint does. What it refuses, loudly: a non-SwiGLU block, `rope_scaling`, MLP biases, a `--seq-len` past `max_position_embeddings`, and an `rms_norm_eps` other than the runtime's fixed `1e-5` (Qwen2's `1e-6` is a drift; `--allow-eps-drift` accepts it knowing step 0 will not equal the source model — the attribute is P7, #96). Two tier-2 extras: `--text-corpus` tokenizes a UTF-8 file with the checkpoint's `tokenizer.json` into `S + 1`-token records (needs `tokenizers`), and `--hf-parity` runs a NumPy forward with SeeAI's exact semantics against `transformers` on seeded random tokens and prints the max logit delta (SmolLM-135M: `6.4e-05` at `S = 128`; needs torch + transformers). Neither is needed to import.

Each corpus record is `S + 1` ids: the runtime feeds the first `S` and derives next-token labels from the shifted view, and the embedding gathers on-device. `python3 tool/export_model.py --demo-decoder out/` writes a working example of both files — `decoder.smf` and `decoder_corpus.sds` (NumPy only — no PyTorch needed); the corpus is deliberately *not* named `corpus.sds`, so both demos can share one output directory. Pass those names to the compile and update steps below in place of `model.smf` / `corpus.sds`. Remember that `--data-batch` counts *rows* (tokens), so it must be a multiple of `seq_len`.

## Step 2: Compile the Update Plan (build host)

Here's a full-featured invocation; we'll unpack it flag by flag:

```bash
seeml-update-compile \
  --source model.smf --out pkg/ \
  --data-batch 32 --loss xent \
  --lora-rank 8 --lora-alpha 16 \
  --optimizer adamw --lr 1e-3 --clip-norm 1.0 \
  --lr-schedule cosine --warmup 100 --min-lr-factor 0.1 \
  --quantize-base \
  --steps 1000 --report pkg/report.json --build
```

**What to train on.** `--data-batch` (default 32) fixes the batch size — how many samples are processed together in each training step — *into the plan*; shapes are compile-time facts in SeeAI, so this isn't a runtime knob. `--loss` picks the objective, the measure of wrongness that training drives down: `xent` (softmax cross-entropy, needs class labels), `mse` (mean squared error, dense labels), `kl` (**distillation**: the model learns to imitate a teacher model's output probabilities rather than hard labels — add `--teacher teacher.smf` and use an unlabeled corpus), or `xent+kl` (both, blended by `--distill-weight`, default 0.5; `--temperature`, default 2.0, softens both distributions, and the KL term is scaled by `T²` so that `--distill-weight` means the same thing at every temperature — see [runtime.md](runtime.md) for why).

**What to adapt.** `--lora-rank` (default 8) and `--lora-alpha` (default 16) set the adapter geometry — the update lives in `r·(K+M)` parameters per adapted matmul instead of `K·M` ([compiler.md](compiler.md) does the math). `--targets substr1,substr2` restricts grafting to weights whose names match a substring; by default every eligible frozen matmul is adapted. Every substring must match at least one weight LoRA can adapt, or the compile fails and says why: a weight read any other way — above all an LM head tied to the embedding table, which the gather also reads — cannot take `W + Δ` without changing that other site, so `--targets emb` on a tied model is an error rather than a silently unmet filter. `--lora-seed` (default 42) makes the adapter initialization reproducible.

**How to optimize.** `--optimizer adamw|sgd` (default adamw), `--lr` (default 1e-3), `--weight-decay` (default 0.01), and `--clip-norm` (default 0 = off; a positive value bakes per-tensor gradient clipping instructions into the stream). The schedule — `--lr-schedule const|cosine`, `--warmup N`, `--min-lr-factor F` (default 0.1) — travels in the plan header and is evaluated per step on the device. The cosine anneals over **the steps the update actually runs** — `model_update --steps 200` on a plan compiled with `--steps 1000` reaches the floor at step 200, not never — and the compiler refuses a schedule that would misbehave silently: `--steps 0`, a warmup as long as the budget, `--warmup` or `--min-lr-factor` under `const`, and a floor of 0 (the last step would train at a learning rate of zero) unless `--allow-zero-lr` asks for it.

**How big.** `--quantize-base` stores eligible frozen weights as int8 in the plan (4× smaller, dequantization fused into the GEMM for free); `--bf16-base` stores them as bfloat16 instead (2× smaller, float32's exponent range with 8 bits of mantissa, widened exactly inside the GEMM — the gentler choice for outlier-heavy weights; the two flags are mutually exclusive). Either way, because commit patches the *original file's* floats, storage rounding never reaches the committed model — and since E12 (#95) the gate measures that committed model: a plan compiled from a model file evaluates the student's f32 weights straight from the file at run time, so the validation losses the gate compares (and the best-state choice) are the shipped model's, not the narrow copies' (`validation_scores: "shipped"` in both reports). int8 scales are per output column, which keeps outlier-heavy projections' bulk at full resolution; teacher weights stay f32. `--steps` (default 1000) is the default optimizer-step count baked into the plan (the device may override it). `--grad-accum G` (default 1) compiles at the micro-batch `--data-batch` and accumulates `G` micro-batch gradients per optimizer step: the effective batch is `data-batch × G` while activation memory stays that of one micro-batch, at the cost of one gradient-sized accumulator per adapter in the persistent segment (the compile report lists `effective_batch` and `step_instructions`). The report also splits the arena into its segments (`persistent_bytes`, `io_bytes`, `transient_bytes`), and the transient one — the activations the backward reads, held across the whole forward, plus adjoints and the merged deltas — is about 98 % of a decoder's arena at 512 tokens. `--remat auto|off|on` (roadmap 2b, F5 #133) trades compute for that memory: the cheap forward activations (the norm outputs, `silu(gate)`, the SwiGLU product, folded elementwise chains — never a GEMM or attention output) are recomputed in the backward, right before the instruction that reads them, by the same opcode over the same operands, so the plan trains the same bits — the same loss curve, the same committed model — in a smaller arena, at the cost of those extra instructions in the train program (the eval program is untouched). `auto` (the default) recomputes only when the arena plus the plan would exceed a configured budget (`--memory-budget-mib N`; `UpdateConfig::memory_budget_bytes` from the C++ API) — never the host's physical memory, which the gate's refusal may consult but a decision that changes the plan's bytes may not; with no budget `auto` leaves the plan plain and says so; `on` forces it so the saving can be measured at any size; the report records `remat: {policy, applied, recomputed_values, bytes_saved}`.

**How precise.** `--precision f32` (the default) is the reference: every kernel exact, every backend's bits its own. `--precision certified-bf16` (plan v18) marks the frozen-weight GEMMs of the train and step programs — the forward and dX products through the base (and teacher) weights, in f32, int8 or bf16 storage — as *relaxed*: a permission for the backend to run them on a certified relaxed kernel. On Metal that is the tensor-op family (the M5's neural accelerators): the f32 activations are rounded to bfloat16 once at the GEMM input and multiplied against the exact stored weight into f32 accumulators, 2–2.5× the exact kernels' rate. On the CPU, a relaxed plan's frozen-weight GEMMs run on Accelerate's SGEMM (the AMX / SME matrix units) when the package is built on an Apple host (`SEEML_NO_ACCELERATE=1` keeps the portable kernels): an f32 weight directly, an int8 weight widened panel by panel into a 4 MiB f32 buffer with its scales applied as it widens (F4 #132); a bf16 base stays on the portable widening kernels. A plan that runs the tiled attention family relaxes its training attention too (plan v20, F3 #131): on Metal, at head widths up to 64, each block's products run on `simdgroup_matrix` tiles — the same expressions and stats row, the f32 sums in another order. The adapter GEMMs, the cached attention family, norms, losses and the optimizer are exact everywhere, the eval program scores what ships in exact arithmetic, and a relaxed kernel is still bitwise run-to-run on its device — but its bits are not the reference's, so *no certificate, no relaxed plan*: `tool/pack_update.py` refuses to package the plan until `tool/certify_numerics.py certify pkg/ --corpus corpus.sds [--source model.smf]` has priced its relaxed sites on the plan's own operands and granted. The certificate states the plan's own contract (F2 #130): it enforces exactly the sites the plan relaxes — `gemm.relaxed` (the bf16 rounding is 2⁻⁹ relative per element; default tolerance 2⁻⁷) and, for a tiled plan, `attn.fwd.tiled` .. `attn.dv.tiled` (any-order f32 sums; default 1e-5) — and its free run relaxes those and nothing else, with a loss default of 1e-3, F2's own acceptance line; every other site is measured and reported as evidence only. A few steps of a toy model can move the loss past 1e-3, so a test-sized run states its own `--max-loss-deviation`, and `pack_update.py` prints a NOTE for any tolerance looser than its contract's default. Every build path holds that line: the compiler refuses `--build` for a relaxed plan outright, and the emitted `build.sh` checks, before its first compile line, that a granted certificate for exactly these plan bytes sits beside the plan. The compile report carries `precision`, `relaxed_gemms` and `relaxed_attention`; `seeml-seeu-dump` shows the bit as `relaxed`.

**How it's optimized.** By default the compiler fuses each frozen `X@W → +bias → activation` chain into a single matmul instruction with a fused write-back epilogue, so an MLP layer's three arena round-trips become one. Fusion is bitwise-neutral by construction: it only matches chains no backward instruction reads (the frozen teacher subgraph, the bias step of unadapted layers), and the runtime applies the epilogue with the same per-element expressions as the standalone kernels. `--report out.json` carries, beside the plan's geometry, a `"passes"` array — every compiler pass and driver phase with its op count and wall time — and `--dump-sir out.txt` writes the final SIR (debug output; nothing is rendered unless asked). `--no-fuse-epilogue` disables the pass — the plan gets more instructions and more transient arena, never different bits; useful when diffing `seeml-seeu-dump` output across compiler versions or isolating a kernel while debugging. Same-shape elementwise chains whose intermediates have a single reader fold into one `kFusedMap` instruction (plan v13; `--no-fuse-elementwise`); the LoRA scale `α/r` itself is no longer an instruction at all but the rank-r GEMMs' own alpha (plan v19). Two more passes of the same kind ship on by default (plan v12), each with its switch for bit-for-bit comparison: the RoPE angles are hoisted into one device-built table per sequence geometry instead of being recomputed by every rotation (`--no-rope-table`), and `--clip-norm` is folded into the optimizer step, which removes a full read+write pass over every gradient and refuses to step a tensor whose gradient norm is not finite (`--no-fuse-clip` restores the standalone clip instruction). At every LoRA site the two activation-sized sums — `C + (α/r·t)@B` forward, `dC@Wᵀ + dt@Aᵀ` backward — are written by the rank-r GEMM itself rather than by a separate add over the whole tensor (plan v14; bit-identical, `--no-fuse-addend` restores the separate instructions).

**How fast (the kernel policy).** The CPU GEMM tile geometry — a throughput knob only, never a bit — travels *in the plan* (header v11), decided here. By default it is the runtime's own defaults. `--kernel-policy table.json` consults the table `tool/autotune.py` measured on this host (keyed on the host's identity; the compile note prints the key, and a table with no entry for it is a note and the defaults), `--target-host KEY` looks another host up when cross-compiling, and `--gemm-tiles K,N` sets the geometry outright (K a multiple of 4). Nothing is read when no table is given, so compile time is unchanged; a table that does not parse or names a geometry the kernels reject is a hard error. See [benchmarks.md](benchmarks.md) for tuning a host.

**What comes out.** The emitted `pkg/` is **self-contained**: the plan (`update_plan.seeu`), the same plan embedded as a C array (in the default mode — the two paragraphs below are the other two), a generated driver `main`, the vendored runtime sources, and a `build.sh`. `--build` runs that script immediately; on any machine, `sh pkg/build.sh` builds the `model_update` binary with nothing but a C++23 compiler — set `CXX` to cross-compile for the device. `--report pkg/report.json` writes a machine-readable summary (arena bytes and their persistent / io / transient split — the transient segment is the activations, adjoints and deltas, the part that scales with batch and model (F5 #133) —, instruction counts, per-adapter shapes and scales, the kernel policy — source, host key, tiles — and which embedded TU the package carries) worth archiving with each release.

**Big plans: embed with `.incbin` instead.** The C-array embedding renders every plan byte as ~4 characters of C++ that the host compiler must parse at `-O2` — fine at kilobytes, but a 135M-parameter plan becomes a ~950 MB translation unit and a 16-minute `--build`. For anything past a few tens of megabytes, pass `--no-embed` (no C array is written; `--build` is refused, since there is nothing yet to link) and let the build-host packer embed the plan as an assembly stub that splices the `.seeu` in verbatim:

```bash
seeml-update-compile --source model.smf --out pkg/ ... --no-embed
python3 tool/pack_update.py pkg/ --build          # writes update_plan_embedded.S, runs build.sh
```

The packer is standard-library Python on the build host only; the package it leaves behind is the same dependency-free C++ (the stub needs only the preprocessor and assembler every C++ toolchain carries), the `.seeu` is untouched, and the binary trains the very same bits as the C-array build — CI asserts all three on every change. The embedded plan is page-aligned (`--page-align`, default 16 KiB) and, on Apple, placed in writable data — exactly what the Metal backend's zero-copy residency wants: a packed package trains from the plan's own pages, a heap-loaded plan is copied once. `build.sh` prefers the stub whenever one is present, so `pack_update.py` also upgrades an unmodified package emitted before `--no-embed` existed (it rewrites the one compile line and drops the old C array); a hand-edited `build.sh` is refused rather than guessed at.

**Very big plans: the plan stays beside the binary.** An embedded plan is the binary's size, and on macOS arm64 that stops working near 1.8 GB: the default linker segfaults on a 1.85 GB plan (f32 SmolLM2-360M), and a 2.72 GB plan (f32 Qwen2.5-0.5B) links only with `-ld_classic` and then cannot be loaded — the image runs past the address where dyld maps the shared cache, and arm64 executables are position-independent, so no base address fixes it (#158). So the packer has a second mode, chosen by size: from **1 GiB** (`pack_update.EMBED_LIMIT`, a margin under the first failure) it writes no stub at all and says so; `build.sh` compiles a two-line translation unit that defines an *empty* plan; and `model_update` maps `update_plan.seeu` **beside itself** at start-up (located through the OS relative to the executable, never the working directory) — a private, page-aligned, copy-on-write mapping the page cache serves, which is byte-for-byte the memory the stub provides, so the Metal backend's zero-copy residency and the one-arena-allocation doctrine are unchanged. The disk plan is validated exactly as an embedded one — magic, version gate, the seal (`plan_hash`), the source-model hash, both contracts — so a swapped or corrupt file is refused before anything runs; a missing file is exit 1 with the mode and both remedies named. `--plan FILE` on `model_update` names the plan elsewhere (and overrides an embedded one). Ship the two files together; `pack_update.py --no-embed` forces this mode at any size (what CI does on the demo package), `--embed` forces the stub at any size with a warning above the limit, and the pack report (schema 2) records `mode`, `mode_forced` and `embed_limit`. Measured on the f32 Qwen2.5-0.5B plan (2,724,184,064 bytes, Apple M5): the disk-mode package builds in 11 s, loads, validates and source-binds the plan in 5.1 s (`lifecycle.load_s`), and trains and commits on the CPU (two steps: 4.22 GB maximum resident, against a 4.51 GB arena and the 2.72 GB plan) — the row the Real Analysis Delta study could not run. On Metal the mapped plan is page-aligned and writable like the `.incbin` embedding, so the backend wraps its frozen weights zero-copy as before (verified: a disk-loaded SmolLM-135M package reports `rodata zero-copy` and commits the same bytes as the embedded one after 20 steps on Metal and on the CPU).

**What it prints.** Four statistics screens, described in [What the Compiler Tells You](#what-the-compiler-tells-you) below; `--verbose` adds the detail, `--peak-gflops G` a predicted step time.

Compilation is also your first line of defense: infeasible memory footprints, shape mismatches, a loss that can't see any trainable parameter — all fail *here*, on the build host, with a one-line `"<unit>: <message>"` diagnostic.

Curious what was actually generated? Disassemble any plan:

```bash
seeml-seeu-dump pkg/update_plan.seeu            # header, integrity check, emit table
seeml-seeu-dump pkg/update_plan.seeu --instrs   # all three instruction streams
```

Reading a training loop as thirty-ish opcodes of straight-line code is a genuinely instructive way to see what the compiler did — and a good sanity check that, say, your clip instructions exist.

## What the Compiler Tells You

Every compile prints four screens — what the frontend read, what the analysis passes did, the plan the backend emitted, and the runtime as planned — and nothing else unless you ask. Each line is a number, its unit, and the clause it comes from. `--report report.json` carries the same rows by key under `"screens"`, and `build/seeml-abi` publishes their schema (each row's key, kind, unit and flags), so a tool reading a report can check what it reads. `--verbose` adds the verbose rows (the step-0 footprint's four parts, alpha, gradients, the arena's segments, each phase's time), a table of every pass with the ops after it, a table of every adapter site, and each pass's own note. `--peak-gflops G` states the target's peak and turns the work per step into a predicted step time; the compiler never guesses a peak.

SmolLM-135M, compiled with the nightly's flags (`--data-batch 512 --lora-rank 8 --lora-alpha 16 --quantize-base --steps 300`) and a stated target of `--peak-gflops 1000`:

```
frontend — the model the compiler read
  operators                483 ops                       operators in the SMF model file
  tensors                  274                           tensors the file declares, the input included
  frozen weights           273                           the file's constant tensors
  weight payload           704,390,400 B (671.8 MiB)     those tensors' bytes in the file
  parameters               176,097,600                   weight elements: the payload ÷ 4 B (f32)
  input                    token ids, 128 per sequence   token ids per sequence, or features per row
  vocabulary               49,152 tokens                 rows of the embedding table
  step-0 bound             1,600,448,832 B (1.5 GiB)     the accountant's lower bound the early gate proved fits
  memory budget            17,179,869,184 B (16.0 GiB)   the configured budget, else this host's memory
analysis — what the passes did
  passes                   18                            passes and driver phases run, each timed
  ops in                   757 ops                       the forward SIR the frontend built
  ops out                  4,426 ops                     the training SIR after every pass, as lowered
  adapters                 211                           LoRA adapters grafted, one per targeted matmul weight
  rank                     8                             r, each adapter's inner width
  trainable                3,024,384                     the sum of r·(k + m) over the adapters
  trainable share          1.717 %                       trainable ÷ the model's parameters
  epilogue fusions         0                             bias and activation chains folded into a GEMM
  addend fusions           419                           GEMMs folded into the add that reads them
  chain fusions            0                             elementwise chains folded into one kernel
  dead ops removed         419 ops                       ops nothing reads, swept before lowering
  attention                cached, 30 op(s)              cached probabilities, or the tiled family's stats rows
  rematerialization        off: auto, no memory budget   forward values the backward recomputes instead of keeping
  narrow weights           211 int8                      frozen weights stored int8 or bf16 in rodata
backend — the plan it emitted
  train program            2,887 instructions            one micro-batch: forward, backward, optimizer
  eval program             907 instructions              the forward and the loss, for validation
  merge program            422 instructions              each adapter's delta, (alpha ÷ r)·A·B
  arena                    1,576,898,688 B (1.5 GiB)     the one allocation the device makes
  rodata                   262,048,512 B (249.9 MiB)     frozen weights packed into the plan
  plan file                298,631,168 B (284.8 MiB)     the sealed .seeu
  arena + plan             1,875,529,856 B (1.7 GiB)     what the device holds: the arena and the mapped plan
  bound ÷ exact            0.8533                        the step-0 bound ÷ arena + plan
  GEMM work                310,851,403,776 FLOP          per step: 2·M·N·K over the train GEMMs, times G
  attention work           13,589,544,960 FLOP           per step: 4·B·H·S²·d forward + 8·B·H·S²·d backward
  target peak              1000.0 GFLOP/s                the target's peak, as passed with --peak-gflops
  predicted step           324.441 ms                    (GEMM + attention work) ÷ the target peak
  kernel policy            runtime default               the CPU GEMM tiles the header carries
  plan hash                1063fab4621f7f4a              the plan's self-hash, which checkpoints bind to
runtime — planned by the compiler, measured by the device
  planned memory           1,875,529,856 B (1.7 GiB)     arena + plan: the resource contract the device checks
  work per step            324,440,948,736 FLOP          GEMM + attention work of one optimizer step
  predicted step           324.441 ms                    the work ÷ the target peak
```

Reading it top to bottom:

- **frontend.** The file holds 483 operators and 273 frozen weights — 672 MiB of f32, 176 million parameters — and its input is token ids, 128 per sequence, over a 49,152-token vocabulary. The step-0 bound (1.5 GiB) is the accountant's lower bound on the training footprint, proven to fit before any pass runs; with no `--memory-budget-mib` it is judged against this host's memory, which is what the budget row says.
- **analysis.** LoRA grafted 211 adapters at rank 8: 3.0 million trainable parameters, 1.7% of the model. The 419 addend fusions are every LoRA site's activation-sized sums folded into the GEMM that produces them, and the 419 dead ops are the GEMMs they orphaned. Attention keeps its probabilities (the 30 layers' caches fit the cache budget at this shape); rematerialization stays off because no memory budget was given; `--quantize-base` stored the 211 matmul weights as int8, which is why rodata is 250 MiB rather than the file's 672.
- **backend.** One micro-batch is 2,887 instructions; validation runs 907, the merge 422. The arena is the one allocation the device makes; arena + plan, 1.7 GiB, is what the device holds, and the step-0 bound was 85% of it (bound ÷ exact — the accountant's honesty at this shape). A step is 311 GFLOP of GEMMs and 14 GFLOP of attention (model FLOPs: causal masking not discounted, recomputation not counted); at the stated 1000 GFLOP/s that predicts 324 ms. The plan hash is what checkpoints bind to.
- **runtime.** What the device must hold and do, as planned. The device prints the same screen with what it measured beside it (Step 3).

The default output stays near 46 lines whatever the model's size: the rows are aggregates, and the per-item detail is `--verbose`'s. The suite pins the four screens as golden blocks (`test/fixtures/golden/screens/`: two MLPs and the `dec_wide` decoder), so a pass that silently changes what it does changes a number the suite sees; after a deliberate change, `SEEML_UPDATE_GOLDEN=1 ./build/seeml_screens_test` rewrites them for review.

## Step 3: Run the Update (device)

```bash
model_update --model model.smf --data corpus.sds --out updated.smf \
  --val-frac 0.1 --seed 7 \
  --checkpoint state.ckpt --checkpoint-every 100 --resume \
  --loss-log curve.csv
```

| flag | meaning (default) |
|---|---|
| `--model source.smf` | the model the plan was compiled from (required; must hash-match the plan) |
| `--data corpus.sds` | the training corpus (required) |
| `--out updated.smf` | where the committed model is written (`updated_model.smf`) |
| `--steps N` | optimizer steps (0 = the plan's compiled default); under `--grad-accum G` each consumes `G` micro-batches |
| `--seed S` | shuffle-permutation seed (0) |
| `--val-frac F` | held-out fraction for the regression gate (0.1); 0 gates on the training-loss trend instead |
| `--checkpoint path` | checkpoint file, hash-bound to the plan |
| `--checkpoint-every N` | steps between checkpoints (0 = off) |
| `--resume` | resume from `--checkpoint` if present. With no `--steps` it trains exactly the **remainder** of the interrupted run, on that run's learning-rate schedule — the checkpoint records the horizon — so the result is bit-identical to the run left uninterrupted; with `--steps N` it trains N *further* steps and the schedule is stretched to cover them |
| `--loss-log curve.csv` | write the per-step loss curve |
| `--plan FILE` | the plan to run, for a package built in the disk mode (default: `update_plan.seeu` beside the executable); also overrides an embedded plan. Validated in full either way — version gate, seal, source-model hash; a missing or swapped file is exit 1 |
| `--backend cpu\|metal\|auto` | the executor (cpu; or `$SEEML_BACKEND`). `cpu` is the bitwise-deterministic reference and builds anywhere; `metal` runs the update on an Apple GPU and is an error where none exists; `auto` takes the GPU when present and says so, else cpu |
| `--min-improvement F` | commit only if the gated loss fell by at least the fraction F of its initial value (0 = any strict fall) |
| `--require-accuracy` | additionally require held-out accuracy not to drop; a usage error on plans without class labels or without a validation split |
| `--attention auto\|cached\|tiled` (compiler) | attention memory: `cached` keeps the `[B·H·S, S]` probability cache from forward to backward, `tiled` keeps four floats per query row and recomputes (about 3× the attention backward, 1.2–1.7× a whole step in the measurements of `docs/benchmarks.md`); `auto` (default) tiles when the caches of all layers exceed `--attention-cache-budget-mib` (256) or, with `--memory-budget-mib N` given, when the step-0 footprint would not fit that budget with them — never the compiling host's memory, so identical inputs give identical plan bytes on every host (F3 #131); a footprint the host cannot hold is still refused by the memory gate, which says when `--attention tiled` would fit. Identical bits either way |
| `--remat auto\|off\|on` (compiler) | rematerialization (roadmap 2b, F5 #133): `on` recomputes the cheap forward activations (norm outputs, `silu(gate)`, the SwiGLU product, folded chains; never a GEMM or attention output) in the backward instead of holding them across the forward — the same bits, a longer train program, a smaller arena; `auto` (default) does so only when the arena plus the plan would exceed a budget given as `--memory-budget-mib N` — never the compiling host's memory, so identical inputs give identical plan bytes on every host; with no budget, `auto` leaves the plan plain and says so; `off` never. The compile report's `remat` object records the decision and the bytes saved |
| `--memory-budget-mib N` (compiler) | the memory gate's budget for compile-time decisions that change the plan's bytes (`--remat auto`, `--attention auto`); without it the gate still refuses a compile the host cannot run, but takes no decision that would make the plan host-dependent |
| `--adapter-only` (compiler) | the adapter-only package (F6 #134, plan v21): every frozen weight is a source ref — read at run time from the model file the plan is bound to by hash, in every program — and the plan carries no rodata: the instructions, the adapters' initial image and the emit table, at adapter scale (SmolLM-135M r8: 35 MB against the 299 MB int8 package). The device commits onto its own copy of the model. An f32 base only, exact arithmetic, a model loaded from a file; the driver refuses the other combinations by name. On Metal the file is bound to the GPU when it fits beside the arena, else its readers run on the CPU (`docs/runtime.md`). |
| `--validation-scores shipped\|plan` (compiler) | where a narrow-base plan compiled from a file reads the frozen weights in its eval program (E12, plan v17): `shipped`, the default, scores the source model's f32 — the function that ships, exactly; `plan` scores the plan's own int8 / bf16 copies, a proxy for a device that cannot hold the f32 file beside the arena (the 1.5B rung on 16 GB, F5 #133; a phone). The compile report records which (`validation_scores`); an adapter-only plan has no weights of its own and refuses `plan`. |
| `--eval-every N` | with a validation split, evaluate every N steps and commit the **best** evaluated state rather than the last; default a tenth of the run (at least 1); `0` evaluates only before and after and commits the last state, bit for bit as before |
| `--patience K` | stop after K consecutive evaluations without a new best (0 = never); needs `--eval-every > 0` and a split |
| `--report report.json` | write the backend, the gate parameters, the loss and accuracy pairs (`validation_loss[1]` is the committed state's; `best_state` carries its step, the endpoint's loss and the evaluation settings; `validation_scores` says what they score; `relaxed_executions` counts the instructions the backend ran on a relaxed kernel, 0 on the exact kernels — schema 3), the verdict and whether the commit landed as JSON (written last; an unwritable path is a warning, the exit code still carries the verdict) |
| `--peak-gflops G` | the target's peak in GFLOP/s, for the runtime screen's predicted step and utilization (default: the compile's `--peak-gflops`, baked into the runner; unstated means no prediction) |
| `--force` | commit even if the gate shows no improvement |

What happens, in order:

1. **Load + verify** — the plan's hash is checked, then every instruction operand is bounds-validated *before anything executes*. One arena allocation, sized at compile time. A corrupt or foreign plan is refused at this door.
2. **Split + shuffle** — the last `--val-frac` (here 10%) of the corpus is held out for validation; training batches are then served through a seeded per-epoch permutation. Same `--seed`, same batches, same bits — on any machine.
3. **Train** — N steps of forward + backward + clip + optimizer. Interruptible: checkpoints (`--checkpoint`, `--checkpoint-every`) are hash-bound to the plan and fsync-durable, and `--resume` picks up from one — optimizer momentum, step counter, *and* the data-shuffle position, which is replayed so a resumed run produces the same bits as one that was never interrupted. `--steps` after `--resume` counts further steps, not a total. A non-finite loss aborts immediately rather than continuing on garbage.
4. **Gate** — validation loss is evaluated before training, every `--eval-every` steps, and after, with the plan's eval program; the state that scored best is the one gated and committed (the log line names its step and the endpoint's loss; the report's `best_state` records both). A resume keeps the *source model's* "before" from the checkpoint rather than re-scoring the resumed adapter, and refuses a `--seed` or `--val-frac` other than the first run's (exit 2). No improvement — or less than `--min-improvement`, or a dropped accuracy under `--require-accuracy` — → exit code 3 and the device is left *untouched* (`--force` overrides, if you must). The gate line names the failing gate and the backend that produced the numbers.
5. **Merge + commit** — deltas `Δ = (α/r)·A@B` are materialized and added to the pristine f32 weights of the source file (which must hash-match the plan), written durably, renamed atomically. A power cut at any moment leaves the old model or the new one — never a torn file.

After the gate, the device prints the runtime screen: the plan's own rows — recomputed from the plan it loaded, with the compile's `--peak-gflops` baked in (a run's `--peak-gflops` overrides it) — beside what this run measured. The demo package (`tool/export_model.py --demo`, compiled with `--peak-gflops 500`), on an M5 on battery:

```
runtime — planned by the compiler, measured by the device
  planned memory           55,680 B (54.4 KiB)           arena + plan: the resource contract the device checks
  work per step            84,992 FLOP                   GEMM + attention work of one optimizer step
  predicted step           0.00017 ms                    the work ÷ the target peak
  backend                  cpu                           the executor that ran the plan
  steps                    300                           optimizer steps trained
  measured step            0.011 ms                      training wall time ÷ steps, evaluations excluded
  achieved                 7.4 GFLOP/s                   the work per step ÷ the measured step
  utilization              1.5 %                         achieved ÷ the target peak
  peak RSS                 6,799,360 B (6.5 MiB)         the process's resident peak, as the OS reports it
  RSS ÷ planned            122.115                       the resident peak ÷ the planned memory
  train loss, start        1.3096 nats                   the mean loss over the run's first window of steps
  train loss, end          0.1664 nats                   the mean loss over the run's last window of steps
  validation loss, start   1.4304 nats                   the held-out loss before training
  validation loss, end     0.0988 nats                   the held-out loss of the state the gate weighed
  input-blind loss         1.3261 nats                   the training labels' entropy: an input-blind predictor's loss
  validation accuracy      0.9559                        the held-out accuracy of the state the gate weighed
  verdict                  accepted                      the gate's decision on the update
```

The measured step is the steps alone — staging the batch, the train program, the optimizer — with evaluations and checkpoints excluded; achieved is the work ÷ that step, and utilization is achieved ÷ the peak (here a toy plan on a 500 GFLOP/s statement: the gap is the point of the row). Peak RSS is the whole process, so a 54 KiB plan sits under 6.5 MiB of runtime. The input-blind loss is the entropy of the training labels — what a model that ignored its input would score, a reference level rather than a bound: this run ends at 0.0988 nats on held-out data against 1.3261, so it learned from its inputs. `--report` carries the same rows under `"runtime"`.

With `--checkpoint` and `--checkpoint-every`, the storage role adds three rows after the resident peak. The same package, checkpointing every 100 of its 300 steps:

```
  checkpoints              3                             the durable checkpoints the run saved
  per checkpoint           16,240 B (15.9 KiB)           bytes written per checkpoint, the mean
  fsyncs per checkpoint    2.0                           the file's and its directory's, the mean
```

Each checkpoint is the plan's persistent segment behind a header, staged beside the destination, synced, renamed into place, and its directory synced: two fsyncs, or one on a filesystem that refuses a directory fsync. The bytes are a mean because a checkpoint also carries the best state while it differs from the segment.

Exit codes are the API for your update orchestrator: `0` committed, `1` runtime error, `2` bad arguments, `3` regression-gate rejection, `4` committed but `--report` or `--loss-log` could not be written. `--loss-log` writes the full per-step CSV loss curve once training completes (it is not appended live, so tail it after the run, not during).

## Threading and Determinism

Both halves parallelize: the compiler's byte-heavy passes (int8 quantization, adapter initialization, plan embedding) and the runtime's kernels, plus a feeder thread that stages the next batch while the current step computes. Control it with one variable:

```bash
SEEML_THREADS=1 model_update ...   # fully serial: no thread is ever created
SEEML_THREADS=4 model_update ...   # pin the pool width; default = all cores
SEEML_METAL_PROFILE=1 model_update --backend metal ...   # per-kernel GPU time at exit (serializes; diagnostic only)
SEEML_METAL_PROFILE=1 model_update --backend metal --val-frac 0 --eval-every 0 ...   # the train-only profile: every pass a training pass
SEEML_METAL_SPANS=1 model_update --backend metal ...     # per command buffer: GPU busy vs CPU wall, the time outside kernels
```

The profile's header counts the forward and backward passes it saw (at the loss opcodes), so a reader can tell a train-only profile from one taken with validation on, where the evaluations' forward passes outnumber the backward ones many to one and every per-family share reads too forward-heavy. Kernel decisions are taken on the train-only profile.

Here's the property that makes this knob safe to turn: parallel execution is **bitwise-deterministic**. Work is split into chunks whose boundaries depend only on the problem shape (never the thread count), and reductions combine per-chunk partials in a fixed order — so the same plan, data, and seed produce the *same bits* at any thread count. Thread count is a throughput knob, not a numerics knob. Practically: a loss curve from an 8-core dev board reproduces exactly on a single-core target, and any bug you find is reproducible by construction. ([runtime.md](runtime.md) explains the mechanism.)

This is a tested contract, not an aspiration:
`ParallelFor.OrderedPartialReductionIsBitwiseThreadCountInvariant`
(`test/source/parallel/parallel_for_test.cc`) proves the substrate, and
the `ParallelDeterminism` suite (`test/runtime/executor/kernels_test.cc`)
re-proves every kernel family bit-for-bit across thread counts.

## Development

Working on SeeAI itself:

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
# without CMake (builds every per-module SeeTest suite):
sh build/build.sh && for t in build/seeml_*_test; do "$t"; done
# one suite, one test:
./build/seeml_update_engine_test --filter=UpdateEngineCheckpoint
# sanitizers / fuzzing:
cmake -B build -DSEEML_SANITIZE="address;undefined"
cmake -B build -DSEEML_SANITIZE="thread"   # proves the pool + feeder sync
cmake -B build -DSEEML_FUZZ=ON && ./build/seeml_fuzz_formats
```

The test tree mirrors the source tree one-to-one — a suite lives where its subject lives — and [test/README.md](../test/README.md) explains the layout and the in-tree harness. The thread sanitizer build is not decoration: it's the mechanical proof that the worker pool and the feeder handoff are race-free, and the fuzzer hammers the binary-format parsers with hostile bytes — the same parsers whose paranoia [formats.md](formats.md) describes.
