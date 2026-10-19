# Benchmarks — The Metrics That Gate Frontier Development

As the frontier program lands (token-native training now; QLoRA-class
quantization, GPU dispatch, chunked cross-entropy, and scale passes next),
"is it faster / does it still fit" stops being something you can eyeball.
This document defines the metric program: what to measure, why each metric
gates a specific frontier decision, and the measurement discipline that
keeps numbers comparable across commits.

## Measurement discipline (non-negotiable)

- **Fixed seeds, fixed inputs, pinned `SEEML_THREADS`.** Every benchmark
  is a compiled plan + seeded corpus; determinism makes medians of 5 runs
  tight enough that a >3% delta is signal, not noise.
- **Medians, not means; report the machine.** Apple M-series vs x86 AVX2
  are different regimes — a number without its host string is not a
  benchmark.
- **Per-commit trend, not absolutes.** The gate is regression against the
  previous commit's number on the same host (CI can carry the baseline as
  an artifact the way the fuzz corpus is cached).
- **Measure what ships, and say what ran.** The CPU GEMM tile geometry
  travels in the plan (v11), so the harness compiles every fixture with the
  policy a package would carry — the runtime defaults, or `--gemm-tiles` /
  `--kernel-policy` exactly as `seeml-update-compile` takes them — and the
  report (schema 3) records the policy, the host key a table is keyed on,
  and the host description. `bench_compare.py` refuses to compare two runs
  whose policies differ: that delta is a tuning result, not a regression.
- **External definitions, not house units.** Every headline number is
  reported under the definition an outside harness already prints for it
  (next section). A metric only this repository defines cannot be set
  beside an MLX-LM or llama.cpp log, so it cannot answer "how far behind
  the frontier are we" — the question the whole program exists to answer.

## External standards the harness reports in

`seeml-bench` emits each Tier A/C cell twice: once under the internal key
the regression gate has always compared (`rows_per_s`), and once under the
key and definition the external tool uses. The numbers are identical by
construction where the definitions coincide; the point is that a reader
of an `mlx_lm.lora` log or a `llama-bench` table needs no unit translation.

| field | external standard | definition here | note |
|---|---|---|---|
| `tokens_per_s` (token/decoder fixtures), `samples_per_s` (feature fixtures) | MLX-LM `lora` **Tokens/sec** | loss-target rows per wall second of training | MLX counts loss-masked target tokens only; every row of a SeeAI batch is a target (next-token label per position), so this is exactly `rows_per_s` — the gate key is unchanged, the unit is now named |
| `it_per_s` | MLX-LM **It/sec** | optimizer steps per second = `1000 / step_ms` | per-step, from the steps-regression slope, so lifecycle cost is excluded as MLX's warm iterations exclude it |
| `step_ms_min` / `step_ms_max` | `llama-bench` **t/s ± σ** | spread of the per-step slope across `--repeats` | medians stay the headline; the spread says whether a delta is signal |
| `train_loss_first` / `train_loss_last`, `trained_tokens` | MLX-LM **Train loss**, **Trained Tokens** | first/last windowed mean training loss over the whole sweep; rows trained | a sanity anchor that the timed work was real training, not a stalled loop |
| `peak_rss_bytes`, `rss_over_planned` | MLX-LM **Peak mem**; Tier C "peak RSS vs arena" | OS-observed peak resident set (`getrusage`); ratio to `arena + plan` | process-cumulative, so it rises monotonically across fixtures; on the MB-scale fixtures the ~20 MB process floor dominates the ratio, so read it at model scale (where the field audits found 1.00–1.13×) |
| `mfu` (only with `--peak-gflops F`) | PaLM / Megatron **model-FLOPs utilization** | achieved GEMM FLOP/s ÷ the host peak you pass | the harness never guesses a peak — a guessed denominator is not a standard. Published anchors (arXiv 2502.05317): Apple M4 GPU ≈ 2.9 TFLOP/s measured, M-series CPU AMX ≈ 1.49, NEON-only ≈ 0.84 |
| `gemm_gflops` | (internal) | `gemm_flops_per_step / step_ms`, FLOPs summed as 2·M·N·K over the plan's own GEMM instructions | the numerator of `mfu`; kept because it needs no external input |
| `host` | every standard's "report the machine" rule | `uname` sysname + machine | the JSON is self-describing; a number without its host string is not a benchmark |

What is deliberately *not* reported: a validation perplexity. The fixtures
train on seeded synthetic corpora, so a val loss here would be an internal
number dressed as lm-eval output. Quality metrics against a real model
(SmolLM-135M val loss/accuracy vs the torch reference) live in the
validation field reports, where the corpus is real text.

Reference frontier numbers, measured on this host class (Apple M5, 16 GB,
2026-09-22, same model, corpus, split and 512 tokens per step as SeeAI —
"The frontier row, measured" below): on SmolLM-135M, `mlx_lm`'s LoRA
trainer reaches 4,605 tok/s in bf16 (9.0 it/s; 1.34 GB Metal-allocator
peak, 1.01 GB resident) and 2,240 in
true f32; `torch.compile` reaches 3,761 tok/s on MPS in bf16, 2,099 in f32,
and 847 on the CPU in f32 (Accelerate/AMX). SeeAI's Metal backend trains
the same model at 1,802 tok/s (f32 math, int8 base) — 0.80–0.86× the f32
frontier and 0.39–0.48× the bf16 one (MLX-LM's bf16 is all bf16;
`torch.compile`'s is a bf16 base under f32 adapters) — and its CPU
package at 332, 0.39× of `torch.compile`. Half precision (F2) and the CPU
matrix units (F4) are the gaps those ratios price. (The v1.2.4 report's M4 anchors — MLX ~1,600
tok/s "bf16", torch.compile ~40% MFU — were carried from another machine;
the MLX run was f32, since SmolLM-135M's safetensors store F32.)

## Tier A — Training throughput (the headline numbers)

| metric | definition | what it gates |
|---|---|---|
| **tokens/sec** (token plans) | `batch × steps / wall` on the token tiny-GPT and a mid decoder (e.g. D=512, 8 heads, S=256) | every kernel/GPU/fusion decision rolls up here; THE frontier comparison number against MLX / llama.cpp-finetune |
| **step latency breakdown** | wall time of one `ExecuteTrainOnce`, split fwd / bwd / optimizer (instrument at the instruction-stream level: the three programs' boundaries are known offsets) | where the next kernel dollar goes — if bwd ≫ 2×fwd, attention backward or GEMM-TN is the target |
| **samples/sec** (feature plans) | the same for MLP-class plans | regression canary for the classic path while transformer work churns shared kernels |
| **scaling efficiency** | tokens/sec at `SEEML_THREADS=1,2,4,8` vs ideal | the parallel_for chunking + feeder overlap health; a flat curve says grain constants (`kGrainCheap/kGrainMath`, `RowGrain`) need retuning |

## Tier B — Kernel-level (why Tier A moved)

| metric | definition | what it gates |
|---|---|---|
| **GEMM GFLOP/s vs peak** | blocked cores at the shapes the decoders actually emit (projections, SwiGLU, lm-head) | the explicit-SIMD kernel project (frontier pillar C2): if autovectorized cores sit at <30% of peak, hand SIMD is worth it; if 60%+, it is not |
| **attention μs and bytes** | `AttnFwd` + backward chain per (B,H,S,d) sweep of S ∈ {64, 256, 1024, 2048} — `seeml-bench --attention-sweep` | the flash-attention decision — run and recorded below (E11, #94): the tiled family is a memory decision, not a speed one |
| **EmbedFwd GB/s** | gather bandwidth vs `memcpy` bandwidth | it should be memory-bound; if not, the row-grain is wrong |
| **softmax-xent μs at vocab ∈ {1k, 32k, 128k}** | fwd+bwd per row | the chunked-CE project (pillar A3): this curve IS the justification |
| **q8 dequant overhead** | q8 GEMM / f32 GEMM time ratio | NF4/int4 design: if int8 dequant already costs >15%, block-wise 4-bit needs a fused design, not a naive port |
| **Metal vs CPU GEMM** | the G1a harness at real shapes | the G1b engine-integration go/no-go: dispatch overhead amortization point (at which M×N×K does GPU win?) |
| **backend split** | `seeml-bench --backend metal` vs `cpu` on the same fixtures; `bench.json` carries `"backend"` and the gate keys rows by it | where the GPU pays: on this suite's tiny fixtures Metal runs at the dispatch floor (0.64–3.50× the CPU, Apple M5); on the SmolLM-135M-shaped `tok_smollm135m_q8` fixture 7.7× (1,786 vs 233 tok/s; 27.5× before #104 fixed the CPU's q8 NT kernel) — see "The first GPU baseline row" below |

### The GEMM row after the redesign (E1, #80 — Apple M5, 2026-09-17)

The Tier B question above, answered for the cores as redesigned — still
portable C++ with no intrinsics, and bit-identical to the cores they
replace on every shape, tiling and thread count (so no baseline, plan or
checkpoint moved):

| GEMM (M × K × N) | 1 thread before → after | 8 threads before → after |
|---|---:|---:|
| NN projection 512 × 512 × 512 | 29.5 → 48.9 GFLOP/s | 175 → 332 |
| NN FFN-up 512 × 512 × 2048 | 22.5 → 50.9 | 142 → 357 |
| NN lm-head 512 × 512 × 4096 | 19.7 → 49.5 | 127 → 351 |
| NN SmolLM head 512 × 576 × 49152 | 13.8 → 48.8 | 67 → 260 |
| NT dX 512 × 512 × 512 | 28.0 → 45.8 | 183 → 288 |
| NT dX FFN 512 × 2048 × 512 | 26.9 → 46.6 | 177 → 300 |
| TN adapter dB 16 × 512 × 512 (small M) | 27.6 → 39.2 | 101 → 134 |
| int8 weights, NN 512 × 512 × 512 | 21.6 → 50.9 | 164 → 348 |

Against the ~840 GFLOP/s NEON envelope the field report derived, the
large-shape cores moved from 15–22% to **31–43% of peak** at eight threads —
inside the 30–60% band where this document says hand-written SIMD is no
longer clearly worth it. End to end: `dec_wide` 1,808 → 2,637 tok/s (8T)
and 390 → 622 (1T); the SmolLM-135M-shaped `tok_smollm135m_q8` fixture
196 → 269 tok/s on the CPU backend, losses identical to the last printed
digit. `seeml-plan-probe --profile N` attributes a plan's step per opcode
and GEMM shape: on `dec_wide` the GEMM family is now ~65% of the step and
the attention family ~24% — the f64 score reductions of #77 and the S²
cache of E11 (#94) are where the next CPU kernel dollar goes, not GEMM.

### The flash-attention decision, measured (E11, #94, 2026-09-18)

`seeml-bench --attention-sweep` times the attention kernels alone —
forward, then the whole backward chain — for both families at B = 1,
H = 8, d = 64 on an Apple M5 (median of 3):

| S | threads | cached fwd / bwd (ms) | tiled fwd / bwd (ms) | kept fwd→bwd, cached → tiled |
|---|---|---|---|---|
| 64 | 1 | 1.07 / 1.45 | 0.97 / 3.24 | 128 KiB → 8 KiB |
| 256 | 1 | 10.7 / 16.0 | 10.7 / 47.6 | 2 MiB → 32 KiB |
| 1024 | 1 | 172 / 252 | 171 / 777 | 32 MiB → 128 KiB |
| 2048 | 1 | 684 / 1,036 | 684 / 3,107 | 128 MiB → 256 KiB |
| 64 | 8 | 0.15 / 0.41 | 0.16 / 0.59 | |
| 256 | 8 | 1.63 / 2.53 | 1.62 / 7.07 | |
| 1024 | 8 | 25.3 / 39.4 | 26.4 / 116 | |
| 2048 | 8 | 103 / 185 | 109 / 476 | |

The forward costs the same; the tiled backward recomputes the scores in
each of its three passes and dP in two, and runs at about 3× the cached
backward at every length and both widths. **There is no crossover on this
CPU**: no S where tiling is faster, because the cached family's S² traffic
is served from a large unified memory without becoming the bottleneck.
What tiling buys is memory. End to end, on a 4-block, D = 512, H = 8
decoder at batch = one sequence, trained by the emitted `model_update`
(`--eval-every 0`, second of two runs each):

| S | step time, cached → tiled | peak footprint, cached → tiled | committed model |
|---|---|---|---|
| 256 | 1.55 → 1.84 s / 20 steps (1.19×) | 88 → 80 MiB | byte-identical |
| 2048 | 8.99 → 14.91 s / 6 steps (1.66×) | 1,039 → 324 MiB | byte-identical |

The probability caches are 8 MiB of the 86 MiB arena at S = 256 and
512 MiB of 1,036 MiB at S = 2048 — the S² term that makes long sequences
a memory-gate refusal on a device. So the compiler's rule is a memory
rule: `--attention auto` keeps the cache while all layers' caches fit
`--attention-cache-budget-mib` (256 by default) and the whole footprint
fits local memory with them, and tiles otherwise. A GPU, where the
cached family's S² traffic does cost bandwidth, may move the crossover;
that is #63's measurement to make on the tiled family.

### The first GPU baseline row (v1.3.0 gate, 2026-09-15)

`tok_smollm135m_q8` on an idle Apple M5 (10 GPU cores), `--threads 10`,
per-step medians by steps-regression; the Metal row at the harness
defaults (20→80 steps, 3 repeats), the CPU row at `--steps-lo 5
--steps-hi 15 --repeats 1` because a default sweep of it takes the better
part of an hour. The step trains 512 tokens; `gemm_gflops` sums 2·M·N·K
over the plan's own GEMM instructions.

| backend | step ms | tok/s | it/s | GEMM GFLOP/s | fwd / bwd / opt ms |
|---|---|---|---|---|---|
| metal | 286.6 | 1,786 | 3.49 | 1,084 | 127 / 158 / 1.9 |
| cpu | 2,194 | 233 | 0.456 | 141.7 | 1071 / 1134 / 13 |
| cpu, before #104 | 7,841 | 65 | 0.128 | 39.6 | 1253 / 6591 / 14 |

The same binary on the real SmolLM-135M package (the T2 import, the docs
corpus, `model_update --backend metal`) measured 1,749 tok/s by the
same regression on 2026-09-15 (1,802 on 2026-09-22, "The frontier row,
measured" below); on Metal the fixture's seeded weights and corpus
reproduce the model's per-step cost, not its loss. On the CPU the fixture
under-reads: the real package measured 332 tok/s on 2026-09-22 ("The
frontier row, measured" below) against the fixture's 233 here and 269
after E1, so the CPU baselines of F4 (#132) are set from the real-model
row, not from this table. The CPU row was re-measured on
2026-09-15 after #104: `GemmNTQ8` — the dX backward through every frozen
projection, 208 of them per step at this geometry — had run scalar since
#66 because the int8 cast sat inside the vectorized lane loop; widening
each block first (bit-identical, see `gemm.cc`) took the backward from
6.6 s to 1.1 s and the row from 65 to 233 tok/s.

### The frontier row, measured (2026-09-22)

The first same-host frontier measurement: the emitted `model_update`
package against `torch.compile` and `mlx_lm`'s LoRA trainer on four
Hugging Face checkpoints imported with `export_model.py --hf`, all at one
configuration — this repository's documentation as the corpus (129-id
records, the tail 10 % held out, identical ids in every stack), LoRA r8 /
α16 on the seven projections and the head, AdamW 1e-4 with weight decay
0.01, constant LR, no clipping, 4 records × 128 targets = 512 loss tokens
per step, 300 steps, 10 threads. Apple M5 (10-core GPU, 16 GB), main at
1d6b8ed (the code that ran), driven by the F1 harness `tool/frontier_run.py`
(#129, merged in #135), torch 2.14.0, mlx 0.32.2 / mlx-lm 0.31.3, warm chip, strictly
serial. SeeAI's step is the steps-regression slope on the package binary
(the 15→300-step slope where a 300-step run exists); PyTorch's is the
median step after warm-up; MLX's is the trainer's own `It/sec` × 512.
The two cells marked † are a single 5→15-step slope with `--repeats 1`
and no 300-step run; on the same models the Metal 15→300 slope came out
23–38 % slower than the 5→15 one, so read those two cells as optimistic.
Tokens per second, higher is better:

| model | SeeAI `metal` (int8 base) | SeeAI `cpu` (int8 base) | torch.compile MPS f32 / bf16 base with f32 adapters | torch.compile CPU f32 | MLX-LM true f32 / all-bf16 |
|---|---:|---:|---:|---:|---:|
| SmolLM-135M | 1,802 | 332 | 2,099 / 3,761 | 847 | 2,240 / 4,605 |
| SmolLM2-135M | 1,754 | 335 | 2,184 / 3,833 | 842 | 2,234 / 4,690 |
| SmolLM2-360M | 830 | 133 † | 1,041 / 2,121 | 445 | 1,009 / 2,324 |
| Qwen2.5-0.5B | 636 | 102 † | 767 / 1,668 | 345 | 734 / 1,768 |

Read across a row: SeeAI's f32-math Metal package runs at 0.79–0.87× the
f32 frontier and 0.36–0.48× the bf16 one (0.36–0.39× of MLX-LM's all-bf16
trainer, 0.38–0.48× of `torch.compile`'s bf16 base); its CPU package at
0.39–0.40× of `torch.compile` on the two 135M models and 0.30× on the two
† cells (PyTorch eager is 2.2–3.2× SeeAI there, so the matrix units are
the whole CPU gap; `torch.compile` adds 1.06–1.16× over eager). Half
precision is worth 1.75–2.17× to `torch.compile` end to end on this GPU
(bf16 base, f32 adapters) and 2.06–2.41× to MLX-LM (all bf16) — the F2
budget, two experiments, not one. SeeAI's base storage barely moves its Metal step
(int8 1,802, bf16 1,756, f32 1,776 on SmolLM-135M) but the per-tensor int8
base costs +0.10–0.14 nats of validation loss at step 0 on every model and
+0.05 at step 300 on SmolLM-135M (E12, #128, is the fix). Where the
arithmetic matches, the answers match: SeeAI f32 and PyTorch start from the
same validation loss (4.5743 vs 4.5744) and end within 0.005; two Metal
runs of each 135M model are bitwise identical; the CPU and Metal backends
agree to 1e-5. Memory on SmolLM-135M, like against like: peak resident
set (`peak_rss_bytes`, the definition above) SeeAI 1.89 GB — its arena,
plan and Metal buffers are all shared host memory and all resident —
against MLX-LM 1.01 GB (bf16) / 1.06 GB (f32) and PyTorch MPS 0.6–1.4 GB;
the frameworks keep their weights and activations in the Metal
allocator, which the resident set does not see: MLX-LM's allocator peak
is 1.34 GB (bf16) / 2.11 GB (f32) and PyTorch's device peak 2.4–2.6 GB.
Neither number alone ranks the three. `torch.compile`
pays 40–54 s of warm-up whenever a graph is new to Inductor's cache (6–9 s
when it is not); SeeAI's package has none.

Parity rules a frontier row must pin, learned the hard way: cast the base
dtype explicitly (SmolLM-135M's safetensors are stored F32 although its
config says bfloat16, so `mlx_lm` runs it in f32 unasked); pass
`bias_correction=True` to `mlx.optimizers.AdamW` (its default is off, which
roughly triples the first update and changes the loss trajectory);
`MLX_ENABLE_TF32=0` for a true-f32 MLX row (the default matmul is 1.3–1.5×
faster and not f32); adapt the head in the frameworks too (the importer
writes the tied head as its own weight and SeeAI adapts it); and note that
`mlx_lm`'s `evaluate` drops the ragged validation tail (48 of 50 records).
The harness, raw JSON and loss curves are in `out/frontier-2026-09-22/`
(not committed); the full report is the SeeAI Frontier Harness page of
2026-09-22.

### The relaxed GEMM family on Metal, measured (F2 #130, 2026-10-15)

The first Metal number for the certified half-precision family (plan v18,
`--precision certified-bf16`, #136): the SmolLM-135M int8-base package of
the frontier row above, compiled exact and relaxed from the same source
and flags (the two plans are byte-identical to the ones certified on
2026-09-23 — exact `c9181488851bad75`, relaxed `12a25add58faee3f`, 419
frozen-weight GEMMs carrying `kFlagRelaxed`; certificate `gemm.relaxed`
observed 6.2e-3 against 2⁻⁷), run by `tool/frontier_run.py matrix` on the
same Apple M5 (macOS 26.6.2, 10 threads): two 40→160-step slopes and a
300-step run per package, the relaxed package twice, an exact run again
at the end as the thermal-drift control, and `mlx_lm`'s all-bf16 trainer
re-run in the same session. SeeAI's step is the 160→300 slope, MLX's the
trainer's `It/sec` × 512.

| run | step (ms) | tokens/s | validation 0 → 300 | gate | committed model |
|---|---:|---:|---:|---|---|
| SeeAI `metal`, exact (f32 math, int8 base) | 287.2 | 1,783 | 4.5743 → 4.0137 | accepted | `a82b22e0…` |
| SeeAI `metal`, relaxed (certified-bf16), run 1 | 203.3 | 2,519 | 4.5743 → 4.0136 | accepted | `93fb4a37…` |
| SeeAI `metal`, relaxed, run 2 | 206.8 | 2,476 | 4.5743 → 4.0136 | accepted | `93fb4a37…` |
| SeeAI `metal`, exact, drift control (40→160 only) | 288.4 | 1,775 | — | — | — |
| MLX-LM all-bf16 | 114.7 | 4,465 | 4.5759 → 4.0427 | — | — |

Read against F2's acceptance: the relaxed package runs **1.41× the exact
one** and **0.56× MLX-LM's all-bf16 trainer** on the same chip in the same
hour — the ≥ 0.9× target is not met, and the number says why, not the
kernels: the frozen GEMMs are 47.6 % of the exact step's kernel time
(the profile below) and the relaxed kernels run them 2.8× faster, which
predicts 1.38× on the measured step; a relaxed family that cost *nothing*
would top out at 1.76× on the measured step (1.91× on kernel time), and
0.9× of MLX-LM's 4,465 needs 2.25×, so the line is out of this family's
reach by construction.
What remains of the step is the rank-8 LoRA family (F7 #138) and the
dispatch model with its CPU-resident loss (F8 #139), which F2 does not
touch. The numerical acceptance holds: the 300-step validation loss
differs by 2.4e-5 relative (bound 1e-3), the gate decides identically,
the two relaxed runs commit byte-identical models, and the exact package
is where the 2026-09-22 row left it (1,783 vs 1,802 tok/s; the MLX
reference itself read 4,465 today against 4,605 then). Peak resident set
is 2.59 GB on every SeeAI row against 1.89 GB on 2026-09-22: E12 (#128,
merged after that row) made the eval program read the shipped f32 weights
from the 704 MB source model, which is resident while the gate scores it;
the arena is unchanged at 1.56 GB.

The train-only Metal profile (`SEEML_METAL_PROFILE=1`, `--val-frac 0
--eval-every 0`, 20 steps and the commit, so every pass is a training
pass — the mode F9 #140 asks for) puts the gain where the family says it
is. Kernel time per step, summed from the per-kernel spans:

| family | exact (ms) | relaxed (ms) | relaxed share |
|---|---:|---:|---:|
| frozen-weight GEMMs (q8; tensor-op `k_to_bf16+k_gemm_rx_*` when relaxed) | 124.0 | 44.5 | 25.8 % |
| rank-8 LoRA GEMMs with their scale and add | 96.7 | 89.5 | 51.9 % |
| elementwise, norms, fills | 14.4 | 15.2 | 8.8 % |
| attention's batched GEMMs | 11.2 | 9.8 | 5.7 % |
| CPU-resident opcodes (softmax cross-entropy 11/12, embedding, RoPE table) | 9.3 | 9.4 | 5.5 % |
| attention softmax, optimizer | 4.6 | 4.2 | 2.4 % |
| **kernel time summed** | **260.2** | **172.4** | |
| measured step | 287.2 | 203.3 | |

The relaxed kernels run the frozen GEMMs 2.8× faster; nothing else
moved; the LM head's `512×49152×576` product alone is 10.3 ms of the
relaxed step. After F2 the rank-8 family is half the kernel time and the
time outside kernels (the measured step less the sum, 27–31 ms, including
the 9 ms the CPU-resident loss costs plus its drain) is the next
sixth — F7 and F8 by name. Raw JSON, profiles and logs in
`out/f2-2026-10-15/` (not committed).

### The nightly frontier row's recipe, run by hand (F1 #129, 2026-10-15)

The opt-in frontier job in `nightly.yml` needs a self-hosted Apple-silicon
runner and had never been dispatched, so its recipe was run on the M5
verbatim: `export_model.py --hf` on the SmolLM-135M checkpoint over the
docs/ tree (531 records of 129 ids), the compile and `pack_update.py
--build`, then the five-row matrix. It is the same configuration as the
2026-09-22 row on today's docs, main at 0832bdd plus this branch, torch
2.14.0, mlx 0.32.2 / mlx-lm 0.31.3; the CPU row is the job's 5→15-step
slope with no 300-step run.

| row | tokens/s | step (ms) | validation 0 → 300 |
|---|---:|---:|---:|
| SeeAI `metal` (int8 base) | 1,782 | 287.3 | 4.5436 → 4.1534, accepted |
| SeeAI `cpu` (int8 base) | 338 | 1,514.0 | — |
| torch.compile MPS, bf16 base, f32 adapters | 4,033 | 126.9 | 4.5439 → 4.1446 |
| torch.compile CPU f32 | 852 | 601.1 | 4.5437 → 4.1430 |
| MLX-LM all-bf16 | 4,590 | 111.5 | 4.5442 → 4.1588 |

Every stack starts from the same validation loss and ends within 0.016
of the others; the ratios are the 2026-09-22 row's (SeeAI Metal 0.39× of
MLX-LM bf16 and 0.44× of torch.compile's bf16 base; SeeAI CPU 0.40× of
torch.compile CPU). Two things the run found in the job itself, both
fixed in the workflow: its corpus was `docs/usage.md` alone — 58
records, so 300 steps of 4 are 23 epochs, every stack's validation loss
*rose* (SeeAI 4.19 → 5.72, torch 4.19 → 5.30, MLX 4.19 → 5.23) and the
gate declined — and its CPU row ran no 300-step pass, so the parity check
refused it as "not a measurement" and the matrix exited 3 on every run.
Raw JSON in `out/frontier-2026-10-15b/` (not committed). The job itself
still waits on a runner labelled `self-hosted, macOS, ARM64` with the
checkpoint path passed as `frontier_model_dir` or set in its environment.

### The loss on the GPU, and the dispatch model measured (F8 #139, 2026-10-15)

Softmax cross-entropy (opcodes 11 / 12) runs on Metal: one simdgroup per
row over the 49,152 logits with the fixed shuffle trees, the row losses
folded in order by a compensated single-thread kernel into the loss slot
(the CPU folds in f64; two Metal runs are byte-identical and the loss
agrees with the CPU inside 1e-5, the probabilities inside 1e-4 — the
CPU's serial f32 row sum over 49,152 terms is the less accurate of the
pair). The train-only profile of the SmolLM-135M package now shows **no
`cpu opcode 11` or `12` row**. Two CPU-resident rows remain, and one of
them still drains: the embedding gather (41) is instruction 0 of the
step, with nothing pending; the RoPE angle table (45) is instruction 14,
written into a transient slot the thirteen GPU instructions before it
just used, so the backend flushes and waits there every step — the span
table below shows it as a 16-dispatch command buffer per step, 0.8 ms of
GPU inside 4.8 ms of wall, about 4 ms (1.4 %) per step. The acceptance
line "no CPU-resident opcode in the train program" is therefore not met
yet; porting the angle table (a kernel over B·S·H·d/2 angles) is the
follow-up, and the gather can go with it. The loss kernels cost 4.0 +
1.5 ms where the CPU took 7.4 + 1.8 — and the measured step did not move
(exact 287.2 → 288.4 ms, relaxed 203.3 → 201.2), so the mid-step drain the
review priced at "≈5 % plus the drain itself" was worth about 1 %. The
gradient now descends from GPU-computed probabilities: the 300-step
Metal endpoint moved from 4.013721 to 4.014309 against the CPU's
4.013742, so the run-level CPU–Metal agreement is 1.4e-4 relative where
it had been 5e-6 with the loss on the CPU (the twin test's 1e-5 is on
identical inputs; this is the number the two runs reach by themselves).

The dispatch model, measured without serializing (`SEEML_METAL_SPANS=1`:
per command buffer, the GPU's busy span against the CPU wall from first
encode to completion). One command buffer per training step, 3,862
dispatches (exact) / 3,355 (relaxed):

| package | dispatches / step | GPU busy (ms) | CPU wall (ms) | outside the GPU | Σ per-kernel spans (serialized profile) | gaps inside the busy span |
|---|---:|---:|---:|---:|---:|---:|
| exact | 3,862 | 276.2 | 277.9 | 1.7 ms (0.6 %) | 221.5 | 54.7 ms (19.8 %), 14 µs per dispatch |
| relaxed | 3,355 | 192.4 | 194.2 | 1.7 ms (0.9 %) | 157.6 | 34.8 ms (18.1 %), 10 µs per dispatch |

(The 2,462 + 1,400 pair is the first step, split where the loss forward
grew the row scratch and reopened the encoder.) One finding is firm:
**encoding is not the cost**. The CPU hands the GPU a step's worth of
dispatches 1.7 ms before the GPU finishes them, so the "≈27 % outside any
kernel" the review inferred from the step-timing split is not CPU time,
and a scheduling phase that reorders instructions would recover none of
it. The other needs a denominator the serialized profile does not give.
The GPU is busy 276 ms per step; the kernels' own spans, summed from the
profile taken after this port, are 220.8–221.5 ms over four runs (relaxed
156.6–157.9), which would make the serial encoder's gap between kernels
~55 ms, 14 µs per dispatch, 20 % of the step. But the profile taken
before the port — the same GEMM, LoRA and norm kernels, the loss on the
CPU — summed to 260.2 ms (relaxed 172.4), a 15 % swing with no change to
those kernels: one command buffer per instruction lets the GPU clock
down in the gaps, and how far depends on what sat in them (here a 9 ms
CPU loss per step). Against that profile the gap is ~20 ms, 7 %. So the
honest statement is a range, **7–20 % of the step between kernels**,
straddling the issue's 10 % line, and the serialized profile cannot
resolve it. The direct experiment can: F7's step 1 removes the 7,174
`k_scale` dispatches a profile counts (one per adapter per pass) with no
other change to the step, so the real step's movement per dispatch
removed is the per-dispatch cost; F7 opens with that measurement. What
is already decided: a concurrent dispatch type is not the answer (the
serial encoder is what keeps two runs byte-identical without barriers),
and for SeeAI S6 (#152) the time outside kernels, whatever its size, is
per-dispatch latency rather than ordering, so scheduling stays a
function inside selection and fusion owns the lever.

### Measurement rules, and the rows they were missing (F9 #140, 2026-10-15)

Three rules, each learned from a decision taken on the wrong object
(the 2026-09-24 systems review):

1. **Kernel decisions are taken on the train-only Metal profile**
   (`SEEML_METAL_PROFILE=1 --val-frac 0 --eval-every 0`): with validation
   on, the evaluations' forward passes outnumber the backward ones many to
   one (30 : 4 in the profile F7 and F8 were first priced from), and every
   per-family share reads too forward-heavy. The profile's header now counts
   the passes it saw, so a reader can tell which kind of profile it is.
2. **CPU decisions are gated on the real-package row** (the SmolLM-135M
   package from the HF import, `model_update --backend cpu`, the frontier
   table above); the `tok_smollm135m_q8` bench fixture is a regression
   canary — it moves when the kernels move — not a baseline. Its under-read
   is explained below.
3. **The gate's share of the update is a reported number**: every package's
   `--report` carries a `lifecycle` object (plan load and validation with
   the source-model bind, the dataset, training, the seconds inside every
   evaluation, merge, commit, the whole wall), and `frontier_run.py` copies
   the 300-step run's split into its SeeAI rows.

**The per-column int8 row, rerun.** Per-tensor int8 cost +0.10–0.14 nats
of validation loss at step 0 on every model (the 2026-09-22 row); E12
(#128) landed per-column scales and the row was never rerun. The Real
Analysis Delta study of 2026-09-28 (`out/realanalysis-2026-09-25/`, not
committed) ran it on SmolLM-135M, the runtime gate's own validation loss,
300 steps, the docs corpus:

| base storage | step 0 | step 300 | vs f32 at 300 |
|---|---:|---:|---:|
| f32 | 1.7156 | 1.0800 | — |
| int8, per-column scales (E12) | 1.7156 | 1.0846 | +0.0046 |
| int8, per-tensor scales (pre-E12) | 1.8335 | 1.0961 | +0.0161 |

Per-column scales close the step-0 gap entirely (the gate scores the
shipped f32 model plus the update, so the base rounding no longer enters
the "before") and leave 0.005 nats at step 300, a quarter of the
per-tensor residual; no follow-up issue is needed.

**Tier D, the gate's share of the update wall.** The real SmolLM-135M
package, 300 steps, the 10 % held-out split evaluated before and after
(`--eval-every 0`), from the runner's `lifecycle` object:

| backend | wall (s) | plan load + validate + source bind | dataset | training (incl. evaluations) | the two evaluations | merge | commit | gate's share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `metal`, exact | 91.0 | 0.29 | 0.00 | 88.9 | 3.51 (1.75 each) | 0.13 | 1.65 | 3.9 % |
| `metal`, relaxed | 66.2 | 0.45 | 0.01 | 63.9 | 3.56 | 0.13 | 1.69 | 5.4 % |
| `cpu`, exact | 514.5 | 0.44 | 0.00 | 512.4 | 18.3 (9.15 each) | 0.14 | 1.55 | 3.6 % |

Two evaluations are 3.6–5.4 % of a 300-step update; the plan loads in
under half a second (the 39.5 s "lifecycle intercept" of the F2 CPU run
was its two evaluations and the 5-step regression's own fixed costs, not
the plan). The default `--eval-every auto` evaluates every `steps / 10`
steps, 30 periodic evaluations plus the two: on this package that is 56 s
of a 147 s Metal run (38 %) and 293 s of an 807 s CPU run (36 %) — the
best-state tracking E9 bought is paid for in evaluations, and a product
run should size `--eval-every` against the 1.75 s (Metal) / 9.2 s (CPU)
an evaluation of a 53-record split costs. `torch.compile`'s 40–54 s
Inductor warm-up (6–9 s when its cache is warm) is the counterweight on
the other side of the table.

**The fixture's under-read, explained.** `tok_smollm135m_q8` compiles
to a plan of exactly the real package's size (298,663,936 bytes, 3,309
training instructions, the same arena), so its per-step cost is the
model's. Run on 2026-10-16 on the same host, the same code and the same
`--threads 10` as the real-package row above, with the row's own `5→15
--repeats 1` geometry, the fixture reads 1,603 ms (319 tok/s; forward
647 ms, backward 934, optimizer 12) against the package's 1,514 ms (338):
a 6 % gap, inside what a single 5→15-step slope resolves (the fixture's
own lifecycle intercept came out negative on that run). The 233 and 269
against 332 of the earlier table were readings of different code states
— fixture rows from 2026-09-15 and the days after E1 against a package
row taken on 2026-09-22 — not a property of the fixture. The rule stands
regardless: the fixture is the canary that moves when the kernels move;
the real package sets the baseline.

### The LoRA lowering, steps 1 and 3, and the per-dispatch cost (F7 #138, 2026-10-19)

The rank-r scale is the GEMM's own alpha (plan v19): the grafter sets
`alpha` on the `t@B` matmul, autodiff carries it onto `dt = dC@Bᵀ` and
`dB = tᵀ@dC`, and the lowering writes it into the instruction's `imm`
word — no `kScale` instruction remains in the train program (3,309 →
2,887 instructions on SmolLM-135M, one per adapter per pass). The CPU
applies it where the scale instruction it replaces sat (folded into A on
the NN and TN cores, on the finished product on the NT core), so the CPU
plan is byte-identical for every alpha; Metal applies it in the write-back,
identical for the default power-of-two 16/8. Step 3 then folds the GEMM
addend into the exact Metal kernels' write-back (flag 2048; `d + s` over
the finished dot product, the bits of the GEMM + `k_add_ew` pair it
replaces), so a LoRA site's forward and the dX backward are one dispatch
each instead of two. Every stage below commits the same bytes as main's
package after 20 steps on both backends (CPU
`beb98417…`, Metal `0fef91ef…`; `out/f7-2026-10-19/*/id_sha.txt`), and
the "every alpha" claim was demonstrated, not argued: the demo MLP
compiled with `--lora-alpha 3 --lora-rank 4` (0.75, no power of two)
under main's compiler and this one commits the same bytes after 20 CPU
steps, while the two Metal runs differ at rounding level with the same
loss to six decimals (`out/f7-2026-10-19/alpha3/sha.txt`).

Train-only (`--val-frac 0 --eval-every 0`), the SmolLM-135M int8-base exact
package, Apple M5, every number from this session; GPU busy per step is
the training step's command buffer under `SEEML_METAL_SPANS=1` over three
runs, the step time the 40→160 slope with two repeats (`frontier_run.py`):

| state | train instr. | dispatches / step | GPU busy / step (ms) | outside the GPU (ms) | step (ms) | tok/s | LoRA family, share of kernel time | head LoRA / head GEMM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main (`ee979e7`), the 2026-10-15 package | 3,309 | 3,862 | 275.4–277.7 | 1.72 | 288.4 (10-15) / 285.5 (today) | 1,775 / 1,793 | 37.3 % | 61 % |
| step 1: alpha in the GEMM | 2,887 | 3,443 | 273.6–275.5 | 1.50 | 285.8 | 1,791 | 37.0 % | 61 % |
| step 3: the addend in the write-back | 2,887 | 3,027 | 271.0–271.6 | 1.38 | 280.4 | 1,826 | 33.6 % | 34 % |

**The per-dispatch cost, measured** (the experiment F8's dispatch model
waited on). Step 1 removes 422 dispatches per step and nothing else: the
`k_scale` rows of the profile (8,440 over the 40-pass profile, 1.7 µs
each) were 0.72 ms of kernel span per step, and the GPU's busy span fell
2.0 ms (276.3 → 274.3 ms, means of three runs), so the serial encoder's
gap costs **about 3 µs per dispatch at the means**; the CPU's side of it
is 0.5 µs (outside-the-GPU time 1.72 → 1.50 ms per buffer). The
run-to-run spread is ~2 ms, so this experiment alone allows 0–8 µs; step
3 below bounds it independently — 416 dispatches *and* their traffic
bought 2.9 ms, so a dispatch costs at most 7 µs even if the traffic were
free — and the two together put it at **0–7 µs**. Over the 3,443
dispatches of the step-1 step that is ≈10 ms at the means, **≈4 % of
the step**, and ≤8 % at the ceiling — the "7–20 %" the serialized
profile could not resolve lands at its low end, below the 10 % line
across the whole range. The
train-only profile prices the family at 37 % of kernel time with every
rank-8 shape counted (the A-projections' `512x8x576` and dB's `576x8x512`
rows included) — the ≈36 % the forward-weighted profile read (F9) holds.

**Step 3's rows** (`SEEML_METAL_PROFILE=1`, µs per dispatch, before →
after the fold): `512x1536x8` nn 155 → 92, `512x576x8` nt 51 → 42, nn
46 → 37, `512x1536x8` nt 154 → 107, the head's `512x49152x8` 5,105 →
2,847; 416 dispatches and one activation-sized pass per site gone, GPU
busy −2.9 ms, the measured step −5.4 ms. The rows are still
occupancy-bound — `512x1536x8` moves its 6 MB of C and D in 92 µs, 65
GB/s against the issue's ≥100 GB/s target — so the small-N kernel
rewrite proper (step 3's second half), step 2 (batched A-projections,
which needs strided operands in the ISA) and step 4 (the K-extension,
certified) remain. Against F1's rows the exact Metal package now stands
at 1,826 tok/s: 0.82× MLX-LM's true-f32 2,240 and 0.40× its all-bf16
4,590; the relaxed package was not re-measured this round.

### No CPU-resident opcode in the train step, and the fusion verdict (F8 #139, 2026-10-19)

The two CPU-resident instructions the 2026-10-15 port left — the RoPE
angle table (45), instruction 14 of the step, whose write into a
transient the GPU had just used flushed and drained the queue every
step, and the embedding gather (41), instruction 0 — run on the GPU:
`k_rope_table` is one thread per position running the rotation kernels'
own recurrence through the same `cos` / `sin`, so a rotation through the
table computes the bits of one that recomputes (the twin test asserts
it bitwise), the rotation kernels read it, and `k_embed_fwd` is the
gather. The train-only profile of SmolLM-135M shows no `cpu opcode` row,
and the step is one command buffer:

| state (train-only, M5, this session) | command buffers / step | dispatches / step | GPU busy (ms) | wall per step's buffers (ms) | step (ms) | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| F7 steps 1 + 3 | 2 (the table's drain: 13 dispatches, 0.8 ms GPU inside 4.9–5.0 ms wall) | 3,027 + 13 | 271.0–271.6 + 0.8 | 277.2–278.0 | 280.4 | 1,826 |
| F8: table and gather on the GPU | 1 | 3,039 | 272.6–274.1 | 274.0–275.5 | 278.8 | 1,837 |

The drain was worth the 4 ms its wall said (the 13-dispatch buffer's
82 % outside the GPU) and the step moved 1.6 ms; the table kernel is 9
µs, the gather 13 µs — the busy column's rise against step 3 is the ~2
ms run-to-run spread, not those 22 µs. Committed models are byte-identical to main's
package on both backends. What stays on the CPU: the MSE and KL losses
(not on the SmolLM step) and the tiled attention family (F3).

**The fusion verdict for the GPU, from F7's experiment.** Step 2's
"7–20 % of the step between kernels" could not be resolved by the
serialized profile; F7 step 1 resolved it directly — 422 dispatches
removed per step with no other change, 0.72 ms of kernel span, 2.0 ms
of GPU busy, so **about 3 µs per dispatch at the means**, and 0–7 µs
once the run-to-run spread and step 3's independent ceiling (416
dispatches plus their traffic bought 2.9 ms) are taken together — ≈10 ms
across the 3,443 that remained, **≈4 % of the step**, ≤8 % at the
ceiling, below the issue's 10 % line across the range. `kFusedMap` is
therefore not extended to
Metal for the dispatch count's sake. The lever on the GPU is bytes per
kernel: the addend fold (416 dispatches and one activation-sized pass
per LoRA site gone) moved the step 5.4 ms where the scale removal moved
it 0.3–2.6 ms, because it removed traffic — and that is the E4 case
(2–6 % on the CPU) on both backends, keyed on memory, not dispatches.
For SeeAI S6 (#152): scheduling stays a function inside selection,
decided. Not measured this round: the per-program (train / eval / merge)
table and the `dec_wide` row.

## Tier C — Memory (the gate that refuses compiles)

| metric | definition | what it gates |
|---|---|---|
| **arena bytes / model bytes** | header `arena_size` vs source `.smf` size across model scales | the honest cost of training; grad-accumulation (2a) and remat (2b) success = this ratio falling |
| **arena breakdown** | persistent / IO / transient split (compile report already knows it) | which memory project fires first: transients ⇒ remat; persistent ⇒ 8-bit optimizer states |
| **probs-cache share** | attention P cache bytes / arena bytes as S grows | same flash-attention decision from the memory side |
| **estimator honesty ratio** | step-0 estimate / exact final gate value | must stay ≥ 1 and near 1 — the review found it 16× off for teachers once; this metric keeps the lower-bound claim measured, not asserted |
| **peak RSS vs arena** | OS-observed peak / (arena + plan) | the "one allocation" doctrine, verified: a drifting ratio means hidden allocation crept in |

## Tier D — Lifecycle latencies (the product feel)

| metric | definition | what it gates |
|---|---|---|
| **compile wall time** | `seeml-update-compile` end to end, per model scale | developer loop; autodiff/fusion passes are O(ops²) risks as models deepen |
| **load+validate ms** | `LoadFromMemory` (hash + contracts + validator over every instruction) | on-device startup; the validator is O(instructions) and must stay trivial vs training |
| **plan size / delta size** | `.seeu` bytes; later: adapter-only package bytes | the delta-only update-package project (pillar D2) — the headline there is this number collapsing from model-scale to adapter-scale |
| **gate (eval) time share** | 2×eval-pass wall / total update wall | if the gate costs >20% of the update, eval batching or fused-eval work is justified |
| **merge+commit ms/MB** | `RunMerge`+`CommitToModel` per model MB | the streamed COW commit's O(chunk) claim under load; the new delta-finiteness scan rides this path and must stay invisible |
| **checkpoint save/load ms** | per persistent-segment MB | resume UX; 8-bit optimizer states will change this |

## Tier E — Development-velocity metrics (speeding up the project itself)

| metric | definition | what it gates |
|---|---|---|
| **CI wall time per job** | each ci.yml job's duration trend | the per-diff feedback loop; when build-and-test crosses ~10 min, precompiled-header or unity-build work pays |
| **full local build time** | `build/build.sh` clean | same loop locally; the single biggest dev-speed lever in a -O2 -Werror tree |
| **suite runtime top-10** | slowest tests trend | keeps the 367-test suite honest — one 60 s test taxes every diff forever |
| **fuzz corpus coverage** | edges covered nightly (libFuzzer `-print_final_stats`) | whether the fourth arm (compile-of-SMF) is still finding new ground or needs structure-aware mutators |

## Reference points from this branch (not benchmarks — sanity anchors)

Measured once on the dev host (Apple Silicon, default threads) during
validation, to seed expectations until the harness lands: the demo MLP
e2e package compiles + builds in seconds; 300 training steps of the
16-wide MLP run in ~1 s; the token tiny-GPT (V=16, D=8, S=4) trains 300
steps to 0.7× loss in ~2 s; serial vs 8-thread committed models are
bitwise identical (the determinism overhead is therefore *measurable as
pure speedup*, no correctness tax).

## Pricing the frontier on a plan (`tool/frontier_exec.py`)

The frontier row above was measured with a matched PyTorch program and
`mlx_lm`'s own trainer on four real models. `frontier_exec.py price` regenerates that kind of
comparison for **any compiled plan**, by interpreting the plan's own
instruction stream through the frontier frameworks:

```sh
python3 tool/frontier_exec.py price out/pkg/update_plan.seeu \
    --backends torch,torch-mps --probe build/seeml-plan-probe \
    --cpp-backends cpu,metal --peak-gflops 1490 --report frontier.json
```

Each backend's cell carries `seeml-bench`'s fields under the same
definitions (`step_ms` by steps-regression, `rows_per_s`, `tokens_per_s` /
`samples_per_s`, `it_per_s`, `gemm_gflops` over the plan's own GEMM
instructions, `mfu` against `--peak-gflops`, `step_ms_min/max`), plus
`peak_rss_bytes` and the device string; the C++ numbers come from
`seeml-plan-probe --time` (the instruction stream alone — no feeder, no
evaluation — which is also all the Python side times). Read it with two
rules. **It is a floor on the frontier, not the frontier:** the interpreter
dispatches one opcode at a time, with no `torch.compile`, no `mx.compile`
and no fusion, so a hand-written program on the same framework is faster.
**MLX's default is not f32:** on recent Apple GPUs MLX's matmul runs a
reduced-precision mode (~1e-3 relative error against f64 — the tool
measures and prints it); `mlx` prices true f32 (`MLX_ENABLE_TF32=0`) and
`mlx-tf32` the shipped default, one per process because MLX latches the
switch. Any MLX throughput quoted without naming the mode is quoting the
reduced-precision one.

First row (Apple M5, the Tier A `dec_wide` shape V4096 D512 H8 S256 L4,
LoRA r16, 512 tokens/step, f32 base, 2026-09-17):

| executor | ms/step | tok/s | GEMM GFLOP/s | vs C++ CPU |
|---|---:|---:|---:|---:|
| SeeAI C++ `cpu` | 248.6 | 2,060 | 108 | 1.00× |
| PyTorch CPU (Accelerate/AMX), interpreted | 99.6 | 5,142 | 270 | 2.50× |
| PyTorch MPS, interpreted | 82.1 | 6,239 | 328 | 3.03× |
| MLX f32, interpreted | 58.6 | 8,734 | 459 | 4.24× |
| MLX TF32 default, interpreted | 52.2 | 9,802 | 515 | 4.76× |
| SeeAI C++ `metal` | 29.7 | 17,237 | 905 | 8.37× |

So on this plan the Accelerate exception is worth **at least 2.5×** on the
CPU (on the real models above `torch.compile` is 2.5–3.4× SeeAI's CPU
package and PyTorch eager 2.2–3.2×), and the C++ Metal backend runs
above what op-by-op MLX reaches — the real `mlx_lm` trainer, which
compiles its step, is the 0.80× (f32) / 0.39× (bf16) of the frontier row.

The same tool is the runtime's differential oracle — `frontier_exec.py
diff` — described in [tool/README.md](../tool/README.md).

## Tuning the kernel policy (`tool/autotune.py`)

The one knob the CPU kernels expose — the blocked GEMM's K and N tiles —
is measured offline, per host, by the Python plane, never inside the
compiler (Two-Plane Overhaul P2, #76; the design is in
[compiler.md](compiler.md)). One command tunes a host:

```bash
python3 tool/autotune.py tune --bench build/seeml-bench --out kernel_policy.json \
    --fixtures mlp_128x1024x2,dec_v256_d128_s128,dec_v512_d192_s32 --threads 10
python3 tool/autotune.py show kernel_policy.json
seeml-update-compile ... --kernel-policy kernel_policy.json   # consumes it
```

Every arm is one `seeml-bench --gemm-tiles K,N` run under this document's
discipline; arms are visited round-robin for `--rounds` (3) rounds and
each arm's per-key result is the median over rounds — medians of medians.
The score is the geometric mean over (fixture, threads) keys of rows/s
relative to the kernel-default arm, which is always swept, as is the
compiler's analytic tiling (`analytic_gemm_tiles` in the report — the
hypothesis #90 found 1.3–3.3× slow, now measured instead of emitted). The
winner is recorded only if it beats the defaults by `--min-gain` (3 %);
otherwise the table records that the defaults are best. Either way every
arm's numbers are kept beside the decision (`tuned.arms`), so the table is
an audit trail, not just an answer. The table is keyed on the host key the
bench printed; the compiler recomputes the same key and writes the tiles
into the plan header.

**Measured on the development host (2026-09-15, Apple M5, 10 threads,
fixtures `mlp_128x1024x2`, `dec_v256_d128_s128`, `dec_v512_d192_s32`, 3
rounds x 3 repeats, 18 arms, 10 minutes):** the kernel defaults are the
policy. Scores are geometric-mean rows/s relative to the default arm.

| arm | label | score |
|---|---|---|
| 256x512 | grid (best) | 1.005 |
| 32x256 | grid | 1.004 |
| 64x512 | grid | 1.004 |
| **64x256** | **default (recorded)** | **1.000** |
| 128x128 | grid | 0.989 |
| 64x64 | grid | 0.976 |
| 256x64 | grid | 0.960 |
| 512x16 | analytic (`SuggestGemmTiling`) | 0.903 |

Decision: `default-within-margin` — the best arm's 0.5 % is inside the
3 % noise line, so the table records 64x256 for this host key
(`arm64;Apple M5;cores=10;l1d=65536;l2=6291456;simd=4`). The analytic
tiling that packages used to ship with is 9.7 % slower here, in line with
#90's finding at larger shapes. A tuned entry is therefore not what this
host gets from P2; what it gets is the measurement that says so, and a
gate that cannot be fooled by the tiling again.

## The harness

All three pieces of this program exist:

1. **`tool/seeml_bench.cc`** compiles the standard fixture set in-process
   (the seeded builders the test suites share — two MLP stacks, three
   4-block decoders, one token-native decoder, plus an opt-in
   SmolLM-135M-shaped token decoder, `tok_smollm135m_q8`, that runs only
   when `--fixtures` names it: 30 blocks of D=576 / 9 heads / ffn 1536
   over a 49,152 vocabulary, q8 base, 512 tokens per step — the frontier
   row below, kept out of the default set and the nightly keys because a
   CPU sweep of it takes the better part of an hour) and emits one JSON
   per run:

   ```bash
   cmake -B build -DSEEML_BENCH=ON && cmake --build build --target seeml-bench
   # or: SEEML_BENCH=1 sh build/build.sh
   ./build/seeml-bench --out bench.json --threads 1,2,4,8
   # the geometry a package ships with: --gemm-tiles K,N or
   # --kernel-policy table.json [--target-host KEY], as the compiler takes them
   ```

   Per fixture, per thread width: per-step latency and fixed lifecycle
   cost by **steps-regression** (train `--steps-lo` then `--steps-hi`;
   slope = per-step, intercept = lifecycle — startup never pollutes the
   step number), Tier A `rows_per_s`, effective GEMM GFLOP/s (the FLOPs
   are summed from the plan's own instruction stream, 2·M·N·K over every
   GEMM), the fwd/bwd/optimizer split, and the Tier D lifecycle numbers
   (compile, load+validate, merge, checkpoint save), plus the
   external-standard fields of the table above (`tokens_per_s`,
   `it_per_s`, spread, train loss, peak RSS, and `mfu` when
   `--peak-gflops` names the host peak). Medians of `--repeats`
   (default 3); seeds and thread widths pinned. The CLI is strict, like
   the other tools: an unknown flag or a flag that cannot parse is exit 2,
   never a default. Report schema is 4 (2 added the external-standard
   fields, 3 the kernel policy, 4 the calibration block); every earlier
   key is unchanged, so stored baselines stay comparable.

2. **The nightly `bench` CI job** restores the previous night's numbers
   from the actions cache (the fuzz-corpus trust model), runs the harness,
   and **fails on a >10% regression in any Tier A `rows_per_s`** via
   `tool/bench_compare.py` — which also seeds the baseline on first run
   and skips (never fails) fixtures that exist on only one side, so adding
   a fixture can't fail the night it lands. Two fail-closed rules keep the
   gate honest: a comparison with **zero overlapping keys is an error**
   (a renamed metric must re-seed deliberately, not pass silently), and a
   **pinned epoch baseline** (`--epoch-baseline`, cache key `bench-epoch-v2`,
   saved once and never rolled forward) is diffed at a 15% threshold so a
   sub-10%/night drift cannot compound unnoticed.

   Cloud runners of one class differ by 15–35% on identical work, and a
   rolling baseline that only a green night replaces turns that into a
   high-water trap: one fast runner sets it and every ordinary runner
   fails until an equally fast one lands (Nightly #20–#23; 09-13/09-14 on
   the commit that was green on 09-12). Two mechanisms (P5, #79) keep the
   10% threshold meaningful across machines:

   - **Calibration.** `seeml-bench` (report schema 4) times a frozen,
     single-threaded reference kernel — a naive f32 i-k-j product over
     64 KiB matrices that shares no code with `runtime/executor`, so a
     SeeAI kernel regression cannot slow its own yardstick — before and
     after the fixtures, and records `calibration.calib_rows_per_s` (the
     median of both brackets), the two sides, and their `spread`.
     `bench_compare.py` divides the current run's rows/s by the ratio of
     the two calibration rates before applying the thresholds. A reference
     that predates schema 4 compares un-normalized with a note; a schema-4
     report without the block, two different calibration kernels, or a
     ratio outside [⅓, 3] are errors. The kernel's name
     (`sgemm_ikj_128_f32_v1`) is its identity: changing the loop bumps it.
   - **Two consecutive reds.** With `--state` a key's first regression is
     a warning (a `::warning::` annotation and the uploaded
     `bench-gate.log`); the job fails when the same key regresses against
     the same reference on the next run too. `--promote` makes the comparer
     the one that rolls the baseline (and seeds a missing epoch): only a
     fully green gate does, so a warned run is never the next reference.

   The epoch key is `bench-epoch-v2`, the first epoch that carries a
   calibration block. `test/tool/bench_compare_test.py` replays both
   incidents (all green) and an injected regression (warned, then failed),
   and shows the un-calibrated replay reproducing the four red nights.

3. **The step-latency instrumentation** lives in the engine behind
   `-DSEEML_STEP_TIMING` (set by `-DSEEML_BENCH=ON` / `SEEML_BENCH=1`):
   the training stream's phase boundaries — after the last loss-forward
   instruction, and at the first clip/optimizer instruction — are scanned
   once at load, and each step's wall clock is split across them into
   `UpdateEngine::step_timings()`. Default builds, and every emitted
   package, compile the exact untouched dispatch loop; the split answers
   Tier A's "where does the next kernel dollar go" (if bwd ≫ 2×fwd,
   attention backward or GEMM-TN is the target).
