---
title: "SeeAI F7: LoRA lowering — fold the rank-r scale into the GEMM alpha, batch the A-projections, lift the small-N Metal kernels off the occupancy floor, then fold B(Ax) into the base GEMM"
number: 138
labels: enhancement,efficiency,core-plane,frontier-parity
plane: Core plane
milestone: SeeAI v1.0.0.A
priority: P0
---
## Root cause: the LoRA lowering's structure (2026-09-24 systems review)

`compiler/analysis/algebra/lora_grafter.cc` lowers every adapted
projection `C = X @ W` into four separate ops: `t = X @ A` ([N, r]), `t' =
(α/r) · t` (`sc_high.scale`), `u = t' @ B` ([N, M]), `C += u`. On the
SmolLM-135M package that is 211 adapters, and on Metal the rank-8 family is
a family of kernels at the occupancy floor, not a FLOP cost. From the
per-opcode profile (`out/frontier-2026-09-22/prof_q8.txt`, kernel spans
per training step, estimated):

| family | fwd ms | bwd ms | share of kernel time | share of FLOPs |
|---|---:|---:|---:|---:|
| frozen q8 GEMMs (block + head) | 51 | 53 | ≈50 % | ≈97 % |
| rank-8 LoRA GEMMs + scale + add | 32 | 43 | **≈36 %** | ≈3 % |

- `k_gemm_nn_small+k_add_ew 512x1536x8` runs 158 µs: 3 MB of f32 moved
  at ≈40 GB/s, a quarter of the M5's bandwidth — occupancy-bound, not
  bandwidth-bound. `k_gemm_nn_rows 512x8x576` (the A-projection) is 37 µs,
  5,430 dispatches over the profile.
- The head's LoRA (`512x49152x8` + add, 5.1 ms) costs 60 % of the head
  GEMM itself (8.5 ms).
- `k_scale` on 4,096 elements (512 × r) is dispatched 7,174 times over the
  profile at 2.7 µs each: a whole kernel for a scalar the GEMM's `alpha`
  argument already carries (`KArgs.f[0]`).
- The profile over-weights the forward (30 forward passes to 4 backward,
  validation on); the shares above are per-step estimates and the first
  step of the work is a train-only profile (F9). The finding survives
  either way.

After F2 halves the frozen-GEMM half of the step, this family is the
largest term left.

**What is already done, and what this issue integrates.** E10 (#93,
closed, landed in #123) moved `α/r` off the activation-sized `dC` copy
onto the `[N, r]` operand in the backward and merged `dX += dh·Aᵀ`
through `gemm_acc` (on Metal: `k_gemm_nt_small+k_add_ew 512x576x8`, 712
rows of the profile). The 7,174 `k_scale` dispatches on 4,096 = 512 × r
elements are exactly the scales E10 left as their own op — the forward
`t' = (α/r)·t` the grafter still emits (`lora_grafter.cc`, "s == 1 needs
no op") and the backward's `dh` scale — so step 1 below is E10's fold
carried the last inch, into the GEMM's own alpha. (The profile's
`k_gemm_acc_small 576x1536x8` / `576x576x8` rows are the merge program's
`W += α·A@B` at commit, not E10's dX path.) `docs/roadmap.md` Project 5
already lists "the LoRA residual chain (four dispatches per adapter)"
and "fusing x·A·B per adapter" as what is left on the GPU side; this
issue is that remainder, now priced.

## Design — four steps, each measured on the Metal profile

1. **Scale into alpha** (compiler, bit-neutral where `alpha == 1`,
   otherwise a rounding change the certificate prices): the grafter emits
   no `scale` op; the `t @ B` GEMM carries `α/r` as its alpha, the
   backward's `dC` scaling likewise. Removes every `k_scale` dispatch.
2. **Batch the A-projections** per block: `X @ [A_q | A_k | A_v]` is one
   512×24×576 GEMM instead of three 512×8×576; gate/up likewise. Same
   per-element reduction order (each column's K-sum is unchanged), so
   bit-identical.
3. **Small-N / small-K Metal kernels** rewritten for occupancy: the
   `nn_small`, `nn_rows`, `nt_small`, `tn_rows`, `tn_colsg` family at N or
   K ≤ 16 splits rows across simdgroups and vectorizes the add-epilogue;
   target ≥ 100 GB/s effective on `512x1536x8+add`. Bit-identical: the
   per-output K order does not change.
4. **Fold B(Ax) into the base GEMM** as a K-extension (`[X | t'] @ [W ;
   B]`, K = 576 + r) or as a second accumulate in the epilogue. This
   changes the reduction order of `C`, so it is a new opcode / flag the
   old path never emits, admitted under the same certificate as F2's
   relaxed family; the four-op lowering stays the exact reference.

## Status 2026-10-19

Steps 1 and 3 landed and measured (`docs/benchmarks.md`, "The LoRA
lowering, steps 1 and 3, and the per-dispatch cost"). Step 1: the scale is
the GEMM's own alpha (plan v19, the instruction's `imm` word on the three
f32 forms; the grafter sets it on `t@B`, autodiff carries it onto `dt` and
`dB`, the addend fuser keeps it), so no `kScale` instruction remains —
3,309 → 2,887 train instructions, 3,862 → 3,443 dispatches per step on
SmolLM-135M. The CPU applies alpha where the scale instruction it replaces
sat, so the CPU plan is byte-identical for every alpha (demonstrated on
the demo MLP at alpha/r = 0.75: main's compiler and this one commit the
same bytes after 20 CPU steps); Metal applies it in the write-back,
identical for the power-of-two default and at rounding level otherwise. Step 3, first
half: the GEMM addend is summed in the exact Metal kernels' write-back
(flag 2048) instead of a second `k_add_ew` dispatch — 3,027 dispatches per
step, the LoRA rows 155 → 92 µs (`512x1536x8`), 5,105 → 2,847 µs (the
head). Every stage commits the same bytes as main's package after 20
steps on both backends. The step: 285.5 (main's package, re-measured today) → 285.8 → 280.4 ms
(1,793 → 1,826 tok/s; 0.82× MLX-LM true-f32, 0.40× all-bf16); the LoRA family's
share of the train-only profile's kernel time 37.3 → 33.6 % with every
rank-8 shape counted (the ≈36 % above holds on the train-only profile
too), the head's LoRA 61 → 34 % of the head GEMM. **The per-dispatch cost, the experiment F8 waited on:
about 3 µs per dispatch at the means** (step 1 removed 422 dispatches
and 0.72 ms of kernel span, and GPU busy fell 2.0 ms; the run-to-run
spread allows 0–8 µs, and step 3's 416 dispatches plus their traffic
bought 2.9 ms, capping it at 7 µs, so 0–7 µs across the two
experiments), ≈4 % of the step across the 3,443 that remained and ≤8 %
at the ceiling — F8's verdict is recorded there. Remaining here: the small-N kernel rewrite itself (the
rows are still occupancy-bound, 65 GB/s on `512x1536x8` against the ≥100
target), step 2 (needs strided operands in the ISA), step 4 (certified),
and the relaxed package's row.

Step 3, second half, later the same day (`docs/benchmarks.md`, "The LoRA
family's Metal kernels, off the issue floor"): the K ≤ 16 form computes
four outputs per thread and moves B, D and C as 16-byte vectors; the
rows and columns forms keep 8 accumulators at a rank-8 site and read an
8-wide B row as two vectors; two staged forms (`_rows_tg`, `_colsg_tg`)
load the [32 k × 8] block of the strided operand through threadgroup
memory at the dB and dA sites. Per-output reduction orders unchanged, so
the committed model stays byte-identical on both backends. GPU busy per
step 272.6–274.1 → 238.1–242.8 ms, the step 278.8 → 245.5 ms, 1,837 →
**2,085 tok/s** (0.93× MLX-LM true-f32, 0.45× all-bf16); the family
33.6 → 24.1 % of kernel time; `512x1536x8` 92 → 44 µs, the head's 2,855
→ 1,882, the A-projection 36 → 16. Left: the `nt_rows` sites (dt, 42 →
38 µs: latency-bound at 512 simdgroups), step 2, step 4, the relaxed
row.

## Acceptance

- [x] Train-only Metal profile of SmolLM-135M with the LoRA family's share
  of kernel time before and after each step — 37.3 % (main) → 37.0 %
  (step 1) → 33.6 % (step 3, first half) → 24.1 % (step 3, second half):
  "The LoRA lowering, steps 1 and 3, and the per-dispatch cost" and "The
  LoRA family's Metal kernels, off the issue floor" (2026-10-19).
- [x] Steps 1 and 3 commit the same bytes as main — `beb98417…` (CPU) and
  `0fef91ef…` (Metal) at every stage.
- [x] Design item 3's target, ≥ 100 GB/s effective on `512x1536x8+add` —
  44 µs for the 6 MB of C and D the row moves (the doc's own figure: 92 µs
  was 65 GB/s), so about 137 GB/s.
- [ ] The `nt_rows` sites (`dt = dC@Bᵀ`, 42 → 38 µs, latency-bound at 512
  simdgroups) — the family's next floor.
- [ ] Step 2, the batched A-projections — not started; needs strided
  operands in the ISA.
- [ ] Step 4, B(Ax) folded into the base GEMM, certified — not started.
- [x] SmolLM-135M `metal` tok/s against F1's MLX-LM rows — 2,085 tok/s,
  0.93× true-f32 and 0.45× all-bf16.
- [x] The F2 relaxed package's tok/s on these kernels — 4,392–4,395 tok/s,
  0.74× of MLX-LM all-bf16 in the same session; the exact package 0.97×
  of true-f32 (2026-10-24, "The relaxed package on the F7-era kernels").
- [x] The head's LoRA cost as a fraction of the head GEMM — 61 % → 34 %
  after step 3.
