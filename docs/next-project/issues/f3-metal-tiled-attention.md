---
title: "SeeAI F3: tiled attention on Metal — port opcodes 47-50 to the GPU, then fuse forward and backward flash-style"
number: 131
labels: enhancement,efficiency,core-plane,frontier-parity
plane: GPU backend
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

E11 (#94, plan v15) added the tiled attention family (`kAttnFwdTiled`,
`kAttnDQTiled`, `kAttnDKTiled`, `kAttnDVTiled`) so long sequences need no
O(S²) probability cache (S=2048 peak 1,039 → 324 MiB). On Metal those four
opcodes still return `false` from the backend's coverage switch
(`runtime/executor/metal_backend.mm`, "runs on the CPU until #63 ports it"),
so a long-context plan either keeps the cached family or leaves the GPU for
every attention op. Attention is ~24% of a decoder step; MLX and PyTorch
both ship fused SDPA on Apple GPUs.

## Design

1. Port: Metal kernels for the four tiled opcodes with the CPU form as the
   bit-for-bit reference in `seeml_metal_backend_test` (stats row of 4
   floats per query row, same tile order); `HazardWithPending` extents from
   `DescribeInstruction` as for every other kernel.
2. Fuse: one flash-style forward kernel (QKᵀ, online softmax, PV in
   threadgroup memory) and a fused backward (dQ, dK, dV from the saved row
   stats), keyed on head dim ≤ 128. New opcodes if the reduction order
   changes; the unfused port stays the reference.
3. `--attention auto` extends its decision to the Metal backend: tiled
   whenever the cached family's `probs_cache_bytes` exceeds the device
   budget.

## Integrated from the 2026-09-24 systems review

Root cause 3, *backend coverage and the dispatch model*: a CPU-resident
opcode is not merely slower — `MetalBackend::Execute` flushes the command
buffer and waits (`HazardWithPending` → `Flush` → `waitUntilCompleted`)
before every CPU-resident instruction whose operands the GPU still owns.
On a SmolLM-shaped long-sequence plan the four tiled opcodes drain the GPU
four times per block per step, ≈120 mid-step drains, the same mechanism
that costs the softmax cross-entropy its drain every pass in F8. The port
is therefore a pipeline fix before it is a kernel fix: the acceptance adds
that a long-S train program on Metal contains no CPU-resident opcode.

## Status 2026-10-20

Step 1 landed: the four tiled opcodes run on Metal (`k_attn_fwd_tiled`,
`k_attn_dq_tiled`, `k_attn_dk_tiled`, `k_attn_dv_tiled`: one simdgroup per
query or key row, every probability recomputed from Q, K and the stats
row with the CPU kernels' expressions, no S × S matrix and no threadgroup
memory), bitwise run-to-run and against the CPU family at tolerance (the
twin test, three shapes; a tiled decoder in the program-family test). A
SmolLM-135M step at S = 2048 (2,048 tokens per step) runs entirely on the
GPU with a 4.12 GB arena against the cached family's 9.03 GB (peak RSS
5.5 vs 10.1 GB; MLX-LM bf16 peaks at 6.2 GB of device memory, torch.compile
MPS at 10.3). **But it is slow**: 9.8 s per step against the cached
family's 1.7 s on the same 20-step run (MLX-LM bf16 789 ms, f32 1,823 ms;
torch.compile MPS bf16 998 ms). The serialized profile puts every tiled
kernel at 75–90 ms per layer against 1.4 ms for a frozen GEMM, and the
cause is traffic, not arithmetic: each lane streams whole K / V (or Q /
dO) rows from L2 — about 14 GB per kernel at this length — and
vectorizing those reads (float4, same order) changed nothing. Step 2 is
therefore the whole of the remaining work: a blocked kernel that stages
K / V blocks in threadgroup memory once per 128 query rows (lane = query
row, serial j — the CPU's own per-output order) divides the traffic by
the block size; the fused flash-style form follows from it. The
acceptance's "runs entirely on the GPU" line is met; its tok/s line is
reported, not met.

Step 2, first form, later the same day: the blocked kernels
(`k_attn_*_tiled_blk`, head width ≤ 64) — a threadgroup of 128 query or
key rows, the other operand's rows staged through threadgroup memory 32
at a time, one thread per row, serial over the other index (the CPU's own
order, so no cross-lane reduction anywhere) — bring the S = 2048 step
from 9,763 to **3,473 ms** (590 tok/s) against the cached family's 1,659
in the same session, with the same bytes as the per-row form (the twin
test drives both; wider heads keep the per-row kernels). Per layer the
forward, dQ, dK and dV cost 17.4 / 36.2 / 25.4 / 10.9 ms — 2.7 s of the
step — and the bound is now the scalar dot, one threadgroup-memory read
per multiply-add. The remaining step is the same blocking on
`simdgroup_matrix` tiles for QKᵀ and PV (the fused flash-style kernel),
which changes the reduction order and takes the certificate the issue
names for it. Memory line met; tok/s line reported, not met.

## Status 2026-10-30

Design item 2 landed, in the form the status above named: the blocked
kernels' passes with every block's products on `simdgroup_matrix` tiles
(`k_attn_{fwd,dq,dk,dv}_tiled_mma`, head widths up to 64), behind
`kFlagRelaxed` on the four tiled opcodes (plan v20), set under certified
precision, priced by the certificate's `attn.*.tiled` sites. A
threadgroup of four simdgroups owns 32 query or key rows as 8 × 8
fragments over the head columns; each block's dot products are one
8 × 32 tile of fragment products and each accumulation a fragment product
into 8 × 64 accumulators, the elementwise work between on the block staged
in threadgroup memory. Nothing is S × S, every kernel is bitwise run to
run, and the CPU and wider heads keep the exact family. On SmolLM-135M at
S = 2048 the attention kernels run 5.4× faster than the blocked ones
(2,541 → 468 ms per step, serialized) and the certified tiled plan's step
is 1,098 ms: 1.48× the cached family's 1,629 at less than half its arena,
0.75× MLX-LM all-bf16 and 1.58× MLX-LM true f32 measured the same day
(`docs/benchmarks.md`, "The relaxed tiled attention family: certified,
then measured at S = 2048"). Every acceptance line is met.

## Acceptance

- Superseded 2026-10-23, the line as first written: *GPU and CPU tiled
  attention agree bit-for-bit (the unfused port) at model shapes including
  ragged S.* The doctrine is per-backend determinism: the Metal backend is
  not bitwise against the CPU library (`runtime/executor/metal_kernels.h`),
  and the twin test compares the two at tolerance. The line it was
  checked against follows.
- [x] GPU tiled attention agrees with the CPU family at tolerance (1e-4
  relative, 1e-5 absolute on O, the stats row, dQ, dK and dV) at model
  shapes including ragged S, and two Metal runs are byte-identical — the
  twin test in `seeml_metal_backend_test` (three shapes; d = 16 and 48
  drive the blocked form, d = 96 the per-row one) and the tiled decoder
  in the program-family test.
- [x] The fused flash-style form (design item 2: QKᵀ and PV on
  `simdgroup_matrix` tiles) within the certified bound — 2026-10-30, plan
  v20. The twin test drives it against the CPU family at the exact
  kernels' tolerance (1e-4 / 1e-5, five shapes, ragged S, d 16 / 48 / 64)
  and run to run bitwise; `dec_wide` tiled under certified precision is
  granted at the certifier's defaults (the four attention sites observed
  2.5e-7 to 5.6e-7 against 1e-5), and a Metal diff of that plan that ran
  all 71 relaxed instructions on relaxed kernels passes `verify
  --observed` (attention sites 3.9e-7 to 9.0e-7).
- [x] S=2048 SmolLM-shaped step runs entirely on the GPU — no CPU-resident
  opcode in the step; arena 4.12 GB against the cached family's 9.03 GB.
- [x] Peak memory and tok/s reported against the cached family and F1's
  MLX row — "Tiled attention on Metal at S = 2048" (2026-10-20): 3,473 ms
  per step (590 tok/s) against the cached family's 1,659 ms and MLX-LM
  bf16's 789 ms. Reported, not at parity: the blocked kernels are still
  2.1× the cached family, and the simdgroup-matrix form above is the
  remaining step. Re-measured 2026-10-30 with that form: the certified
  tiled plan's step is 1,098 ms (1,865 tok/s) against the cached
  family's 1,629 and the blocked kernels' 3,417 on the same day, and 0.75×
  / 1.58× of MLX-LM all-bf16 / true f32 (823 / 1,732 ms) in the same
  session.
- [x] `--attention auto` chooses the tiled family for either backend from
  `--attention-cache-budget-mib` (design item 3) and, when
  `--memory-budget-mib` is given, from the step-0 footprint against that
  budget — never from the compiling device's memory, since the plan's
  bytes must not depend on the host (the #163 review's rule for `--remat
  auto`). Until 2026-10-29 the second leg read the host's memory when no
  budget was configured; `DecideAttention` now follows the remat rule, the
  memory gate's refusal says when `--attention tiled` would fit, and
  `UpdateCompiler.AttentionAutoDecidesAgainstAConfiguredBudgetOnly` pins it.
