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

## Acceptance

- GPU and CPU tiled attention agree bit-for-bit (unfused port) or within
  the certified bound (fused), at model shapes including ragged S.
- S=2048 SmolLM-shaped step runs entirely on the GPU; peak memory and
  tok/s reported against the cached family and against F1's MLX row.
