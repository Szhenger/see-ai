---
title: "SeeAI F4: CPU matrix units — SME kernels (M4+) or a certified Accelerate exception for the frozen-weight GEMMs"
number: 132
labels: enhancement,efficiency,core-plane,doctrine,frontier-parity
plane: Core plane
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

SeeAI's own SIMD GEMMs peak at 332 GFLOP/s (8 threads, E1 #80) and the
SmolLM-135M CPU row is 332 tok/s on the real package (the 2026-09-22
frontier row in `docs/benchmarks.md`, 10 threads; the `tok_smollm135m_q8`
fixture under-reads it at 233–269, #105 / E1), 0.39× of `torch.compile`'s
847 on the same host. PyTorch reaches the AMX/SME
matrix units through Accelerate: even interpreted op by op it runs the
dec_wide plan 2.5× faster than SeeAI's CPU backend (5,142 vs 2,060 tok/s,
`frontier_exec.py price`, 2026-09-17). The speculated gap to
`torch.compile` on the CPU is 2–3×.

## Options (decide in this issue)

- **A. Native SME kernels** (Apple M4 and later, Armv9 SME/SME2): owned
  code, fixed accumulation order, so the result can stay bit-identical run
  to run; a new opcode family if the per-element order differs from the
  NEON kernels; runtime capability check with the NEON path as fallback.
- **B. Accelerate exception**: `cblas_sgemm` for frozen-weight GEMMs only,
  behind the P3 certificate (F2's relaxed family), because Accelerate's
  reduction order is not ours and may change across OS releases —
  determinism holds per OS build, not across them. Packages record the
  requirement.

Recommendation: A first — it keeps the determinism contract whole; B only
if A cannot reach ≥ 2× on M4/M5.

## Integrated from the 2026-09-24 systems review

Root cause 1, *arithmetic doctrine*, CPU side — measured on #136 (option
B, the Accelerate exception): the f32-base SmolLM-135M package on
Accelerate's SGEMM runs 681–708 ms/step (724–752 tok/s) against the exact
kernels' 1,830 ms (280 tok/s), 2.6×, i.e. **0.85–0.89× of
`torch.compile`'s 847**, validation equal to six decimals, two runs
byte-identical. But only an f32-weight relaxed GEMM takes the Accelerate
path; int8 and bf16 bases stay on the portable NEON kernels (#136), and
the int8 package is the one that ships (rodata 262 MB vs a 741 MB f32
plan). So the shipped package still trains at 332 tok/s, 0.39×.

Executable next step: give the int8 base the same exception — widen a
K×nc panel of the int8 weight into the arena-planned pack segment (E1's
buffer, already f32) and hand that panel to SGEMM, or SME kernels that
consume int8 directly (option A). Measure against the real-package row
(332 tok/s), never the `tok_smollm135m_q8` fixture, which under-reads it
(233–269; F9).

## Status 2026-10-19

The executable next step above landed: a relaxed int8 GEMM on the CPU
widens its weight panel by panel (≤ 4 MiB of f32, the per-column scale
applied as it widens) into a backend-owned buffer and runs each panel on
SGEMM — option B extended to the int8 base, behind the same certificate
(f32 products over the exact levels sit inside the bound priced for the
bf16-rounding kernel). Measured (`docs/benchmarks.md`, "The int8 base on
the CPU matrix units"): the shipped SmolLM-135M int8 package goes from
1,591.6 ms/step (322 tok/s) to 761–777 ms (659–673 tok/s), 2.05–2.09×,
two runs byte-identical, the 300-step validation 1.0e-5 relative from
the exact run — **0.77–0.79× of `torch.compile`'s CPU row**, just under
the ≥ 0.8× acceptance line. The gap to the f32-base package's 0.85–0.89×
is the widening of the 262 MB int8 base twice per step (the head's 113
MB for the forward and again for dX); widening it once, or SME kernels
over int8 (option A), is what remains. Not done: the GEMM GFLOP/s table
per shape and the serial-vs-8-thread identity on the new path (SGEMM's
threading is Accelerate's; the two 10-thread runs above are identical).

## Status 2026-10-23

The three measurable lines are met (`docs/benchmarks.md`, "The relaxed
int8 CPU step past 0.8× of `torch.compile`, with the same bits"; raw in
`docs/ledger/f4-2026-10-23/`). The 10-19 package commits the same model
(`025f3f3c…` after 20 steps) fully serial, serial with Accelerate's
default pool, and at 8 and 10 threads, and a system test pins serial
against 8-wide byte identity on the relaxed f32 and int8 bases. A
`sample` of the step refuted the widening hypothesis above (2.7 % of busy
samples): the cost was SGEMM on the calling thread, and an SGEMM
microbenchmark found the LM head's dX product running as 28 skinny calls
of 21 columns each (the 4 MiB panel spans the 49,152-long reduction). A
256-column panel floor and an eight-chain schedule for the cached
attention's double score dots — both bitwise-neutral, checked by the
10-19 plan committing `025f3f3c…` (20 steps) and `d459b541…` (300 steps)
on the new runtime — take the step from 761–777 ms to 663–678 ms: 755–773
tok/s, **0.89–0.91×** of F1's `torch.compile` CPU row (852 tok/s, the
2026-09-22 row; not re-run today). The GEMM table at SmolLM shapes is in
the same section: relaxed 623–972 GFLOP/s against the portable kernels'
74–315. Option A (native SME over int8) stays open for the determinism
contract across OS releases; the exact f32 head dX (74 GFLOP/s) is the
portable kernels' next lever.

## Acceptance

- [x] The option decided: B, the Accelerate exception, for the f32 base
  (#136, 2026-09-23) and the int8 base (2026-10-19), behind the P3
  certificate; A (native SME kernels) stays open as the route to the 0.8×
  line and to a determinism contract that holds across OS releases.
- [x] GEMM GFLOP/s at SmolLM shapes (q8 and f32, NN/NT/TN) against the
  committed kernels, timed against the committed object — `seeml-plan-probe
  --profile` on four plans, 2026-10-23: relaxed 623–972 GFLOP/s, portable
  74–315 ("The relaxed int8 CPU step past 0.8×").
- [x] SmolLM-135M CPU tok/s against F1's `torch.compile` CPU row, target
  ≥ 0.8× — **0.89–0.91×** (755–773 against 852 tok/s), "The relaxed int8
  CPU step past 0.8×" (2026-10-23); it was 0.77–0.79× on 2026-10-19.
- [x] Determinism: serial against 8-thread committed models byte-identical
  on the new path — `025f3f3c…` at 1/1, 1/default, 8 and 10 threads (the
  pool and Accelerate's), and `UpdateSystem.RelaxedCpuBitsAreThreadCountInvariant`
  (2026-10-23).
