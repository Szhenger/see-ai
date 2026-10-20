---
title: "SeeAI F5: scale ladder — SmolLM-360M, Qwen2.5-0.5B, then 1.5B on a 16 GB device with rematerialization and 8-bit optimizer states"
number: 133
labels: enhancement,core-plane,frontier-parity
plane: Core plane
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

Only SmolLM-135M has been validated end to end (import #69, Metal #103,
CPU #105). The arena doctrine — one allocation, refused compiles instead
of OOM — is the property that should beat frontier frameworks as models
grow on a fixed-memory device, and it has never been exercised past 135M.
At 135M on the M5 (2026-09-22) MLX-LM peaks at 1.34 GB in the Metal
allocator with 1.01 GB resident, while SeeAI's whole footprint is 1.89 GB
resident (arena, plan and Metal buffers are one shared allocation) — two
quantities, to be compared like against like in this ladder; nothing is
known about SeeAI's arena at 0.5B–1.5B.

## Design

1. Import and field-run SmolLM-360M and Qwen2.5-0.5B through `--hf`
   (Qwen2 is a supported walker family); record arena bytes / model bytes,
   step-0 estimate honesty, tok/s cpu/metal, val-loss parity vs torch.
2. **Rematerialization (roadmap 2b):** a compile-time policy that drops
   selected activations from the primal snapshot and recomputes them in
   the backward; chosen by the memory gate when the arena would exceed the
   budget. Bit-identical to the non-remat plan.
3. **8-bit optimizer states:** block-quantized AdamW moments (new opcode;
   SEKP change), opt-in, measured on loss curves.
4. Qwen2.5-1.5B on a 16 GB M-series machine with q8 base + remat + 8-bit
   states + grad accumulation (#70): the compile must either fit or refuse
   with the exact shortfall.

## Integrated from the 2026-09-24 systems review

Root cause 4, *memory planning* — the arena at each rung, measured
2026-09-22 (`out/frontier-2026-09-22/summary.json`, the q8 packages):

| model | arena | persistent | cached probs | rodata |
|---|---:|---:|---:|---:|
| SmolLM-135M | 1.56 GB | 36.3 MB | 70.8 MB | 262 MB |
| SmolLM2-360M | 3.17 GB | | | |
| Qwen2.5-0.5B | 4.20 GB | | | |

At 135M ≈1.2–1.5 GB of the arena is activations and adjoints for 512
tokens; a linear extrapolation puts 1.5B past 12 GB, a refused compile on
a 16 GB device. Like against like at 135M SeeAI is *below* the frameworks
in device memory (MLX-LM f32 allocator peak 2.11 GB, PyTorch MPS device
peak 2.4–2.6 GB), so this is a scaling problem, not a parity problem.
Executable order: first the compile report's persistent / IO / transient
split at each rung and adjoint liveness in the arena binder (bit-identical,
no new opcode), then rematerialization (2b) for what liveness cannot
recover, then the 8-bit states.

The review's N29 — the early memory gate's "lower bound" over-estimates and
rejects compiles that fit — is the accountant's to fix (S1); this ladder
reads the estimate-vs-exact ratio at each rung from that fix.

## Status 2026-10-20

Step 1 measured on both rungs, with torch parity (`docs/benchmarks.md`,
"The scale ladder's first two rungs"): SmolLM2-360M compiles in 1.45 s
to a 3.19 GB arena (225 adapters) and trains at 971 tok/s on Metal
(0.86× `torch.compile` MPS f32) and 129 on the CPU; Qwen2.5-0.5B in 2.59
s to 4.23 GB (169 adapters), 726 tok/s on Metal (0.92×), 95 on the CPU;
both start from the torch row's validation loss to five decimals and end
within 0.013 nats of it. Peak RSS is 1.8× the arena at both rungs — the
mapped f32 source file the eval program scores, the bytes the frameworks
hold as weights. The executable order's first item landed: the compile
report states the arena's persistent / io / transient split
(`UpdateCompiler` exposes `io_size`), and at these rungs the transient
segment is 98 % of the arena — the number step 2 works against. The
split also exposed a binder flaw: a pinned value (the loss slot) placed
first-fit at its birth could split the hole a dead 100 MB buffer left,
and did on the F7 step-1 layout (+100 MB at 135M); pinned values are
now bound first, below the scan — no bit changes, and the arenas above
are 80 MB (360M), 288 MB (0.5B) and 390 MB (135M at S=2048) smaller
than before it. Left: adjoint liveness beyond that, rematerialization
(step 2), 8-bit optimizer states (step 3), the 1.5B rung (step 4, which also
needs #158's plan loading above the 1.8 GB link limit), and the
estimate-vs-exact ratio once S1 fixes N29.
Step 2, rematerialization (design item 2, roadmap 2b), landed:
`--remat auto|off|on`. `Rematerializer` (`compiler/analysis/algebra/`)
runs after the fusers, the sweep and a first arena binding, walks the
primal snapshot in reverse and, for each cheap forward op whose results
the backward reads — the norms, the activations and elementwise
arithmetic, folded chains, the embedding gather; never a GEMM, attention
or a loss — re-emits the op as it stands (same mnemonic, attributes and
operands) right before its first backward reader and points every
backward reader at the copy, so the binder frees the original in the
forward; then the transient segment alone is rescanned
(`RebindTransients`) and the deltas re-pinned. Which ops qualify is a
liveness rule, not a list: a recompute happens only where the bytes it
frees exceed the operand bytes it would newly keep (an operand the
backward reads anyway, or one whose own producer is free to recompute,
costs nothing) — on the decoder that admits the RMSNorm outputs,
`silu(gate)`, the SwiGLU product and the first norm's embedding input,
and declines the RoPE outputs and the residual adds. `auto` fires when
the plain binding's arena plus the plan (bounded from above before
lowering) would fail the memory gate; `on` forces it. The compile report
states the three segments (`persistent_bytes` / `io_bytes` /
`transient_bytes`) and `remat: {policy, applied, recomputed_values,
bytes_saved}`, the last the measured difference of the two bindings;
`UpdateConfig::remat` and `CompiledUpdate::remat_*` carry the same.

Bit-identity is the acceptance and it holds: the system suite trains the
remat and the plain plan for 20 CPU steps (f32 and int8 base) and
compares the loss curve, the validation loss and the persistent segment
byte for byte; the compiler suite checks the arena shrinks, the eval
program is the same instruction sequence, and `auto` at a budget one
byte short of the plain footprint produces the forced plan byte for
byte. Measured (`docs/benchmarks.md`, "Rematerialization at three
rungs"): the arena 1.67 → 1.33 GB at SmolLM-135M (−20.3 %), 3.27 → 2.75
GB at SmolLM2-360M (−15.9 %), 4.51 → 3.66 GB at Qwen2.5-0.5B (−18.9 %),
for 98–130 extra train instructions and an unchanged eval program; the
packaged SmolLM-135M plans commit the same SHA-256 after 20 CPU steps
(`beb98417…`), the run's peak RSS falling by the arena saving to the
megabyte (2,688 → 2,350 MB) for a step 1,641 → 1,780 ms (+8.5 %, one
sample).
What remains of the transient segment is the GEMM outputs the backward
needs (the Q/K/V rotations' inputs, the attention output, gate and up),
the adjoints and the pinned deltas — the next lever is step 3's 8-bit
optimizer states for the persistent segment, and for transients a
recompute of the rank-r projections' inputs is not worth its GEMM.
Metal was not measured (the GPU was in use); the pass is backend-neutral
(the same opcodes), so the byte-identity claim on Metal is a test to
run, not a design question.

## Acceptance

- A field-report table per model: arena, peak RSS / arena, tok/s,
  val-loss parity, compile time.
- 1.5B either trains on 16 GB or is refused at compile time with a correct
  estimate — never an OOM at run time.
