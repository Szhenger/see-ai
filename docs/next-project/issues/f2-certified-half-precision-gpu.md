---
title: "SeeAI F2: certified half-precision GPU GEMMs — bf16/f16 simdgroup-MMA and M5 neural-accelerator kernels as a relaxed opcode family behind a P3 certificate"
number: 130
labels: enhancement,efficiency,core-plane,doctrine,frontier-parity
plane: GPU backend
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P0
---
## Why

SeeAI's Metal backend computes every GEMM in f32 (bf16/int8 are
*storage* formats widened into f32 panels, #71/#102). MLX runs bf16 or
TF32 by default and, on M5-class GPUs, can reach the per-core neural
accelerators. That precision gap — not compiler design — is the largest
single cause of the speculated 0.55–0.8× SmolLM-135M GPU ratio versus
MLX-LM (Frontier Outlook, 2026-09-18). GEMMs are ~65% of a decoder step
(`seeml-plan-probe --profile`, E1 #80); fusion work moved step time <6%.

## Doctrine

Bitwise determinism stays the default. `docs/roadmap.md` already states
the rule: a relaxed computation is an **opt-in opcode family** that the
old path never emits, admitted only for a plan that
`tool/certify_numerics.py` has certified on its own operands. The tool
and the contract exist (P3, #77); the opcode family does not. This issue
builds the first member of it. Relaxed kernels must still be
run-to-run deterministic on a given device (fixed reduction order, no
atomics), so two runs of the same relaxed plan stay byte-identical.

## Design

1. Plan: relaxed GEMM opcodes (or a `kFlagRelaxedPrecision` bit on
   kGemmNN/NT/TN) carrying the compute type (bf16 or f16 inputs, f32
   accumulate); a header field or section recording the certificate hash;
   the validator refuses relaxed instructions without it. SEEU version bump.
2. Compiler: `--precision f32|certified-bf16|certified-f16`; lowering picks
   the relaxed opcode only for the frozen-weight GEMMs and the LoRA
   projections the certificate covers; optimizer, norms, softmax and loss
   stay f32.
3. Metal: bf16/f16 `simdgroup_matrix` GEMM templates (same tile family as
   the f32 kernels, #103); a Metal 4 tensor-op path for M5 neural
   accelerators behind a device-capability check, falling back to simdgroup.
4. `certify_numerics.py`: price the relaxed GEMM sites against the f64
   oracle, emit the certificate; `pack_update.py` refuses an uncertified
   relaxed plan (it already checks certificates).
5. CPU backend: executes relaxed opcodes by widening to f32 (reference
   semantics), so every plan still runs everywhere.

## Integrated from the 2026-09-24 systems review

Root cause 1, *arithmetic doctrine*: every GEMM is f32 math. The Metal
per-opcode profile of the SmolLM-135M q8 package
(`out/frontier-2026-09-22/prof_q8.txt`, `SEEML_METAL_PROFILE`) shows the
exact kernels are already at the M5's f32 ceiling — the head
512×49152×576 in 8.46 ms (≈3.4 TFLOP/s), the block GEMMs 512×1536×576 in
299 µs and 512×576×576 in 122 µs (2.8–3.0 TFLOP/s) — so no further f32
kernel work can close the 0.36–0.48× bf16 gap; only this issue can. Base
storage does not move the step (int8 1,802 / bf16 1,756 / f32 1,776
tok/s), so dequantization is not the bottleneck either. Per training step
the frozen GEMMs are ≈50 % of kernel time for ≈97 % of the FLOPs: this
issue's ceiling on the step is ≈2× on that half, after which the LoRA
family (F7) and the dispatch model (F8) are the majority of what remains.

Status 2026-09-24: plan v18 `kFlagRelaxed`, the Metal 4 tensor-op GEMMs,
the Accelerate CPU path and the certificate gate at packaging landed in
#136; the SmolLM-135M certificates priced 419 relaxed GEMMs (gemm.relaxed
6.2e-3 / 5.5e-3 against 2^-7, validation within 3e-6). **The Metal rows
are unmeasured**: every row of `out/frontier-2026-09-22/f2_measure/metal.json`
is exit 1 and marked suspect (the session had no Metal access). The CPU
row is the only real F2 number so far (F4 carries it). The first
executable step is therefore the measurement itself, on a host with Metal
access, with validation off so the profile weights forward and backward
as a training step does (F9).

Status 2026-10-15: **measured** (`docs/benchmarks.md`, "The relaxed GEMM
family on Metal, measured"): on the M5 the relaxed SmolLM-135M package
runs 2,519 tok/s against the exact package's 1,783 (1.41×) and MLX-LM
all-bf16's 4,465 in the same session (0.56×); the 300-step validation loss
deviates 2.4e-5 relative, the gate decides identically, two relaxed runs
commit byte-identical models, and the exact package is unchanged. The
≥ 0.9× line is not reachable from this issue alone — the frozen GEMMs are
47.6 % of the exact step's kernel time, so a relaxed family that cost
nothing would top out at 1.76× on the measured step where 0.9× of MLX-LM
needs 2.25× — and passes to F7 and F8. The code-review holes closed the same day: the compiler refuses
`--build` for a relaxed plan and the emitted `build.sh` checks for a
granted certificate over its exact plan bytes before compiling (N17); the
certifier refuses a zero or non-finite tolerance, a diff report carries
each opcode's relative error and `verify` reads that rather than
`worst_ratio × rtol`, so NaN cannot pass (F41); the report names its plan,
its structural mismatches refuse it, and every uncertified opcode is held
to the diff tolerance (F42; whether the C++ arm of a diff actually ran the
relaxed kernels joined the report on 2026-10-29 — see that status); relaxed GEMMs are keyed `gemm.relaxed`
on both sides (F72); the certificate binds to the source model by SHA-256
and `verify --source` refuses another file (N35, with the ContentHash64
check of "is this the model the plan was compiled from" left to the C++
probe, which runs it on every eval section). The certificate carries the
per-opcode maximum relative error within each site.

## Integrated from the 2026-09-24 code review

The certified family has gate holes the review reached, and they are this
issue's to close: a relaxed (`--precision certified-bf16`) plan is built into
a runnable binary by the C++ `--build` path with **no numerics certificate**
(N17) — only `pack_update.py` refuses; the certifier's `worst_ratio * rtol`
is a lower bound on relative error, not an upper bound, and with `rtol = 0`
every site measures 0 and NaN passes (F41); `--observed` is not bound to the
plan being verified and its own failures are ignored (F42); `check_observed`
looks for `gemm.relaxed` while reports use opcodes (F72); `run`/`certify`
accept a `--source` whose hash differs from the plan's (N35). Acceptance
gains: every build path refuses an uncertified relaxed plan; the certificate
carries the per-opcode max relative error and refuses `rtol <= 0` and
non-finite values; the certificate binds to the plan hash and the source
hash.

## Status 2026-10-24

Re-measured on the F7-era kernels beside MLX-LM in the same session
(`docs/ledger/f2-2026-10-23/`): relaxed 4,392–4,395 tok/s, 0.74× of MLX-LM
all-bf16 (5,929 today), exact 0.97× of true-f32. The host itself read
about 1.3× faster than on 10-15 for every row, SeeAI's and MLX's alike,
so only same-session ratios are quoted. Open, and newly so: the plan is
granted under the settings the 10-15 and 10-19 certificates used
(`--steps 2 --max-site-error 1e-4 --max-loss-deviation 1e-2`) and refused
under `certify_numerics.py`'s defaults (20 steps, 1e-5, 1e-4: the
softmax cross-entropy site at 4.5e-5, the loss deviation at 3.8e-4). The
milestone has to decide which contract a shipped relaxed package
carries.

## Status 2026-10-29

F42's residual closed. Every executor backend counts the instructions it
ran on a relaxed kernel (`ExecutorBackend::relaxed_executions`: the
Accelerate SGEMM paths on the CPU, the tensor-op GEMMs on Metal; the CPU's
device line now says which it has for them, as Metal's did).
`seeml-plan-probe` writes the count per instruction into trace v2 and into
its `--time` / `--profile` JSON; the diff report carries
`relaxed_instructions` and `cpp_relaxed_executed`; the generated driver's
run report carries `relaxed_executions` and the frontier row repeats it.
`certify_numerics.py verify --observed` refuses a report, for a certificate
that covers `gemm.relaxed`, whose C++ arm ran no relaxed GEMM on a relaxed
kernel or whose reference arm modelled the rounding (`--relaxed`) — the two
ways a diff agrees with itself and vouches for nothing. The system suite
asserts that the Accelerate build takes the permission for every relaxed
GEMM of every step and the reference build for none, so the path is
tested rather than tolerated. On the 10-15 relaxed SmolLM-135M plan
(`out/f2-2026-10-15/pkg_relaxed`, 3,309 train instructions) the probe's
`--time` reports `relaxed_executions: 419` per train step on Metal and on
the Accelerate CPU alike — the 419 relaxed GEMMs the certificate priced, so
neither backend leaves a tensor-op or SGEMM shape on the exact kernels at
SmolLM's shapes. The contract decision above is unchanged and
now visible where the binary is built: `pack_update.py` records the
certificate's full tolerances and evidence steps in its report and prints
a NOTE for each tolerance looser than the certifier's defaults
(`DEFAULT_TOLERANCES`, the one table the CLI defaults read).

## Status 2026-10-30

The contract is decided: a certificate certifies the relaxations its plan
takes (`docs/benchmarks.md`, "The certificate contract, decided"). A plan
with `kFlagRelaxed` instructions carries `relaxed-flagged-instructions`:
the flagged sites enforced, the free run's relaxed arm relaxing exactly
those instructions, the loss default 1e-3 (this issue's acceptance line);
every other site measured as evidence. Exact plans keep P3's
`relaxed-reductions-f32`. `check_certificate` refuses a certificate that
leaves any relaxed site of the plan uncovered, and `pack_update.py`'s
tolerance NOTE compares against the certificate's own family
(`FAMILY_DEFAULTS`, one row per family, replaces 10-29's single
`DEFAULT_TOLERANCES` table). The measured 10-24 package (plan hash
`a0d2d967…`) is granted under it at the defaults: `gemm.relaxed` 6.75e-3
against 2⁻⁷, the 20-step free run's loss deviation 2.57e-4 against 1e-3,
the softmax cross-entropy's 4.49e-5 printed as evidence; the branch tip's
`verify` and `pack_update.py --build` accept the certificate
(`docs/ledger/f2-2026-10-30/`). Every acceptance line is met.

## Acceptance

- [x] SmolLM-135M GPU tok/s measured against F1's MLX-LM all-bf16 row on
  the same M5 — relaxed 2,519 tok/s against the exact package's 1,783
  (1.41×) and MLX-LM all-bf16's 4,465 (0.56×), `docs/benchmarks.md`, "The
  relaxed GEMM family on Metal, measured (2026-10-15)".
- Superseded 2026-10-22, the line as first written: *target ≥ 0.9× on the
  same M5.* It was set against the 2026-09-18 Outlook's guess, and the
  measurement shows it is out of this family's reach by construction: the
  frozen GEMMs are 47.6 % of the exact step's kernel time, so a relaxed
  family that cost nothing would top out at 1.76× on the measured step,
  and 0.9× of 4,465 needs 2.25×. The 0.9× line is the milestone's, carried
  by F2 + F7 (#138) + F8 (#139) together and read from the relaxed package
  after each; what this issue still owes is the next line.
- [x] The relaxed package re-measured on the F7-era kernels — 4,392–4,395
  tok/s, **0.74×** of MLX-LM all-bf16 measured in the same session
  (5,929), 1.55× the exact package; "The relaxed package on the F7-era
  kernels, beside MLX-LM in the same session" (2026-10-24). Granted
  under the 10-15 certificate settings; refused under the tool's
  defaults — see that section.
- [x] Certificate: per-site error inside the stated bound (`gemm.relaxed`
  observed 6.2e-3 against 2⁻⁷); 300-step val loss within 1e-3 relative of
  the f32 run (2.4e-5); gate decision unchanged.
- [x] Two runs of a relaxed plan byte-identical (`93fb4a37…` twice); f32
  plans bit-identical to today (the exact package unchanged, 1,783 against
  the 2026-09-22 row's 1,802).
- [x] Every build path refuses an uncertified relaxed plan; the certificate
  carries the per-opcode max relative error, refuses `rtol <= 0` and
  non-finite values, and binds to the plan hash and the source hash (N17,
  F41, F72, N35 — closed 2026-10-15).
- [x] F42's residual: the diff report records whether the C++ arm ran the
  relaxed kernels (`cpp_relaxed_executed`, from the probe's per-instruction
  count), and `verify --observed` refuses an exact-vs-exact diff, or a
  `--relaxed`-modelled one, for a certificate that covers the site
  (2026-10-29; `test_an_observed_run_must_have_measured_the_relaxed_kernels`
  and the relaxed-plan C++ test in `test/tool/certify_numerics_test.py`).
