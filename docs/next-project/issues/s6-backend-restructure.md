---
title: "SeeAI S6: backend restructure — architecture, allocation, selection, assembly, packaging; assembly out of the driver; tuner and kernel emitter retired; the GPU half of the target description"
number: 152
labels: enhancement,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

Plan assembly, the seeded persistent image and the integrity seal are
implemented inside the driver's `CompileImpl`, a 560-line function the
Style Audit named; the tuner directory holds a table reader that belongs to
the target description; `kernel_emitter.cc` generates Metal source nobody
ships since the runtime took ownership of the kernel library; and the host
description cannot say the device has a GPU, so the backend screen's
predicted step time has no GPU denominator. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §5.

## Design

1. Directories `architecture/` (host + GPU target description, analytic
   tiling, policy-table read), `allocation/`, `selection/`, `assembly/`
   (moved out of the driver: layout, persistent image, emit table, seal;
   the header gains the identity manifest (S4) and the ledger (S5)),
   `packaging/` (the emitter, now also vendoring `source/`'s contract
   check, sanitizer and tokenizer). `tuner/` folds into architecture;
   `kernel_emitter.cc` retires or its GPU tiling clamp moves under
   architecture as data.
2. Selection: ISA drops MSE and conv opcodes, adds the masked cross-entropy
   pair (S3) and the attention segment operand; SEEU version bump with the
   header fields above.
3. Scheduling stays a candidate gated on F8 (#139): if the measured step
   time outside kernels is real, it becomes a directory that runs before
   allocation; otherwise a function inside selection.
4. The backend screen: arena by segment with the five largest tensors, the
   honesty ratio against the accountant, the GEMM census (shape, FLOPs,
   share), dispatch count per program, plan and package bytes, predicted
   step time at the host peak.

## Status 2026-11-11

The moves of design item 1 are complete on `claude/v100b-s4s5s6` (step 1 of
the S4/S5/S6 plan), every plan bit-identical. `backend/assembly/
plan_assembly.{h,cc}` holds what the driver did inline — the seeded
persistent image, the emit table, the section layout, the blob and its seal
— as five pure functions. `CompileImpl`, 1,390 lines, is now one
`Compilation` (what the stages hand each other) and 24 stage methods run in
order; the longest is under 140 lines and no driver function exceeds 150
except `FillCompileScreens` (192 lines of row assignments, left for the
screen work of item 4). One ordering difference: the config rules all run
before the weight sweep, which used to sit between two groups of them.
`kernel_emitter.{h,cc}` is retired: the GPU tiling clamp is architecture
data (`GpuTilingFromHost` in `host_arch`), and the Metal source generator,
used only by the G1a hardware harness test, is `test/support/
metal_gemm_source`. Verified: plan content identical to main 523e4fc on 17
fixtures; goldens and the ABI manifest unchanged by this step. Open: the
header's manifest and ledger with the v23 bump (item 2; the masked
cross-entropy pair already landed with S3), dropping the MSE opcodes
(deferred to its own issue — MSE is live in the demo MLP, an identity
fixture and the frontier matrix; conv had no opcode to drop), the GPU half
of the target description beyond the tiling clamp, the backend screen's new
rows and the GPU-denominated predicted step (item 4), and scheduling (item
3, gated on F8).

## Acceptance

- Bit-identical `.seeu` and packages for every fixture before and after the
  move, except the header fields the bump adds.
- The driver contains orchestration only: no function over 150 lines.
- The backend screen's predicted step time for SmolLM-135M on the M5 GPU is
  printed and the runtime's measured line lands beside it.
