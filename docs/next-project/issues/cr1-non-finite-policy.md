---
title: "SeeAI CR1: a non-finite policy — NaN and Inf refused at the boundary: ingest, int8 selection and packing, clip norms on both backends, KL temperature, q8 scales, and a test comparison NaN cannot pass"
number: 142
labels: bug,correctness,core-plane,doctrine
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

Ten findings of the 2026-09-24 review are one missing rule. A NaN weight is
selected for int8 because `std::max` ignores NaN (F12) and then cast to
int8, undefined behavior that makes plans differ across hosts (F34); the
standalone `kClipNorm` does not refuse a non-finite norm on the CPU (F09)
and the GPU fused clip writes NaN into p, m and v where the CPU refuses the
step (F11); `max_norm == 0` multiplies by stale scratch (F13); the KL
temperature is never validated, and a negative T inverts both softmaxes
(F17, F10); the standalone clip threshold is unvalidated, −1.0 is gradient
ascent (F18); a NaN q8 dequant scale passes and the test expects it (F33);
`--temperature` is unbounded (N26); `ExpectClose` accepts NaN because
`err > tol` is false (F29). Full text: the "SeeML Code Review" page, first
review detail. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

One rule, applied at each layer: **non-finite is refused at the boundary
it crosses.** Ingressor: non-finite weights refused (S1). Verifier: every
immediate float (clip threshold, KL temperature, q8 scale) must be finite
and in range; the validator's `clip_word_ok` checks applied to every site.
Executor: `ClipNorm` returns a status, CPU and Metal both map a non-finite
norm to `kNonFiniteNorm` with a flag checked after `Flush`; the sum of
squares accumulates in f64. Tests: `!(err <= tol)` everywhere a tolerance
is compared. The threat model on the screens (S0) states which classes the
sanitizer covers.

## Status 2026-11-05

Landed in two commits on `claude/v100a-cr` (#178), one rule at every layer.
The boundaries: the validator proves every immediate float — `kClipNorm`'s
threshold a finite positive float (F13 / F18), the KL words' temperature
finite and within `[kKlTemperatureMin, kKlTemperatureMax]` = [1e-3, 1e3] at
every version (F17 / F10), the q8 per-tensor dequant scale finite and
positive (F33) — and the load contract reads each quantized GEMM's column
scales from rodata and refuses a non-finite or non-positive one by index;
the header's seven optimizer floats are finite and within the CLI's ranges.
The compiler refuses a config whose floats are not finite and in range (N26
on the library path; the CLI's `--temperature` is bounded at both ends by
the same constant) and sweeps every constant tensor of the student and the
teacher before any pass, naming the first non-finite element (F12 / F34 —
the acceptance's UBSan case: the int8 cast is never reached; the quantizer's
own scan also never selects a NaN tensor); the exporter refuses a non-finite
tensor at write (S1). The executor: the standalone `ClipNorm` returns a
status and the CPU maps a non-finite norm, or a bad threshold, to a refusal
(F09); on Metal `k_clip_finish` raises a refusal flag in a buffer of its own
when the norm is not finite, every parameter-writing kernel honors it, and
`Flush` returns the CPU's words (F11); a zero threshold is refused before
anything is encoded (F13). The twin test drives both backends by hand on a
gradient with an infinity — fused SGD, fused AdamW, a standalone clip before
a plain step, a zero threshold — and asserts one verdict and a bit-for-bit
untouched arena on both; `ExpectClose` records a NaN as the worst error
(F29). Each rule was removed in turn and its test failed (the table is in
the PR). Known boundary: the GPU's clip norm sums in f32 and the CPU's in
f64, so a finite gradient whose squares overflow f32 (|g| beyond ~1.8e19)
is refused on Metal and clipped on the CPU. Not here: the Python plane's
plan checks (`pack_update.py check_plan`, `formats.py`) do not mirror the
new validator rules — CR2's open Python remainder; and the runtime binds a
source model (the adapter-only package) without sweeping it, so a
hand-written SMF with a NaN base trains to a NaN loss and the finite-loss
guard stops it, as before.

## Acceptance

- Each of the ten reproductions is a refusal or a defined result, with a
  test; the twin test pins CPU and Metal to the same verdict.
- UBSan clean on a plan compiled from a model with a planted NaN (it is
  refused before packing).
