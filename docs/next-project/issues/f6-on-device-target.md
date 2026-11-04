---
title: "SeeAI F6: on-device target — iPhone/iPad package runs, energy per token as a Tier A metric, adapter-only update packages"
number: 134
labels: enhancement,frontier-parity
plane: Gates & docs
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P2
---
## Why

SeeAI's claim is on-device training with no Python, but every measurement
so far is from a MacBook. The frontier frameworks are not designed to run
on a phone; this is where SeeAI's doctrine (one arena, validated plans,
gates that score what ships, #95) should turn into a product advantage —
and it is unmeasured.

## Design

1. iOS/iPadOS: build the emitted package for arm64 iOS (Metal backend
   vendored, `.incbin` plan), a minimal host app that runs `model_update`
   on a bundled corpus; thermal-throttling behaviour recorded
   (the M5 laptop already shows two regimes).
2. **Energy per token** (J/token) as a Tier A metric: powermetrics on
   macOS, MetricKit / Xcode energy gauges on iOS; recorded next to tok/s in
   `seeml-bench` and in F1's frontier rows where the frontier can run.
3. **Adapter-only update packages** (Frontier Bridge pillar D2, the
   "plan size / delta size" row of `docs/benchmarks.md`): ship the LoRA
   delta and the plan, not the model; the device commits onto its own
   copy, hash-checked (`UpdateEngine::VerifySourceModel` already
   verifies the source against the plan's recorded hash).
   Headline: package bytes from model-scale to adapter-scale.

## Status 2026-11-03

The two lines that need no device landed (`docs/benchmarks.md`, "The
adapter-only package, and energy per token"; `docs/ledger/f6-2026-11-03/`).
The adapter-only package is plan v21's `--adapter-only`: every frozen
weight — the embedding table too — is a source ref in every program (the
address space the eval program has read since v17), the plan lays out no
rodata, and what ships is the instructions, the adapters' initial image
and the emit table; the commit path is unchanged, the device commits onto
its own copy of the model. SmolLM-135M r8 on the f32 base: a 37.4 MB
binary (36.6 MB of plan, 36.3 MB of it the adapters' initial image)
against 301.5 MB for today's int8 package and 747 MB for the f32 rodata
one — 8.1× and 20× smaller. The same f32 bits read from
another address compute the same bits: the rodata and adapter-only plans
commit the same model on both backends. The limits, each a named refusal:
a model loaded from a file, the f32 base the file holds, exact
arithmetic. An int8 adapter-only package — the device quantizing the
file's weights at load into a second allocation — is the follow-up the
one-arena doctrine has to rule on. Energy per token: `tool/seeml/energy.py`
(the frontier rows) and `tool/energy_meter.h` (`seeml-bench`) read
IOReport's "Energy Model" counters — the library behind `powermetrics`,
opened at run time, no root — by the step time's own regression, whole-SoC
and stated so in every row; on this macOS 26 the GPU counter advances
with every sample while the CPU, DRAM and ANE counters arrive in bursts
(747 J landing in one 8-second window, none in the next; the probe and
the rows are in the ledger), so a Metal row carries a GPU-scoped number
and a row that computes on the CPU says `unavailable` rather than borrow
the GPU's idle draw. SmolLM-135M r8 on Metal: 2.98 mJ per token on the
rodata plan (1.52 J per 512-token step, 6.8 W), 3.04 on the adapter-only
one — the host on battery with Low Power Mode on, that state's joules, the
comparison between the plans the measurement. The device line
stays open: this host has no iPhone or iPad and no iOS SDK (`xcrun` finds
none), and nothing in the tree pretends otherwise.

## Acceptance

- [ ] SmolLM-135M update runs on a current iPhone and iPad: tok/s, J/token,
  peak memory, gate result, committed model hash — not started: no device
  and no iOS SDK on the build host (2026-11-03). The package it would run
  is the adapter-only one below, and the metric it would report is the
  energy meter's.
- [x] Energy per token in `seeml-bench` and the frontier rows on macOS
  (design item 2's first half, measurable without a device) — 2026-11-03:
  `energy_j_per_step` / `energy_j_per_token` in the bench's per-thread
  block and an `energy` object on every frontier row, from IOReport's
  counters without root, by the lo/hi regression; whole-SoC, GPU-scoped on
  an unprivileged macOS 26, `unavailable` with the reason elsewhere.
  SmolLM-135M r8 on Metal reads 2.98 mJ per token (the rodata plan) and
  3.04 (the adapter-only plan), the host on battery with Low Power Mode on;
  the CPU rows say `unavailable`, since
  the CPU domain reaches an unprivileged reader in bursts.
- [x] Adapter-only package for SmolLM-135M r8 measured in bytes against
  today's package — 2026-11-03, `--adapter-only` (plan v21): a 37,404,848-
  byte binary (plan 36,585,472, rodata 0) against 301,495,056 for today's
  int8 package and 747,391,920 for the f32 rodata plan's — 8.1× and 20×
  smaller (`docs/benchmarks.md`, "The adapter-only package, and energy
  per token").
  The rodata and adapter-only plans commit the same model after 20 steps
  on Metal and on the CPU.
