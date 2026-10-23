---
title: "Python plane P8: the Tier A gate went red twice, then green at the same commit (nightly 2026-10-20 → 10-22) and promoted the baseline — decide real or variance, and what clears a red"
number: 165
labels: bug,testing,python-plane
plane: Python plane
origin: Nightly eval
milestone: SeeAI v1.0.0.B
priority: P1
---
## What the nightly gate recorded

Four consecutive `bench (Tier A regression gate)` jobs of `nightly.yml`,
on the hosted `ubuntu-latest` runner, `tool/bench_compare.py` against the
rolling baseline and the pinned epoch anchor (`bench-epoch-v2`):

| run | date | main at | verdict against the rolling baseline |
|---|---|---|---|
| 39271309779 | 2026-10-19 | `ee979e7` (#160 merged) | no Tier A regression; promoted |
| 39424640243 | 2026-10-20 | `32c8ac6` (#161, #162 merged) | **WARN**, first red on 12 rows: `dec_v256_d128_s128` @1t/2t/4t −37.7 / −27.9 / −28.9 %, `dec_v512_d192_s32` @1t/2t/4t −28.9 / −25.3 / −21.2 %, `mlp_128x1024x2` @1t/2t/4t −18.5 / −17.8 / −17.1 %, `dec_v256_d128_s32` @1t/4t −14.1 / −15.3 %, `mlp_64x512x3` @4t −10.6 % (normalized); 10 rows red against the epoch too; "22 Tier A metric(s) red for the first time — warning only" |
| 39603771547 | 2026-10-21 | `4b7ce89` (#163 merged) | **FAIL**, second consecutive red on 7 rows: `dec_v256_d128_s128` @1t/2t/4t −37.5 / −28.1 / −18.8 %, `dec_v512_d192_s32` @1t/2t −26.9 / −22.2 %, `mlp_128x1024x2` @1t/2t −18.0 / −15.5 %; 6 rows red against the epoch ("compounding drift") |
| 39782901348 | 2026-10-22 | `4b7ce89` (the same commit) | no Tier A regression against either anchor; **promoted `bench.json` to the baseline** |

The runner calibrated at 0.947× (10-19), 1.024× (10-20), 1.214× (10-21)
and 1.063× (10-22) of the baseline's host, with median raw key ratios
0.918 / 0.868 / 1.185 / 1.096. The raw `dec_v256_d128_s128@1t` readings
were 1,532 (10-20), 1,820 (10-21) and a green value on 10-22 against a
baseline of 2,400 rows/s. Nothing was filed when the gate fired; this
issue is the record.

## Two questions, neither answered here

1. **Real or variance.** The merges in the window touch the CPU path:
   F7 step 1 folds alpha into A's elements on the CPU NN and TN cores
   (#161), #162's int8 Accelerate path is relaxed-only and does not run
   in a fixture, #163 changes the arena binder's pinned-value order and
   adds `--remat auto`. Each is bit-identical by test, and bit-identity
   says nothing about speed. The 10-22 run at the same commit read green,
   which argues variance; the 10-21 calibration ratio of 1.214× is the
   largest in the window, and the normalization at that ratio is exactly
   the P5 (#79) assumption that Nightly #20–#23 first broke.
2. **What clears a red.** A green run at the same commit promoted the
   baseline, so if the slowdown was real it is now the reference and the
   next real regression is measured from it. The rolling baseline has no
   rule for an unresolved red: two consecutive reds fail, one green
   promotes.

## Design

1. Reproduce or refute on a quiet host: `seeml-bench` at `ee979e7` and
   `4b7ce89` on `dec_v256_d128_s128` and `dec_v512_d192_s32` at 1, 2 and 4
   threads, the same fixtures and the steps-regression slope — a number
   from one machine, not a runner's. If real, name the commit and file it
   under its root cause; if variance, restate the P5 assumption with this
   incident beside Nightly #20–#23 and widen the band or re-derive the
   normalization from the two incidents' calibration ratios.
2. A promotion rule in `tool/bench_compare.py`: a red at commit X is
   cleared only by a green at X whose calibration ratio is inside the band
   the baseline was taken at, or by `--reseed-baseline` with a named
   cause in the workflow dispatch; a green that clears a red says so in
   the log. `docs/benchmarks.md`, "Measurement discipline", states the
   rule.

## Status 2026-10-22

Variance, of the host class. The 40 nightly `bench` jobs since
2026-09-13 sorted by the runner's CPU model (`host_key`): every FAIL and
every Intel WARN is an Intel Xeon (8573C ×5, 8370C, 6973P-C) measured
against an AMD EPYC baseline, a 7763 night's or a 9V74 night's (this
incident's 2,400 is the 9V74's 10-19 reading), and the one other red
(10-14) an AMD EPYC 9V45 against a 7763 — the models' fixture ratio is
not their calibration ratio (the 8573C reads `dec_v256_d128_s128` @1t
at 0.62–0.73× of the 7763 while its calibration kernel reads
0.99–1.15×; the 6973P-C 0.82× and 1.39×), so the P5 normalization adds
to the red; the 19 calibrated 7763 nights are
within ±1.4 % across six commits. The 10-20 → 10-22 runs were Xeon 8370C, Xeon 8573C, EPYC 7763:
two different Intel hosts made the "two consecutive" reds. On one host
(Apple M5) `ee979e7`, `32c8ac6` and `4b7ce89` read 16 of the 18 rows
within 2.2 % of each other; of the other two, `mlp_128x1024x2` @4t
re-measured quiet at +0.6 %, and `dec_v256_d128_s32` @4t reads 15.3 %
and 14.3 % *faster* at the two red commits (the 10-20 gate named it red
at −15.3 %), unexplained and not re-measured, but opposite to the red.
Numbers in `docs/benchmarks.md`, "The nightly gate's reds were the
runner's CPU model, not the commit"; raw in `docs/ledger/p8-2026-10-22/`.

The rule: `bench_compare.py` names the host a report measured on
(`host_arch`: model and cores — not the host_key, which is the kernel
policy's lookup key and names `--target-host` when given) and refuses a
baseline from another host unless `--across-hosts`, comparing slugs so
two spellings of one model are one host; `nightly.yml` benches first and
restores the rolling baseline and gate state under
`bench-rolling-<generation>--<host>--<release>--<policy>--<calibration>--`
(`--cache-key`), saved after every gate; each host's epoch is a report
committed at `docs/ledger/bench-epochs/` — the first night on a host
seeds its rolling baseline, never its epoch. (Amended 2026-10-23 after
the `/code-review` of this change: the epoch moved from the cache to git,
the key gained the setup segments and `--` separators, the state is
saved on failure, and an unmeasured red stays red.) What clears a red: a green on
the same model; the state carries the commit (`--sha`, `$GITHUB_SHA`),
and a cleared key is announced with both commits — the same commit twice
is the runner's variance on record, a later one is a fix. The design's
calibration-ratio band is not needed once the comparison is within a
model (the 9V74 at 712k and 889k rows/s normalizes to the same row).
Tests: `test/tool/bench_compare_test.py` (the 10-08 → 10-22 replay red
on one baseline and green keyed by model, the refusal, the slug, the
cleared-red record, the schema-1 state; since 2026-10-23 also a cleared
red after a FAIL night, an unmeasured red, the measuring host against a
`--target-host` key, two spellings of one model, the key segments, and
the interleaved command line). Written into "Measurement discipline".
The next nights seed each host's rolling baseline; until a host has met
itself twice its rolling gate is a seed, not a verdict — and an Actions
cache unused for seven days is evicted, so the rare models (8573C nine
days apart, 9V45 three weeks, 8370C and 6973P-C once each) will mostly
re-seed their rolling baseline (10 of the 35 calibrated nights, replayed).
The committed epochs of the 7763 and the 9V74, which is where the nights
are, compare on every visit.

## Acceptance

- [x] The 2026-10-20 and 10-21 rows reproduced or refuted on one host at
  the two commits, the numbers in `docs/benchmarks.md` beside the runner's
  — refuted: 16 of 18 rows within 2.2 % across the three commits on the
  M5, one re-measured quiet at +0.6 %, one 15.3 % faster at the red
  commits (2026-10-22, corrected 2026-10-23).
- [x] Real: the commit named and an issue under its root cause. Variance:
  the P5 (#79) assumption restated with this incident, and the gate's
  band or normalization changed so the same readings would not fail —
  variance of the host class; the gate compares within a CPU model only,
  so an Intel night never meets an AMD baseline (2026-10-22).
- [x] The promotion rule decided, implemented and tested in
  `tool/bench_compare.py`, and written into "Measurement discipline" — a
  red is cleared by a green on the same model, both commits named
  (2026-10-22).
