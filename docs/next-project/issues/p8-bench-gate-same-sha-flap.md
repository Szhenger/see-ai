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

## Acceptance

- [ ] The 2026-10-20 and 10-21 rows reproduced or refuted on one host at
  the two commits, the numbers in `docs/benchmarks.md` beside the runner's.
- [ ] Real: the commit named and an issue under its root cause. Variance:
  the P5 (#79) assumption restated with this incident, and the gate's
  band or normalization changed so the same readings would not fail.
- [ ] The promotion rule decided, implemented and tested in
  `tool/bench_compare.py`, and written into "Measurement discipline".
