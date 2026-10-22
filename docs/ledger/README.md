# The measurement ledger

Every dated section of [`docs/benchmarks.md`](../benchmarks.md) names the
`out/` directory its run wrote. `out/` is ignored by git and exists on one
machine, so this directory carries, by the same name, the files a
section's numbers are read from: the frontier harness's JSON
(`tool/frontier_run.py`), the compile reports (`report.json`), one
serialized profile (`prof_1.txt`) and one span table (`spans_1.txt`) per
measured state, and the identity hashes (`id_sha.txt`, `sha.txt`) the
byte-identity claims rest on. Raw packages, plans, checkpoints and loss
curves stay in `out/` (102 GB on 2026-10-22). Home-directory prefixes in
the copies are written `~`. Added 2026-10-22; a section that lands after
that date commits its summaries here in the same change.

| `docs/benchmarks.md` section | ledger directory | the files its numbers come from |
|---|---|---|
| The frontier row, measured (2026-09-22) | `frontier-2026-09-22/` | `summary.json` (every row, every stack) |
| The relaxed GEMM family on Metal, measured (F2 #130, 2026-10-15) | `f2-2026-10-15/` | `metal.json` (the matrix), `plan_metal.json`, `pkg_exact/report.json`, `pkg_relaxed/report.json`, `pkg_relaxed/numerics_certificate.json` |
| The nightly frontier row's recipe, run by hand (F1 #129, 2026-10-15) | `frontier-2026-10-15b/` | `frontier.json`, `plan.json` |
| The loss on the GPU, and the dispatch model measured (F8 #139, 2026-10-15) | `f8-2026-10-15/` | `metal.json`, `plan_metal.json`, `cpu_lifecycle.json` |
| Measurement rules, and the rows they were missing (F9 #140, 2026-10-15) | `f8-2026-10-15/`, `realanalysis-2026-09-25/` | `bench_fixture.json` (the fixture's under-read), `cpu_lifecycle.json`; `results.json`, `results.md`, `manifest.json`, `bootstrap.json` (the per-column int8 row) |
| The LoRA lowering, steps 1 and 3, and the per-dispatch cost (F7 #138, 2026-10-19) | `f7-2026-10-19/baseline/`, `step1/`, `step3/`, `alpha3/` | `metal.json`, `plan.json`, `prof_1.txt`, `spans_1.txt`, `id_sha.txt` per state; `alpha3/sha.txt` (the every-alpha demonstration); `step1/compile_report.json` |
| No CPU-resident opcode in the train step, and the fusion verdict (F8 #139, 2026-10-19) | `f7-2026-10-19/f8/` | `metal.json`, `prof_1.txt`, `spans_1.txt`, `id_sha.txt` |
| The int8 base on the CPU matrix units (F4 #132, 2026-10-19) | `f7-2026-10-19/f4/` | `cpu.json` (the matrix), `plan_cpu.json` |
| The LoRA family's Metal kernels, off the issue floor (F7 #138 step 3, second half, 2026-10-19) | `f7-2026-10-19/f7b/` | `metal.json`, `prof_1.txt`, `spans_1.txt`, `id_sha.txt` |
| The scale ladder's first two rungs (F5 #133, 2026-10-20) | `f7-2026-10-19/f5/`, `split/`, `binder/` | `frontier_*.json`, `plan_*.json`, `pkg_*/report.json`; the compile reports with the three-segment split; the binder fix's identity hashes |
| Tiled attention on Metal at S = 2048 (F3 #131, 2026-10-20) | `f7-2026-10-19/f3/` | `frontier.json` (the matrix with MLX-LM and torch), `plan.json`, `plan2.json`, `steps*.json` (the 20-step runs, cached and tiled) |
| Rematerialization at three rungs (F5 #133 step 2, 2026-10-20) | `f7-2026-10-19/gpu_checks/` | `*_off.json` / `*_on.json` (the six compile reports), `disk_metal.json` (#158's zero-copy check), `id_*.json`, `id_sha.txt` |
| The nightly gate's reds were the runner's CPU model, not the commit (P8 #165, 2026-10-22) | `p8-2026-10-22/` | `nightly/runs-by-host.csv` (40 runs by CPU model), `nightly/<run>/bench.json` + `bench-gate.log` (the four runs of the incident), `m5/bench-<sha>.json` (the three commits on one host), `m5/mlp4-*.json` (the quiet re-measure) |

The 2026-09-15 GPU baseline row, the E1 GEMM row and the E11 flash
decision predate the ledger and are not in it; their `out/` directories
were not kept.
