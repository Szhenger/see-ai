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
| The relaxed int8 CPU step past 0.8× of `torch.compile`, with the same bits (F4 #132, 2026-10-23) | `f4-2026-10-23/` | `sha_threads.txt` (the four thread settings and the rebuilt runtime), `row_new_*.json` (the harness rows), `gemm_profile.json` + `gemm_table.md` (the probe's per-shape profile), `sgemm_panels.txt` (the SGEMM microbenchmark), `cpu_samples.txt` + `main_thread.txt` (the `sample` summaries) |
| Dispatch overhead of the eval and merge programs, and `dec_wide` (F8 #139, 2026-10-23) | `f8-2026-10-23/` | `spans_smollm.txt`, `spans_dec_wide.txt`, `report_*.json` (the two compile reports) |
| The relaxed package on the F7-era kernels, beside MLX-LM in the same session (F2 #130 / F7 #138, 2026-10-24) | `f2-2026-10-23/` | `row_exact*.json`, `row_relaxed*.json` (SeeAI rows), `row_f7b_1019pkg.json` (the 10-19 package as the host control), `row_mlx_bf16.json` + `row_mlx_f32.json` (MLX-LM), `numerics_certificate.json` + `certify.log` (granted, 10-15 settings), `*_defaults_refused.*` (refused, tool defaults), `report_*.json`; `disturbed/` (the rows taken while the machine slept, quoted nowhere) |
| The certificate contract, decided (F2 #130, 2026-10-30) | `f2-2026-10-30/` | `numerics_certificate.json` + `certify.log` (the 10-24 plan granted at the defaults under the flagged contract), `verify.log` + `pack.log` + `pack_report.json` (the branch tip's verify and build path accept it), `report.json` (the recompile, plan hash `a0d2d967…`) |
| The dec_wide row, and the frontier rows on GitHub's hosted Apple silicon (F1 #129, 2026-10-30) | `f1-2026-10-30/`, `f1-2026-10-30/hosted/` | `frontier.json` + `plan.json` (the M5 matrix), `report_dec_wide.json`, `export.log` (the `--hf-parity` line); `hosted/frontier_smollm.json`, `hosted/frontier_dec_wide.json`, `hosted/plan_*.json`, `hosted/probe.json` (dispatch 41093112857 on the hosted VM) |
| The relaxed tiled attention family: certified, then measured at S = 2048 (F3 #131 design item 2, 2026-10-30) | `f3-2026-10-30/` | `numerics_certificate.json` + `certify.log` (dec_wide tiled, granted at the defaults), `diff_metal.json` + `verify.log` (the Metal diff `verify --observed` accepts), `report_dec_wide_tiled_relaxed.json`; `time_s2048_*.json` + `profile_s2048_tiled*.json` (the probe rows, the shipped kernels' window), `window2/` (the earlier window, the first-form kernels), `diff_metal_shipped.json` (the diff again on the shipped kernels), `report_s2048_*.json`, `row_mlx_s2048_*.json` (MLX-LM, same session) |
| The nightly gate's reds were the runner's CPU model, not the commit (P8 #165, 2026-10-22) | `p8-2026-10-22/` | `nightly/runs-by-host.csv` (40 runs by CPU model), `nightly/<run>/bench.json` + `bench-gate.log` (the four runs of the incident), `m5/bench-<sha>.json` (the three commits on one host), `m5/mlp4-*.json` (the quiet re-measure) |

`bench-epochs/` is not a section's evidence but the nightly gate's
pinned reference: one `seeml-bench` report per runner host, named by
`tool/bench_compare.py --host-slug`, each a calibrated green night
copied from this ledger (the AMD EPYC 7763's 2026-10-22 run 39782901348
and the 9V74's 2026-10-19 run 39271309779, from `p8-2026-10-22/nightly/`).
The nightly reads it and never writes it; replacing a file is starting
a new epoch for that host, deliberately, in a commit. After a kernel
policy or calibration kernel change the nightly sets a host's file aside
with a notice until it is replaced.

The 2026-09-15 GPU baseline row, the E1 GEMM row and the E11 flash
decision predate the ledger and are not in it; their `out/` directories
were not kept.
