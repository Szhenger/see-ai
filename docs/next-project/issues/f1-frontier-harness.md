---
title: "SeeAI F1: measured frontier harness — real mlx_lm.lora and torch.compile runs on the same host, model, corpus and tokens/step, plus a nightly frontier row"
number: 129
labels: enhancement,python-plane,frontier-parity
plane: Python plane
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P0
---
## Why

Every frontier comparison SeeAI has today is either from another host
(`mlx_lm.lora` ~1,600 tok/s on an M4, `torch.compile` ~40% CPU MFU, the
v1.2.4 field report) or a **floor**: `tool/frontier_exec.py price`
interprets a plan op by op with no `mx.compile`, no `torch.compile` and no
fusion. The 2026-09-18 Frontier Outlook therefore had to *speculate* every
frontier cell (SmolLM-135M GPU: SeeAI 1,749 tok/s measured vs MLX-LM
2,200–3,200 guessed). The rest of this milestone (F2–F6) is priced against
those guesses; it must be priced against measurements instead.

## Design

1. `tool/frontier_run.py` (python plane, optional tier — needs mlx-lm /
   torch, never imported by the core tools): given an SMF model or an HF
   directory, a corpus and `--seq-len/--batch/--rank`, runs
   - `mlx_lm.lora` with the matching LoRA targets, rank, alpha, batch,
     sequence length and step count, in both `MLX_ENABLE_TF32=0` (f32) and
     the default mode (one process each — MLX latches the switch);
   - a hand-matched PyTorch LoRA loop under `torch.compile` on `cpu` and
     `mps` (inductor on MPS where it compiles; eager MPS otherwise, labelled);
   - SeeAI `model_update` on `cpu` and `metal` from the same package.
2. One report per run in `seeml-bench`'s units and definitions (`tokens_per_s`,
   `it_per_s`, `step_ms` by steps-regression, `peak_rss_bytes`, train loss
   first/last, precision mode named on every row), schema-versioned; the
   loss trajectories must agree within a stated tolerance or the row is
   refused (a faster run that trains a different function is not a number).
3. `nightly.yml`: an opt-in **frontier row** on a self-hosted Apple-silicon
   runner (the Linux runner cannot run MLX / MPS / Metal) for
   `tok_smollm135m_q8` and `dec_wide`; trend only, never a merge gate.
4. `docs/benchmarks.md`: replace the carried M4 anchors with the measured
   same-host table; the Frontier Outlook's asterisks become numbers.

## Acceptance

- [x] Same-host table for SmolLM-135M (r8, S=128, 512 tok/step): SeeAI
  cpu/metal, MLX-LM true-f32/all-bf16, torch.compile cpu, torch mps —
  `docs/benchmarks.md`, "The frontier row, measured (2026-09-22)" and
  "The nightly frontier row's recipe, run by hand (2026-10-15)". (MLX-LM's
  *default* mode on this F32-stored checkpoint is f32 under TF32-class
  matmul, so the rows are labelled true-f32 and all-bf16 instead.)
- [ ] The `dec_wide` column of that table — never run through the harness.
- [x] Every row names device, precision mode, library versions and host
  key — the harness's schema-versioned rows (`tool/frontier_run.py`).
- [x] Loss-agreement check fails loudly on a mismatched configuration —
  `test/tool/frontier_run_test.py`, `test_a_wrong_rank_is_refused_by_name`.
- [ ] The nightly frontier row (design item 3) — the job is in
  `nightly.yml` and has been skipped on every run since 2026-09-23 (29 as
  of 2026-10-22). Of the two things that kept it off, one closed on
  2026-10-29: its `if:` now also admits the schedule while the repository
  variable `SEEML_FRONTIER_NIGHTLY` is `true`, taking the checkpoint from
  the runner's `SEEML_FRONTIER_MODEL_DIR`. The other stands: no runner
  labelled `self-hosted, macOS, ARM64` is registered, and the variable
  must stay unset until one is (a scheduled job with no runner queues
  until GitHub times it out). Its recipe was run by hand on 2026-10-15
  (two workflow bugs fixed from that run). Blocked on the runner, or on
  dropping the job.
