#!/bin/sh
# The defaults ablation (S5 #151 item 4): SmolLM-135M, the nightly's setting
# (batch 512 x 128, rank 8, alpha 16, int8 base, 300 steps), docs corpus with
# a compiled 10 % held-out split, Metal backend. One row per candidate default.
WT=~/SeeAI/see-ai/.claude/worktrees/review-2026-11-09
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/ablate
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
mkdir -p "$OUT"
cd "$WT"
pmset -g batt | head -2
row() {
  name=$1; shift
  d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --lora-rank 8 --lora-alpha 16 --quantize-base --steps 300 --report "$d/compile.json" --build "$@" > "$d/compile.log" 2>&1
  echo "$name compile rc=$?"
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep 'validation loss' "$d/run.log" | head -2
  rm -f "$d/updated.smf" "$d/updated.smf.export.json"
}
row baseline
row clip --clip-norm 1.0
row lr1e4cos --lr 1e-4 --lr-schedule cosine --warmup 15
row lr2e4cos --lr 2e-4 --lr-schedule cosine --warmup 15
row cos_only --lr-schedule cosine --warmup 15
row rslora --lora-scaling alpha_over_sqrt_r
row all --clip-norm 1.0 --lr 1e-4 --lr-schedule cosine --warmup 15 --lora-scaling alpha_over_sqrt_r
row all2e4 --clip-norm 1.0 --lr 2e-4 --lr-schedule cosine --warmup 15 --lora-scaling alpha_over_sqrt_r
# Determinism: the baseline package once more.
d="$OUT/baseline"
"$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force > "$d/run2.log" 2>&1
echo "baseline rerun rc=$?"
grep 'validation loss' "$d/run2.log" | head -2
rm -f "$d/updated.smf" "$d/updated.smf.export.json"
echo "ablate done"
