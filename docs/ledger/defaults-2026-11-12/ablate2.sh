#!/bin/sh
# Baseline vs alpha / sqrt(r) on two more adapter seeds (the first run used 42).
WT=~/SeeAI/see-ai/.claude/worktrees/review-2026-11-09
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/ablate
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
cd "$WT"
row() {
  name=$1; shift
  d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --lora-rank 8 --lora-alpha 16 --quantize-base --steps 300 --build "$@" > "$d/compile.log" 2>&1
  echo "$name compile rc=$?"
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep 'validation loss [0-9.]* ->' "$d/run.log"
  rm -f "$d/updated.smf" "$d/updated.smf.export.json"
}
row base_s7 --lora-seed 7
row rs_s7 --lora-seed 7 --lora-scaling alpha_over_sqrt_r
row base_s123 --lora-seed 123
row rs_s123 --lora-seed 123 --lora-scaling alpha_over_sqrt_r
echo "ablate2 done"
