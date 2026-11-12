#!/bin/sh
# alpha / r vs alpha / sqrt(r) at rank 16, alpha 32 (scale 2 vs 8), and at
# rank 8 with alpha / r at alpha 45.2548 (the same scale 5.657 as the rank-8
# alpha / sqrt(r) row: the same program but for the recipe).
WT=~/SeeAI/see-ai/.claude/worktrees/review-2026-11-09
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/ablate
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
cd "$WT"
row() {
  name=$1; shift
  d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --quantize-base --steps 300 --build "$@" > "$d/compile.log" 2>&1
  echo "$name compile rc=$?"
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep 'validation loss [0-9.]* ->' "$d/run.log"
  rm -f "$d/updated.smf" "$d/updated.smf.export.json"
}
row r16_base --lora-rank 16 --lora-alpha 32 --lora-scaling alpha_over_r
row r16_rs --lora-rank 16 --lora-alpha 32 --lora-scaling alpha_over_sqrt_r
row r4_base --lora-rank 4 --lora-alpha 8 --lora-scaling alpha_over_r
row r4_rs --lora-rank 4 --lora-alpha 8 --lora-scaling alpha_over_sqrt_r
echo "ablate3 done"
