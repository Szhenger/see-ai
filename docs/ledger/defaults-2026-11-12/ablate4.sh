#!/bin/sh
# The learning-rate claim at a longer horizon (S5 #151): 1,200 steps, warmup
# scaled to 5 % (60), lr 1e-4 / 2e-4 cosine against the baseline (lr 1e-3
# constant) and against lr 1e-4 constant (the nightly's own F1 setting).
# The compiler binary is a frozen copy so the tree can change meanwhile.
WT=~/SeeAI/see-ai/.claude/worktrees/review-2026-11-09
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/ablate
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
COMPILER=~/.claude/jobs/8eaa3413/tmp/frozen/seeml-update-compile
cd "$WT"
pmset -g batt | head -2
row() {
  name=$1; shift
  d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  "$COMPILER" --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --lora-rank 8 --lora-alpha 16 --quantize-base --steps 1200 --report "$d/compile.json" --build "$@" > "$d/compile.log" 2>&1
  echo "$name compile rc=$?"
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force --eval-every 30 --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep 'best state at' "$d/run.log"
  rm -f "$d/updated.smf" "$d/updated.smf.export.json"
}
row h1200_base
row h1200_lr1e4cos --lr 1e-4 --lr-schedule cosine --warmup 60
row h1200_lr2e4cos --lr 2e-4 --lr-schedule cosine --warmup 60
row h1200_lr1e4const --lr 1e-4
echo "ablate4 done"
