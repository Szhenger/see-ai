#!/bin/sh
# The 1,200-step rows with the split scored every 30 steps (S5 #151): the two
# packages already built are re-run; the other two are compiled first.
WT=~/SeeAI/see-ai/.claude/worktrees/review-2026-11-09
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/ablate
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
COMPILER=~/.claude/jobs/8eaa3413/tmp/frozen/seeml-update-compile
cd "$WT"
pmset -g batt | head -2
build() {
  d="$OUT/$1"; shift
  rm -rf "$d"; mkdir -p "$d"
  "$COMPILER" --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --lora-rank 8 --lora-alpha 16 --quantize-base --steps 1200 --report "$d/compile.json" --build "$@" > "$d/compile.log" 2>&1
  rc=$?
  echo "$(basename "$d") compile rc=$rc"
}
run() {
  d="$OUT/$1"
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force --eval-every 30 --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$1 run rc=$?"
  grep 'best state at' "$d/run.log"
  rm -f "$d/updated.smf" "$d/updated.smf.export.json"
}
build h1200_lr2e4cos --lr 2e-4 --lr-schedule cosine --warmup 60
build h1200_lr1e4const --lr 1e-4
echo "builds done"
run h1200_base
run h1200_lr1e4cos
run h1200_lr2e4cos
run h1200_lr1e4const
echo "ablate5 done"
