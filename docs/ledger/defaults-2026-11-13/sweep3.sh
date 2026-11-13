#!/bin/sh
# The candidate default (lr 2e-3, cosine, warmup 15) held to more seeds and
# ranks, against each row's own baseline (alpha / r, lr 1e-3 constant):
# seed 7 3.6369, seed 123 3.6439 (rank 8), rank 4 3.6802, rank 16 3.6040.
# And one #190 row: the global clip at 1.0 (rank 8, seed 42; baseline 3.6396).
WT=~/.claude/jobs/8eaa3413/tmp/snap2
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/sweep
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
COMPILER=$WT/build/seeml-update-compile
cd "$WT"
pmset -g batt | head -2
row() {
  name=$1; rank=$2; alpha=$3; shift 3
  d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  "$COMPILER" --source "$SRC/smollm.smf" --out "$d/pkg" --data "$FRONTIER/docs_corpus.txt" --val-frac 0.1 --data-batch 512 --lora-rank "$rank" --lora-alpha "$alpha" --quantize-base --steps 300 --report "$d/compile.json" --build "$@" > "$d/compile.log" 2>&1
  rc=$?
  echo "$name compile rc=$rc"
  [ $rc -eq 0 ] || return
  "$d/pkg/model_update" --model "$SRC/smollm.smf" --data "$d/pkg/corpus.sds" --out "$d/updated.smf" --backend metal --force --eval-every 30 --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep 'best state at' "$d/run.log"
  rm -rf "$d/updated.smf" "$d/updated.smf.export.json" "$d/pkg/update_plan_embedded.cc" "$d/pkg/update_plan.seeu"
}
C="--lr 2e-3 --lr-schedule cosine --warmup 15"
row r8_lr2e3cos_s7 8 16 $C --lora-seed 7
row r8_lr2e3cos_s123 8 16 $C --lora-seed 123
row r16_lr2e3cos 16 32 $C
row r4_lr2e3cos 4 8 $C
row r8_gclip1 8 16 --clip-mode global --clip-norm 1.0
echo "sweep3 done"
