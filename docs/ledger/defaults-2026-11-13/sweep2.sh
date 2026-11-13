#!/bin/sh
# #190 acceptance 3 and S5 #151: the global clip measured on SmolLM-135M at
# the #151 setting (300 steps, the split scored every 30 steps), against no
# clip (3.6396) and the per-tensor clip at 1.0 (3.6396).
WT=~/.claude/jobs/8eaa3413/tmp/snap2
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/sweep
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
COMPILER=$WT/build/seeml-update-compile
mkdir -p "$OUT"
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
row r8_gclip1 8 16 --clip-mode global --clip-norm 1.0
row r8_gclip01 8 16 --clip-mode global --clip-norm 0.1
row r8_gclip001 8 16 --clip-mode global --clip-norm 0.01
echo "sweep2 done"
