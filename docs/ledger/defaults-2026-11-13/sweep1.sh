#!/bin/sh
# S5 #151, the defaults search, sweep 1: SmolLM-135M, nightly shapes, 300
# steps, the split scored every 30 steps, Metal. Each row: name rank alpha
# then extra compiler flags. Baselines (alpha / r, lr 1e-3 const, seed 42):
# r4 3.6802, r8 3.6396, r16 3.6040 (committed).
WT=~/.claude/jobs/8eaa3413/tmp/snap
SRC=~/.claude/jobs/8eaa3413/tmp/accept_s6
OUT=~/.claude/jobs/8eaa3413/tmp/sweep
FRONTIER=~/SeeAI/see-ai/out/frontier-2026-09-22
COMPILER=~/.claude/jobs/8eaa3413/tmp/snap/build/seeml-update-compile
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
row r8_lr5e4 8 16 --lr 5e-4
row r8_lr2e3 8 16 --lr 2e-3
row r8_lr2e3cos 8 16 --lr 2e-3 --lr-schedule cosine --warmup 15
row r8_wd01 8 16 --weight-decay 0.1
row r8_rs_lr5e4 8 16 --lora-scaling alpha_over_sqrt_r --lr 5e-4
row r8_rs_lr3e4 8 16 --lora-scaling alpha_over_sqrt_r --lr 3e-4
row r8_rs_lr5e4cos 8 16 --lora-scaling alpha_over_sqrt_r --lr 5e-4 --lr-schedule cosine --warmup 15
row r4_rs_lr5e4 4 8 --lora-scaling alpha_over_sqrt_r --lr 5e-4
row r16_rs_lr5e4 16 32 --lora-scaling alpha_over_sqrt_r --lr 5e-4
row r4_rs_lr3e4 4 8 --lora-scaling alpha_over_sqrt_r --lr 3e-4
row r16_rs_lr3e4 16 32 --lora-scaling alpha_over_sqrt_r --lr 3e-4
echo "sweep1 done"
