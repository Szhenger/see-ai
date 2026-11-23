#!/bin/sh
# #192: 8-bit AdamW moments against f32 moments, losses only. SmolLM-135M at
# the #151 setting — rank 8 (alpha 16), 512 tokens a step, int8 base, 300
# steps, a compiled split of a tenth scored every 30 steps — on the related
# corpus of the #191 rows (this repository's docs, one paragraph a document;
# docs/ledger/segment-mask-*/run2.sh wrote it), the plans compiled with
# --no-embed and packed by tool/pack_update.py --build (seconds; embedding a
# 135 MB plan as a C array takes minutes). For each adapter seed and each
# backend named, the same plan but for --moments: f32 (the default) and
# int8. The host was on battery in Low Power Mode: no time here is a
# measurement, and none is recorded.
#   sh run.sh WORKTREE OUT SMF CORPUS.jsonl "SEEDS" "BACKENDS"
set -u
WT="$1"; OUT="$2"; SMF="$3"; CORPUS="$4"; SEEDS="$5"; BACKENDS="$6"
PY="${PY:-python3}"
COMPILER="${COMPILER:-build/seeml-update-compile}"
mkdir -p "$OUT"
cd "$WT" || exit 1
pmset -g batt | head -2 > "$OUT/power.txt"
train() {  # seed, backend, moments
  s=$1; b=$2; mom=$3
  d="$OUT/s$s/$b/$mom"; rm -rf "$d"; mkdir -p "$d"
  "$COMPILER" --source "$SMF" --out "$d/pkg" \
    --data "$CORPUS" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --lora-seed "$s" --quantize-base --steps 300 \
    --moments "$mom" --no-embed --verbose > "$d/compile.log" 2>&1
  rc=$?
  "$PY" -B tool/pack_update.py "$d/pkg" --build >> "$d/compile.log" 2>&1 || rc=9
  "$d/pkg/model_update" --model "$SMF" --data "$d/pkg/corpus.sds" \
    --out "$d/out.smf" --backend "$b" --force --eval-every 30 \
    --loss-log "$d/loss.csv" > "$d/run.log" 2>&1
  echo "seed $s $b moments $mom compile rc=$rc run rc=$? :: $(grep -E 'best state at' "$d/run.log" | cut -c15-90)"
  rm -rf "$d/pkg" "$d/out.smf" "$d/out.smf.lock" "$d/out.smf.export.json"
}
for b in $BACKENDS; do
  for s in $SEEDS; do
    train "$s" "$b" f32
    train "$s" "$b" int8
  done
done
echo "done"
