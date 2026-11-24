#!/bin/sh
# #191: the segment mask's seeds. For each corpus of run2.sh (its
# related.jsonl / unrelated.jsonl, reused as written) and each adapter seed
# given, train unmasked and masked at the #151 setting, then score the
# unmasked-trained model under the mask. The masked-trained model's own
# committed loss is already its masked score (run2.sh's diagonal reproduces
# it to the last digit). The comparison is the "scored masked" column: one
# document in its context, the model as it is used.
# Losses only — the host is on battery.
set -u
WT="$1"; OUT="$2"; SMF="$3"; shift 3
PY="${PY:-python3}"
cd "$WT" || exit 1
train() {  # corpus, seed, row name, extra flags
  c=$1; s=$2; name=$3; shift 3
  d="$OUT/$c/s$s/train_$name"; rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$SMF" --out "$d/pkg" \
    --data "$OUT/$c.jsonl" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --lora-seed "$s" --quantize-base --steps 300 --no-embed "$@" \
    > "$d/compile.log" 2>&1
  rc=$?; "$PY" -B tool/pack_update.py "$d/pkg" --build >> "$d/compile.log" 2>&1 || rc=9
  "$d/pkg/model_update" --model "$SMF" --data "$d/pkg/corpus.sds" \
    --out "$OUT/$c/s$s/$name.smf" --backend metal --force --eval-every 30 \
    > "$d/run.log" 2>&1
  echo "$c seed $s train $name compile rc=$rc run rc=$? :: $(grep -E 'best state at' "$d/run.log" | cut -c15-80)"
  rm -rf "$d/pkg"
}
score_masked() {  # corpus, seed, trained model name
  c=$1; s=$2; model=$3
  d="$OUT/$c/s$s/score_${model}_as_masked"; rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$OUT/$c/s$s/$model.smf" --out "$d/pkg" \
    --data "$OUT/$c.jsonl" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --lr-schedule const --steps 1 --no-embed \
    --segment-mask on > "$d/compile.log" 2>&1
  rc=$?; "$PY" -B tool/pack_update.py "$d/pkg" --build >> "$d/compile.log" 2>&1 || rc=9
  "$d/pkg/model_update" --model "$OUT/$c/s$s/$model.smf" --data "$d/pkg/corpus.sds" \
    --out "$d/discard.smf" --backend metal --force --steps 1 --eval-every 0 \
    > "$d/run.log" 2>&1
  echo "$c seed $s score trained=$model scored=masked compile rc=$rc run rc=$? :: $(grep -E '^seeml-update: validation loss' "$d/run.log" | tail -1 | cut -c15-50)"
  rm -rf "$d/discard.smf" "$d/discard.smf.export.json" "$d/discard.smf.lock" "$d/pkg"
}
for s in "$@"; do
  for c in related unrelated; do
    mkdir -p "$OUT/$c/s$s"
    train $c $s unmasked
    train $c $s masked --segment-mask auto
    score_masked $c $s unmasked
    rm -f "$OUT/$c/s$s"/*.smf "$OUT/$c/s$s"/*.lock "$OUT/$c/s$s"/*.export.json
  done
done
echo "done"
