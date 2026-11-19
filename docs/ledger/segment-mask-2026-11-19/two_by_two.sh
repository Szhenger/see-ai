#!/bin/sh
# #191: the segment mask's 2 x 2 — train {unmasked, masked} x score
# {unmasked, masked} — on two corpora:
#   related:    this repository's docs, one paragraph a document (the
#               2026-11-18 row's corpus);
#   unrelated:  Python standard-library docstrings, one a document, in a
#               seeded shuffle, so neighbours in a window have nothing to do
#               with each other.
# SmolLM-135M, the #151 setting: rank 8, 512 tokens a step, int8 base, 300
# steps, a compiled split of a tenth scored every 30 steps, Metal. Each
# committed model is then scored on the same held-out records under both
# attentions: a plan compiled from the committed model (f32 base, constant
# schedule) reports the held-out loss before its first step.
# "Scored masked" is the model as it is used: one document in its context.
# Losses only — the host is on battery; no time here is a measurement.
set -u
WT="$1"; OUT="$2"; SMF="$3"; DOCS="$4"
PY="${PY:-python3}"
mkdir -p "$OUT"
cd "$WT" || exit 1
pmset -g batt | head -2 > "$OUT/power.txt"

"$PY" -B - "$DOCS" "$OUT/related.jsonl" "$OUT/unrelated.jsonl" <<'EOF'
import importlib, inspect, json, pkgutil, random, sys, warnings
warnings.simplefilter("ignore")
text = open(sys.argv[1], encoding="utf-8").read()
paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 40]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for p in paras:
        f.write(json.dumps({"text": p}) + "\n")
# Standard-library docstrings: one document each, 200..1200 characters,
# deduplicated, shuffled with a fixed seed, capped near the related
# corpus's size so the two rows train on comparable amounts of text.
names = sorted(m.name for m in pkgutil.iter_modules()
               if m.name in sys.stdlib_module_names and not m.name.startswith("_")
               # Not these: importing them prints, opens a browser or a window.
               and m.name not in ("this", "antigravity", "idlelib", "turtledemo",
                                  "tkinter", "turtle"))
docs, seen = [], set()
for name in names:
    try:
        mod = importlib.import_module(name)
    except BaseException:
        continue
    for _, obj in sorted(vars(mod).items()):
        if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj)):
            continue
        if getattr(obj, "__module__", None) != name:
            continue
        d = inspect.getdoc(obj)
        if d and 200 <= len(d) <= 1200 and d not in seen:
            seen.add(d)
            docs.append(d)
random.Random(191).shuffle(docs)
budget, kept, used = sum(len(p) for p in paras), [], 0
for d in docs:
    if used >= budget:
        break
    kept.append(d)
    used += len(d)
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for d in kept:
        f.write(json.dumps({"text": d}) + "\n")
print("related documents:", len(paras), "chars", budget)
print("unrelated documents:", len(kept), "of", len(docs), "chars", used)
EOF

train() {  # corpus name, row name, extra compile flags
  c=$1; name=$2; shift 2
  d="$OUT/$c/train_$name"; rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$SMF" --out "$d/pkg" \
    --data "$OUT/$c.jsonl" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --quantize-base --steps 300 --build "$@" > "$d/compile.log" 2>&1
  echo "$c train $name compile rc=$?"
  grep -E 'segment mask  |straddling|validation   ' "$d/compile.log" | cut -c1-120
  "$d/pkg/model_update" --model "$SMF" --data "$d/pkg/corpus.sds" \
    --out "$OUT/$c/$name.smf" --backend metal --force --eval-every 30 \
    > "$d/run.log" 2>&1
  echo "$c train $name run rc=$?"
  grep -E 'best state at|^seeml-update: validation loss' "$d/run.log" | cut -c1-150
  rm -rf "$d/pkg/update_plan_embedded.cc" "$d/pkg/update_plan.seeu" "$d/pkg/runtime" "$d/pkg/source"
}
score() {  # corpus name, trained model name, scoring name, extra flags
  c=$1; model=$2; name=$3; shift 3
  d="$OUT/$c/score_${model}_as_$name"; rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$OUT/$c/$model.smf" --out "$d/pkg" \
    --data "$OUT/$c.jsonl" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --lr-schedule const --steps 1 --no-embed "$@" > "$d/compile.log" 2>&1
  rc=$?
  # An f32 plan of this model is too large to embed as a C array: the
  # packer embeds the bytes instead.
  "$PY" -B tool/pack_update.py "$d/pkg" --build >> "$d/compile.log" 2>&1 || rc=9
  "$d/pkg/model_update" --model "$OUT/$c/$model.smf" --data "$d/pkg/corpus.sds" \
    --out "$d/discard.smf" --backend metal --force --steps 1 --eval-every 0 \
    > "$d/run.log" 2>&1
  echo "$c score trained=$model scored=$name compile rc=$rc run rc=$? :: $(grep -E '^seeml-update: validation loss' "$d/run.log" | tail -1 | cut -c1-90)"
  rm -rf "$d/discard.smf" "$d/discard.smf.export.json" "$d/discard.smf.lock" "$d/pkg"
}
for c in related unrelated; do
  mkdir -p "$OUT/$c"
  train $c unmasked
  train $c masked --segment-mask auto
  for model in unmasked masked; do
    score $c $model unmasked
    score $c $model masked --segment-mask on
  done
  rm -f "$OUT/$c/unmasked.smf" "$OUT/$c/masked.smf" "$OUT/$c"/*.smf.lock
done
echo "done"
