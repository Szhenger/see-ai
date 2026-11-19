#!/bin/sh
# #191 design item 3: the segment mask's measured row. SmolLM-135M, the
# #151 setting (rank 8, 512 tokens a step, int8 base, 300 steps, the split
# scored every 30), over the repository's docs as one document per
# paragraph, so windows hold several documents. Masked (auto decides on)
# against unmasked (off, the default). Losses only: the host is on battery.
# The logs beside this script were taken while `auto` was still the default
# under test (the masked row with no flag, the unmasked one with
# --segment-mask off); the rows below name the flags as they are now.
set -u
WT="$1"; OUT="$2"
PY=~/SeeAI/see-ai/.venv/bin/python
SNAP=~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/1d461723eec654e65efdc40cf49301c89c0c92f4
DOCS=~/SeeAI/see-ai/out/frontier-2026-09-22/docs_corpus.txt
mkdir -p "$OUT"
cd "$WT" || exit 1
pmset -g batt | head -2 > "$OUT/power.txt"
if [ ! -f "$OUT/smollm.smf" ]; then
  "$PY" -B tool/export_model.py --hf "$SNAP" "$OUT/smollm.smf" --seq-len 128 > "$OUT/export.log" 2>&1
  echo "export rc=$?"
fi
"$PY" -B - "$DOCS" "$OUT/docs.jsonl" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 40]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for p in paras:
        f.write(json.dumps({"text": p}) + "\n")
print("paragraph documents:", len(paras))
EOF
row() {
  name=$1; shift
  d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"
  build/seeml-update-compile --source "$OUT/smollm.smf" --out "$d/pkg" \
    --data "$OUT/docs.jsonl" --val-frac 0.1 --data-batch 512 --lora-rank 8 \
    --lora-alpha 16 --quantize-base --steps 300 --report "$d/compile.json" \
    --build "$@" > "$d/compile.log" 2>&1
  echo "$name compile rc=$?"
  grep -E 'segment mask|straddling|validation  |records' "$d/compile.log" | cut -c1-150
  "$d/pkg/model_update" --model "$OUT/smollm.smf" --data "$d/pkg/corpus.sds" \
    --out "$d/updated.smf" --backend metal --force --eval-every 30 \
    --report "$d/run.json" > "$d/run.log" 2>&1
  echo "$name run rc=$?"
  grep -E 'best state at|validation loss' "$d/run.log" | cut -c1-160
  rm -rf "$d/updated.smf" "$d/updated.smf.export.json" "$d/updated.smf.lock" \
    "$d/pkg/update_plan_embedded.cc" "$d/pkg/update_plan.seeu"
}
row masked_auto --segment-mask auto
row unmasked_off
echo "done"
