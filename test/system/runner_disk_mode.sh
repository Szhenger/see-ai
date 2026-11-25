#!/bin/sh
# The device's main, built by the tree and run end to end (review
# 2026-11-25): runtime/host/update_main.cc was compiled only inside emitted
# packages, so no sanitizer job, no -Werror cell and no static analysis ever
# saw it. build/seeml-model-update is that file in the disk mode (no
# embedded plan; --plan names one). This script compiles the golden MLP,
# runs the runner over the golden corpus and holds it to the exit-code
# table: 0 committed or 3 rejected by the gate for a real run, 1 for a plan
# that is not one, 2 for a bad command line. Under the sanitizer build
# (ctest's seeml_runner_disk_mode) any report fails the run.
#
#   sh test/system/runner_disk_mode.sh COMPILER RUNNER REPO_ROOT
set -eu
COMPILER=$1
RUNNER=$2
REPO=$3
TMP=$(mktemp -d "${TMPDIR:-/tmp}/seeml-runner-disk.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
GOLDEN="$REPO/test/fixtures/golden"

"$COMPILER" --source "$GOLDEN/mlp.smf" --out "$TMP/pkg" --data-batch 2 \
  --loss xent --lora-rank 2 --lr 1e-2 --lr-schedule const --steps 4 \
  --no-embed > "$TMP/compile.log" 2>&1 || {
  cat "$TMP/compile.log"; echo "runner_disk_mode: the compile failed"; exit 1; }

expect() {
  want=$1; shift
  set +e
  "$@" > "$TMP/run.log" 2>&1
  got=$?
  set -e
  case " $want " in
    *" $got "*) ;;
    *) cat "$TMP/run.log"
       echo "runner_disk_mode: exit $got, expected one of: $want ($*)"
       exit 1 ;;
  esac
}

# A real run: trained, gated, and committed or left untouched.
expect "0 3" "$RUNNER" --plan "$TMP/pkg/update_plan.seeu" \
  --model "$GOLDEN/mlp.smf" --data "$GOLDEN/class.sds" \
  --out "$TMP/updated.smf" --report "$TMP/report.json" --threads 2
test -s "$TMP/report.json"
# Forced, so the merge and the durable commit run whatever the gate said.
expect "0" "$RUNNER" --plan "$TMP/pkg/update_plan.seeu" \
  --model "$GOLDEN/mlp.smf" --data "$GOLDEN/class.sds" \
  --out "$TMP/forced.smf" --force
test -s "$TMP/forced.smf"
# A plan that is not a plan is a runtime error, by name.
printf 'not a plan' > "$TMP/garbage.seeu"
expect "1" "$RUNNER" --plan "$TMP/garbage.seeu" --model "$GOLDEN/mlp.smf" \
  --data "$GOLDEN/class.sds" --out "$TMP/never.smf"
test ! -e "$TMP/never.smf"
# A bad command line is 2.
expect "2" "$RUNNER" --plan "$TMP/pkg/update_plan.seeu" --no-such-flag
expect "0" "$RUNNER" --help
echo "runner_disk_mode: ok"
