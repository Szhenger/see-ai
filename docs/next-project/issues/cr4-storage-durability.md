---
title: "SeeAI CR4: storage and egress durability — checkpoint metadata dropped on save, partial temp files left on failure, a writer that mutates offsets before a failable write, runners that drop report write failures"
number: 145
labels: bug,correctness,core-plane
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

The durability story has edges the review reached: `has_binding` is dropped
on checkpoint save so v5 loads report true (F71, previously refuted, now
real); `WriteFileDurable` leaves the partial temp file on every failure
path (F08); `SaveSmf` changes the model's offsets before a write that can
fail, so a failed save leaves new offsets paired with the old hash (F05);
`export_smf` leaves a truncated file when a non-CPU tensor write fails
(F55); the generated runner drops `--loss-log` open failures and `--report`
write errors, so a full disk leaves truncated JSON with exit 0 (F66);
`--attention-sweep` writes unstaged and ignores write errors (F61). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §6
(storage), §8.

## Design

One rule for every writer on both planes: stage to a temp file, check every
write and close, replace atomically, unlink the temp on every error path,
and commit in-memory state only after the file is durable. Checkpoint
header round-trip tested field by field.

## Status 2026-11-05

Landed on `claude/v100a-cr` (#178) under one rule for every writer on both
planes — stage beside the destination, check every write and the close,
replace atomically, remove the stage on every error path, commit in-memory
state only once the file is in place — and one flag the checkpoint lacked.
F71: `kCkptHasBinding` (bit 4 of the v5 flags) is written by the engine from
its run binding and read by the load and the peek; before it every v5 file
read as bound, and one saved through the v3/v4-shaped call made a resume
refuse any real seed or split against zeros (a v5 file written before the
bit reads as unbound). F08: `WriteFileDurable` removes its sidecar on every
failure past the open, checks the close, and `FailNextDurableWriteForTesting`
injects the one failure a test can name (ENOSPC). F05: `SaveSmf` lays the
data out into a local table, stages the file (every status checked, fsync'd),
renames it, and only then assigns the model's offsets, sizes and hash. F55:
the exporter's `_atomic_output` stages, syncs and replaces every file it
writes. F66: the generated runner checks `--loss-log` and stages `--report`,
and a failed output after a commit exits 4 (the usage text lists the codes;
the e2e job runs the built runner against an unwritable report and expects
the 4 and the same committed bytes). F61: the bench's attention sweep stages
its report like the main one. Each rule was removed in turn and its test
failed (the table is in the PR). Not here: the storage role's screen line
(bytes written, fsyncs per checkpoint) is S0's; a write that fails midway
through `SaveSmf` has no injection seam on the compiler plane (the open
failure is what its test exercises).

## Acceptance

- Each reproduction fixed with a test that injects the failure (full disk,
  failed write, kill mid-save).
- The storage role's screen line reports bytes written and fsyncs per
  checkpoint.
