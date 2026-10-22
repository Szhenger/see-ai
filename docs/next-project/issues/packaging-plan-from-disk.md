---
title: "SeeAI packaging: load the update plan from disk — embedded plans above ~1.8 GB cannot be linked (ld segfault) or loaded (dyld shared-region overlap) on macOS arm64"
number: 158
labels: bug,design,sev:medium,core-plane
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

The package reaches its update plan only through the binary: the generated
main consumes `kSeemlUpdatePlan[]` / `kSeemlUpdatePlanSize`
(`compiler/backend/packaging/native_emitter.cc:66`, `:131`), and the emitted
`build.sh` (`native_emitter.cc:666-711`) links `update_plan_embedded.o` from
either the compiler's decimal byte-array TU or the `.incbin` stub
`tool/pack_update.py` writes, and errors out when neither exists. `--no-embed`
only chooses the stub. There is no path that loads `update_plan.seeu` from
disk, so the plan's size is the binary's size, and on macOS arm64 that
breaks at about 1.8 GB:

| package (Real Analysis Delta study, 2026-09-28) | plan | default `ld` | `-Wl,-ld_classic` |
|---|---|---|---|
| per-column int8, Qwen2.5-0.5B (largest in the study) | 1.15 GB | links, runs | — |
| f32, SmolLM2-360M | 1.85 GB | **segfault in the linker** | links, runs |
| f32, Qwen2.5-0.5B | 2.72 GB | **segfault in the linker** | links; **dyld cannot load it** |

The classic-linked 2.72 GB image is based at `0x100000000` and runs past
`0x180000000`, where dyld maps the shared cache (`dyld cache '(null)' not
loaded: syscall to map cache into shared region failed`, then every framework
"not loaded"). `-image_base` does not help: arm64 executables must be PIE,
so the base is the kernel's. The f32 SeeAI control for Qwen2.5-0.5B therefore
does not exist in the study, and any f32 or larger-model package will hit the
same wall. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md)
§5, the **packaging** row ("embedded plan, generated main, vendored runtime").

Evidence: `out/realanalysis-2026-09-25/pkg_f32_{smollm2_360,qwen25_05b}/build_ldclassic.log`,
`loadtest.err`, and the "Method and verification" bullet of the published report.

## Design

1. **A disk-resident plan as a first-class package mode.** The host entry
   (generated main) takes `--plan <path>`, defaulting to `update_plan.seeu`
   beside the binary; it maps the file read-only (it is already 64-byte
   aligned and page-alignable via `pack_update.py --page-align`) and hands the
   same `(pointer, size)` the embedded symbols provide today. Nothing
   downstream changes: the verifier, loader and dispatcher already consume a
   byte range.
2. **The plan's identity is checked either way.** The identity manifest
   (S4 #150) already hashes every input into the plan; the verifier checks
   the plan's own hash on load so a disk plan is trusted exactly as much as an
   embedded one, and a swapped file fails closed.
3. **`build.sh` never errors on a missing embed TU.** When neither
   `update_plan_embedded.{S,cc}` exists, the emitter writes a two-line TU that
   defines `kSeemlUpdatePlan` as null and `kSeemlUpdatePlanSize` as 0, and the
   host falls through to the disk path.
4. **Policy, not a flag the user has to know.** `seeml-update-compile` embeds
   when the plan is under a threshold (1 GB by default, documented against
   the measured limits above) and emits the disk mode otherwise, printing
   which it chose; `--embed` / `--no-embed` force it.
5. **Tests and docs.** A unit test for the mmap path with a small plan (hash
   check, wrong-file failure, missing-file exit code per the contract); a
   `docs/formats.md` / `docs/runtime.md` note on the two modes and the
   linker/dyld limits; the study's `run_controls_f32.sh` gains the Qwen row
   once the mode exists.


## Status 2026-10-20

Landed in #163 (`07aa9ff`), filed by hand on 2026-09-29 and given this
file on 2026-10-22 so the reconciler owns it. The generated driver maps
`update_plan.seeu` beside itself (located relative to the executable,
never the working directory) as a private, page-aligned, copy-on-write
mapping (`LoadFromFileMapped`), or the file `--plan FILE` names, and
validates it exactly as an embedded plan — magic, version gate, the seal,
the source-model hash, both contracts; a missing file is exit 1 with the
mode and both remedies named. `pack_update.py` chooses the mode by size
(`EMBED_LIMIT`, 1 GiB, a margin under the first failure), `--embed` /
`--no-embed` force it, the pack report (schema 2) records `mode`,
`mode_forced` and `embed_limit`, and the emitted `build.sh` compiles a
two-line empty-plan TU when no stub exists; the compiler prints which
mode a plan will take. Measured (`docs/usage.md`, "Very big plans"): the
f32 Qwen2.5-0.5B plan (2,724,184,064 bytes) builds in 11 s, loads,
validates and source-binds in 5.1 s, and trains and commits on the CPU —
the row the Real Analysis Delta study could not run; a disk-loaded
SmolLM-135M package reports `rodata zero-copy` on Metal and commits the
same bytes as the embedded one after 20 steps on both backends
(`docs/benchmarks.md`, "Rematerialization at three rungs").

## Acceptance

- [x] Design item 1, the disk-resident mode: `--plan FILE`, default beside
  the binary, mapped private and copy-on-write; measured on the 2.72 GB
  f32 Qwen2.5-0.5B plan.
- [x] Design item 2, identity: the disk plan is validated as an embedded
  one; a swapped or corrupt file is refused before anything runs
  (`update_engine_test.cc` flips the last payload byte).
- [x] Design item 3: `build.sh` never errors on a missing embed TU — the
  empty-plan TU, with the pre-#158 script refused by the packer
  (`check_disk_capable`).
- [x] Design item 4, policy by size with `--embed` / `--no-embed` forcing
  it, the choice printed and recorded.
- [x] Design item 5, tests and docs: `update_engine_test.cc`,
  `native_emitter_test.cc`, `pack_update_test.py`; `docs/usage.md`,
  `docs/formats.md`, `docs/runtime.md`.
- [ ] Design item 5's last clause, the study's `run_controls_f32.sh`
  gaining the Qwen row — the script lives in
  `out/realanalysis-2026-09-25/` (not committed) and names Qwen; whether
  that row was run is not recorded in the tree. Closed without it: the
  Qwen f32 row the clause exists for is the measurement above.
