# runtime/loader — the security boundary

Plan load (`docs/next-project/seeai.md` §6): from the plan's bytes to a
proven, decoded plan and its one arena. Everything after it trusts the plan;
the verifier proves instructions, the loader proves the container and the
environment.

- `plan_loader.{h,cc}` — `DecodePlan`: the seal hash, the version window,
  the plan contract, the metadata section (the identity manifest's words and
  the reduction-order ledger), the instruction streams and emit table, the
  executor contract, and the scans the feeder enforces (the narrowest softmax
  width, the narrowest embedding table, the eval program's loss tail, the
  source-model extent). `AllocateArena`: the one allocation, bounded by the
  address space and the host's memory, seeded with the persistent image.
- `contract.{h,cc}` — the plan, executor and feeder contracts: every operand
  bounds-proven (via the verifier), the staged slots no program writes, the
  label and token bindings, the segment starts (#191), the masked-loss flag,
  the optimizer steps (in the train and step programs alone; an 8-bit moment
  step's codes and scales in the persistent segment, #192:
  `EngineContract.EightBitMomentsArePersistentStateOfTheTrainingPrograms`),
  the quantized column scales, the emit table (CR2 #143's coverage table:
  `test/runtime/dispatcher/engine_test.cc`, `CoverageTable.*`). And the
  relaxed permit (F2 #206, `RelaxedPlanPermit` / `VerifyRelaxedPermit`): a
  plan with a `kFlagRelaxed` instruction loads only when the caller names
  its seal — the generated runner passes the hash its build compiled in
  after checking a granted numerics certificate, zero otherwise — or says it
  is a measuring instrument (`UncertifiedForMeasurement`: the probe under
  `--relaxed-for-measurement`, the fuzzer, the suites that prove the relaxed
  kernels). Every load defaults to no permit (`RelaxedPermit.*` in the same
  suite).

`UpdateEngine::Initialize` calls both, binds the backend, and commits the
result to its members only after every rule has passed — a refused re-load
leaves the previous plan fully usable. `VerifySourceModel` /
`BindSourceModel` (the source model's hash against the header) stay on the
engine, which owns the mapping.
