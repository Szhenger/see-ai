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
  the quantized column scales, the emit table (CR2 #143's coverage table:
  `test/runtime/dispatcher/engine_test.cc`, `CoverageTable.*`).

`UpdateEngine::Initialize` calls both, binds the backend, and commits the
result to its members only after every rule has passed — a refused re-load
leaves the previous plan fully usable. `VerifySourceModel` /
`BindSourceModel` (the source model's hash against the header) stay on the
engine, which owns the mapping.
