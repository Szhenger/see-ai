# The SeeAI Source Language & Substrate

## Why isn't this under `compiler/`?

Because it isn't a stage of compilation. `source/` holds the abstractions of
the *source language itself* — the model container, the plan ABI — and the
low-level substrate both halves of the product share. The
[compiler](../compiler/README.md) consumes these; the
[runtime](../runtime/README.md) consumes these; neither *owns* them. Putting
them here, outside both, is what lets the two halves agree on every byte
without depending on each other.

```
source/
  language/               the containers a model and a corpus travel in
    model_format          SMF: the parsed structs + on-disk layout constants
    smf_preamble          the SMF header every reader checks first
    dataset_format        SDS: the corpus container
  plan/                   the compiler <-> runtime ABI (façade: update_types.h)
    config                the compilation request (UpdateConfig)
    instruction           the 64-byte instruction word + opcode vocabulary
    schema                the .seeu container header + emit table
    metadata              the trailing metadata section (v23): manifest, policy, ledger
    opcode_names bf16     the opcode's printed name; the bf16 conversions
  contract/               the load rules both halves run, as pure functions
    plan_contract         what a plan header must satisfy
    split                 the seeded held-out partition
    bounds                MulOk / RangeOk, the one way bounds math is written
  recipe/                 the declarative recipe: schema, file parser, run flags
  tokenizer/              byte-level BPE and the text ingest (sanitize,
                          deduplicate, encode, pack), vendored into packages
  statistics/             the statistics screens' structs and their rendering
  platform/               the POSIX seam: file mapping, memory queries,
                          durable replace
  parallel/               deterministic chunked data-parallelism
    parallel_for          the leaked worker pool, chunk geometry by shape
  identity/               content hashing for integrity, and the release version
    hash                  ContentHash64 + the in-blob PlanSelfHash seal
    version               kSeemlVersion
```

## What everyone shares

**`language/`** defines what a model *is* on disk — tensors, ops, the
byte layout — so the exporter, the compiler's loader, and the tools all
speak one format. **`plan/`** is the crucial seam: it is the entire
vocabulary the compiler and runtime share — the instruction set the compiler
*writes* and the runtime *reads*, plus the container header addressing every
section. Change it in one place and both halves move together.

**`parallel/`** and **`identity/`** are the substrate that makes the
product's headline promises possible. `parallel_for` chunks work by problem
shape and never by thread count — the single reason results are
**bitwise-identical** whether one core ran or eight. `hash` is the
integrity layer: a parallel FNV-1a variant that fingerprints a model (so
commit refuses the wrong file) and seals a plan (so a flipped bit is caught
at load, not as silent garbage on-device).

**`contract/`**, **`recipe/`**, **`tokenizer/`**, **`statistics/`** and
**`platform/`** arrived with SeeAI v1.0.0.B (`docs/next-project/seeai.md`):
every rule the build host checks is re-run on the device, so the rule lives
here once — the plan-header contract and the held-out split, the recipe and
the runner's flags, the tokenizer and the text ingest, the FLOP count the
screens print — and the package vendors it with the runtime.

The discipline to notice: `plan/` is an *ABI*, so its structs are
`#pragma pack(1)` with `static_assert`s on their sizes — a layout change is
a versioned event, negotiated by the runtime's version policy, never an
accident.

## Where to go next

The byte-level formats — every field, offset, and magic number, and *why*
each is there — are specified in **[docs/formats.md](../docs/formats.md)**.
The parallelism and hashing machinery is taught where it matters most, in
[docs/runtime.md](../docs/runtime.md). How the compiler consumes all of this
is in [docs/compiler.md](../docs/compiler.md). The substrate's own suites
live under [test/source/](../test/README.md) (`parallel_for`, `hash`); the
`language/` and `plan/` structs are exercised throughout the compiler and
runtime suites.
