# frontend/tokenization — the training-data ingress

The frontend's second input path (`docs/next-project/seeai.md` §3). The
model path is `ingressor → accountant → topology → computation`; the data
path is this directory, and the two meet at the accountant.

Stages, in order, each a pure deterministic function so the package can
vendor them and the device's pipeline re-runs the same checks on its own
data (security at every layer):

1. **ingress** — a bounded reader over raw text (or a pre-tokenized SDS
   whose header carries the tokenizer hash).
2. **normalize** — Unicode normalization as the model's tokenizer section
   specifies.
3. **sanitize / filter** — invalid UTF-8, control characters, bidirectional
   overrides, empty and degenerate records; every drop recorded by index and
   reason in the manifest.
4. **deduplicate** — exact, by record hash (near-duplicates are the Python
   plane's job).
5. **encode** — byte-level BPE from the SMF tokenizer section only (no
   `--tokenizer` flag exists); proven bit-exact against the reference
   tokenizer at export.
6. **pack** — concatenate documents with EOS and cut fixed S+1 windows;
   never pad, because every position is a loss target.

Where the code lives: the normalize, sanitize, deduplicate, encode and pack
functions are implemented under `source/` (beside the contract check), the
only tree the packaging phase vendors into a package; this directory is the
compile-time driver that runs them over the user's corpus and fills the
accountant's statistics. Code placed here alone would never reach the device.

Artifact: a canonical SDS, its manifest, and its hash. The tokenizer
definition's hash binds every corpus on both layers (plan header, SDS
header). What this directory does **not** defend against, and says so on
its screen: well-formed poisoned data — the gate and a held-out set the
training source cannot influence are that defense.

Where it stands (S2 #148, 2026-11-09): the stages are code —
`source/tokenizer/text_ingest.{h,cc}` (sanitize, deduplicate, encode, pack,
manifest) over `source/tokenizer/byte_level_bpe.{h,cc}` — and this directory's
`corpus_ingress.{h,cc}` is the driver: `seeml-update-compile --data corpus.txt |
corpus.jsonl | corpus.sds` runs them after topology, holds the corpus to the
model (tokenizer identity, geometry, every id below the table), writes
`<out>/corpus.sds` + `<out>/corpus_manifest.json` for raw text, pins the plan
to the corpus by hash (plan v22 `corpus_hash`, `kDataFlagCorpusPinned`; the
feeder contract refuses any other corpus) and fills the frontend screen's
corpus rows. Normalization is the one stage with no code: the core refuses
a tokenizer that needs one (SmolLM / GPT-2-class tokenizers have none). The
device-side re-run of the stages on raw text is open: the sources are
vendored into the package, but the emitted runner takes an SDS.
