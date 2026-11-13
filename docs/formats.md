# SeeAI Binary Formats

## Why bytes, and how to read this document

Every artifact SeeAI produces or consumes — models, corpora, plans, checkpoints — is a binary file with a fixed, documented layout. Why binary, when JSON exists? Because these files are mostly *tensors* — n-dimensional arrays of floats, millions of them — and because the device-side loader must be tiny, allocation-conscious, and paranoid. A format you can parse with bounds-checked `memcpy` is a format you can *prove* things about.

Before the individual formats, three conventions that apply everywhere, each worth understanding once:

- **Little-endian, always.** A multi-byte integer like `0x31464D53` is stored least-significant byte first. Rather than swap bytes on big-endian machines, the loaders simply refuse to compile there (`static_assert`) — a deliberate simplification: every target SeeAI cares about is little-endian, so byte-swapping code would be untested dead weight.
- **Packed layouts.** Every multi-byte integer is packed without padding; where a C struct is shown, it is `#pragma pack(1)` and part of the ABI. Normally compilers insert invisible padding between struct fields for alignment; packing turns the struct into an exact byte-for-byte contract, so `sizeof` is the wire size and a `static_assert` can pin it forever.
- **Magic numbers.** Each format opens with a four-byte signature — read the little-endian `u32` as ASCII and you get the name back (`"SMF1"`, `"SDS1"`, `"SEEU"`, `"SEKP"`). It's the file introducing itself, and it means a mix-up (feeding a corpus where a model belongs) dies on byte 0 with a clear message, not on byte 40,000 with a weird one.

## One statement per plane, checked by machine

Every layout below is declared twice — once in the C++ headers, once in `tool/seeml/formats.py` for the build-host tools — and never a third time. The two are held together without anyone remembering to: `seeml-abi` prints the C++ side's magics, versions, enums and packed-struct offsets as JSON; that output is committed as `tool/seeml/abi.json`; CI regenerates it and fails on a diff; and `test/tool/formats_test.py` compares every Python declaration with it. Changing a format is therefore a three-step edit that cannot be half done: the header, `formats.py`, and `build/seeml-abi > tool/seeml/abi.json`. `seeml-seeu-dump --json` is the same idea for whole plans: the C++ decode as data, compared field for field with the Python reader on freshly compiled plans.

## First, a word about hashes

Several formats below carry 64-bit hashes, so let's establish what they are and — just as important — what they are *not*.

The workhorse is **FNV-1a**, a classic non-cryptographic hash chosen for being almost embarrassingly simple:

```
h = 0xcbf29ce484222325                    (the "offset basis")
for each byte b:  h = (h XOR b) × 0x100000001b3   (the "FNV prime")
```

Two constants, one XOR, one multiply per byte. Any single flipped bit avalanches through the multiplications and changes the digest — which is exactly the property needed to detect *corruption*.

For whole-model identity there's a faster sibling, **`ContentHash64`** (`source/identity/hash.h`), which fixes FNV's one weakness — it's inherently serial, one byte after another — with two layers of parallelism:

1. **Striping (instruction-level):** run 8 independent FNV lanes, byte i feeding lane i mod 8, each lane seeded differently (the offset basis XOR a golden-ratio multiple, so lanes never collide). Eight independent multiply chains keep a modern core's pipeline full — roughly 8× the throughput of the serial loop.
2. **Chunking (thread-level):** split the input into 1 MiB chunks, hash each chunk (with the striped kernel) in parallel, then fold the per-chunk digests together *in chunk order*, finishing with the total length. Because chunk boundaries depend only on the input *size* — never the thread count — the digest is bitwise-identical on one core or eight, in keeping with SeeAI's determinism rule ([runtime.md](runtime.md)).

Note that `ContentHash64` of some bytes deliberately does **not** equal plain `Fnv1a64` of the same bytes — they are distinct identity contracts, and the format version bumps below track which one a field uses.

Now, the caveat, stated as bluntly as possible: **these hashes detect accidents, not adversaries.** FNV is trivially forgeable by anyone who wants to; there are no keys and no signatures here. SeeAI's hashes answer "is this the same file, uncorrupted?" — never "do I trust whoever sent this?" Authenticate plans in your update transport (TLS, signed manifests — whatever your deployment already uses for software updates).

## SMF — SeeAI Model Format (`.smf`, v7)

The dependency-free model container consumed by `seeml-update-compile` (source and teacher models), produced by `tool/export_model.py`. It answers exactly two questions: *what are the tensors?* and *what is the computation graph over them?*

```
u32 magic  "SMF1" (0x31464D53)
u32 version         1..7 accepted; both writers emit the lowest version
                    that can carry the model (3, 4, 5, 6, or 7 with a
                    tokenizer section)
u32 num_tensors
u32 num_ops
str input_name      (str = u16 length + bytes, no terminator)
str output_name
u64 seq_len         (v3+) rows per sequence; 0 = non-sequential
u64 tokenizer_offset (v7+) absolute offset of the tokenizer section, 0 = none
u64 tokenizer_size   (v7+) its byte size; the section is an opaque blob laid
                    out like a payload (64-aligned, after the metadata,
                    overlapping nothing) that source/tokenizer/ defines; its
                    ContentHash64 is the tokenizer identity (SDS v3, plan v22)
tensors[num_tensors]:
  str  name
  u8   rank
  u8   flags        bit0 = constant (weight); else graph I/O
  i64  dims[rank]   -1 = dynamic batch (non-const tensors only)
  u64  data_offset  absolute file offset of the f32 blob (0 if not constant)
  u64  byte_size    must equal volume × 4 for constant tensors
data section: each constant tensor's f32 blob at its 64-aligned offset
ops[num_ops] (topologically ordered):
  u8   kind         0 MatMul  1 AddBias  2 Relu
                    3 Gelu    4 Silu     5 Mul    6 LayerNorm    (v2)
                    7 Add     8 RmsNorm  9 Rope   10 Attention   (v3)
                    11 Embedding                                  (v4)
  str  name
  u8   num_inputs
  str  inputs[num_inputs]
  str  output
  u32  attr0        (v3+) num_heads for Rope/Attention, else 0
  u32  attr1        (v5+) Rope: rotary base θ as IEEE-754 f32 bits (0 = 10000); else 0
  u32  attr2        (v6+) LayerNorm / RmsNorm: epsilon as IEEE-754 f32 bits (0 = 1e-5); else 0
```

Op signatures: `MatMul(x, W)`, `AddBias(x, b)`, unary activations `(x)`, `Mul(x, y)` / `Add(x, y)` (same shape), `LayerNorm(x, gamma, beta)` and `RmsNorm(x, gamma)` over the last dim, `Rope(x)` (rotary position embedding), and causal `Attention(q, k, v)` (v3, with model-level `seq_len`), plus `Embedding(tokens, table)` (v4). For a token-native model, the graph input is a rank-1 dynamic (`{-1}`) non-const tensor meaning i32 token ids, consumed *only* by embedding ops that gather rows of a constant `[vocab, dim]` table.

A few design choices worth noticing:

- **Strings are length-prefixed** (u16 + bytes, no NUL terminator). A parser reading length-prefixed strings can bounds-check *before* reading; a parser scanning for terminators can run off the end. This is the safer of the two classic conventions, chosen on purpose.
- **Ops must arrive topologically ordered** — every input produced before it's consumed. This moves a whole class of work (dependency resolution) out of every loader and into the single writer, and turns the loader's check into a linear scan ([compiler.md](compiler.md)).
- **Data offsets are absolute and 64-byte aligned.** Aligned so a mapped tensor starts on a cache line; *absolute* for a deeper reason: the offset of every weight is preserved through the entire compilation and lands in the plan's emit table — it is literally the patch address that commit uses to apply `W′ = W + Δ` to this very file. The format is designed for being *updated in place*, not just read.

`LoadSmf` records the whole file's `ContentHash64` as the model's identity; the compiled plan carries it, and commit refuses any file that doesn't match.

### The tokenizer section (SMTK, v7)

A v7 file may carry its tokenizer: the bytes at `tokenizer_offset` are one self-describing blob (`source/tokenizer/tokenizer_format.h`) the exporter writes from the checkpoint's own `tokenizer.json` and `tokenizer_config.json`, and the compiler's tokenizer is instantiated from — and from nothing else; no `--tokenizer` flag exists. Its `ContentHash64` is the tokenizer identity every corpus is bound to (SDS v3 `tokenizer_hash`, plan v22 `tokenizer_hash`).

```
u32 magic "SMTK"; u32 version (1)
u32 kind            1 = byte-level BPE (the GPT-2 family)
u32 pre_tokenizer   1 = the GPT-2 split pattern; 2 = Digits(individual), then it
u32 flags           bit 0 add_prefix_space, bit 1 NFC normalizer — both refused
u32 vocab_size; u32 num_merges; u32 num_special
i32 eos_id; i32 bos_id; i32 unk_id; i32 pad_id     (-1 = none)
u32 chat_template_len; bytes chat_template         (Jinja source, verbatim, uninterpreted)
vocab[vocab_size]:    u16 len; bytes     the token's BYTES (the byte-level string mapped
                                         back through the GPT-2 byte<->unicode table)
merges[num_merges]:   u32 a, u32 b, u32 c   in rank order; c's bytes are a's then b's
special[num_special]: u32 id              split before the pattern, longest match first
```

The core implements one family — byte-level BPE with the GPT-2 pattern, optionally preceded by the Digits splitter, special tokens split first — and refuses every other construct by name on both planes: the exporter will not write a normalizer, an explicit `Split` pattern, byte fallback, an unknown token or a non-BPE model, and the reader will not take a kind, pre-tokenizer or flag it does not run. The encoder is proven against the reference tokenizer at export (`tool/export_model.py --tokenizer-parity`, over `test/fixtures/tokenizer/adversarial.txt`, through `seeml-tokenize`), and the section's vocabulary must be the embedding table's rows and the head's width (topology refuses otherwise).

## SDS — SeeAI Dataset (`.sds`, v3)

The corpus container — deliberately the simplest format in the family, because a dataset is just samples:

```
u32 magic "SDS1"; u32 version (1, 2 or 3)
u64 num_samples; u64 input_dim
u32 label_kind    0 none | 1 class index (i32) | 2 dense (f32[label_dim])
u32 input_kind    v2: 0 = f32 feature rows | 1 = i32 token records
                  (this word was padding in v1, always 0)
u64 label_dim
u64 tokenizer_hash  (v3) the SMF tokenizer section the ids came from; 0 = none
u32 flags           (v3) bit 0 = token records carry a loss mask
u32 reserved        (v3) must be 0
u64 reserved        (v3) must be 0
records[num_samples]:
  input_kind 0: f32 input[input_dim], then the label
  input_kind 1: i32 tokens[input_dim + 1] and NO stored label — one
                sequence per record; inputs are tokens[0..S) and the
                next-token class labels are the shifted view tokens[1..S],
                derived at serving time (label_kind must be 1);
                under the mask flag, then u8 mask[S] padded to 4 bytes:
                1 where the next token is a loss target, 0 where the
                feeder stages the label -1 the masked loss ignores
                (every record keeps at least one target)
```

The sample section fills the file exactly: a writer that appended bytes, or a file cut short, is refused rather than read around (the review's F19). A writer emits the lowest version that carries its content, so a plain token corpus is still the v2 file it always was; naming a tokenizer or carrying a mask makes it v3, and a plan bound to a tokenizer (v22) refuses a corpus whose hash differs — the ids would mean other strings.

A 40-byte header (64 bytes from v3), then fixed-size records — which means sample k lives at a *computable* offset, no index needed. `label_kind` covers the three training modes: `1` (a class index) for cross-entropy, `2` (a dense vector) for regression/MSE — predicting continuous values rather than classes, `0` (no label at all) for distillation, where the teacher model provides the target. The `u32` after `label_kind` was header padding in v1; v2 gives it meaning as `input_kind`, so old files (always zero there) still read correctly. Token corpora (input_kind 1) shuffle, split, and replay at record (sequence) granularity, so no sequence is ever cut or mixed.

## SEEU — Update Plan (`.seeu`, v24)

The star of the show: the fully AOT-compiled update. One file containing three instruction streams (train / eval / merge) — four under gradient accumulation, when the train section is the grad program and a step section holds the optimizer program — the frozen weights, the persistent segment's initial image, and the emit table — every section addressed by a single `PlanHeader` at offset 0 (authoritative definition: `source/plan/schema.h`; every section 64-byte aligned). Versioning is additive: see the version history below (currently v24).

Key header fields:

| field | meaning |
|---|---|
| `plan_hash` | `PlanSelfHash` (chunked-parallel FNV-1a) of the whole blob with this field zeroed; verified on load |
| `source_model_hash` | `ContentHash64` of the source `.smf`; commit refuses other files (0 = unbound) |
| `arena_size`, `persistent_size` | the single device allocation, and its checkpointable prefix |
| `input_ref/floats`, `label_ref/bytes/kind`, `loss_ref` | the I/O slots — where the feeder writes and the engine reads |
| `train/eval/merge_instr_offset/count` | the three programs; eval is the forward+loss program for validation gating |
| `rodata_offset/size`, `persist_init_offset/size` | frozen weights; initial adapter/moment image |
| `emit_table_offset`, `emit_count` | the patch map (below) |
| `lr, beta1, beta2, eps, weight_decay` | optimizer hyperparameters, read at dispatch |
| `lr_schedule, warmup_steps, min_lr_factor` | runtime LR schedule (0 = constant) |
| `clip_norm` | informational; clip instructions are baked into the stream |
| `gemm_tile_k`, `gemm_tile_n` | the CPU blocked-GEMM tile geometry (v11) the compiler decided from the offline tuner's host-keyed table or `--gemm-tiles`; 0 = the runtime's compiled-in default; throughput only, never bits (the K tile must be a multiple of 4) |

Notice the "zeroed field" trick in `plan_hash`: you can't hash a file that contains its own hash (the act of writing the digest would change it), so the digest is computed with that one field held at zero, then patched in. The verifier replays the same convention.

Version history: v2 added the eval program, integrity hashes, and LR schedule; v3 moved `source_model_hash` to `ContentHash64`; v4 moved `plan_hash` to the chunked-parallel `PlanSelfHash`; v5 gave the instruction `flags` word meaning (fused GEMM epilogues); v6 added the transformer opcode family (RMSNorm, RoPE, causal attention and its backward primitives); v7 added token-native input — `input_kind` and `seq_len` carved from `reserved`, plus the `kEmbedFwd` gather over a rodata-only table; v8 gave the high half of the `kKLDistill{Fwd,Bwd}` temperature word meaning as the distillation loss scale (`T²`, so the soft-target gradient stays commensurate with a hard-label term) — zero there, as in every pre-v8 plan, reads as 1.0; v9 added gradient accumulation — `grad_accum_steps` (from a pad word) and the step program's offset and count (from `reserved`), plus the `kAccumulate` opcode (`dst += src`), version-gated: when `grad_accum_steps > 1` the train section is the grad program and the step section the optimizer program, and the two must appear together; v10 added bf16 frozen weights — `kGemmNNBF16` / `kGemmNTBF16`, whose B operand is bfloat16 rodata (2 bytes per element, rodata-pinned like int8, widened exactly in the kernel), version-gated; v11 gave the header's two pad words meaning as the CPU GEMM tile geometry, `gemm_tile_k` and `gemm_tile_n` — the compiler's kernel-policy decision (`tool/autotune.py`'s table, `--gemm-tiles`, or zero for the runtime default), proven on load (the K tile on the kernel's 4-wide unroll) and handed to the CPU backend; it changes throughput only, never bits. v12 is the bitwise-safe kernel batch (E3), carried entirely in instruction words that were `kNullRef` / zero in every earlier plan, so the header is unchanged: one opcode, `kRopeTable` (`[S, d/2, 2]` cos/sin of the RoPE angles, built on the device with the recurrence the rotation kernels run), an optional third operand on `kRopeFwd` / `kRopeBwd` naming its result, and an optional per-tensor clip threshold in `out[1]` of `kSgdStep` / `kAdamWStep` that folds the preceding `kClipNorm` into the step (the gradient is consumed clipped and never rewritten; a non-finite norm refuses that tensor's step). All three are rejected below v12, and a v12 program computes the same bits as the v11 program it replaces (`--no-rope-table`, `--no-fuse-clip` compile the old shapes for comparison). v13 added `kFusedMap` (E4): a chain of up to four elementwise stages — add, mul, scale, relu, gelu, silu — as one instruction, the stage bytes in `out[1]` (kind in the low nibble; operand slot or immediate index in bits 4–5; bit 7 = the running value is the right operand), two scale immediates in `out[2]`, operands `in[0]` = x, `in[1..2]` = the binary stages' tensors, `in[3]` = out; the validator proves the program (known kinds, zero-terminated, every named slot present and every unnamed one absent, finite immediates) and the runtime runs each stage as its own loop over a cache-resident block, so the result is bit-identical to the instruction sequence it replaces. v14 added the GEMM addend (E10): `kFlagGemmAddend` (flag bit 3) on `kGemmNN` / `kGemmNT` / `kGemmTN` makes the instruction `C = D + A@B`, with `D`'s ref (`[M, N]` f32, read, disjoint from `C`) in the otherwise-free `in[3]` — every LoRA site's forward `C + ts@B` and backward `dC@Wᵀ + dt@Aᵀ` without the activation-sized `kAddEW` and the transient it read. Each element is `d + s` over the complete dot product, so the fold is bit-identical to the GEMM + add pair; the bit excludes the bias / activation epilogue (one slot, one tenant), is refused on the int8 / bf16 GEMMs and every other opcode, and is corruption below v14. v15 added the tiled attention family (E11): `kAttnFwdTiled` (47), `kAttnDQTiled` (48), `kAttnDKTiled` (49), `kAttnDVTiled` (50). Instead of the `[B·H·S, S]` probability cache the cached family keeps from forward to backward, they keep a stats row of four floats per query row (`kAttnStatsWidth`: the score max, the inverse softmax denominator, the softmax-backward rowsum delta — written by the dQ pass, read by the dK pass, so a stream runs dQ first — and a pad), and every backward pass recomputes the probabilities it needs with the cached kernels' own expressions, so the two families compute identical bits. The forward keeps the cached forward's slots (stats where the cache was); the backward takes q, k, v, dO in `in[0..3]`, the stats row in `out[0]`, its result in `out[1]` and the geometry packed 16 bits a field in `out[2]` (`B<<48 | S<<32 | H<<16 | d`). All four are corruption below v15. v16 gave the instruction's pad word meaning (P7, #96): it is now `imm`, an opcode-defined 32-bit immediate, and `kLayerNormFwd` / `kRmsNormFwd` carry the model's normalization epsilon there as f32 bits (0 = 1e-5, which every earlier plan meant; the backwards read the cached statistics and need none). A nonzero `imm` on any other opcode, or below v16, is corruption, and a set epsilon must be finite and positive. v17 added a third address space for tensor refs (E12, #95): bit 62 marks a **source ref**, an absolute byte offset into the source model file the plan was compiled from (the offsets the emit table patches). Source refs are read-only f32, admitted in the eval program only, and resolved against a file the engine maps at run time (`BindSourceModel`: verified against `source_model_hash` and checked to cover every extent the eval program reads, before anything executes). A plan compiled with `--quantize-base` / `--bf16-base` from a model file lowers its eval program's student GEMMs as plain f32 GEMMs over source refs, so every evaluation — the gate's before and after, and every best-state evaluation — scores `W_f32 + Δ`, the function the commit ships, rather than the narrow copies training ran against; a plan compiled from memory keeps the in-plan proxy. v18 added relaxed arithmetic (F2 #130 / F4 #132): `kFlagRelaxed` (flag bit 5) on the six frozen-weight GEMMs — `kGemmNN` / `kGemmNT` and their int8 and bf16 forms — in the train and step programs only, is the compiler's permission for the backend to run that product on a *certified relaxed kernel*: on Metal the tensor-op family (the f32 activations rounded to bf16 once at the GEMM input, multiplied against the exact stored weight into f32 accumulators), on the CPU Accelerate for f32 weights; the exact portable kernel is always a valid execution. It is set only under `--precision certified-bf16`, never by the old path, is corruption below v18 and in the eval program (which scores what ships in exact arithmetic), and `tool/pack_update.py` refuses to package a plan carrying it without a numerics certificate — the certificate binds to the plan's hash, so it lives beside the plan, not in it. v19 added the GEMM alpha (F7 #138): `kGemmNN` / `kGemmNT` / `kGemmTN` carry a scalar alpha's f32 bits in `imm` (0 = 1.0, what every earlier plan meant), making the instruction `C = alpha·A@B [+ D]`. It replaces the two rank-r `kScale` instructions a LoRA site spent per step — the forward's `(α/r)·t` and the backward's `(α/r)·(dC@Bᵀ)` — a dispatch per site per pass for a scalar the GEMM already multiplies by. The CPU's NN and TN cores fold alpha into A's elements as they enter the product and its NT core multiplies the finished dot product — at each site the multiply the `kScale` it replaces performed — so a v19 program computes the v18 program's bits on the CPU for every alpha; Metal multiplies the finished product in every form, identical for a power-of-two alpha (the default 16/8) and equal to rounding otherwise. A set alpha must be finite and nonzero; it is corruption below v19, on the int8 / bf16 forms, on every other opcode, and beside `kFlagRelaxed`. v20 extended relaxed arithmetic to the tiled attention family (F3 #131): `kFlagRelaxed` is also admitted on `kAttnFwdTiled` .. `kAttnDVTiled` (47–50), in the train and step programs only, set by the compiler under certified precision. On Metal, at head widths up to 64, it lets the backend run the blocked kernels' passes with every block's products — `QKᵀ`, `dO·Vᵀ`, their transposes, and the `P·V`, `dS·K`, `dSᵀ·Q`, `Pᵀ·dO` accumulations — on `simdgroup_matrix` tiles: the same stats row and causal mask, the f32 sums in another order, and the forward's `1/l` applied once to each accumulated output row rather than to every probability. The CPU always runs the exact family. It is corruption below v20, on the cached attention opcodes, and in the eval program; the certificate that must accompany it prices the `attn.*.tiled` sites. v21 added the adapter-only plan (F6 #134): source refs are admitted in every program, not the eval program alone — `--adapter-only` binds every frozen weight, the embedding table included, to its f32 payload in the model file the plan is bound to by hash, and the plan carries no rodata: the instructions, the adapters' initial image and the emit table, at adapter scale (SmolLM-135M r8: 35 MB against the 299 MB int8 package). Reads only, as before — a source ref as a destination, or in a header I/O slot, is corruption — and `BindSourceModel` checks the file covers every program's extent, not the eval program's alone. An f32 base and exact arithmetic (the narrow bases are copies the plan would carry; the relaxed kernels read rodata weights); below v21 a source ref outside the eval program is corruption. v22 (S2 #148 / S3 #149) grew the header from 280 to 320 bytes — into the padding every plan since G1b-2 carried before its first 64-aligned section, so no section offset moved and the readable floor stays at v4 — with the data contract: `tokenizer_hash` (the SMF tokenizer section's ContentHash64 the corpus must match; 0 = unbound), `corpus_hash` (the canonical corpus compiled over, provenance unless the pinned flag enforces it), the compiled held-out split as a function the device recomputes — `split_seed`, `val_frac_bits` (the fraction's f32 bits) and `val_split_hash` (over the sorted held-out record indices) under the split flag — and `data_flags` (bit 0 masked loss: the labels carry -1 at ignored positions; bit 1 corpus pinned; bit 2 split present). Every word is zero below v22 and the load contract proves it; a set flag with an unknown bit, a split without a fraction in (0, 1), split fields without the flag or a pin without a hash are refused. v22 also added the masked cross-entropy pair, `kSoftmaxXEntMaskedFwd` (51) / `kSoftmaxXEntMaskedBwd` (52): the operands of 11 / 12 with a label of -1 marking a position the loss ignores — the mean runs over the target rows alone (0 for a batch with none), the probabilities are written for every row, an ignored row's gradient is zero — present exactly when the header carries the masked-loss flag (a masked opcode without the flag, a plain one under it, or the flag without a masked opcode is refused on load), corruption below v22. The header is now 320 bytes, the last size that moves no section: the next growth is a layout change. v23 (S4 #150 / S5 #151 / S6 #152) therefore adds a **metadata section** instead of header words: under `data_flags` bit 3 the plan carries, at the first 64-aligned byte after the emit table (inside the page padding every plan already ends with, so no offset moves), a 64-byte `PlanMetadata` — magic `"SEEM"`, format, the section's byte count, `recipe_hash` (the resolved recipe's `ContentHash64`: the request with every decision the compile took written in — attention family, rematerialization, the mask, which weights the eval program reads — and the budgets those decisions consumed dropped, so two requests that resolve to one plan share it), `export_report_hash` with `export_flags` (bit 0 a report was verified, bit 1 the tokenizer-parity harness ran, bit 2 the logits-parity check ran), `tokenizer_parity_ids` and `logits_parity_max_delta` — followed by `ledger_count` 96-byte `LedgerEntry` records, the **reduction-order ledger**: one line per pass in the order they ran (`pass[32]`, `arithmetic` 0 = bit-identical to its input / 1 = reduction order may change under the numerics certificate, the count of instructions carrying the change, `why[56]`). With the header's `source_model_hash`, `tokenizer_hash`, `corpus_hash`, `val_split_hash`, `split_seed` and the masked-loss flag, that is the identity manifest: every input of the fine-tune, hashed. The loader proves the section before reading it (`PlanMetadataViolation`, `source/plan/metadata.h`): in bounds, sized by its ledger, reserved words zero, flags known, export fields only under their flags, every ledger text NUL-padded printable ASCII, and the ledger's reordered-instruction count equal to the number of `kFlagRelaxed` instructions in the train and step programs — an exact plan's ledger names none, a relaxed plan's names every one. The flag is corruption below v23; a plan without it reads as every earlier plan did. The seal covers the section like every other byte. v24 (#190) added the **global-norm clip**: `kSqNormAccum` (53, `acc += ‖g‖²`, operands g and a one-float arena slot) and `kClipGlobal` (54, `g *= min(1, max/√acc)` in place, the threshold's f32 bits in `in[2]`, finite and positive). Under `--clip-mode global` the optimizer program accumulates every trainable gradient's squared norm into the one slot, then scales every gradient by the one factor it implies — so the update keeps its direction, where the per-tensor clip rescales each tensor on its own — then steps unclipped and zeroes the slot after the steps, as the gradient accumulators are zeroed. A non-finite norm refuses the step, as `kClipNorm`'s does. The CPU sums the squared norm with the clip's f64 chunk partials; Metal with the same chunk geometry in f32, so the backends agree to rounding, as their per-tensor clips do. Both opcodes are corruption below v24; a plan that does not clip globally is the v23 plan with its version word raised. Layout note, no version: since G1b-2 the rodata section starts on a 16 KiB boundary and the blob is padded to a 16 KiB multiple (`kSeeuRodataAlignment`), so a page-aligned embedded plan lets the Metal backend wrap the frozen weights zero-copy — the header's explicit offsets already describe it, and older runtimes read such plans unchanged. The same padding is what makes the file itself mappable as the plan: in the disk-resident package mode (#158, for plans the macOS arm64 linker and loader cannot carry embedded, from ~1.8 GB) the generated driver maps `update_plan.seeu` private and copy-on-write, page-aligned by construction, and the Metal backend wraps the frozen weights zero-copy exactly as it does the `.incbin` embedding; the validation — magic, version gate, the seal, both contracts, the source-model hash — is the same whichever way the bytes arrive. Each new opcode is version-gated: a plan carrying one below its introducing version is corruption, not forward compatibility. The gather's *index* bound is the feeder contract's runtime job — every token id is proven inside both the narrowest embedding table and the narrowest softmax width before anything executes, exactly how class labels are bounded.

Why do hyperparameters live in the *header* while clip lives in the *stream*? Because a learning rate is a number the runtime consults, but clipping changes which instructions exist — structure belongs to the program, parameters to the header, and each fact has exactly one home ([compiler.md](compiler.md)).

**Tensor references** are 64-bit words: bit 63 selects the address space (0 = mutable arena, 1 = read-only rodata), bits 0..62 are a byte offset. Two flat address spaces and an offset — the entire memory model, and the reason the validator can prove write-safety with a single bit test.

**Instructions** are exactly 64 bytes — one cache line — laid out as:

```
u16 opcode | u16 flags | u32 imm | u64 in[4] | u64 out[3]
```

with 51 opcodes (`kNop` through `kAttnDVTiled`, contiguous from 0; `seeml-abi` prints them all). The `in[]` slots hold tensor refs; scalars (a GEMM's α, a fill value, clip's max-norm) are f32 *bit-cast* into a spare slot, and `imm` (v16) carries a normalization forward's epsilon; the `out[]` words carry dimensions (a GEMM's M, N, K; LayerNorm packs `(rows << 32) | cols`). Inspect any plan's streams with `seeml-seeu-dump --instrs` ([usage.md](usage.md)).

Frozen weights selected by `--quantize-base` are stored in rodata as symmetric int8. Since v17 the scale is per output column: the `M` f32 scales of a weight `W [K, M]` follow its int8 levels, and the q8 GEMMs carry `kFlagQ8ColScale` (flag bit 4) with a rodata ref to them in `in[3]` — the forward applies them as its output columns' scales, the dX GEMM as its reduction index's. Earlier plans carry one per-tensor scale's f32 bits in `in[3]` (scale = max|w|/127) and run unchanged.

**The emit table** (`EmitEntry[]`, 24 bytes each: `smf_data_offset`, `byte_size`, `arena_offset`) is the bridge back to the model file: it maps each adapter's **delta** (`Δ = (α/r)·A@B`, materialized by the merge program at `arena_offset`) to the f32 byte range of its weight inside the source `.smf`. Commit applies `W′ = W + Δ` onto the file's pristine weights — which is why a quantized plan never bakes quantization error into the committed model: the int8 copy trains, but the original floats get patched.

## Checkpoint (`SEKP`, v5)

Training state you can power-cycle through:

```
u32 magic "SEKP"; u32 version = 5      (v3 and v4 files are still read)
u64 plan_hash        must match the plan's PlanHeader::plan_hash
u64 step             1-indexed AdamW timestep at save
u64 persistent_size  payload length
u64 payload_hash     ContentHash64 of the payload (v3; v2 used serial FNV-1a)
u64 horizon_steps    v4: the LR-schedule horizon of the run that saved it
v5 tail (64 bytes):
  u64 shuffle_origin       the training set's shuffle stream (0 = sequential)
  u64 train_samples        the split: training samples ...
  u64 val_samples          ... and validation samples (0 = no split)
  u64 best_step            the step of the best evaluated state
  u64 best_payload_hash    ContentHash64 of the best payload, when present
  u32 flags                1 val_initial present, 2 accuracy present,
                           4 best payload follows, 8 best_* are live
  u32 val_initial_loss     f32 bits: the SOURCE model's validation loss
  u32 val_initial_accuracy f32 bits
  u32 best_loss            f32 bits
  u32 best_accuracy        f32 bits
  u32 stale_evals          evaluations since the best (patience)
payload              the arena's persistent segment (adapters + moments)
best payload         v5, flag 4: the best evaluated segment, same length
```

The payload is a raw byte-copy of the arena's persistent segment — possible only because the compiler put everything resumable (LoRA parameters *and* AdamW moments) contiguously at arena offset 0. Three fields guard the restore, each against a different failure: `plan_hash` against the wrong plan (offsets into someone else's arena layout would be garbage), `persistent_size` against a layout drift, `payload_hash` against bit rot. A foreign or bit-flipped checkpoint is rejected before any byte reaches the arena. Saving `step` matters more than it looks: AdamW's bias correction depends on t, so resuming at the wrong step would silently distort the next updates. The **horizon** (v4) is the same kind of fact for the schedule: the step count the interrupted run was annealing over. A resume with no step count trains exactly the remainder to it, so the resumed run's learning rates — and its bits — are the uninterrupted run's; a v3 file has none and resumes on the plan's compiled budget, as it always did. The **v5 tail** makes a resume honest in two more ways. The run binding — which shuffle stream, and how many samples on each side of the train/validation split — lets a resume *refuse* a different `--seed` or `--val-frac`: those would silently move the boundary and train on the first run's validation rows. The stored `val_initial_*` is the source model's score, measured once by the first run, so the gate keeps comparing the whole update against it instead of scoring the resumed adapter as its own "before". And the **best payload** is the state the update will commit — the best evaluated one, not the last — which must survive an interruption exactly as the last one does; it is written only while it differs from the main payload (a checkpoint taken at a new best holds it once).

## Durability

Model commits and checkpoints are written as `fsync`'d sidecar files followed by an atomic `rename` and a best-effort directory `fsync` — the full liturgy, and why each step exists, is in [runtime.md](runtime.md). The contract to remember: a power cut leaves either the old file or the new file, never a torn one.

## To recap

- Four formats, one philosophy: fixed layouts you can bounds-check before trusting, magic numbers so mix-ups die at byte 0, and hashes so corruption dies at load — while authentication remains the transport's job.
- SMF is a model designed to be *patched* (absolute weight offsets survive compilation into the emit table); SDS is samples at computable offsets; SEEU is the entire training job as data; SEKP is the resumable slice of the arena, triple-guarded.
- Every artifact is verified before its first byte is acted upon — the same boundary discipline as the code that reads them.
