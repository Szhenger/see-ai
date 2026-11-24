#!/usr/bin/env python3
"""Writes the fuzz harness's seed corpus, test/fuzz/seeds/ (CR3 #208): one
small VALID input for every arm of test/fuzz/binary_formats.cc, in that
harness's framing, so a fuzzing run starts from inputs its consumers accept
and the replay suite (test/fuzz/fuzz_replay_test.cc) can assert that each
arm is reached.

    python3 test/fuzz/make_seeds.py            regenerate test/fuzz/seeds/
    python3 test/fuzz/make_seeds.py --check    compare with what is checked in

A plan-version bump (or any change that moves a plan byte) is the first
command, run after a build; the second is what test/tool/fuzz_seeds_test.py
runs. The standard library and the built seeml-update-compile only: no
NumPy, so the script runs wherever the tree builds.

What is compared byte for byte, and what is not. The models, corpora,
tokenizer section, text, recipe and report are written here from integers
(every weight a dyadic rational, exact in float32, drawn from a fixed
linear congruential sequence): --check holds them to the checked-in bytes on
every host. A plan is the compiler's output and the compiler reads its host
(the kernel-policy table's tiles, the CPU's cache geometry), so a plan seed
is NOT byte-compared: the replay suite holds each one to being ACCEPTED by
the loader and executed, on every host, and names this script when one is
refused.

The framing (binary_formats.cc is its definition): byte 0 selects the arm,
`byte % ARMS`; the rest is the arm's input. An arm that reads more than one
file takes them joined by SEPARATOR.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tool"))
import export_model as em  # noqa: E402
from seeml import content_hash, formats  # noqa: E402
from seeml import tokenizer_section as ts  # noqa: E402

SEEDS = os.path.join(HERE, "seeds")

# --- The harness's framing (test/fuzz/binary_formats.cc) ----------------------
ARM_SMF, ARM_PLAN, ARM_SDS, ARM_COMPILE = 0, 1, 2, 3
ARM_PLAN_RAW, ARM_TOKENIZER, ARM_RECIPE, ARM_REPORT, ARM_COMPILE_DATA = (
    4, 5, 6, 7, 8)
ARMS = 9
SEPARATOR = b"\xf5seeml\xf5\xf5"
# The data-compile arm's corpus kinds (the byte before the corpus).
DATA_SDS, DATA_JSONL, DATA_TXT = 0, 1, 2
# The tokenizer arm's text kinds (the byte before the text).
TEXT_JSONL, TEXT_ONE_DOCUMENT = 0, 1


def frame(arm, *parts):
    for p in parts:
        if SEPARATOR in p:
            raise ValueError("a seed part holds the part separator")
    return bytes([arm]) + SEPARATOR.join(parts)


# --- Deterministic values ------------------------------------------------------
class Lcg:
    """Knuth's MMIX sequence: integers only, so every host draws the same."""

    def __init__(self, seed):
        self.state = seed & (2**64 - 1)

    def next(self):
        self.state = (self.state * 6364136223846793005 +
                      1442695040888963407) & (2**64 - 1)
        return self.state >> 33

    def below(self, n):
        return self.next() % n

    def weight(self):
        # A multiple of 1/256 in [-0.25, 0.25]: exact in float32.
        return (self.below(129) - 64) / 256.0

    def floats(self, count):
        return struct.pack(f"<{count}f", *(self.weight() for _ in range(count)))


def ones(count):
    return struct.pack(f"<{count}f", *([1.0] * count))


# --- Models --------------------------------------------------------------------
MLP_IN, MLP_HIDDEN, MLP_OUT = 4, 8, 3
VOCAB, DIM, HEADS, SEQ, FFN = 16, 8, 2, 4, 8
EOS = 3  # the segment-mask model's EOS id
TEXT_SEQ = 4


def mlp_builder():
    """4 -> 8 -> 3 with a bias and a ReLU: the feature-model family."""
    rng = Lcg(1)
    b = em._SmfBuilder("x", MLP_IN)
    b.add_tensor("w0", [MLP_IN, MLP_HIDDEN], rng.floats(MLP_IN * MLP_HIDDEN))
    b.add_op(em.OP_MATMUL, "mm0", ["x", "w0"], "z0")
    b.add_tensor("b0", [MLP_HIDDEN], rng.floats(MLP_HIDDEN))
    b.add_op(em.OP_ADDBIAS, "ab0", ["z0", "b0"], "zb0")
    b.add_op(em.OP_RELU, "a0", ["zb0"], "h0")
    b.add_tensor("w1", [MLP_HIDDEN, MLP_OUT], rng.floats(MLP_HIDDEN * MLP_OUT))
    b.add_op(em.OP_MATMUL, "mm1", ["h0", "w1"], "z1")
    b.add_tensor("b1", [MLP_OUT], rng.floats(MLP_OUT))
    b.add_op(em.OP_ADDBIAS, "ab1", ["z1", "b1"], "logits")
    return b


def mlp_smf():
    return mlp_builder().serialize()


def decoder_builder(vocab, seed, tokenizer=b""):
    """A one-block token-native pre-norm decoder — the graph
    export_model._emit_decoder_graph writes, built from bytes so nothing
    here needs NumPy."""
    rng = Lcg(seed)
    b = em._SmfBuilder("x", DIM, seq_len=SEQ, token_input=True)

    def tensor(name, dims, data=None):
        count = 1
        for d in dims:
            count *= d
        b.add_tensor(name, dims, data if data is not None
                     else rng.floats(count))
        return name

    tensor("emb", [vocab, DIM])
    b.add_op(em.OP_EMBEDDING, "embed", ["x", "emb"], "e")
    p = "l0."
    tensor(p + "ln1_g", [DIM], ones(DIM))
    b.add_op(em.OP_RMSNORM, p + "ln1", ["e", p + "ln1_g"], p + "n1")
    for w in ("wq", "wk", "wv"):
        tensor(p + w, [DIM, DIM])
        b.add_op(em.OP_MATMUL, p + "mm_" + w, [p + "n1", p + w], p + w[1])
    b.add_op(em.OP_ROPE, p + "rope_q", [p + "q"], p + "qr", attr0=HEADS)
    b.add_op(em.OP_ROPE, p + "rope_k", [p + "k"], p + "kr", attr0=HEADS)
    b.add_op(em.OP_ATTENTION, p + "attn", [p + "qr", p + "kr", p + "v"],
             p + "a", attr0=HEADS)
    tensor(p + "wo", [DIM, DIM])
    b.add_op(em.OP_MATMUL, p + "mm_wo", [p + "a", p + "wo"], p + "o")
    b.add_op(em.OP_ADD, p + "res1", ["e", p + "o"], p + "x1")
    tensor(p + "ln2_g", [DIM], ones(DIM))
    b.add_op(em.OP_RMSNORM, p + "ln2", [p + "x1", p + "ln2_g"], p + "n2")
    tensor(p + "w_gate", [DIM, FFN])
    tensor(p + "w_up", [DIM, FFN])
    b.add_op(em.OP_MATMUL, p + "mm_gate", [p + "n2", p + "w_gate"], p + "g")
    b.add_op(em.OP_MATMUL, p + "mm_up", [p + "n2", p + "w_up"], p + "u")
    b.add_op(em.OP_SILU, p + "silu", [p + "g"], p + "s")
    b.add_op(em.OP_MUL, p + "swiglu", [p + "s", p + "u"], p + "m")
    tensor(p + "w_down", [FFN, DIM])
    b.add_op(em.OP_MATMUL, p + "mm_down", [p + "m", p + "w_down"], p + "d")
    b.add_op(em.OP_ADD, p + "res2", [p + "x1", p + "d"], p + "x2")
    tensor("lnf_g", [DIM], ones(DIM))
    b.add_op(em.OP_RMSNORM, "lnf", [p + "x2", "lnf_g"], "nf")
    tensor("w_head", [DIM, vocab])
    b.add_op(em.OP_MATMUL, "mm_head", ["nf", "w_head"], "logits")
    b.tokenizer = tokenizer
    return b


def decoder_smf(vocab, seed, tokenizer=b""):
    return decoder_builder(vocab, seed, tokenizer).serialize()


def head_only_section():
    """A tokenizer section that is its head alone: a vocabulary of VOCAB
    ids and an EOS — what the segment mask needs of a section."""
    return formats.SMTK_HEAD.pack(
        formats.SMTK_MAGIC, formats.SMTK_VERSION, 1, 1, 0, VOCAB, 0, 0, EOS,
        -1, -1, -1, 0)


def text_section(work):
    """A byte-level BPE over the 256 bytes with two merges, an EOS and the
    two ChatML markers (so a prompt / response document is ingestible)."""
    b2u = {v: k for k, v in ts._UNICODE_TO_BYTE.items()}
    specials = ["<|eot|>", "<|im_start|>", "<|im_end|>"]
    vocab = {s: i for i, s in enumerate(specials)}
    for byte in range(256):
        vocab[b2u[byte]] = len(specials) + byte
    vocab["th"] = len(vocab)
    vocab["the"] = len(vocab)
    tok = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": i, "content": s, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for i, s in enumerate(specials)],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "ByteLevel", "add_prefix_space": False,
             "trim_offsets": True, "use_regex": True}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False,
                  "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": ["t h", "th e"]},
    }
    tok_dir = os.path.join(work, "tok")
    os.makedirs(tok_dir, exist_ok=True)
    with open(os.path.join(tok_dir, "tokenizer.json"), "w") as f:
        json.dump(tok, f)
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"eos_token": "<|eot|>"}, f)
    return ts.build_section(tok_dir), len(vocab)


def export_report(smf, section):
    """The exporter's report for exactly these bytes, as
    export_model.write_export_report spells it."""
    report = {
        "schema": 1,
        "smf_hash": f"{content_hash.content_hash64(smf):016x}",
        "tokenizer_hash": f"{content_hash.content_hash64(section):016x}",
        "tokenizer_parity_checked": True,
        "tokenizer_parity_ids": 64,
        "logits_parity_checked": True,
        "logits_parity_max_delta": 0.0001,
    }
    return (json.dumps(report, indent=1) + "\n").encode()


# --- Corpora -------------------------------------------------------------------
def class_sds(rows=8):
    rng = Lcg(2)
    out = formats.SDS_HEADER.pack(formats.SDS_MAGIC, 1, rows, MLP_IN, 1, 0, 0)
    for r in range(rows):
        out += rng.floats(MLP_IN) + struct.pack("<i", r % MLP_OUT)
    return out


def dense_sds(rows=8):
    rng = Lcg(3)
    out = formats.SDS_HEADER.pack(formats.SDS_MAGIC, 1, rows, MLP_IN, 2, 0,
                                  MLP_OUT)
    for _ in range(rows):
        out += rng.floats(MLP_IN) + rng.floats(MLP_OUT)
    return out


def token_records(rows, seed, eos_every=0):
    rng = Lcg(seed)
    records = []
    for r in range(rows):
        ids = [EOS + 1 + rng.below(VOCAB - EOS - 1) for _ in range(SEQ + 1)]
        if eos_every and r % eos_every == 0:
            ids[2] = EOS  # a document ends inside the window
        records.append(ids)
    return records


def token_sds(records, mask=False):
    version = 3 if mask else 2
    out = formats.SDS_HEADER.pack(formats.SDS_MAGIC, version, len(records),
                                  SEQ, 1, 1, 0)
    if mask:
        out += formats.SDS_HEADER_V3_TAIL.pack(0, formats.SDS_FLAG_MASKED, 0,
                                               0)
    pad = formats.sds_mask_bytes(SEQ)
    for r, ids in enumerate(records):
        out += struct.pack(f"<{SEQ + 1}i", *ids)
        if mask:
            # A prompt-shaped mask: the first positions are not targets.
            flags = [0 if k < 1 + r % 2 else 1 for k in range(SEQ)]
            out += bytes(flags) + b"\x00" * (pad - SEQ)
    return out


JSONL = "".join(json.dumps(doc) + "\n" for doc in [
    {"text": "the cat sat by the door and ate 3 plums."},
    {"text": "the owl sat by the barn; the fox by the gate."},
    {"prompt": "where is the cat?", "response": "by the door."},
    {"text": "the cat sat by the door and ate 3 plums."},  # a duplicate
    {"text": "café ☃ tab\there"},
]).encode()
PLAIN_JSONL = "".join(json.dumps({"text": t}) + "\n" for t in [
    "the cat sat by the door and ate 3 plums.",
    "the owl sat by the barn; the fox by the gate.",
    "the dog sat by the tree and the dog ate 12 plums.",
]).encode()
TXT = ("the cat sat by the door and the cat ate 3 plums. "
       "the owl sat by the barn.").encode()

RECIPE = (json.dumps({
    "data_batch": 8, "steps": 8, "grad_accum": 2, "loss": "xent",
    "lora_rank": 2, "lora_alpha": 4.0, "lora_scaling": "alpha_over_sqrt_r",
    "lora_seed": 7, "targets": ["wq", "wv"], "optimizer": "adamw",
    "lr": 0.002, "weight_decay": 0.01, "clip_norm": 0.5,
    "clip_mode": "global", "lr_schedule": "cosine", "warmup": 2,
    "min_lr_factor": 0.1, "allow_zero_lr": False, "quantize_base": False,
    "bf16_base": False, "precision": "f32", "attention": "auto",
    "attention_cache_budget_mib": 1, "remat": "auto", "adapter_only": False,
    "validation_scores": "shipped", "masked_loss": False,
    "segment_mask": "auto", "memory_budget_mib": 64,
}, indent=1) + "\n").encode()
# What the data-compile arm's seeds compile under.
RECIPE_TEXT = (json.dumps({
    "data_batch": 2 * TEXT_SEQ, "steps": 4, "lora_rank": 2,
    "val_frac": 0.25, "split_seed": 1, "memory_budget_mib": 64,
}) + "\n").encode()
RECIPE_SDS = (json.dumps({
    "data_batch": 2 * SEQ, "steps": 4, "lora_rank": 2,
    "memory_budget_mib": 64,
}) + "\n").encode()

# --- Plans ---------------------------------------------------------------------
# (seed name, model, corpus or None, data batch, compiler flags): one per
# fixture family. The memory budget is pinned so the gate does not read the
# host's memory; everything else the compiler reads of its host is why a
# plan is held to acceptance, not to its bytes.
DEC = ["--lora-rank", "2", "--steps", "8"]
PLANS = [
    ("plan-mlp", "mlp", None, 4, ["--clip-norm", "0.5", "--steps", "8"]),
    ("plan-mlp-mse-sgd", "mlp", None, 4,
     ["--loss", "mse", "--optimizer", "sgd"]),
    ("plan-mlp-clip-global", "mlp", None, 4,
     ["--clip-mode", "global", "--clip-norm", "0.05"]),
    ("plan-mlp-accum", "mlp", None, 4, ["--grad-accum", "2"]),
    ("plan-mlp-int8", "mlp", None, 4, ["--quantize-base"]),
    # 8-bit moments (#192, plan v27): kAdamW8Step with the fused clip.
    ("plan-mlp-moments-int8", "mlp", None, 4,
     ["--moments", "int8", "--clip-norm", "0.5", "--steps", "8"]),
    ("plan-dec-cached", "dec", None, 2 * SEQ, DEC + ["--attention", "cached"]),
    ("plan-dec-tiled", "dec", None, 2 * SEQ, DEC + ["--attention", "tiled"]),
    ("plan-dec-bf16", "dec", None, 2 * SEQ, DEC + ["--bf16-base"]),
    ("plan-dec-relaxed", "dec", None, 2 * SEQ,
     DEC + ["--precision", "certified-bf16", "--attention", "tiled"]),
    ("plan-dec-masked", "dec", "masked.sds", 2 * SEQ,
     DEC + ["--attention", "cached"]),
    ("plan-dec-segment", "segdec", None, 2 * SEQ,
     DEC + ["--segment-mask", "on", "--attention", "cached"]),
    ("plan-dec-segment-tiled", "segdec", None, 2 * SEQ,
     DEC + ["--segment-mask", "on", "--attention", "tiled"]),
    ("plan-dec-clip-global-accum", "dec", None, 2 * SEQ,
     DEC + ["--clip-mode", "global", "--clip-norm", "0.05", "--grad-accum",
            "2"]),
]


def compile_plan(compiler, work, name, model, corpus, batch, flags):
    out = os.path.join(work, name)
    cmd = [compiler, "--source", os.path.join(work, model + ".smf"), "--out",
           out, "--data-batch", str(batch), "--no-embed",
           "--memory-budget-mib", "64"] + flags
    if corpus:
        cmd += ["--data", os.path.join(work, corpus)]
    done = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if done.returncode != 0:
        raise SystemExit(f"make_seeds: compiling {name} failed\n{done.stdout}")
    with open(os.path.join(out, "update_plan.seeu"), "rb") as f:
        return f.read()


def is_plan_seed(name):
    return name.startswith("plan-")


# --- Regression inputs ---------------------------------------------------------
REGRESSIONS = "regressions"


def poke_header(seed, field, value):
    """A plan seed with one header field rewritten (the seal left stale:
    the plan arm re-seals)."""
    h = formats.PLAN_HEADER
    at, size = 1 + h.offsets[field], h.sizes[field]
    return seed[:at] + value.to_bytes(size, "little") + seed[at + size:]


def build_regressions(seeds):
    """{name: bytes} for seeds/regressions/: inputs that once hung or
    crashed the harness, each derived from a seed so that a plan-version
    bump regenerates them with everything else — but for an input a
    fuzzing run itself wrote out, which is kept byte for byte. Named by
    libFuzzer's prefix for the kind of finding."""
    return {
        # CR3 #207: one bit of grad_accum_steps (2 -> 2^30 + 2) in an
        # accumulating plan. One optimizer step was then a billion passes of
        # the train program, which the harness ran; since #223 the load
        # contract bounds the count (kMaxGradAccumSteps = 2^16) and refuses
        # this plan by name.
        "slow-unit-grad-accum": poke_header(
            seeds["plan-mlp-accum"], "grad_accum_steps", 2 + (1 << 30)),
        # #223: one bit lower (2 -> 2^15 + 2) the count is inside the
        # contract's bound, and a step is still 32,770 train passes — too
        # slow for one fuzz unit. The harness's own, tighter cap turns it
        # away: accepted, and nothing run.
        "slow-unit-grad-accum-in-contract": poke_header(
            seeds["plan-mlp-accum"], "grad_accum_steps", 2 + (1 << 15)),
        # CR3 #207, acceptance 3: the nightly's hang, reproduced on
        # 2026-11-24 by running main 043112d's harness over the nightly's
        # cached corpus (3,589 units, cache key fuzz-corpus-44918953982)
        # with a 120 s unit limit: libFuzzer wrote this unit out at 164 s,
        # two mutations from a unit of that corpus. A 325-byte plan — a v24
        # header whose two instruction streams lie inside the header
        # itself, a kNop and a kSqNormAccum over 786,432 floats — with
        # grad_accum_steps = 0x41000000: 1,090,519,040 passes of the train
        # program for one step. It is the candidate cause of 2026-11-20
        # reproduced where the nightlies hung; whether the cancelled nights
        # held these very bytes cannot be known (they uploaded nothing).
        # Since #223 the load contract refuses it by name. Verbatim but for
        # the selector byte: 137 selected the plan arm of the four-arm
        # harness (137 % 4), and is written here as the nine-arm one's (1).
        # Not derived from a seed: it is evidence, kept as it was found.
        "timeout-nightly-grad-accum": bytes.fromhex(
            "01534545551800000000005a0000000000000000000000000000100000000000"
            "000000100000000000000000000000ffffff1000000000000000000000000000"
            "4100000000000000007800000000000000020000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000010000000000000000000000000000000000000000000010000000000000"
            "0000000000ff0002000002020000000000000001000000000035000000000000"
            "0010000000000000000000000000000000000010000000000000000001000000"
            "0000000c0000000000000000000000ffffff1000000000000000000000000000"
            "4100000000000000007800000000000000020000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "000008002c00"),
    }


def build_seeds(compiler):
    """{seed name: bytes}, every seed of the corpus."""
    seeds = {}
    with tempfile.TemporaryDirectory(prefix="seeml-fuzz-seeds-") as work:
        section, text_vocab = text_section(work)
        seg_section = head_only_section()
        models = {
            "mlp": mlp_smf(),
            "dec": decoder_smf(VOCAB, 4),
            "segdec": decoder_smf(VOCAB, 5, seg_section),
            "textdec": decoder_smf(text_vocab, 6, section),
        }
        reports = {"segdec": export_report(models["segdec"], seg_section),
                   "textdec": export_report(models["textdec"], section)}
        corpora = {
            "class.sds": class_sds(),
            "dense.sds": dense_sds(),
            "tokens.sds": token_sds(token_records(4, 7)),
            "masked.sds": token_sds(token_records(4, 8), mask=True),
            "segments.sds": token_sds(token_records(4, 9, eos_every=2)),
        }
        for name, data in models.items():
            path = os.path.join(work, name + ".smf")
            with open(path, "wb") as f:
                f.write(data)
            if name in reports:
                with open(path + ".export.json", "wb") as f:
                    f.write(reports[name])
        for name, data in corpora.items():
            with open(os.path.join(work, name), "wb") as f:
                f.write(data)

        # [0] the model reader, [3] the compile with no corpus.
        for name in ("mlp", "dec", "segdec", "textdec"):
            seeds["smf-" + name] = frame(ARM_SMF, models[name])
        for name in ("mlp", "dec", "segdec"):
            seeds["compile-" + name] = frame(ARM_COMPILE, models[name])
        # [2] the corpus reader.
        for name, data in corpora.items():
            seeds["sds-" + name[:-len(".sds")]] = frame(ARM_SDS, data)
        # [5] the tokenizer section and the text ingest.
        seeds["tokenizer-jsonl"] = frame(
            ARM_TOKENIZER, section, bytes([TEXT_JSONL, TEXT_SEQ]) + JSONL)
        seeds["tokenizer-txt"] = frame(
            ARM_TOKENIZER, section, bytes([TEXT_ONE_DOCUMENT, TEXT_SEQ]) + TXT)
        # [6] the recipe file.
        seeds["recipe"] = frame(ARM_RECIPE, RECIPE)
        # [7] the export report, held to its model.
        seeds["report-segdec"] = frame(ARM_REPORT, models["segdec"],
                                       reports["segdec"])
        seeds["report-textdec"] = frame(ARM_REPORT, models["textdec"],
                                        reports["textdec"])
        # [8] the compile with a recipe and a corpus.
        seeds["compile-data-jsonl"] = frame(
            ARM_COMPILE_DATA, models["textdec"], RECIPE_TEXT,
            bytes([DATA_JSONL]) + PLAIN_JSONL)
        seeds["compile-data-chat"] = frame(
            ARM_COMPILE_DATA, models["textdec"], RECIPE_TEXT,
            bytes([DATA_JSONL]) + JSONL)
        seeds["compile-data-txt"] = frame(
            ARM_COMPILE_DATA, models["textdec"], RECIPE_TEXT,
            bytes([DATA_TXT]) + TXT)
        seeds["compile-data-sds"] = frame(
            ARM_COMPILE_DATA, models["segdec"], RECIPE_SDS,
            bytes([DATA_SDS]) + corpora["segments.sds"])
        seeds["compile-data-masked"] = frame(
            ARM_COMPILE_DATA, models["dec"], RECIPE_SDS,
            bytes([DATA_SDS]) + corpora["masked.sds"])
        seeds["compile-data-class"] = frame(
            ARM_COMPILE_DATA, models["mlp"],
            b'{"data_batch": 4, "memory_budget_mib": 64}\n',
            bytes([DATA_SDS]) + corpora["class.sds"])
        # [1] one sealed plan per fixture family; [4] the same bytes with no
        # re-seal, one family being enough to hold the seal check's accept.
        for name, model, corpus, batch, flags in PLANS:
            plan = compile_plan(compiler, work, name, model, corpus, batch,
                                flags)
            seeds[name] = frame(ARM_PLAN, plan)
        seeds["plan-raw-mlp"] = frame(ARM_PLAN_RAW, seeds["plan-mlp"][1:])
    return seeds


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", default=os.environ.get(
        "SEEML_BUILD_DIR", os.path.join(REPO, "build")),
        help="the directory holding seeml-update-compile")
    ap.add_argument("--out", default=SEEDS, help="the seed directory")
    ap.add_argument("--check", action="store_true",
                    help="write nothing: fail when --out is not what this "
                    "script writes (plans and the regression inputs "
                    "derived from them: the same names; every other seed: "
                    "the same bytes)")
    args = ap.parse_args(argv)
    compiler = os.path.join(args.build, "seeml-update-compile")
    if not os.path.isfile(compiler):
        raise SystemExit(f"make_seeds: no built {compiler}")
    seeds = build_seeds(compiler)
    regressions = build_regressions(seeds)
    reg_dir = os.path.join(args.out, REGRESSIONS)
    if args.check:
        have = {n for n in os.listdir(args.out)
                if os.path.isfile(os.path.join(args.out, n))}
        problems = [f"{n}: not in {args.out}" for n in sorted(set(seeds) - have)]
        problems += [f"{n}: not a seed this script writes"
                     for n in sorted(have - set(seeds))]
        for name in sorted(have & set(seeds)):
            if is_plan_seed(name):
                continue
            with open(os.path.join(args.out, name), "rb") as f:
                if f.read() != seeds[name]:
                    problems.append(f"{name}: bytes differ")
        # The regression inputs derive from plans: held to their names.
        have_reg = set(os.listdir(reg_dir)) if os.path.isdir(reg_dir) else set()
        problems += [f"{REGRESSIONS}/{n}: not in {args.out}"
                     for n in sorted(set(regressions) - have_reg)]
        problems += [f"{REGRESSIONS}/{n}: not an input this script writes"
                     for n in sorted(have_reg - set(regressions))]
        for p in problems:
            print("make_seeds: " + p, file=sys.stderr)
        if problems:
            print("make_seeds: regenerate with `python3 "
                  "test/fuzz/make_seeds.py`", file=sys.stderr)
        return 1 if problems else 0
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.out):
        path = os.path.join(args.out, name)
        if os.path.isfile(path):
            os.remove(path)
    for name, data in seeds.items():
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(data)
    os.makedirs(reg_dir, exist_ok=True)
    for name in os.listdir(reg_dir):
        os.remove(os.path.join(reg_dir, name))
    for name, data in regressions.items():
        with open(os.path.join(reg_dir, name), "wb") as f:
            f.write(data)
    total = sum(len(d) for d in seeds.values())
    print(f"wrote {len(seeds)} seeds, {total} bytes, and "
          f"{len(regressions)} regression input(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
