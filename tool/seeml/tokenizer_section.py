"""The SMF tokenizer section (SMTK, source/tokenizer/tokenizer_format.h),
built from a Hugging Face checkpoint's own tokenizer files (S2 #148).

The core implements one tokenizer family — byte-level BPE with the GPT-2
split pattern, optionally preceded by the Digits(individual) pre-tokenizer,
special tokens split first — and refuses every construct it does not
implement by name. This module writes only what the core reads: a
tokenizer.json that needs a normalizer, a different pre-tokenizer, a
non-BPE model, byte fallback, a dropout or an unknown token is refused here,
at export, with the construct named, never silently approximated.

Standard library only (tier 0): the exporter imports it, the tests too.
"""
import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

from . import formats

# The section's constants are formats.py's (held to the C++ header through
# abi.json); the names here are this module's spelling of them.
SMTK_MAGIC = formats.SMTK_MAGIC
SMTK_VERSION = formats.SMTK_VERSION
KIND_BYTE_LEVEL_BPE = formats.SMTK_KINDS["byte_level_bpe"]
PRE_GPT2 = formats.SMTK_PRE_TOKENIZERS["gpt2"]
PRE_DIGITS_THEN_GPT2 = formats.SMTK_PRE_TOKENIZERS["digits_then_gpt2"]
FLAG_ADD_PREFIX_SPACE = formats.SMTK_FLAGS["add_prefix_space"]
FLAG_NFC = formats.SMTK_FLAGS["nfc"]


class UnsupportedTokenizer(ValueError):
    """A tokenizer.json construct the core does not implement."""


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte -> printable-character table (the ByteLevel map)."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}


def token_bytes(token: str) -> bytes:
    """A byte-level token string back to the bytes it stands for."""
    out = bytearray()
    for ch in token:
        try:
            out.append(_UNICODE_TO_BYTE[ch])
        except KeyError:
            raise UnsupportedTokenizer(
                f"token {token!r} carries {ch!r}, which is not a byte-level "
                "character") from None
    return bytes(out)


def _pre_tokenizer_kind(pre: Optional[Dict[str, Any]]) -> int:
    """PRE_GPT2 or PRE_DIGITS_THEN_GPT2, or a refusal naming the construct."""
    def byte_level(p: Dict[str, Any]) -> None:
        if p.get("type") != "ByteLevel":
            raise UnsupportedTokenizer(
                f"pre-tokenizer {p.get('type')!r} is not implemented (the "
                "core runs the byte-level GPT-2 pattern)")
        if not p.get("use_regex", True):
            raise UnsupportedTokenizer(
                "ByteLevel without use_regex (an explicit Split pattern) is "
                "not implemented")
        if p.get("add_prefix_space", False):
            raise UnsupportedTokenizer(
                "ByteLevel add_prefix_space is not implemented")
    if pre is None:
        raise UnsupportedTokenizer("a tokenizer without a pre-tokenizer")
    if pre.get("type") == "ByteLevel":
        byte_level(pre)
        return PRE_GPT2
    if pre.get("type") == "Sequence":
        steps = pre.get("pretokenizers", [])
        if (len(steps) == 2 and steps[0].get("type") == "Digits" and
                steps[0].get("individual_digits", False)):
            byte_level(steps[1])
            return PRE_DIGITS_THEN_GPT2
        if len(steps) == 1:
            byte_level(steps[0])
            return PRE_GPT2
        raise UnsupportedTokenizer(
            "pre-tokenizer Sequence " +
            repr([s.get("type") for s in steps]) +
            " is not implemented (Digits(individual) then ByteLevel, or "
            "ByteLevel alone)")
    raise UnsupportedTokenizer(
        f"pre-tokenizer {pre.get('type')!r} is not implemented")


def build_section(model_dir: str) -> bytes:
    """The SMTK bytes for the checkpoint in `model_dir` (tokenizer.json and,
    when present, tokenizer_config.json for eos / bos / chat template)."""
    with open(os.path.join(model_dir, "tokenizer.json"), "r",
              encoding="utf-8") as f:
        tok = json.load(f)
    config: Dict[str, Any] = {}
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    if tok.get("normalizer") is not None:
        raise UnsupportedTokenizer(
            f"normalizer {tok['normalizer'].get('type')!r} is not implemented")
    pre = _pre_tokenizer_kind(tok.get("pre_tokenizer"))
    post = tok.get("post_processor")
    if post is not None and post.get("type") != "ByteLevel":
        raise UnsupportedTokenizer(
            f"post-processor {post.get('type')!r} is not implemented (it "
            "would add tokens the core does not)")
    model = tok["model"]
    if model.get("type") != "BPE":
        raise UnsupportedTokenizer(
            f"model {model.get('type')!r} is not implemented (byte-level BPE "
            "only; SentencePiece / Unigram / WordPiece are refused)")
    for key in ("dropout", "unk_token", "continuing_subword_prefix",
                "end_of_word_suffix"):
        if model.get(key) not in (None, ""):
            raise UnsupportedTokenizer(f"BPE {key}={model[key]!r} is not implemented")
    if model.get("byte_fallback", False):
        raise UnsupportedTokenizer("BPE byte_fallback is not implemented")
    if model.get("fuse_unk", False):
        raise UnsupportedTokenizer("BPE fuse_unk is not implemented")
    if model.get("ignore_merges", False):
        raise UnsupportedTokenizer("BPE ignore_merges is not implemented")

    vocab: Dict[str, int] = model["vocab"]
    size = len(vocab)
    if sorted(vocab.values()) != list(range(size)):
        raise UnsupportedTokenizer("vocabulary ids are not 0..V-1")
    by_id = [""] * size
    for s, i in vocab.items():
        by_id[i] = s
    # Added tokens are strings in the vocabulary too (ids below size), not
    # byte-level; their bytes are the UTF-8 of the string itself.
    added = {a["id"]: a for a in tok.get("added_tokens", [])}
    for a in added.values():
        # The reference extracts every added token whole before the
        # pre-tokenizer runs; the core matches by longest bytes alone, so
        # a token whose match depends on normalization, on stripping the
        # whitespace around it or on a word boundary (single_word) would
        # be found where the reference does not, or missed.
        if (a.get("normalized", False) or a.get("lstrip", False) or
                a.get("rstrip", False) or a.get("single_word", False)):
            raise UnsupportedTokenizer(
                f"added token {a['content']!r} with normalized / lstrip / "
                "rstrip / single_word is not implemented")
        if a["id"] >= size or by_id[a["id"]] != a["content"]:
            raise UnsupportedTokenizer(
                f"added token {a['content']!r} is not vocabulary id {a['id']}")
    token_blobs: List[bytes] = []
    for i, s in enumerate(by_id):
        token_blobs.append(s.encode("utf-8") if i in added else token_bytes(s))
    merges: List[Tuple[int, int, int]] = []
    for m in model["merges"]:
        parts = m.split(" ") if isinstance(m, str) else list(m)
        if len(parts) != 2:
            raise UnsupportedTokenizer(f"merge {m!r} is not a pair")
        a, b = parts
        try:
            merges.append((vocab[a], vocab[b], vocab[a + b]))
        except KeyError:
            raise UnsupportedTokenizer(
                f"merge {m!r} names a token outside the vocabulary") from None
    # SMTK's special list is the split-first list: EVERY added token, as the
    # reference's AddedVocabulary extracts them all (its `special` flag
    # changes no id; it only marks what decoding may skip).
    specials = sorted(added)

    def special_id(name: str) -> int:
        value = config.get(name)
        if isinstance(value, dict):
            value = value.get("content")
        if value is None:
            return -1
        if value not in vocab:
            raise UnsupportedTokenizer(
                f"{name} {value!r} is not in the vocabulary")
        return vocab[value]
    eos, bos, unk, pad = (special_id("eos_token"), special_id("bos_token"),
                          special_id("unk_token"), special_id("pad_token"))
    template = config.get("chat_template") or ""
    if not isinstance(template, str):
        raise UnsupportedTokenizer("a chat_template that is not one string")
    template_bytes = template.encode("utf-8")

    out = bytearray()
    out += formats.SMTK_HEAD.pack(SMTK_MAGIC, SMTK_VERSION, KIND_BYTE_LEVEL_BPE,
                                  pre, 0, size, len(merges), len(specials),
                                  eos, bos, unk, pad, len(template_bytes))
    out += template_bytes
    for blob in token_blobs:
        if len(blob) > 0xFFFF:
            raise UnsupportedTokenizer("a token longer than 65535 bytes")
        out += struct.pack("<H", len(blob)) + blob
    for a, b, c in merges:
        out += struct.pack("<III", a, b, c)
    for i in specials:
        out += struct.pack("<I", i)
    return bytes(out)


def summarize(section: bytes) -> Dict[str, Any]:
    """The head of a section as a dict (for reports and tests)."""
    if len(section) < formats.SMTK_HEAD.size:
        raise UnsupportedTokenizer(
            f"a {len(section)}-byte section is shorter than the "
            f"{formats.SMTK_HEAD.size}-byte head")
    (magic, version, kind, pre, flags, size, merges, specials, eos, bos, unk,
     pad, template_len) = formats.SMTK_HEAD.unpack_from(section, 0)
    return dict(magic=magic, version=version, kind=kind, pre_tokenizer=pre,
                flags=flags, vocab_size=size, num_merges=merges,
                num_special=specials, eos_id=eos, bos_id=bos, unk_id=unk,
                pad_id=pad, chat_template_bytes=template_len)
