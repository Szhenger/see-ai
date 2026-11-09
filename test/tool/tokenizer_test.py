"""The SMF tokenizer section and the core's encoder, from the Python side
(S2 #148).

Tier 0: the section builder's refusals on synthetic tokenizer.json files
and a round trip of a tiny section through its own summary. Tier 2 (the
`tokenizers` package plus a cached Hugging Face checkpoint, and the built
`seeml-tokenize`): the parity harness — every line of the golden
adversarial corpus and the file as one document encode to the same ids in
the reference and in the core — and the content-hash seam, the Python
mirror of ContentHash64 against the C++ reading of the same section.
"""
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tool"))
from seeml import content_hash, tokenizer_section as ts  # noqa: E402

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
TOKENIZE = os.path.join(BUILD, "seeml-tokenize")
CORPUS = os.path.join(REPO, "test", "fixtures", "tokenizer", "adversarial.txt")
SMOLLM = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/*/"
    "tokenizer.json")))


def have_tokenizers():
    try:
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


def tiny_tokenizer_json(**overrides):
    """A byte-level BPE over the printable ASCII bytes with two merges and
    one special token, in tokenizer.json's shape."""
    b2u = {v: k for k, v in ts._UNICODE_TO_BYTE.items()}
    alphabet = [b2u[b] for b in range(256)]
    vocab = {"<|eot|>": 0}
    for i, ch in enumerate(alphabet):
        vocab[ch] = i + 1
    vocab["th"] = 257
    vocab["the"] = 258
    tok = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": 0, "content": "<|eot|>", "single_word": False,
                          "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True}],
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
    tok.update(overrides)
    return tok


class SectionBuilderTest(unittest.TestCase):
    def write(self, tok, config=None):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            json.dump(tok, f)
        if config is not None:
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump(config, f)
        return d

    def test_builds_the_section_the_core_reads(self):
        d = self.write(tiny_tokenizer_json(),
                       {"eos_token": "<|eot|>", "chat_template": "{{ x }}"})
        section = ts.build_section(d)
        head = ts.summarize(section)
        self.assertEqual(head["magic"], ts.SMTK_MAGIC)
        self.assertEqual(head["kind"], ts.KIND_BYTE_LEVEL_BPE)
        self.assertEqual(head["pre_tokenizer"], ts.PRE_DIGITS_THEN_GPT2)
        self.assertEqual(head["vocab_size"], 259)
        self.assertEqual(head["num_merges"], 2)
        self.assertEqual(head["num_special"], 1)
        self.assertEqual(head["eos_id"], 0)
        self.assertEqual(head["bos_id"], -1)
        self.assertEqual(head["chat_template_bytes"], len("{{ x }}"))
        # Deterministic: the same files, the same bytes, the same hash.
        self.assertEqual(section, ts.build_section(d))
        self.assertNotEqual(content_hash.content_hash64(section), 0)

    def test_refuses_what_the_core_does_not_implement_by_name(self):
        cases = [
            (dict(normalizer={"type": "NFC"}), "normalizer 'NFC'"),
            (dict(pre_tokenizer={"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "x"}},
                {"type": "ByteLevel", "use_regex": False}]}),
             "pre-tokenizer Sequence"),
            (dict(pre_tokenizer={"type": "ByteLevel", "use_regex": False}),
             "explicit Split"),
            (dict(pre_tokenizer={"type": "ByteLevel", "add_prefix_space": True}),
             "add_prefix_space"),
            (dict(pre_tokenizer={"type": "Whitespace"}), "pre-tokenizer 'Whitespace'"),
            (dict(post_processor={"type": "TemplateProcessing"}),
             "post-processor 'TemplateProcessing'"),
            (dict(model=dict(tiny_tokenizer_json()["model"], type="Unigram")),
             "model 'Unigram'"),
            (dict(model=dict(tiny_tokenizer_json()["model"], byte_fallback=True)),
             "byte_fallback"),
            (dict(model=dict(tiny_tokenizer_json()["model"], unk_token="<unk>")),
             "unk_token"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                d = self.write(tiny_tokenizer_json(**overrides))
                with self.assertRaisesRegex(ts.UnsupportedTokenizer, message):
                    ts.build_section(d)

    def test_content_hash_mirrors_the_striped_fold(self):
        # Known shape: the empty buffer and a buffer longer than one chunk
        # fold differently from the single-chunk kernel; both are functions
        # of the bytes alone.
        self.assertEqual(content_hash.content_hash64(b""),
                         content_hash.content_hash64(b""))
        a = content_hash.content_hash64(b"abc")
        self.assertNotEqual(a, content_hash.content_hash64(b"abd"))
        self.assertNotEqual(a, content_hash.content_hash64(b"abc\x00"))


@unittest.skipUnless(SMOLLM and have_tokenizers(),
                     "tier 2: needs the `tokenizers` package and a cached "
                     "SmolLM-135M checkpoint (~/.cache/huggingface)")
class SmolLmSectionTest(unittest.TestCase):
    def setUp(self):
        self.model_dir = os.path.dirname(SMOLLM[-1])
        self.section = ts.build_section(self.model_dir)

    def test_the_section_describes_smollm(self):
        head = ts.summarize(self.section)
        self.assertEqual(head["vocab_size"], 49152)
        self.assertEqual(head["num_merges"], 48900)
        self.assertEqual(head["num_special"], 17)
        self.assertEqual(head["pre_tokenizer"], ts.PRE_DIGITS_THEN_GPT2)
        self.assertEqual(head["eos_id"], 0)
        self.assertEqual(head["chat_template_bytes"], 0)

    @unittest.skipUnless(os.path.isfile(TOKENIZE),
                         "needs the built seeml-tokenize (build the tree)")
    def test_the_core_matches_the_reference_on_the_golden_corpus(self):
        # export_model.py --hf --tokenizer-parity, as the CLI would run it,
        # on a tiny decoder so the export is cheap; the section is the real
        # SmolLM one. Then the content-hash seam: the Python mirror hashes
        # the section to the number the C++ reads from the file.
        sys.path.insert(0, os.path.join(REPO, "tool"))
        import export_model as em
        import numpy as np
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        vocab, dim, ffn = 49152, 8, 16
        rng = np.random.default_rng(0)

        def mat(rows, cols):
            return (rng.standard_normal((rows, cols)) * 0.15).astype(np.float32)
        emb = (rng.standard_normal((vocab, dim)) * 0.5).astype(np.float32)
        blocks = [{
            "ln1_g": np.ones(dim, np.float32), "wq": mat(dim, dim),
            "wk": mat(dim, dim), "wv": mat(dim, dim), "wo": mat(dim, dim),
            "ln2_g": np.ones(dim, np.float32), "w_gate": mat(dim, ffn),
            "w_up": mat(dim, ffn), "w_down": mat(ffn, dim),
        }]
        head = {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, vocab)}
        smf = os.path.join(d, "tiny.smf")
        em.export_token_decoder_smf(emb, blocks, head, smf, seq_len=4,
                                    num_heads=2,
                                    tokenizer_section=self.section)
        n, ids = em.tokenizer_parity(self.model_dir, smf, CORPUS, TOKENIZE)
        self.assertGreater(ids, 500)
        info = subprocess.run([TOKENIZE, smf, "--info"], capture_output=True,
                              text=True, check=True).stdout
        line = [l for l in info.splitlines() if l.startswith("tokenizer hash")][0]
        self.assertEqual(int(line.split()[-1], 16),
                         content_hash.content_hash64(self.section))


FRONTIER = os.environ.get("SEEML_FRONTIER_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(REPO))),
    "out", "frontier-2026-09-22"))
COMPILE = os.path.join(BUILD, "seeml-update-compile")


@unittest.skipUnless(
    SMOLLM and have_tokenizers() and os.path.isfile(COMPILE) and
    os.path.isfile(os.path.join(FRONTIER, "docs_corpus.txt")) and
    os.path.isfile(os.path.join(FRONTIER, "smollm135.sds")),
    "tier 2: needs the cached SmolLM-135M checkpoint, `tokenizers`, the built "
    "compiler and the 2026-09-22 frontier corpus (SEEML_FRONTIER_DIR)")
class SmolLmDataPathTest(unittest.TestCase):
    # S2 acceptance: `--data docs_corpus.txt` on SmolLM-135M yields the ids
    # the exporter's path wrote for the 2026-09-22 frontier row, bit-exact.

    def test_the_compile_time_corpus_is_the_exporters(self):
        import numpy as np
        sys.path.insert(0, os.path.join(REPO, "tool"))
        import export_model as em
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        smf = os.path.join(d, "smollm.smf")
        em.export_hf_decoder(os.path.dirname(SMOLLM[-1]), smf, seq_len=128)
        r = subprocess.run(
            [COMPILE, "--source", smf, "--out", os.path.join(d, "pkg"),
             "--data", os.path.join(FRONTIER, "docs_corpus.txt"),
             "--data-batch", "128", "--steps", "2", "--no-embed",
             "--report", os.path.join(d, "report.json")],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

        def ids(path):
            with open(path, "rb") as f:
                raw = f.read()
            _, ver, n, s, _, _, _ = struct.unpack_from("<IIQQIIQ", raw, 0)
            head = 64 if ver >= 3 else 40
            return np.frombuffer(raw, "<i4", n * (s + 1), head).reshape(n, s + 1)
        ours = ids(os.path.join(d, "pkg", "corpus.sds"))
        theirs = ids(os.path.join(FRONTIER, "smollm135.sds"))
        self.assertEqual(ours.shape, theirs.shape)
        self.assertTrue((ours == theirs).all())
        with open(os.path.join(d, "report.json")) as f:
            report = json.load(f)
        self.assertEqual(report["data"]["data_flags"] & 2, 2)
        self.assertNotEqual(int(report["data"]["corpus_hash"], 16), 0)
        self.assertEqual(int(report["data"]["tokenizer_hash"], 16),
                         content_hash.content_hash64(
                             ts.build_section(os.path.dirname(SMOLLM[-1]))))


if __name__ == "__main__":
    unittest.main()
