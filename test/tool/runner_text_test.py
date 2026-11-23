"""Raw text on the device (#202 item 2): model_update --data takes a .txt or
a .jsonl as well as a pre-tokenized .sds, ingested with the model's own
tokenizer section. The acceptance: the committed model of a run on the raw
text is byte-identical to the committed model of the same package run on
the canonical SDS the compiler wrote from that text (pkg/corpus.sds) —
also for a corpus with duplicate windows and for a dirty .txt, which both
sides sanitize and deduplicate by the same vendored stages (#222), and for
a chat corpus on an instruct-style tokenizer, whose documents both sides
join by the same document separator (#218).
"""
import hashlib
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
sys.path.insert(0, HERE)

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
CXX = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
       or shutil.which("g++"))
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

TEXT = " ".join(
    f"the {w} sat by the {v} and the {w} ate {n} plums."
    for n, (w, v) in enumerate([("cat", "door"), ("dog", "tree"),
                                ("owl", "barn"), ("fox", "gate")] * 40))
# TEXT with what the sanitizer removes (#222): a control byte, a bidi
# override (U+202E) and two bytes of invalid UTF-8.
DIRTY = (TEXT[:100].encode() + b"\x01" + TEXT[100:300].encode() +
         "\u202e".encode() + TEXT[300:500].encode() + b"\xff\xfe" +
         TEXT[500:].encode())


def records(path):
    """The records of an unmasked token SDS, each as its bytes."""
    with open(path, "rb") as f:
        raw = f.read()
    _, ver, n, seq, _, _, _ = struct.unpack_from("<IIQQIIQ", raw, 0)
    head, width = (64 if ver >= 3 else 40), (seq + 1) * 4
    return [raw[head + i * width:head + (i + 1) * width] for i in range(n)]


def masked_records(path):
    """The records of a token SDS, masked or not: each as its ids."""
    with open(path, "rb") as f:
        raw = f.read()
    _, ver, n, seq, _, _, _ = struct.unpack_from("<IIQQIIQ", raw, 0)
    flags = struct.unpack_from("<I", raw, 48)[0] if ver >= 3 else 0
    head = 64 if ver >= 3 else 40
    stride = (seq + 1) * 4 + (((seq + 3) & ~3) if flags & 1 else 0)
    return [list(struct.unpack_from(f"<{seq + 1}i", raw, head + i * stride))
            for i in range(n)]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@unittest.skipUnless(os.path.isfile(COMPILER) and HAVE_NUMPY,
                     "needs the built seeml-update-compile and NumPy")
class RunnerRawTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_model as em
        from seeml import tokenizer_section as ts
        from tokenizer_test import tiny_tokenizer_json
        cls.em = em
        cls.dir = tempfile.mkdtemp(prefix="seeml-runner-text-")
        # tearDownClass does not run when setUpClass raises: clean up here,
        # as runner_stop_test does.
        try:
            cls._make_fixtures(em, ts, tiny_tokenizer_json)
        except BaseException:
            shutil.rmtree(cls.dir, True)
            raise

    @classmethod
    def _make_fixtures(cls, em, ts, tiny_tokenizer_json):
        tok_dir = os.path.join(cls.dir, "tok")
        os.makedirs(tok_dir)
        with open(os.path.join(tok_dir, "tokenizer.json"), "w") as f:
            json.dump(tiny_tokenizer_json(), f)
        # The EOS that joins a .jsonl's documents (the ingest refuses a
        # multi-document corpus without one).
        with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"eos_token": "<|eot|>"}, f)
        section = ts.build_section(tok_dir)
        vocab, dim, ffn = 259, 8, 16
        rng = np.random.default_rng(0)

        def mat(rows, cols):
            return (rng.standard_normal((rows, cols)) * 0.15).astype(np.float32)
        blocks = [{
            "ln1_g": np.ones(dim, np.float32), "wq": mat(dim, dim),
            "wk": mat(dim, dim), "wv": mat(dim, dim), "wo": mat(dim, dim),
            "ln2_g": np.ones(dim, np.float32), "w_gate": mat(dim, ffn),
            "w_up": mat(dim, ffn), "w_down": mat(ffn, dim),
        }]
        head = {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, vocab)}
        cls.model = os.path.join(cls.dir, "m.smf")
        em.export_token_decoder_smf(mat(vocab, dim), blocks, head, cls.model,
                                    seq_len=4, num_heads=2,
                                    tokenizer_section=section)
        # A model with a tokenizer section compiles only beside the report
        # the exporter writes for exactly these bytes (S4 #150).
        em.write_export_report(cls.model, section)
        cls.txt = os.path.join(cls.dir, "story.txt")
        with open(cls.txt, "w") as f:
            f.write(TEXT)
        cls.dirty = os.path.join(cls.dir, "dirty.txt")
        with open(cls.dirty, "wb") as f:
            f.write(DIRTY)
        # An instruct-style tokenizer (#218): the EOS is the ChatML turn end
        # <|im_end|>, and <|endoftext|> (id 0) is in the vocabulary — the
        # document separator both sides join a .jsonl's documents by.
        inst_dir = os.path.join(cls.dir, "inst_tok")
        os.makedirs(inst_dir)
        inst = tiny_tokenizer_json()
        vocab_map = dict(inst["model"]["vocab"])
        del vocab_map["<|eot|>"]
        vocab_map["<|endoftext|>"] = 0
        vocab_map["<|im_start|>"] = 259
        vocab_map["<|im_end|>"] = 260
        inst["model"]["vocab"] = vocab_map
        inst["added_tokens"] = [
            dict(inst["added_tokens"][0], id=i, content=c)
            for i, c in ((0, "<|endoftext|>"), (259, "<|im_start|>"),
                         (260, "<|im_end|>"))]
        with open(os.path.join(inst_dir, "tokenizer.json"), "w") as f:
            json.dump(inst, f)
        with open(os.path.join(inst_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"eos_token": "<|im_end|>"}, f)
        inst_section = ts.build_section(inst_dir)
        inst_vocab = 261
        cls.inst_model = os.path.join(cls.dir, "inst.smf")
        em.export_token_decoder_smf(
            mat(inst_vocab, dim), blocks,
            {"lnf_g": np.ones(dim, np.float32), "w_head": mat(dim, inst_vocab)},
            cls.inst_model, seq_len=8, num_heads=2,
            tokenizer_section=inst_section)
        em.write_export_report(cls.inst_model, inst_section)
        # Prompt/response records and plain text already rendered in ChatML:
        # each holds <|im_end|> after every turn.
        cls.chat = os.path.join(cls.dir, "chat.jsonl")
        with open(cls.chat, "w") as f:
            for n, (w, v) in enumerate([("cat", "door"), ("dog", "tree"),
                                        ("owl", "barn"), ("fox", "gate")] * 6):
                if n % 2:
                    f.write(json.dumps({
                        "prompt": f"where is the {w} {n}?",
                        "response": f"the {w} is by the {v}."}) + "\n")
                else:
                    f.write(json.dumps({"text": (
                        f"<|im_start|>user\nwhere is the {w} {n}?<|im_end|>\n"
                        f"<|im_start|>assistant\nthe {w} is by the {v}."
                        "<|im_end|>\n")}) + "\n")
        cls.jsonl = os.path.join(cls.dir, "docs.jsonl")
        with open(cls.jsonl, "w") as f:
            for i in range(0, len(TEXT), 120):
                f.write(json.dumps({"text": TEXT[i:i + 120]}) + "\n")
            f.write(json.dumps({"text": TEXT[:120]}) + "\n")  # a duplicate

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def package(self, name, data, model=None):
        pkg = os.path.join(self.dir, name)
        r = subprocess.run(
            [COMPILER, "--source", model or self.model, "--out", pkg, "--data",
             data, "--data-batch", "16", "--steps", "30", "--build"],
            capture_output=True, text=True, env=dict(os.environ, CXX=CXX))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(os.path.isfile(os.path.join(pkg, "corpus.sds")))
        self.compile_stderr = r.stderr
        return pkg

    def update(self, pkg, data, out, model=None):
        return subprocess.run(
            [os.path.join(pkg, "model_update"), "--model", model or self.model,
             "--data", data, "--force", "--out", out],
            capture_output=True, text=True)

    def assert_text_run_is_the_sds_run(self, name, data, model=None):
        pkg = self.package(name, data, model)
        by_sds = os.path.join(self.dir, name + "_sds.smf")
        by_text = os.path.join(self.dir, name + "_text.smf")
        r = self.update(pkg, os.path.join(pkg, "corpus.sds"), by_sds, model)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        r = self.update(pkg, data, by_text, model)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("seeml-update: ingested '" + data + "'", r.stderr)
        self.assertEqual(sha256(by_text), sha256(by_sds))
        return r.stderr

    def test_a_corpus_that_cannot_take_its_name_is_not_a_short_write(self):
        # The compiler stages corpus.sds and renames it into place. With a
        # directory holding the name, every byte was written and the rename
        # is what failed: the refusal says that (it said "short write to"
        # until the /code-review of #213), exits 1, and leaves no staged
        # file behind.
        pkg = os.path.join(self.dir, "blocked")
        corpus = os.path.join(pkg, "corpus.sds")
        os.makedirs(corpus)
        r = subprocess.run(
            [COMPILER, "--source", self.model, "--out", pkg, "--data",
             self.txt, "--data-batch", "16", "--steps", "30", "--no-embed"],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr[-2000:])
        self.assertIn(f"seeml-update-compile: cannot move '{corpus}.tmp' to "
                      f"'{corpus}'\n", r.stderr)
        self.assertNotIn("short write", r.stderr)
        self.assertFalse(os.path.exists(corpus + ".tmp"))
        self.assertTrue(os.path.isdir(corpus))

    def manifest(self, pkg):
        with open(os.path.join(pkg, "corpus_manifest.json")) as f:
            return json.load(f)

    def test_a_txt_trains_as_its_canonical_sds(self):
        self.assert_text_run_is_the_sds_run("txt", self.txt)
        # The story repeats itself: windows of five ids recur, and both
        # sides drop the same ones (#222).
        man = self.manifest(os.path.join(self.dir, "txt"))
        self.assertGreater(man["dropped_duplicate_windows"], 0)
        recs = records(os.path.join(self.dir, "txt", "corpus.sds"))
        self.assertEqual(len(set(recs)), len(recs))

    def test_a_dirty_txt_trains_as_the_clean_texts_corpus(self):
        # #222: a .txt is sanitized as a .jsonl is — on the build host and,
        # by the same vendored function, on the device. The dirty text's
        # canonical corpus is the clean text's, byte for byte, and the
        # device's run on either text equals the run on that corpus.
        pkg = os.path.join(self.dir, "dirty")
        self.assert_text_run_is_the_sds_run("dirty", self.dirty)
        clean = os.path.join(self.dir, "clean_ref")
        r = subprocess.run(
            [COMPILER, "--source", self.model, "--out", clean, "--data",
             self.txt, "--data-batch", "16", "--steps", "30", "--no-embed"],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(sha256(os.path.join(pkg, "corpus.sds")),
                         sha256(os.path.join(clean, "corpus.sds")))
        man = self.manifest(pkg)
        self.assertEqual((man["invalid_utf8_bytes"], man["control_chars"],
                          man["bidi_controls"]), (2, 1, 1))
        by_sds = os.path.join(self.dir, "dirty_sds.smf")
        by_clean = os.path.join(self.dir, "dirty_clean_text.smf")
        r = self.update(pkg, self.txt, by_clean)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertEqual(sha256(by_clean), sha256(by_sds))

    def test_a_given_sds_with_twins_is_compiled_as_is_and_states_them(self):
        # #222: a given SDS is pinned by its hash, never rewritten; with
        # --val-frac the held-out records with a byte-identical twin in
        # training are counted on the screen and in the report.
        src = os.path.join(self.dir, "twins_src")
        r = subprocess.run(
            [COMPILER, "--source", self.model, "--out", src, "--data",
             self.txt, "--data-batch", "16", "--steps", "30", "--no-embed"],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        with open(os.path.join(src, "corpus.sds"), "rb") as f:
            raw = f.read()
        recs = records(os.path.join(src, "corpus.sds"))[:8]
        twins = os.path.join(self.dir, "twins.sds")
        with open(twins, "wb") as f:  # every record twice
            f.write(raw[:8] + struct.pack("<Q", 16) + raw[16:64] +
                    b"".join(recs) + b"".join(recs))
        before = sha256(twins)
        pkg = os.path.join(self.dir, "twins_pkg")
        report = os.path.join(self.dir, "twins_report.json")
        r = subprocess.run(
            [COMPILER, "--source", self.model, "--out", pkg, "--data", twins,
             "--val-frac", "0.25", "--split-seed", "3", "--data-batch", "16",
             "--steps", "30", "--no-embed", "--report", report],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(sha256(twins), before)
        self.assertFalse(os.path.exists(os.path.join(pkg, "corpus.sds")))
        with open(report) as f:
            data = json.load(f)["data"]
        sys.path.insert(0, os.path.join(REPO, "tool"))
        import frontier_exec as fx
        held = fx.held_out_indices(16, 3, 0.25)
        expect = sum(1 for i in held if (i ^ 8) not in held)
        self.assertGreater(expect, 0)
        self.assertEqual(data["held_out_twins"], expect)
        self.assertIn("held-out twins", r.stderr)
        self.assertRegex(r.stderr, rf"held-out twins +{expect}\b")

    def test_a_jsonl_trains_as_its_canonical_sds(self):
        err = self.assert_text_run_is_the_sds_run("jsonl", self.jsonl)
        self.assertIn("1 duplicate", err)

    def test_a_chat_corpus_on_an_instruct_tokenizer_trains_as_its_sds(self):
        # #218: on a tokenizer whose EOS is the turn end <|im_end|>, both
        # sides join documents by <|endoftext|> (id 0), the plan's segment
        # mask scans for it — `auto` masks — and the device's run on the
        # raw text equals its run on the compiler's corpus.
        self.assert_text_run_is_the_sds_run("chat", self.chat,
                                            self.inst_model)
        self.assertRegex(self.compile_stderr, r"segment mask +on")
        recs = masked_records(os.path.join(self.dir, "chat", "corpus.sds"))
        ids = [i for rec in recs for i in rec]
        self.assertIn(0, ids)    # the separator joins the documents
        self.assertIn(260, ids)  # and the turn ends stay inside them

    def test_raw_text_needs_a_plan_with_a_tokenizer(self):
        # A decoder exported without a tokenizer section compiles from an
        # SDS; its plan names no tokenizer, so raw text is refused by name.
        d = os.path.join(self.dir, "bare")
        os.makedirs(d)
        self.em._demo_decoder(d, vocab=40, dim=24, heads=4, seq=6,
                              blocks_n=2, samples=48, seed=1)
        pkg = os.path.join(d, "pkg")
        r = subprocess.run(
            [COMPILER, "--source", os.path.join(d, "decoder.smf"), "--out", pkg,
             "--data-batch", "12", "--steps", "8", "--build"],
            capture_output=True, text=True, env=dict(os.environ, CXX=CXX))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        r = subprocess.run(
            [os.path.join(pkg, "model_update"), "--model",
             os.path.join(d, "decoder.smf"), "--data", self.txt,
             "--out", os.path.join(d, "u.smf")],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr[-1500:])
        self.assertIn("the plan names no tokenizer", r.stderr)


if __name__ == "__main__":
    unittest.main()
