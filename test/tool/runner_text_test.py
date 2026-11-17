"""Raw text on the device (#202 item 2): model_update --data takes a .txt or
a .jsonl as well as a pre-tokenized .sds, ingested with the model's own
tokenizer section. The acceptance: the committed model of a run on the raw
text is byte-identical to the committed model of the same package run on
the canonical SDS the compiler wrote from that text (pkg/corpus.sds).
"""
import hashlib
import json
import os
import shutil
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
        cls.jsonl = os.path.join(cls.dir, "docs.jsonl")
        with open(cls.jsonl, "w") as f:
            for i in range(0, len(TEXT), 120):
                f.write(json.dumps({"text": TEXT[i:i + 120]}) + "\n")
            f.write(json.dumps({"text": TEXT[:120]}) + "\n")  # a duplicate

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def package(self, name, data):
        pkg = os.path.join(self.dir, name)
        r = subprocess.run(
            [COMPILER, "--source", self.model, "--out", pkg, "--data", data,
             "--data-batch", "16", "--steps", "30", "--build"],
            capture_output=True, text=True, env=dict(os.environ, CXX=CXX))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(os.path.isfile(os.path.join(pkg, "corpus.sds")))
        return pkg

    def update(self, pkg, data, out):
        return subprocess.run(
            [os.path.join(pkg, "model_update"), "--model", self.model,
             "--data", data, "--force", "--out", out],
            capture_output=True, text=True)

    def assert_text_run_is_the_sds_run(self, name, data):
        pkg = self.package(name, data)
        by_sds = os.path.join(self.dir, name + "_sds.smf")
        by_text = os.path.join(self.dir, name + "_text.smf")
        r = self.update(pkg, os.path.join(pkg, "corpus.sds"), by_sds)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        r = self.update(pkg, data, by_text)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("seeml-update: ingested '" + data + "'", r.stderr)
        self.assertEqual(sha256(by_text), sha256(by_sds))
        return r.stderr

    def test_a_txt_trains_as_its_canonical_sds(self):
        self.assert_text_run_is_the_sds_run("txt", self.txt)

    def test_a_jsonl_trains_as_its_canonical_sds(self):
        err = self.assert_text_run_is_the_sds_run("jsonl", self.jsonl)
        self.assertIn("1 duplicate", err)

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
