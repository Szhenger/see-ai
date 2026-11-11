"""The recipe at the command line (S4 #150): one recipe file compiles to the
plan its flags compile to, byte for byte; a flag overrides the field it
names; every input the review found accepted is refused naming the field;
and the manifest publishes the schema the compiler applies.
"""
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

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(os.path.isfile(COMPILER) and HAVE_NUMPY,
                     "needs the built seeml-update-compile and NumPy")
class RecipeCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_model as em
        cls.dir = tempfile.mkdtemp(prefix="seeml-recipe-test-")
        em._demo_decoder(cls.dir, vocab=40, dim=24, heads=4, seq=6,
                         blocks_n=2, samples=48, seed=1)
        cls.model = os.path.join(cls.dir, "decoder.smf")
        cls.corpus = os.path.join(cls.dir, "decoder_corpus.sds")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def compile(self, name, *args, model=None):
        out = os.path.join(self.dir, name)
        r = subprocess.run(
            [COMPILER, "--source", model or self.model, "--out", out,
             "--no-embed", "--report", os.path.join(self.dir, name + ".json"),
             *args], capture_output=True, text=True)
        return r, out

    def plan(self, out):
        with open(os.path.join(out, "update_plan.seeu"), "rb") as f:
            return f.read()

    def report(self, name):
        with open(os.path.join(self.dir, name + ".json")) as f:
            return json.load(f)

    def test_a_recipe_file_is_its_flags_byte_for_byte(self):
        recipe = {"data_batch": 12, "loss": "xent", "lora_rank": 4,
                  "lora_alpha": 8, "targets": ["wq", "wv"], "lr": 1e-4,
                  "lr_schedule": "cosine", "warmup": 2, "steps": 20,
                  "quantize_base": True, "clip_norm": 1.0}
        path = os.path.join(self.dir, "recipe.json")
        with open(path, "w") as f:
            json.dump(recipe, f)
        a, out_a = self.compile("by_file", "--recipe", path)
        self.assertEqual(a.returncode, 0, a.stderr[-1500:])
        b, out_b = self.compile(
            "by_flags", "--data-batch", "12", "--loss", "xent", "--lora-rank",
            "4", "--lora-alpha", "8", "--targets", "wq,wv", "--lr", "0.0001",
            "--lr-schedule", "cosine", "--warmup", "2", "--steps", "20",
            "--quantize-base", "--clip-norm", "1")
        self.assertEqual(b.returncode, 0, b.stderr[-1500:])
        self.assertEqual(self.plan(out_a), self.plan(out_b))
        ra, rb = self.report("by_file"), self.report("by_flags")
        self.assertEqual(ra["recipe_hash"], rb["recipe_hash"])
        self.assertEqual(ra["plan_hash"], rb["plan_hash"])
        self.assertNotEqual(int(ra["recipe_hash"], 16), 0)
        # A flag overrides the field it names: another rank, another recipe.
        c, out_c = self.compile("override", "--recipe", path, "--lora-rank", "2")
        self.assertEqual(c.returncode, 0, c.stderr[-1500:])
        self.assertNotEqual(self.report("override")["recipe_hash"],
                            ra["recipe_hash"])
        self.assertNotEqual(self.plan(out_c), self.plan(out_a))

    def test_every_reviewed_input_is_refused_naming_its_field(self):
        teacher = self.model  # any model file: the rule fires before loading
        cases = [
            (["--steps", " -1"], "--steps (recipe field steps) must be an integer"),
            (["--lora-seed", " -1"], "--lora-seed (recipe field lora_seed)"),
            (["--lora-rank", "5000"], "must be an integer in [1, 4096]"),
            (["--temperature", "2"], "applies to a distillation loss only"),
            (["--distill-weight", "0.5"], "applies to a distillation loss only"),
            (["--teacher", teacher], "--teacher applies to a distillation loss"),
            (["--loss", "kl", "--teacher", teacher, "--temperature", "0"],
             "must be a finite number in [0.001, 1000]"),
            (["--targets", ","], "must name at least one weight"),
            (["--warmup", "3"], "applies to lr_schedule cosine only"),
            (["--lr-schedule", "cosine", "--warmup", "9", "--steps", "9"],
             "must be shorter than steps (9)"),
            (["--lr", "nan"], "--lr (recipe field lr) must be a finite number"),
            (["--val-frac", "0.1"], "pass --data too"),
            (["--quantize-base", "--bf16-base"], "mutually exclusive"),
            (["--lora-rank"], "--lora-rank requires a value"),
            (["--bogus"], "unknown argument '--bogus'"),
        ]
        for args, says in cases:
            with self.subTest(args=" ".join(args)):
                r, _ = self.compile("refused", *args)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(says, r.stderr)

    def test_a_bad_recipe_file_is_refused_by_key(self):
        cases = [
            ('{"rank": 4}', "unknown field 'rank'"),
            ('{"lora_rank": "4"}', "field 'lora_rank' takes a number"),
            ('{"lora_rank": 9999}', "must be an integer in [1, 4096]"),
            ('{"steps": 3, "steps": 4}', "field 'steps' appears twice"),
            ('[]', "not a JSON object"),
        ]
        for text, says in cases:
            with self.subTest(text=text):
                path = os.path.join(self.dir, "bad.json")
                with open(path, "w") as f:
                    f.write(text)
                r, _ = self.compile("refused", "--recipe", path)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(says, r.stderr)
        r, _ = self.compile("refused", "--recipe", os.path.join(self.dir, "no.json"))
        self.assertIn("cannot open recipe", r.stderr)

    def test_the_manifest_publishes_the_schema(self):
        from seeml import formats
        fields = {f["name"]: f for f in formats.load_abi()["recipe"]["fields"]}
        self.assertEqual(fields["lora_rank"],
                         {"name": "lora_rank", "flag": "--lora-rank",
                          "kind": "uint", "min": 1, "max": 4096})
        self.assertEqual(fields["loss"]["choices"], "xent|mse|kl|xent+kl")
        self.assertEqual(fields["val_frac"]["min_open"], True)
        self.assertEqual(fields["temperature"]["max"], 1000)
        # Every flag the schema names is one the compiler's usage lists.
        usage = subprocess.run([COMPILER, "--help"], capture_output=True,
                               text=True).stderr
        for f in fields.values():
            self.assertIn(f["flag"], usage, f["name"])


if __name__ == "__main__":
    unittest.main()
