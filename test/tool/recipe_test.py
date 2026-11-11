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
CXX = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
       or shutil.which("g++"))
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

    def test_a_written_negative_zero_is_the_plans_one_zero(self):
        # "-0" passes a bound closed at 0; stored as -0.0f it was other
        # header bits — another plan and another recipe hash — for the
        # recipe "0" writes.
        a, out_a = self.compile("zero", "--data-batch", "12", "--clip-norm", "0")
        self.assertEqual(a.returncode, 0, a.stderr[-1500:])
        b, out_b = self.compile("negative_zero", "--data-batch", "12",
                                "--clip-norm", "-0")
        self.assertEqual(b.returncode, 0, b.stderr[-1500:])
        self.assertEqual(self.plan(out_a), self.plan(out_b))
        self.assertEqual(self.report("zero")["recipe_hash"],
                         self.report("negative_zero")["recipe_hash"])

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

    def test_a_boolean_flag_overrides_the_file_either_way(self):
        path = os.path.join(self.dir, "bools.json")
        with open(path, "w") as f:
            json.dump({"data_batch": 12, "quantize_base": True}, f)
        on, out_on = self.compile("bool_on", "--recipe", path)
        self.assertEqual(on.returncode, 0, on.stderr[-1500:])
        # --no-quantize-base clears what the file set: the plain plan's bytes.
        off, out_off = self.compile("bool_off", "--recipe", path,
                                    "--no-quantize-base")
        self.assertEqual(off.returncode, 0, off.stderr[-1500:])
        plain, out_plain = self.compile("bool_plain", "--data-batch", "12")
        self.assertEqual(plain.returncode, 0, plain.stderr[-1500:])
        self.assertEqual(self.plan(out_off), self.plan(out_plain))
        self.assertNotEqual(self.plan(out_on), self.plan(out_plain))
        self.assertEqual(self.report("bool_off")["recipe_hash"],
                         self.report("bool_plain")["recipe_hash"])
        # Which makes the other storage choice reachable from the same file.
        bf16, _ = self.compile("bool_bf16", "--recipe", path,
                               "--no-quantize-base", "--bf16-base")
        self.assertEqual(bf16.returncode, 0, bf16.stderr[-1500:])
        both, _ = self.compile("refused", "--quantize-base",
                               "--no-quantize-base")
        self.assertNotEqual(both.returncode, 0)
        self.assertIn("--quantize-base and --no-quantize-base contradict",
                      both.stderr)
        # Only booleans have the form.
        r, _ = self.compile("refused", "--no-lora-rank")
        self.assertIn("unknown argument '--no-lora-rank'", r.stderr)

    def test_a_bad_recipe_file_is_refused_by_key(self):
        cases = [
            ('{"rank": 4}', "unknown field 'rank'"),
            ('{"lora_rank": "4"}', "field 'lora_rank' takes a number"),
            ('{"lora_rank": 9999}', "must be an integer in [1, 4096]"),
            ('{"steps": 3, "steps": 4}', "field 'steps' appears twice"),
            ('[]', "not a JSON object"),
            # Not JSON's number: json.loads refuses both, and so does this.
            ('{"steps": 010}', "field 'steps' takes a JSON number"),
            ('{"lr": .5}', "field 'lr' takes a JSON number"),
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


@unittest.skipUnless(os.path.isfile(COMPILER) and HAVE_NUMPY and CXX is not None,
                     "needs the built seeml-update-compile, NumPy and a C++ "
                     "compiler")
class RunnerFlagsTest(unittest.TestCase):
    """The emitted runner's flag rules through a built model_update: the
    emitter suite reads them in the generated source; here they run."""

    @classmethod
    def setUpClass(cls):
        import export_model as em
        cls.dir = tempfile.mkdtemp(prefix="seeml-recipe-runner-")
        em._demo_decoder(cls.dir, vocab=40, dim=24, heads=4, seq=6,
                         blocks_n=2, samples=48, seed=1)
        cls.model = os.path.join(cls.dir, "decoder.smf")
        cls.corpus = os.path.join(cls.dir, "decoder_corpus.sds")
        pkg = os.path.join(cls.dir, "pkg")
        r = subprocess.run(
            [COMPILER, "--source", cls.model, "--out", pkg, "--data-batch",
             "12", "--lr-schedule", "cosine", "--warmup", "3", "--steps", "8",
             "--build"], capture_output=True, text=True,
            env=dict(os.environ, CXX=CXX))
        if r.returncode != 0:
            shutil.rmtree(cls.dir, True)
            raise AssertionError("the package did not build: " + r.stderr[-2000:])
        cls.runner = os.path.join(pkg, "model_update")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def update(self, *args):
        return subprocess.run(
            [self.runner, "--model", self.model, "--data", self.corpus,
             "--out", os.path.join(self.dir, "updated.smf"), *args],
            capture_output=True, text=True)

    def test_a_flag_that_cannot_apply_is_a_usage_error(self):
        cases = [
            (["--steps", "3"],
             "--steps 3 must exceed the plan's compiled warmup (3 steps)"),
            (["--checkpoint-every", "2"], "--checkpoint-every needs --checkpoint"),
            (["--val-frac", "0", "--eval-every", "2"],
             "--eval-every needs a validation split"),
            (["--steps", " 7"], "invalid numeric argument"),
        ]
        for args, says in cases:
            with self.subTest(args=" ".join(args)):
                r = self.update(*args)
                self.assertEqual(r.returncode, 2, r.stderr[-1500:])
                self.assertIn(says, r.stderr)

    def test_only_a_checkpoint_of_this_plan_is_a_resume(self):
        ckpt = os.path.join(self.dir, "run.ckpt")
        first = self.update("--checkpoint", ckpt, "--checkpoint-every", "2",
                            "--seed", "1", "--steps", "6", "--force")
        self.assertEqual(first.returncode, 0, first.stderr[-1500:])
        with open(ckpt, "rb") as f:
            own = f.read()
        # This plan's checkpoint is a resume, bound to the first run's seed.
        r = self.update("--checkpoint", ckpt, "--resume", "--seed", "2",
                        "--force")
        self.assertEqual(r.returncode, 2, r.stderr[-1500:])
        self.assertIn("--resume refused", r.stderr)
        # The same record bound to another plan (the SEKP header's plan-hash
        # word, bytes 8..15) is one the engine's load refuses before it
        # starts fresh: it binds this run to nothing, so another seed is no
        # usage error ...
        stale = os.path.join(self.dir, "stale.ckpt")
        other = bytearray(own)
        other[8] ^= 0xFF
        with open(stale, "wb") as f:
            f.write(other)
        r = self.update("--checkpoint", stale, "--resume", "--seed", "2",
                        "--force")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("no checkpoint resumed", r.stderr)
        self.assertNotIn("--resume refused", r.stderr)
        # ... and the fresh run's --steps rule is the one that holds.
        with open(stale, "wb") as f:
            f.write(other)
        r = self.update("--checkpoint", stale, "--resume", "--seed", "2",
                        "--steps", "3")
        self.assertEqual(r.returncode, 2, r.stderr[-1500:])
        self.assertIn("must exceed the plan's compiled warmup", r.stderr)
        # A resume's --steps counts further steps: from step 6, two more
        # end at 8, past the warmup, and the engine's rule admits it.
        with open(ckpt, "wb") as f:
            f.write(own)
        r = self.update("--checkpoint", ckpt, "--resume", "--seed", "1",
                        "--steps", "2", "--force")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertNotIn("no checkpoint resumed", r.stderr)
        # A checkpoint of this plan that does not load (truncated): the
        # peek calls it a resume, the engine starts fresh — and the engine's
        # own warmup rule refuses the 2-step horizon the runner's early
        # check could not see.
        broken = os.path.join(self.dir, "broken.ckpt")
        with open(broken, "wb") as f:
            f.write(own[:200])
        r = self.update("--checkpoint", broken, "--resume", "--seed", "1",
                        "--steps", "2", "--force")
        self.assertNotEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("does not exceed the plan's compiled warmup", r.stderr)


if __name__ == "__main__":
    unittest.main()
