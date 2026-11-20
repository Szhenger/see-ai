"""The fuzz harness's seed corpus (CR3 #208): test/fuzz/seeds/ is what
test/fuzz/make_seeds.py writes today.

Every seed the script writes from integers — the models, the corpora, the
tokenizer section, the text, the recipe, the report — is compared byte for
byte with its checked-in copy, on every host. A plan seed is the compiler's
output on the host that generated it and is NOT byte-compared: the C++
replay suite (test/fuzz/fuzz_replay_test.cc) holds each one to being
accepted by the loader and executed. Here the plans are held to compiling
at all, and the directory to holding exactly the seeds the script names.
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "test", "fuzz"))
import make_seeds  # noqa: E402

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")


def run(*argv):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        status = make_seeds.main(["--build", BUILD, *argv])
    return status, out.getvalue()


@unittest.skipUnless(os.path.isfile(COMPILER),
                     f"no built seeml-update-compile under {BUILD}")
class FuzzSeedsTest(unittest.TestCase):
    def test_the_checked_in_seeds_are_what_the_generator_writes(self):
        status, log = run("--check")
        self.assertEqual(status, 0, log)

    def test_a_seed_that_moved_and_a_stray_file_are_both_named(self):
        with tempfile.TemporaryDirectory(prefix="seeml-fuzz-seeds-") as d:
            status, log = run("--out", d)
            self.assertEqual(status, 0, log)
            self.assertEqual(run("--out", d, "--check")[0], 0)
            # A moved byte in a seed the script writes from integers.
            with open(os.path.join(d, "sds-class"), "r+b") as f:
                f.seek(40)
                f.write(b"\x7f")
            # A plan's bytes are the generating host's: not compared.
            with open(os.path.join(d, "plan-mlp"), "r+b") as f:
                f.seek(40)
                f.write(b"\x7f")
            with open(os.path.join(d, "stray"), "wb") as f:
                f.write(b"\x00")
            os.remove(os.path.join(d, "recipe"))
            status, log = run("--out", d, "--check")
            self.assertEqual(status, 1)
            self.assertIn("sds-class: bytes differ", log)
            self.assertIn("stray: not a seed this script writes", log)
            self.assertIn("recipe: not in", log)
            self.assertNotIn("plan-mlp", log)

    def test_every_arm_has_a_seed_and_the_corpus_stays_small(self):
        with tempfile.TemporaryDirectory(prefix="seeml-fuzz-seeds-") as d:
            self.assertEqual(run("--out", d)[0], 0)
            arms, total = set(), 0
            for name in os.listdir(d):
                with open(os.path.join(d, name), "rb") as f:
                    data = f.read()
                arms.add(data[0] % make_seeds.ARMS)
                total += len(data)
            self.assertEqual(arms, set(range(make_seeds.ARMS)))
            # Well under 2 MB: a corpus is cloned by every checkout.
            self.assertLess(total, 1 << 20)


if __name__ == "__main__":
    unittest.main()
