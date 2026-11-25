#!/usr/bin/env python3
"""The exit-code contract of the built C++ tools (S8 #154; STYLE.md):

    0  the work was done
    1  the work failed — an input that cannot be read or is not what it
       claims to be, a refusal, a failed write
    2  a bad command line, refused before any input is read

`seeml-update-compile` exited 1 for an unknown flag and `seeml-seeu-dump`
read a typo'd flag as a file name (also 1) until S8; an orchestrator that
retries on 1 and fixes its invocation on 2 could not tell the two apart.
Each tool has its own test so that a missing binary skips that tool alone.

Run: python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import json
import os
import re
import subprocess
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN = os.path.join(REPO, "test", "fixtures", "golden")
BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))


def tool(name):
    return os.path.join(BUILD, name)


# The five tools every build links. The reason reads "no built ..." on
# purpose: CI's e2e job builds them and fails on a skip that says so.
def built(name):
    return unittest.skipUnless(os.path.isfile(tool(name)),
                               f"no built {name} under {BUILD}")


class CliExitCodes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="seeml-exit-")
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.absent = os.path.join(self.dir, "absent")
        self.garbage = os.path.join(self.dir, "garbage")
        with open(self.garbage, "wb") as f:
            f.write(b"not a model, not a plan\n" * 64)

    def exits(self, code, name, *argv):
        done = subprocess.run([tool(name), *argv], capture_output=True,
                              text=True)
        self.assertEqual(done.returncode, code,
                         f"{name} {' '.join(argv)}\n{done.stderr[-1500:]}")
        if code:  # a refusal says why, on stderr, and prints no result
            self.assertTrue(done.stderr.strip())
        return done

    @built("seeml-update-compile")
    def test_update_compile(self):
        model = os.path.join(GOLDEN, "mlp.smf")
        out = os.path.join(self.dir, "pkg")
        base = ("--source", model, "--out", out)
        for argv in (
                (),                                   # nothing at all
                ("--source", model),                  # --out is required
                (*base, "--no-such-flag"),            # an unknown flag
                (*base, "stray"),                     # an unknown positional
                (*base, "--steps"),                   # a flag without a value
                (*base, "--lr", "--quantize-base"),   # ... taking a flag for one
                (*base, "--steps", "abc"),            # a value that does not parse
                (*base, "--lora-rank", "5000"),       # ... or is out of bound
                (*base, "--gemm-tiles", "x"),
                (*base, "--peak-gflops", "0"),
                (*base, "--quantize-base", "--no-quantize-base"),
                (*base, "--no-embed", "--build"),
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(2, "seeml-update-compile", *argv)
        # A bad command line is refused before anything is written.
        self.assertFalse(os.path.exists(out))
        # The work failing is 1, as before: the inputs, not the invocation.
        recipe = os.path.join(self.dir, "recipe.json")
        with open(recipe, "w") as f:
            f.write("{ not json")
        for argv in (
                ("--source", self.absent, "--out", out),
                ("--source", self.garbage, "--out", out),
                (*base, "--recipe", self.absent),
                (*base, "--recipe", recipe),
                (*base, "--kernel-policy", self.garbage),
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(1, "seeml-update-compile", *argv)
        self.exits(0, "seeml-update-compile", "--version")
        self.exits(0, "seeml-update-compile", "--help")
        # Review 2026-11-25: three lines that were not what the table says.
        # --version in a value flag's slot asked for nothing (as --help).
        done = self.exits(2, "seeml-update-compile", "--out", "--version")
        self.assertIn("--out requires a value", done.stderr)
        # A host key without its table is a contradiction, not a failure.
        done = self.exits(2, "seeml-update-compile", *base,
                          "--target-host", "some-host")
        self.assertIn("pass the table with --kernel-policy", done.stderr)
        # A flag given twice is named as that, not as unknown.
        done = self.exits(2, "seeml-update-compile", *base, "--lr", "1e-3",
                          "--lr", "2e-3")
        self.assertIn("'--lr' is given more than once", done.stderr)
        self.assertFalse(os.path.exists(out))

    def recipe_fields(self):
        with open(os.path.join(REPO, "tool", "seeml", "abi.json")) as f:
            return json.load(f)["recipe"]["fields"]

    # Help is asked for from a slot that is no flag's value. `--out -h` read
    # the -h as a request (usage, exit 0, nothing compiled) until the
    # /code-review of #213; it is --out without a value, as it is in the
    # generated runner. The value flags are the --help text's own entries
    # that show a value, with the recipe's non-boolean fields: one the
    # compiler takes a value for without knowing its next slot is a value
    # slot fails here.
    @built("seeml-update-compile")
    def test_update_compile_help_is_not_read_from_a_value_slot(self):
        model = os.path.join(GOLDEN, "mlp.smf")
        out = os.path.join(self.dir, "pkg")
        base = ("--source", model, "--out", out)
        usage = self.exits(0, "seeml-update-compile", "--help").stderr
        value_flags = {"--source", "--out"}
        value_flags.update(re.findall(r"\[(--[a-z0-9-]+) (?!\|)", usage))
        value_flags.update(f["flag"] for f in self.recipe_fields()
                           if f["kind"] != "bool")
        self.assertGreater(len(value_flags), 35)
        for flag in sorted(value_flags):
            for help_flag in ("-h", "--help"):
                with self.subTest(argv=f"{flag} {help_flag}"):
                    done = self.exits(2, "seeml-update-compile", *base, flag,
                                      help_flag)
                    self.assertIn(f"{flag} requires a value\n", done.stderr)
                    self.assertNotIn("usage:", done.stderr)
        # The two the finding named, with no other flag on the line.
        for argv in (("--source", model, "--out", "-h"),
                     ("--out", out, "--source", "-h")):
            with self.subTest(argv=" ".join(argv)):
                done = self.exits(2, "seeml-update-compile", *argv)
                self.assertIn(f"{argv[2]} requires a value\n", done.stderr)
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists("-h"))
        # As a flag of its own it is help, wherever it stands — also on a
        # line that is otherwise incomplete or wrong, as before.
        for argv in (("-h",), ("--help",), ("-h", *base), (*base, "--help"),
                     ("--source", model, "-h", "--out", out),
                     (*base, "--lr", "1e-3", "-h"),
                     ("--help", "--no-such-flag"), ("-h", "--steps"),
                     (*base, "--lr", "--quantize-base", "--help")):
            with self.subTest(argv=" ".join(argv)):
                done = self.exits(0, "seeml-update-compile", *argv)
                self.assertIn("usage: seeml-update-compile", done.stderr)
        self.assertFalse(os.path.exists(out))
        # Only a flag is refused as a value. A value that merely starts with
        # one dash is still a value, held to its field's bound (recipe_test
        # compiles --clip-norm -0).
        done = self.exits(2, "seeml-update-compile", *base, "--clip-norm", "-1")
        self.assertIn("--clip-norm (recipe field clip_norm)", done.stderr)
        self.assertNotIn("requires a value", done.stderr)

    # --help is the recipe's field table said once: every field's flag (the
    # ABI manifest lists them, tool/seeml/abi.json) is one option entry —
    # `[--flag …` or the second of a `[--a | --b]` pair — and no entry is
    # there twice. Until the /code-review of #213 five were (--weight-decay,
    # --clip-norm, --lr-schedule, --warmup, --allow-zero-lr: two lines in an
    # older and a newer wording), and the --segment-mask entry sat inside
    # the --masked-loss description.
    @built("seeml-update-compile")
    def test_update_compile_help_lists_every_recipe_flag_once(self):
        usage = self.exits(0, "seeml-update-compile", "--help").stderr
        entries = re.findall(r"(?:\[|\| )(--[a-z0-9-]+)(?![\w-])", usage)
        twice = sorted({e for e in entries if entries.count(e) > 1})
        self.assertEqual(twice, [], "option entries --help prints twice")
        flags = [f["flag"] for f in self.recipe_fields()]
        self.assertGreater(len(flags), 30)
        self.assertEqual([f for f in flags if f not in entries], [],
                         "recipe flags --help does not list")
        # An entry's description follows it: the three --masked-loss lines
        # are together, and the next entry starts after them.
        lines = usage.splitlines()
        at = next(i for i, line in enumerate(lines) if "[--masked-loss]" in line)
        self.assertIn("alone (labels of -1 ignored)", lines[at + 1])
        self.assertIn("corpus that carries a mask", lines[at + 2])
        self.assertIn("[--segment-mask auto|off|on]", lines[at + 3])
        self.assertLessEqual(max(map(len, lines)), 80)

    # docs/usage.md says what each recipe field's flag does: a field added
    # to the table is documented there, by its flag.
    def test_usage_md_names_every_recipe_flag(self):
        with open(os.path.join(REPO, "docs", "usage.md")) as f:
            text = f.read()
        missing = [f["flag"] for f in self.recipe_fields()
                   if not re.search(re.escape(f["flag"]) + r"(?![\w-])", text)]
        self.assertEqual(missing, [])

    @built("seeml-seeu-dump")
    def test_seeu_dump(self):
        for argv in (
                (),
                ("--no-such-flag",),                  # was: "cannot open", 1
                ("--json", self.garbage),             # a flag ahead of the file
                (self.garbage, "--no-such-flag"),
                (self.garbage, "stray"),
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(2, "seeml-seeu-dump", *argv)
        for argv in ((self.absent,), (self.garbage,), (self.garbage, "--json")):
            with self.subTest(argv=" ".join(argv)):
                self.exits(1, "seeml-seeu-dump", *argv)
        self.exits(0, "seeml-seeu-dump", "--version")
        self.exits(0, "seeml-seeu-dump", "--help")

    @built("seeml-plan-probe")
    def test_plan_probe(self):
        base = ("--plan", self.garbage, "--section", "train",
                "--arena-in", self.garbage,
                "--arena-out", os.path.join(self.dir, "arena.out"))
        for argv in (
                (),
                (*base, "--no-such-flag"),
                (*base[:-1],),                        # --arena-out without a value
                (*base, "--section", "nowhere"),
                (*base, "--backend", "abacus"),
                (*base, "--threads", "4097"),         # above the pool's bound
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(2, "seeml-plan-probe", *argv)
        self.exits(1, "seeml-plan-probe", *base)      # the plan is no plan
        self.exits(1, "seeml-plan-probe", "--plan", self.absent, *base[2:])

    @built("seeml-tokenize")
    def test_tokenize(self):
        for argv in (
                (),
                (self.garbage,),                      # neither --info nor a text
                (self.garbage, "--no-such-flag"),
                (self.garbage, self.garbage, "stray"),
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(2, "seeml-tokenize", *argv)
        self.exits(1, "seeml-tokenize", self.absent, "--info")
        self.exits(1, "seeml-tokenize", self.garbage, "--info")

    @built("seeml-abi")
    def test_abi(self):
        self.exits(2, "seeml-abi", "--no-such-flag")
        self.exits(2, "seeml-abi", "stray")
        self.exits(0, "seeml-abi")

    # seeml-bench is linked only by a SEEML_BENCH=1 build. The skip reason
    # must not read "no built ...": the e2e job's skip guard fails on that
    # phrase for the tools it builds, and it builds no bench — so in CI today
    # this case runs nowhere: the e2e job skips it, the python-tools job
    # builds nothing and skips every case, and the SEEML_BENCH job, which
    # links all six tools, does not yet run this file (a no-skip step there
    # is proposed with the PR). Locally a SEEML_BENCH=1 build runs all six.
    @unittest.skipUnless(os.path.isfile(tool("seeml-bench")),
                         "seeml-bench absent (a SEEML_BENCH=1 build links it)")
    def test_bench(self):
        out = os.path.join(self.dir, "bench.json")
        for argv in (
                (),                                   # --out is required
                ("--out", out, "--no-such-flag"),
                ("--out",),                           # a flag without a value
                ("--out", out, "--repeats", "abc"),
                ("--out", out, "--backend", "abacus"),
                ("--out", out, "--fixtures", "no_such_fixture"),
        ):
            with self.subTest(argv=" ".join(argv)):
                self.exits(2, "seeml-bench", *argv)
        self.assertFalse(os.path.exists(out))
        self.exits(0, "seeml-bench", "--version")

    # --attention-sweep takes --out, --threads and --repeats only. Each flag
    # of the fixture sweep is given a value that sweep would accept, so the
    # one refusal a case can meet is the sweep's own — which, until the
    # /code-review of #213, was a check no command line could reach: the
    # flags were ignored and the sweep ran (exit 0).
    @unittest.skipUnless(os.path.isfile(tool("seeml-bench")),
                         "seeml-bench absent (a SEEML_BENCH=1 build links it)")
    def test_bench_attention_sweep_refuses_the_fixture_sweeps_flags(self):
        out = os.path.join(self.dir, "sweep.json")
        base = ("--attention-sweep", "--out", out, "--threads", "1",
                "--repeats", "1")
        for flag, value in (
                ("--steps-lo", "5"),
                ("--steps-hi", "90"),
                ("--fixtures", "mlp_64x512x3"),
                ("--peak-gflops", "100"),
                ("--backend", "cpu"),
                ("--gemm-tiles", "64,64"),
                ("--kernel-policy", self.absent),
                ("--target-host", "some-host"),
        ):
            with self.subTest(flag=flag):
                done = self.exits(2, "seeml-bench", *base, flag, value)
                self.assertIn(
                    "--attention-sweep takes --out, --threads and "
                    f"--repeats only, not {flag}\n", done.stderr)
        # The flag ahead of --attention-sweep is the same refusal.
        done = self.exits(2, "seeml-bench", "--backend", "cpu", *base)
        self.assertIn("only, not --backend\n", done.stderr)
        self.assertFalse(os.path.exists(out))

    # --threads is a pool width, and a width is that many workers spawned:
    # the bound is the runner's and seeml-plan-probe's (kMaxParallelThreads,
    # source/parallel/parallel_for.h). seeml-bench had none. Every case also
    # names a fixture that does not exist — the refusal after --threads — so
    # no case can start a run, and which refusal speaks is the test (the
    # harness's exit code is 2 for both).
    @unittest.skipUnless(os.path.isfile(tool("seeml-bench")),
                         "seeml-bench absent (a SEEML_BENCH=1 build links it)")
    def test_bench_threads_is_held_to_the_pools_bound(self):
        out = os.path.join(self.dir, "bench.json")
        base = ("--out", out, "--fixtures", "no_such_fixture")
        for width in ("4097", "1,100000000", "8,4097,2"):
            with self.subTest(threads=width):
                done = self.exits(2, "seeml-bench", *base, "--threads", width)
                self.assertIn(
                    "invalid --threads (an integer from 1 to 4096), got '"
                    + max(width.split(","), key=int) + "'", done.stderr)
                self.assertNotIn("no_such_fixture", done.stderr)
        # The sweep that takes --threads holds it to the same bound.
        done = self.exits(2, "seeml-bench", "--attention-sweep", "--out", out,
                          "--repeats", "1", "--threads", "4097")
        self.assertIn("an integer from 1 to 4096", done.stderr)
        # The bound itself is a width: the next refusal is the one that speaks.
        done = self.exits(2, "seeml-bench", *base, "--threads", "1,4096")
        self.assertIn("unknown fixture 'no_such_fixture'", done.stderr)
        self.assertNotIn("--threads", done.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
