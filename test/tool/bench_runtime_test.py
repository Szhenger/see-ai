"""seeml-bench rows carry the runtime screen (S7 #153 acceptance 3): the very
struct model_update prints — the same rows, filled by the same function
(runtime/profiler/runtime_screen.cc) — from each thread count's last timed
run. Runs the real, built seeml-bench on two tiny fixtures."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tool"))

from seeml import formats  # noqa: E402

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
BENCH = os.path.join(BUILD, "seeml-bench")
KINDS = {"count": int, "bytes": int, "real": (int, float), "text": str}


# seeml-bench is linked only by a SEEML_BENCH=1 build. The skip reason must
# not read "no built ...": the e2e job's skip guard fails on that phrase for
# the tools it builds, and it builds no bench.
@unittest.skipUnless(os.path.isfile(BENCH),
                     "seeml-bench absent (a SEEML_BENCH=1 build links it)")
class BenchRuntimeScreenTest(unittest.TestCase):
    def test_every_row_carries_the_runtime_screen(self):
        rows = {r["key"]: r for r in formats.load_abi()["screens"]["runtime"]["rows"]}
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "bench.json")
            done = subprocess.run(
                [BENCH, "--out", out, "--threads", "1", "--steps-lo", "4",
                 "--steps-hi", "8", "--repeats", "1", "--fixtures",
                 "mlp_64x512x3,tok_v64_d64_s16", "--peak-gflops", "100"],
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            with open(out) as f:
                report = json.load(f)
        for name, fixture in report["fixtures"].items():
            for threads, row in fixture["threads"].items():
                with self.subTest(fixture=name, threads=threads):
                    runtime = row["runtime"]
                    # Every key a published row of its published kind, and
                    # every required row present (the seam test's rule).
                    self.assertTrue(set(runtime) <= set(rows),
                                    set(runtime) - set(rows))
                    for key, value in runtime.items():
                        self.assertIsInstance(value, KINDS[rows[key]["kind"]], key)
                        self.assertNotIsInstance(value, bool, key)
                    for key, spec in rows.items():
                        if not spec["optional"]:
                            self.assertIn(key, runtime)
                    self.assertEqual(runtime["verdict"], "measured")
                    self.assertIn(runtime["backend"], ("cpu", "metal"))
                    # The screen's work per step is the plan's: GEMM +
                    # attention, the same figures the fixture row reports.
                    self.assertEqual(runtime["flops_per_step"],
                                     fixture["gemm_flops_per_step"]
                                     + fixture["attention_flops_per_step"])
                    self.assertGreater(runtime["measured_step_ms"], 0.0)
                    self.assertAlmostEqual(
                        runtime["predicted_step_ms"],
                        runtime["flops_per_step"] / (100.0 * 1e6), places=9)
        # A token plan trains on class labels: the input-blind loss is there.
        self.assertIn("label_entropy",
                      report["fixtures"]["tok_v64_d64_s16"]["threads"]["1"]["runtime"])


if __name__ == "__main__":
    unittest.main()
