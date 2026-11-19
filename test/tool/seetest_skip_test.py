#!/usr/bin/env python3
"""SeeTest's SKIP(), seen from outside the process (S8 #154; STYLE.md).

A hardware-gated test used to `return` early and report green for zero work.
SKIP(reason) says so instead: the test is printed and counted as SKIPPED with
its reason, it is not among the passes, and it does not fail the run. What a
skip looks like is the runner's stdout and exit code, so this drives a built
suite — the two SeeTest.* tests of seeml_diagnostics_test, which skip or
fail only when SEETEST_SELF_TEST asks them to.

Run: python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import os
import subprocess
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
SUITE = os.path.join(BUILD, "seeml_diagnostics_test")
SKIPS = "SeeTest.ASkipIsReportedAndCountedApartFromThePasses"
FAILS = "SeeTest.AFailureBeforeASkipStillFailsTheTest"


# Every build links the suites; "no built ..." is the phrase CI's e2e job
# refuses to see skipped.
@unittest.skipUnless(os.path.isfile(SUITE),
                     f"no built seeml_diagnostics_test under {BUILD}")
class SeeTestSkip(unittest.TestCase):
    def run_suite(self, asked, pattern="SeeTest."):
        env = {k: v for k, v in os.environ.items() if k != "SEETEST_SELF_TEST"}
        if asked:
            env["SEETEST_SELF_TEST"] = asked
        return subprocess.run([SUITE, f"--filter={pattern}"], env=env,
                              capture_output=True, text=True)

    def test_a_run_without_skips_prints_what_it_always_did(self):
        done = self.run_suite(None)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("[  PASSED  ] 2 test(s)", done.stdout)
        self.assertNotIn("SKIPPED", done.stdout)
        self.assertNotIn("FAILED", done.stdout)

    def test_a_skip_is_reported_with_its_reason_and_is_not_a_pass(self):
        done = self.run_suite("skip")
        self.assertEqual(done.returncode, 0, done.stdout)  # not a failure
        out = done.stdout
        # Where it happened and why, as a failure is reported ...
        self.assertRegex(out, r"  SKIPPED at \S*diagnostics_test\.cc:\d+\n"
                              r"    the self-test asked for a skip\n")
        # ... the test's own verdict line, in place of [       OK ] ...
        self.assertRegex(out, rf"\[  SKIPPED \] {SKIPS} \(\d+ ms\)")
        self.assertNotRegex(out, rf"\[       OK \] {SKIPS}")
        # ... and the summary: one pass (the other test), one skip, named.
        self.assertIn("[==========] 2 test(s) ran", out)
        self.assertIn("[  PASSED  ] 1 test(s)", out)
        self.assertIn("[  SKIPPED ] 1 test(s), listed below:", out)
        self.assertIn(f"[  SKIPPED ] {SKIPS}: the self-test asked for a skip",
                      out)
        self.assertNotIn("FAILED", out)

    def test_a_suite_of_skips_alone_passes_nothing_and_fails_nothing(self):
        done = self.run_suite("skip", SKIPS)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("[  PASSED  ] 0 test(s)", done.stdout)
        self.assertIn("[  SKIPPED ] 1 test(s), listed below:", done.stdout)

    def test_a_failure_before_a_skip_still_fails(self):
        done = self.run_suite("fail-then-skip")
        self.assertEqual(done.returncode, 1, done.stdout)
        out = done.stdout
        self.assertRegex(out, rf"\[  FAILED  \] {FAILS} \(\d+ ms\)")
        self.assertIn("[  PASSED  ] 1 test(s)", out)
        self.assertIn("[  FAILED  ] 1 test(s), listed below:", out)
        # A failed test is failed, not also skipped.
        self.assertNotIn("[  SKIPPED ]", out)


if __name__ == "__main__":
    unittest.main()
