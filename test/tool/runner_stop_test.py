"""Cooperative cancellation and the pool width through a built model_update
(#202 item 1): SIGTERM stops the run at a step boundary, checkpoints there
and exits 5 with nothing committed; resuming from that checkpoint commits
the bytes an uninterrupted run commits — without --force, on cpu and on
Metal, the resume re-measuring the gate's stored losses (CR4 #221); and a
checkpoint that cannot be resumed is refused, exit 1, the file untouched. --threads sets the worker pool and
the runtime screen records it.

The stop lands wherever the signal does — the claim holds at any step, so
the test does not pin one; it only refuses a run that finished first.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tool"))
from seeml import formats  # noqa: E402

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
CXX = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
       or shutil.which("g++"))
try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

STEPS = 2000  # long enough that the signal lands with steps to spare


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@unittest.skipUnless(os.path.isfile(COMPILER) and HAVE_NUMPY,
                     "needs the built seeml-update-compile and NumPy")
@unittest.skipIf(sys.platform == "win32", "POSIX signals")
class RunnerStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_model as em
        cls.dir = tempfile.mkdtemp(prefix="seeml-runner-stop-")
        em._demo_decoder(cls.dir, vocab=40, dim=24, heads=4, seq=6,
                         blocks_n=2, samples=96, seed=3)
        cls.model = os.path.join(cls.dir, "decoder.smf")
        cls.corpus = os.path.join(cls.dir, "decoder_corpus.sds")
        pkg = os.path.join(cls.dir, "pkg")
        r = subprocess.run(
            [COMPILER, "--source", cls.model, "--out", pkg, "--data-batch",
             "12", "--steps", str(STEPS), "--build"],
            capture_output=True, text=True, env=dict(os.environ, CXX=CXX))
        if r.returncode != 0:
            shutil.rmtree(cls.dir, True)
            raise AssertionError("the package did not build: " + r.stderr[-2000:])
        cls.runner = os.path.join(pkg, "model_update")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def args(self, out, *extra, backend="cpu"):
        # Evaluations every 20 steps and a best state: the resume must carry
        # the gate's state across the stop, not only the parameters. No
        # --force (CR4 #221): the resume re-measures the gate's stored
        # losses, and a re-measure that refused a legitimate resume — or a
        # gate that weighed it differently — would show here as a failure.
        return [self.runner, "--model", self.model, "--data", self.corpus,
                "--val-frac", "0.25", "--seed", "5", "--eval-every", "20",
                "--backend", backend, "--out", out, *extra]

    def stop_and_resume(self, backend):
        whole = os.path.join(self.dir, backend + "-whole.smf")
        r = subprocess.run(self.args(whole, backend=backend),
                           capture_output=True, text=True)
        if backend == "metal" and r.returncode == 1 and "backend:" in r.stderr:
            self.skipTest("no Metal device: " + r.stderr.strip()[-200:])
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("-> accepted [backend " + backend, r.stderr)

        ckpt = os.path.join(self.dir, backend + "-stop.sekp")
        cut = os.path.join(self.dir, backend + "-cut.smf")
        report = os.path.join(self.dir, backend + "-cut.json")
        # No periodic checkpoint: the stop's own is the only one there is.
        # Two separate process launches: this one is killed by SIGTERM at a
        # step boundary, the next is a cold start with --resume.
        p = subprocess.Popen(
            self.args(cut, "--checkpoint", ckpt, "--checkpoint-every",
                      str(STEPS * 10), "--report", report, backend=backend),
            stderr=subprocess.PIPE, text=True)
        # The engine logs a step line every 100 steps; signal on the second
        # (step 101), with most of the run still ahead.
        lines = 0
        for line in p.stderr:
            if line.startswith("seeml-update: step ") and "validation" not in line:
                lines += 1
                if lines == 2:
                    p.send_signal(signal.SIGTERM)
                    break
        tail = p.stderr.read()
        p.stderr.close()
        rc = p.wait()
        self.assertEqual(lines, 2, "the run ended before the signal was sent")
        self.assertNotEqual(rc, 0, "the run finished before the signal landed")
        self.assertEqual(rc, 5, tail[-1500:])
        self.assertIn("stopped by signal 15", tail)
        self.assertIn("rerun with --resume", tail)
        self.assertTrue(os.path.exists(ckpt), "no checkpoint at the stop")
        self.assertFalse(os.path.exists(cut), "a stopped run committed")
        with open(report) as f:
            rep = json.load(f)
        self.assertEqual(rep["verdict"], "stopped")
        self.assertFalse(rep["committed"])
        self.assertLess(rep["steps"], STEPS)
        self.assertIsNone(rep["validation_loss"][1])  # never scored
        self.assertEqual(rep["runtime"]["verdict"], "stopped")

        resumed = os.path.join(self.dir, backend + "-resumed.smf")
        r = subprocess.run(
            self.args(resumed, "--checkpoint", ckpt, "--resume",
                      backend=backend),
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertNotIn("refused", r.stderr)
        self.assertIn("-> accepted [backend " + backend, r.stderr)
        self.assertEqual(sha256(resumed), sha256(whole))
        return ckpt

    def test_a_sigterm_stops_checkpoints_and_the_resume_finishes_the_run(self):
        self.stop_and_resume("cpu")

    def test_the_stop_and_resume_commits_the_same_bytes_on_metal(self):
        if sys.platform != "darwin":
            self.skipTest("Metal is Apple's")
        self.stop_and_resume("metal")

    def test_a_refused_resume_is_exit_1_and_leaves_the_file(self):
        # CR4 #221: --resume is strict. A checkpoint that is there and
        # cannot be resumed is refused, exit 1, and the file is byte for
        # byte what it was — no fresh run over it, no sidecar left beside
        # it. A checkpoint written on one backend is refused on the other
        # by name: the gate's losses are re-measured bit for bit, which only
        # the backend that wrote them reproduces.
        ckpt = os.path.join(self.dir, "strict-stop.sekp")
        p = subprocess.Popen(
            self.args(os.path.join(self.dir, "strict-cut.smf"), "--checkpoint",
                      ckpt, "--checkpoint-every", str(STEPS * 10)),
            stderr=subprocess.PIPE, text=True)
        for line in p.stderr:
            if line.startswith("seeml-update: step "):
                p.send_signal(signal.SIGTERM)
                break
        p.stderr.read()
        p.stderr.close()
        self.assertEqual(p.wait(), 5)
        with open(ckpt, "rb") as f:
            good = f.read()
        header = formats.SEKP_HEADER_BYTES
        older = bytearray(good)
        older[4:8] = (5).to_bytes(4, "little")
        del older[header - formats.CKPT_HEADER_V6_TAIL.size:header]
        payload_flip = bytearray(good)
        payload_flip[-1] ^= 0x01
        header_flip = bytearray(good)
        header_flip[16] ^= 0x01  # the step's low bit
        cases = [
            ("truncated", good[:-1], "truncated checkpoint"),
            ("payload bit", bytes(payload_flip), "payload is corrupt"),
            ("header bit", bytes(header_flip), "integrity hash"),
            ("older version", bytes(older), "format v5; this runtime reads v6"),
        ]
        for name, data, says in cases:
            with self.subTest(case=name):
                bad = os.path.join(self.dir, "strict-bad.sekp")
                with open(bad, "wb") as f:
                    f.write(data)
                before = sorted(os.listdir(self.dir))
                r = subprocess.run(
                    self.args(os.path.join(self.dir, "strict.smf"),
                              "--checkpoint", bad, "--checkpoint-every", "1",
                              "--resume"),
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 1, r.stderr[-1500:])
                self.assertIn("refused", r.stderr)
                self.assertIn(says, r.stderr)
                with open(bad, "rb") as f:
                    self.assertEqual(f.read(), data)
                self.assertEqual(sorted(os.listdir(self.dir)), before)
        if sys.platform == "darwin":
            r = subprocess.run(
                self.args(os.path.join(self.dir, "strict.smf"), "--checkpoint",
                          ckpt, "--resume", backend="metal"),
                capture_output=True, text=True)
            if "backend:" not in r.stderr:
                self.assertEqual(r.returncode, 1, r.stderr[-1500:])
                self.assertIn("written on the cpu backend and this run is on "
                              "metal", r.stderr)
                with open(ckpt, "rb") as f:
                    self.assertEqual(f.read(), good)

    def test_a_stop_without_a_checkpoint_says_nothing_is_resumable(self):
        p = subprocess.Popen(self.args(os.path.join(self.dir, "none.smf")),
                             stderr=subprocess.PIPE, text=True)
        for line in p.stderr:
            if line.startswith("seeml-update: step "):
                p.send_signal(signal.SIGINT)
                break
        tail = p.stderr.read()
        p.stderr.close()
        rc = p.wait()
        self.assertEqual(rc, 5, tail[-1500:])
        self.assertIn("no --checkpoint, so nothing resumable was kept", tail)

    def test_threads_sets_the_pool_and_the_screen_records_it(self):
        report = os.path.join(self.dir, "threads.json")
        r = subprocess.run(
            self.args(os.path.join(self.dir, "t.smf"), "--threads", "3",
                      "--steps", "150", "--report", report),
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        with open(report) as f:
            runtime = json.load(f)["runtime"]
        self.assertEqual(runtime["threads"], 3)
        # The energy rows (#202 item 3): the meter row is always there; a
        # number only on a GPU run (the GPU's domain; a cpu run never gets
        # one — seeml-bench's rule), and the per-token figure is the energy
        # over the tokens trained.
        meter = runtime["energy_meter"]
        if runtime["backend"] == "cpu":
            self.assertTrue(meter.startswith("unavailable: "), meter)
        if meter.startswith("IOReport "):
            self.assertGreater(runtime["energy_joules"], 0.0)
            self.assertGreater(runtime["energy_per_token"], 0.0)
        else:
            self.assertTrue(meter.startswith("unavailable: "), meter)
            self.assertNotIn("energy_joules", runtime)
            self.assertNotIn("energy_per_token", runtime)
        for bad in ("0", "x", "5000"):
            with self.subTest(threads=bad):
                r = subprocess.run(
                    self.args(os.path.join(self.dir, "t.smf"), "--threads", bad),
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, r.stderr[-800:])
                # The bound is kMaxParallelThreads, printed from it.
                self.assertIn(
                    "invalid --threads (an integer from 1 to 4096)\n", r.stderr)


if __name__ == "__main__":
    unittest.main()
