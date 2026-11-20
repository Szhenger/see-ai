"""The device-side relaxed gate (F2 #206), against real packages.

A relaxed plan (kFlagRelaxed: --precision certified-bf16) ran on a device
with no certificate anywhere: an exact package's `model_update --plan
relaxed.seeu` exited 0, committed, and reported relaxed executions. The
certificate was checked only where a package is built. Now the build that
checked it compiles the certified plan's seal into the runner, and the
runner refuses every other relaxed plan at load — embedded, mapped from
beside the binary, or named by --plan.

  the reproduction   an exact package (embedded and disk mode) given a
                     relaxed plan: exit 1, the missing certificate named,
                     nothing committed — on cpu, and on metal where a
                     device exists;
  the certified one  certify -> pack --build -> run: commits, runs relaxed
                     kernels on metal, takes its own plan file by --plan,
                     refuses another relaxed plan, still runs an exact one;
  the instrument     seeml-plan-probe refuses a relaxed plan as a device
                     does unless the run says it is a measurement.

Needs the built tools (SEEML_BUILD_DIR, else build/), NumPy (the fixture
models, the certifier) and a C++ toolchain (the packages are built).

    python3 -m unittest discover -s test/tool -p 'relaxed_gate_test.py'
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import certify_numerics as cn  # noqa: E402
import frontier_exec as fx  # noqa: E402
import frontier_exec_test as fixtures  # noqa: E402
import pack_update  # noqa: E402

NO_CERTIFICATE = ("relaxed instruction(s) and this package was built with "
                  "no numerics certificate for it")
ANOTHER_PLAN = ("relaxed instruction(s) and this package was built with a "
                "certificate for another plan")


def run_tool(module, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        status = module.main(argv)
    return status, buf.getvalue()


def read(path):
    with open(path, "rb") as f:
        return f.read()


@unittest.skipIf(fixtures.np is None, "NumPy not installed")
@unittest.skipUnless(fixtures.HAVE_CPP,
                     f"no built seeml tools under {fixtures.BUILD}")
class RelaxedGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="seeml-relaxed-gate-")
        cls.dir = cls._tmp.name
        cls.addClassCleanup(cls._tmp.cleanup)
        fixtures._write_models(cls.dir)
        cls.model = os.path.join(cls.dir, "decoder.smf")
        cls.corpus = os.path.join(cls.dir, "decoder_corpus.sds")
        relaxed = ["--precision", "certified-bf16"]
        # The issue's reproduction: one exact package built with its plan
        # embedded, one in the disk mode, and a relaxed plan compiled apart.
        cls.exact = cls.emit("dec_exact")
        cls.build(cls.exact)
        cls.exact_disk = cls.emit("dec_exact_disk")
        cls.build(cls.exact_disk, "--no-embed")
        cls.relaxed = cls.emit("dec_rx", *relaxed)
        # Another relaxed plan of the same model: one more LoRA rank.
        cls.other = cls.emit("dec_rx_other", "--lora-rank", "4", *relaxed)
        # The certified package: certify -> pack --build.
        cls.certified = os.path.join(cls.dir, "dec_cert")
        shutil.copytree(cls.relaxed, cls.certified)
        status, log = run_tool(cn, ["certify", cls.certified, "--corpus",
                                    cls.corpus, "--val-fraction", "0.25"])
        assert status == 0, log
        cls.pack_report = os.path.join(cls.dir, "pack_cert.json")
        cls.build(cls.certified, "--report", cls.pack_report)

    @classmethod
    def emit(cls, name, *flags):
        out = os.path.join(cls.dir, name)
        done = subprocess.run(
            [fixtures.COMPILER, "--source", cls.model, "--out", out,
             "--data-batch", "12", "--attention", "tiled", "--no-embed",
             *flags], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        assert done.returncode == 0, done.stdout
        return out

    @classmethod
    def build(cls, pkg, *flags):
        status, log = run_tool(pack_update, [pkg, "--build", *flags])
        if status != 0:
            # A missing toolchain is a skip; a package that does not build
            # under the toolchain that built the compiler is a failure.
            if shutil.which(os.environ.get("CXX", "c++")) is None:
                raise unittest.SkipTest(
                    "no C++ toolchain to build the emitted package")
            raise AssertionError(f"{pkg} did not build:\n{log[-3000:]}")

    def plan(self, pkg):
        return os.path.join(pkg, pack_update.PLAN_FILE)

    def update(self, binary, name, backend, *flags):
        """One run of a package's model_update; (process, output path,
        report path). Skips the subtest's backend when this host has no
        Metal device — the runner says so before it loads anything."""
        out = os.path.join(self.dir, name + ".smf")
        report = os.path.join(self.dir, name + ".json")
        for stale in (out, report):
            if os.path.exists(stale):
                os.unlink(stale)
        done = subprocess.run(
            [binary, "--model", self.model, "--data", self.corpus, "--steps",
             "60", "--val-frac", "0.25", "--seed", "3", "--force",
             "--backend", backend, "--out", out, "--report", report, *flags],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if backend == "metal" and any(
                line.startswith("backend: ")
                for line in done.stderr.splitlines()):
            self.skipTest("this host has no Metal device")
        return done, out, report

    def refused(self, done, out, report, says):
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("load: ", done.stderr)
        self.assertIn(says, done.stderr)
        self.assertIn("UpdateEngine: the plan carries 35 relaxed",
                      done.stderr)
        # Refused at load: nothing ran, nothing was committed or reported.
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(report))

    def test_an_exact_package_refuses_a_relaxed_plan(self):
        """The reproduction: exit 0, committed and 480 relaxed executions
        before the gate; exit 1 and the missing certificate named now."""
        relaxed = self.plan(self.relaxed)
        self.assertEqual(len(fx.formats.relaxed_opcodes(read(relaxed))), 35)
        # Disk mode's own route: the plan file beside the binary, swapped.
        swapped = os.path.join(self.dir, "swapped")
        os.makedirs(swapped, exist_ok=True)
        shutil.copy2(os.path.join(self.exact_disk, "model_update"), swapped)
        shutil.copy2(relaxed, self.plan(swapped))
        for backend in ("cpu", "metal"):
            with self.subTest(backend=backend):
                for mode, binary, flags in (
                        ("embedded", self.exact, ["--plan", relaxed]),
                        ("disk", self.exact_disk, ["--plan", relaxed]),
                        ("swapped", swapped, [])):
                    done, out, report = self.update(
                        os.path.join(binary, "model_update"),
                        f"repro_{mode}_{backend}", backend, *flags)
                    self.refused(done, out, report, NO_CERTIFICATE)
                    if flags:
                        self.assertIn(f"load: '{relaxed}': ", done.stderr)

    def test_an_exact_package_is_what_it_was(self):
        """An exact plan is outside the rule: both modes commit, and commit
        the same bytes as each other, with no relaxed execution."""
        for backend in ("cpu", "metal"):
            with self.subTest(backend=backend):
                models = []
                for mode, pkg in (("embedded", self.exact),
                                  ("disk", self.exact_disk)):
                    done, out, report = self.update(
                        os.path.join(pkg, "model_update"),
                        f"exact_{mode}_{backend}", backend)
                    self.assertEqual(done.returncode, 0, done.stderr)
                    with open(report) as f:
                        self.assertEqual(json.load(f)["relaxed_executions"],
                                         0)
                    models.append(read(out))
                self.assertEqual(models[0], models[1])

    def test_a_certified_package_runs_its_plan_and_no_other(self):
        with open(self.pack_report) as f:
            packed = json.load(f)
        own = self.plan(self.certified)
        seal = cn.plan_seal(own)
        self.assertEqual(packed["numerics_certificate"]["certified_plan_seal"],
                         seal)
        self.assertEqual(f"{fx.Plan.load(own).plan_hash:016x}", seal)
        with open(os.path.join(self.certified, cn.ATTESTATION_FILE)) as f:
            self.assertEqual(f.read(),
                             cn.attestation_line(
                                 os.path.join(self.certified, cn.CERT_FILE),
                                 own) + "\n")
        binary = os.path.join(self.certified, "model_update")
        other = self.plan(self.other)
        self.assertNotEqual(cn.plan_seal(other), seal)
        self.assertGreater(len(fx.formats.relaxed_opcodes(read(other))), 0)
        for backend in ("cpu", "metal"):
            with self.subTest(backend=backend):
                done, out, report = self.update(binary, f"cert_{backend}",
                                                backend)
                self.assertEqual(done.returncode, 0, done.stderr)
                with open(report) as f:
                    ran = json.load(f)
                if backend == "metal":
                    # The relaxed kernels really ran: the gate admitted the
                    # certified plan as a relaxed plan, not as an exact one.
                    self.assertGreater(ran["relaxed_executions"], 0)
                committed = read(out)
                # Its own plan, by --plan: the same seal, the same bytes.
                done, out, report = self.update(
                    binary, f"cert_own_{backend}", backend, "--plan", own)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(read(out), committed)
                # Another relaxed plan — certified or not, it is not the
                # plan this binary was built for.
                done, out, report = self.update(
                    binary, f"cert_other_{backend}", backend, "--plan", other)
                self.assertEqual(done.returncode, 1, done.stderr)
                self.assertIn(ANOTHER_PLAN, done.stderr)
                self.assertFalse(os.path.exists(out))
                # An exact plan needs no permit, whatever the binary holds:
                # the bytes the exact package commits.
                done, out, report = self.update(
                    binary, f"cert_exact_{backend}", backend, "--plan",
                    self.plan(self.exact))
                self.assertEqual(done.returncode, 0, done.stderr)
                want, exact_out, _ = self.update(
                    os.path.join(self.exact, "model_update"),
                    f"cert_exact_want_{backend}", backend)
                self.assertEqual(want.returncode, 0, want.stderr)
                if backend == "cpu":
                    # The relaxed package's CPU backend may link Accelerate
                    # for its relaxed kernels; an exact plan takes none of
                    # them, so the bytes are the exact package's.
                    self.assertEqual(read(out), read(exact_out))

    def test_the_certified_build_is_the_only_way_in(self):
        """The same relaxed package rebuilt with its certificate's verdict
        flipped does not build; built with no gate at all (an exact
        package's script) it builds a runner that refuses its own plan."""
        pkg = os.path.join(self.dir, "dec_cert_copy")
        shutil.copytree(self.certified, pkg)
        cert = os.path.join(pkg, cn.CERT_FILE)
        with open(cert) as f:
            text = f.read()
        with open(cert, "w") as f:
            f.write(text.replace('"verdict": "granted"',
                                 '"verdict": "refused"'))
        done = subprocess.run(["sh", "build.sh"], cwd=pkg,
                              env=dict(os.environ, CXX="true"),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("does not attest", done.stderr)
        # The exact package's runner beside the relaxed plan: no seal was
        # compiled in, so the swap fails closed (asserted on cpu; the
        # reproduction test covers metal).
        done, out, report = self.update(
            os.path.join(self.exact, "model_update"), "ungated", "cpu",
            "--plan", self.plan(self.certified))
        self.refused(done, out, report, NO_CERTIFICATE)

    def test_the_probe_measures_a_relaxed_plan_only_when_told(self):
        """seeml-plan-probe holds the device's rule and has no certificate
        compiled in; --relaxed-for-measurement is its named permission, the
        one tool/frontier_exec.py's Probe passes."""
        plan = self.plan(self.relaxed)
        header = fx.Plan.load(plan)
        arena = os.path.join(self.dir, "arena.in")
        with open(arena, "wb") as f:
            f.write(bytes(header.arena_size))
        base = [fixtures.PROBE, "--plan", plan, "--section", "train",
                "--arena-in", arena, "--arena-out",
                os.path.join(self.dir, "arena.out")]
        done = subprocess.run(base, capture_output=True, text=True)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn(NO_CERTIFICATE, done.stderr)
        self.assertIn("--relaxed-for-measurement", done.stderr)
        done = subprocess.run(base + ["--relaxed-for-measurement"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        # An exact plan needs no flag, and takes it harmlessly.
        exact = self.plan(self.exact)
        for extra in ([], ["--relaxed-for-measurement"]):
            done = subprocess.run(
                [fixtures.PROBE, "--plan", exact, "--section", "train",
                 "--arena-in", arena, "--arena-out",
                 os.path.join(self.dir, "arena.out"), *extra],
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        probe = fx.Probe(fixtures.PROBE, plan)
        self.addCleanup(probe.close)
        self.assertIn("--relaxed-for-measurement", probe.base)


if __name__ == "__main__":
    unittest.main()
