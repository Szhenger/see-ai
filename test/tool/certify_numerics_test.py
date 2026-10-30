"""Tests for tool/certify_numerics.py, the relaxed-reduction certifier
(P3, #77).

  stdlib   what a certificate binds to and how verification fails closed;
  numpy    the relaxed model itself: the lane reduction is the schedule it
           claims to be, and the any-order bound really bounds it — on
           benign operands and on catastrophic cancellation alike;
  C++      certificates over compiled plans: granted, refused, voided by a
           recompile, enforced by tool/pack_update.py, and checked against
           a measured run (a frontier_exec.py diff report).

    python3 -m unittest discover -s test/tool -p '*_test.py'
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

np = fixtures.np
N = fx.NULL_REF


def run(module, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        status = module.main(argv)
    return status, buf.getvalue()


def granted(plan_path, **over):
    cert = {"schema": 1, "kind": cn.CERT_KIND, "seeml_tool": "test",
            "plan": {"file": "p", "sha256": cn.sha256_file(plan_path),
                     "plan_hash": "0", "version": 11},
            "corpus": {"file": "c", "sha256": "0" * 64},
            "contract": {"family": cn.FAMILY, "model_lanes": 8,
                         "unit_roundoff": cn.UNIT_ROUNDOFF,
                         "sites": ["attn.fwd"]},
            "evidence": {}, "tolerances": {"site_rel": 1e-5,
                                           "loss_rel": 1e-4},
            "verdict": "granted", "reasons": []}
    cert.update(over)
    cert["digest"] = cn.self_digest(cert)
    return cert


class CertificateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.plan = os.path.join(self._tmp.name, "update_plan.seeu")
        with open(self.plan, "wb") as f:
            f.write(fixtures.assemble(64))

    def test_a_granted_certificate_verifies(self):
        self.assertEqual(cn.check_certificate(granted(self.plan), self.plan),
                         [])

    def test_verification_fails_closed(self):
        cert = granted(self.plan)
        edited = dict(cert, tolerances={"site_rel": 1.0, "loss_rel": 1.0})
        self.assertIn("edited", cn.check_certificate(edited, self.plan)[0])
        refusal = granted(self.plan, verdict="refused", reasons=["too loose"])
        self.assertIn("refusal: too loose",
                      cn.check_certificate(refusal, self.plan)[0])
        with open(self.plan, "ab") as f:
            f.write(b"\0")  # "recompiled"
        self.assertIn("different plan",
                      cn.check_certificate(cert, self.plan)[0])
        for junk in ({}, [], {"kind": cn.CERT_KIND, "schema": 99}):
            self.assertIn("not a schema-1",
                          cn.check_certificate(junk, self.plan)[0])
        hollow = {"schema": 1, "kind": cn.CERT_KIND}
        hollow["digest"] = cn.self_digest(hollow)
        self.assertIn("malformed", cn.check_certificate(hollow, self.plan)[0])

    def test_observed_runs_are_held_to_the_tolerances(self):
        cert = granted(self.plan)
        sha = cn.sha256_file(self.plan)

        def report(**over):
            r = {"rtol": 1e-5, "agrees": True, "failures": [],
                 "mismatches": [], "plan_sha256": sha, "plan_hash": "0",
                 "opcodes": {"attn.fwd": {"worst_ratio": 0.4,
                                          "worst_rel": 4e-6},
                             "gemm.nn": {"worst_ratio": 0.2,
                                         "worst_rel": 2e-6}}}
            r.update(over)
            return r

        self.assertEqual(cn.check_observed(cert, report()), [])
        loose = report(opcodes={"attn.fwd": {"worst_ratio": 3.0,
                                             "worst_rel": 3e-5}})
        self.assertIn("attn.fwd: measured", cn.check_observed(cert, loose)[0])
        stray = report(opcodes={"gemm.nn": {"worst_ratio": 1.5,
                                            "worst_rel": 1.5e-5}})
        self.assertIn("not a certified site",
                      cn.check_observed(cert, stray)[0])
        # F41: the relative error is read from the report, not rebuilt from
        # rtol (a lower bound); a NaN cannot pass, nor can a run compared at
        # a tolerance of zero.
        nan = report(opcodes={"attn.fwd": {"worst_ratio": 0.0,
                                           "worst_rel": float("nan")}})
        self.assertIn("attn.fwd: measured", cn.check_observed(cert, nan)[0])
        for rtol in (0.0, -1e-5, float("inf"), float("nan")):
            self.assertIn("rtol", cn.check_observed(cert, report(rtol=rtol))[0])
        old = report(opcodes={"attn.fwd": {"worst_ratio": 0.0}})
        self.assertIn("no relative error", cn.check_observed(cert, old)[0])
        # F42: the report binds to the plan, and its own verdict counts.
        self.assertIn("different plan", cn.check_observed(
            cert, report(plan_sha256="0" * 64))[0])
        self.assertIn("different plan",
                      cn.check_observed(cert, report(plan_hash="1"))[0])
        unnamed = report()
        del unnamed["plan_sha256"], unnamed["plan_hash"]
        self.assertIn("does not name its plan",
                      cn.check_observed(cert, unnamed)[0])
        # A structural mismatch refuses; a certified site past the diff
        # tolerance (agrees=False, no mismatch) is judged by worst_rel
        # alone, since the diff's rtol is not the certified tolerance.
        shape = "step 1 train #3 attn.fwd @0: 8 floats vs 16"
        self.assertIn("structural mismatch", cn.check_observed(cert, report(
            agrees=False, failures=[shape], mismatches=[shape]))[0])
        self.assertEqual(cn.check_observed(cert, report(
            agrees=False, failures=["step 1 train #3 attn.fwd @0: ..."],
            opcodes={"attn.fwd": {"worst_ratio": 3.0,
                                  "worst_rel": 4e-6}})), [])
        old_report = report(agrees=False,
                            failures=["step 1 train #3 gemm.nn @0: ..."])
        del old_report["mismatches"]
        self.assertIn("disagrees", cn.check_observed(cert, old_report)[0])
        # F72: the relaxed GEMM site is judged at its own tolerance.
        relaxed = granted(self.plan, contract={
            "family": cn.FAMILY, "model_lanes": 8,
            "unit_roundoff": cn.UNIT_ROUNDOFF,
            "sites": ["attn.fwd", cn.RELAXED_GEMM_SITE]},
            tolerances={"site_rel": 1e-5, "relaxed_gemm_rel": 2.0 ** -7,
                        "loss_rel": 1e-4})
        inside = report(opcodes={cn.RELAXED_GEMM_SITE: {"worst_ratio": 400.0,
                                                        "worst_rel": 4e-3}},
                        relaxed_instructions=3, cpp_relaxed_executed=3)
        self.assertEqual(cn.check_observed(relaxed, inside), [])
        outside = report(opcodes={cn.RELAXED_GEMM_SITE: {"worst_ratio": 1e3,
                                                         "worst_rel": 1e-2}},
                         relaxed_instructions=3, cpp_relaxed_executed=3)
        self.assertIn("gemm.relaxed: measured",
                      cn.check_observed(relaxed, outside)[0])

    def test_an_observed_run_must_have_measured_the_relaxed_kernels(self):
        # F42's residual: a diff whose C++ arm ran the relaxed GEMMs on the
        # exact kernels, or whose reference arm modelled the bf16 rounding
        # too, agrees with itself and vouches for the certified site
        # vacuously. Both are refused; a certificate over no relaxed site
        # asks nothing of the field.
        relaxed = granted(self.plan, contract={
            "family": cn.FAMILY, "model_lanes": 8,
            "unit_roundoff": cn.UNIT_ROUNDOFF,
            "sites": ["attn.fwd", cn.RELAXED_GEMM_SITE]},
            tolerances={"site_rel": 1e-5, "relaxed_gemm_rel": 2.0 ** -7,
                        "loss_rel": 1e-4})
        sha = cn.sha256_file(self.plan)

        def report(**over):
            r = {"rtol": 1e-5, "agrees": False, "failures": ["x"],
                 "mismatches": [], "plan_sha256": sha, "plan_hash": "0",
                 "cpp_backend": "cpu", "relaxed_modelled": False,
                 "relaxed_instructions": 4, "cpp_relaxed_executed": 4,
                 "opcodes": {cn.RELAXED_GEMM_SITE: {
                     "worst_ratio": 300.0, "worst_rel": 3e-3,
                     "cpp_relaxed_executed": 4}}}
            r.update(over)
            return r

        self.assertEqual(cn.check_observed(relaxed, report()), [])
        # A Metal backend keeps skinny shapes exact by design: one is enough.
        self.assertEqual(cn.check_observed(relaxed,
                                           report(cpp_relaxed_executed=1)), [])
        exact = cn.check_observed(relaxed, report(cpp_relaxed_executed=0))
        self.assertEqual(len(exact), 1)
        self.assertIn("4 relaxed GEMM(s) on the exact kernels", exact[0])
        self.assertIn("--cpp-backend metal", exact[0])
        old = report()
        del old["cpp_relaxed_executed"]
        self.assertIn("regenerate", cn.check_observed(relaxed, old)[0])
        modelled = cn.check_observed(relaxed, report(relaxed_modelled=True))
        self.assertIn("modelled the relaxed rounding", modelled[0])
        plain = granted(self.plan)
        unrelaxed = report(opcodes={"attn.fwd": {"worst_ratio": 0.4,
                                                 "worst_rel": 4e-6}},
                           agrees=True, failures=[], relaxed_modelled=True)
        del unrelaxed["cpp_relaxed_executed"]
        self.assertEqual(cn.check_observed(plain, unrelaxed), [])

    def test_tolerances_looser_than_the_defaults_are_named(self):
        # F2's decision made visible: each contract family has one table of
        # defaults, the CLI leaves a tolerance unset until the plan's family
        # is known, and a certificate granted under looser settings than
        # its family's is named for it (the packer repeats this).
        args = cn.build_parser().parse_args(["certify", "p", "--corpus", "c"])
        self.assertIsNone(args.max_site_error)
        self.assertIsNone(args.max_relaxed_gemm_error)
        self.assertIsNone(args.max_loss_deviation)
        self.assertEqual(cn.FAMILY_DEFAULTS[cn.FAMILY]["loss_rel"], 1e-4)
        self.assertEqual(cn.FAMILY_DEFAULTS[cn.FLAGGED_FAMILY]["loss_rel"], 1e-3)
        for family in cn.FAMILIES:
            self.assertEqual(cn.FAMILY_DEFAULTS[family]["site_rel"], 1e-5)
            self.assertEqual(cn.FAMILY_DEFAULTS[family]["relaxed_gemm_rel"],
                             2.0 ** -7)
        self.assertEqual(cn.tolerance_notes(granted(self.plan)), [])
        loose = granted(self.plan, tolerances={"site_rel": 1e-4,
                                               "relaxed_gemm_rel": 2.0 ** -7,
                                               "loss_rel": 1e-2})
        notes = cn.tolerance_notes(loose)
        self.assertEqual(notes, [
            "site_rel 1.0e-04 is looser than the certify default 1.0e-05",
            "loss_rel 1.0e-02 is looser than the certify default 1.0e-04"])
        # The same tolerances under the flagged contract, against its own
        # defaults: the loss line is 1e-3 there.
        flagged = granted(self.plan, contract={
            "family": cn.FLAGGED_FAMILY, "model_lanes": 8,
            "unit_roundoff": cn.UNIT_ROUNDOFF,
            "sites": [cn.RELAXED_GEMM_SITE]},
            tolerances={"site_rel": 1e-5, "relaxed_gemm_rel": 2.0 ** -7,
                        "loss_rel": 1e-2})
        self.assertEqual(cn.tolerance_notes(flagged), [
            "loss_rel 1.0e-02 is looser than the certify default 1.0e-03"])
        at_default = granted(self.plan, contract=dict(
            flagged["contract"]), tolerances=dict(
                cn.FAMILY_DEFAULTS[cn.FLAGGED_FAMILY]))
        self.assertEqual(cn.tolerance_notes(at_default), [])
        self.assertEqual(cn.check_certificate(at_default, self.plan), [])
        self.assertEqual(cn.tolerance_notes({}), [])
        self.assertEqual(cn.check_certificate(loose, self.plan), [])  # legal
        odd = granted(self.plan, contract=dict(flagged["contract"],
                                               family="relaxed-something"))
        self.assertIn("unknown contract family",
                      cn.check_certificate(odd, self.plan)[0])

    def test_a_vacuous_tolerance_certifies_nothing(self):
        for value in (0.0, -1e-5, float("inf"), float("nan")):
            cert = granted(self.plan, tolerances={"site_rel": value,
                                                  "loss_rel": 1e-4})
            problems = cn.check_certificate(cert, self.plan)
            self.assertTrue(any("tolerance site_rel" in p for p in problems),
                            problems)
            # --flag=value: Python 3.9's argparse reads a bare "-1e-05" as an
            # option and refuses it before the tool sees it.
            status, log = run(cn, ["certify", self.plan, "--corpus", "c",
                                   f"--max-site-error={value!r}"])
            self.assertEqual(status, 2, log)
            self.assertIn("certifies nothing", log)

    def test_the_certificate_binds_to_the_source_model(self):
        other = os.path.join(self._tmp.name, "other.smf")
        with open(other, "wb") as f:
            f.write(b"not the model")
        bound = granted(self.plan, source={"file": "m.smf",
                                           "sha256": cn.sha256_file(self.plan)})
        self.assertEqual(cn.check_certificate(bound, self.plan,
                                              source_path=self.plan), [])
        self.assertIn("different source model", cn.check_certificate(
            bound, self.plan, source_path=other)[0])
        unbound = granted(self.plan)
        self.assertEqual(cn.check_certificate(unbound, self.plan), [])
        self.assertIn("without a source model", cn.check_certificate(
            unbound, self.plan, source_path=self.plan)[0])

    def test_gamma(self):
        self.assertAlmostEqual(cn.gamma(64), 64 * 2.0 ** -24, delta=1e-10)
        self.assertEqual(cn.gamma(2 ** 24), float("inf"))

    def test_pack_update_enforces_a_certificate_that_is_present(self):
        pkg = self._tmp.name
        self.assertIsNone(pack_update.check_certificate(pkg, self.plan))
        path = os.path.join(pkg, pack_update.CERTIFICATE_FILE)
        with open(path, "w") as f:
            json.dump(granted(self.plan), f)
        summary = pack_update.check_certificate(pkg, self.plan)
        self.assertEqual(summary["family"], cn.FAMILY)
        with open(self.plan, "ab") as f:
            f.write(b"\0")
        with self.assertRaisesRegex(pack_update.PackError,
                                    "does not vouch for this plan"):
            pack_update.check_certificate(pkg, self.plan)


@unittest.skipIf(np is None, "NumPy not installed")
class RelaxedModelTest(unittest.TestCase):
    def test_the_lane_sum_is_the_schedule_it_claims(self):
        x = cn.LaneBackend(4)
        v = np.random.default_rng(1).standard_normal((3, 11)).astype(
            np.float32)
        want = []
        for row in v:
            lanes = [np.float32(0)] * 4
            for i, value in enumerate(row):  # strided sequential f32 adds
                lanes[i % 4] = np.float32(lanes[i % 4] + value)
            want.append(np.float32(np.float32(lanes[0] + lanes[1]) +
                                   np.float32(lanes[2] + lanes[3])))
        got = x._lane_sum(v, -1)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got, np.array(want, np.float32))

    def test_the_any_order_bound_bounds_every_schedule(self):
        rng = np.random.default_rng(2)
        benign = rng.standard_normal((64, 777))
        hostile = np.concatenate([benign * 1e6, -benign * 1e6 + 1e-3], -1)
        for data in (benign, hostile):
            exact = data.astype(np.float32).astype(np.float64).sum(-1)
            for lanes in (1, 2, 8, 32):
                x = cn.LaneBackend(lanes)
                got = x.sum(data, -1)
                err = np.abs(got - exact).max() / np.abs(got).max()
                self.assertLessEqual(err, x.bound_rel * (1 + 1e-12))
        self.assertGreater(x.bound_rel, 1e-2)  # cancellation is visible
        a, b = rng.standard_normal((5, 9, 64)), rng.standard_normal((5, 64, 7))
        x = cn.LaneBackend(8)
        got = x.matmul(a, b)
        exact = (a.astype(np.float32).astype(np.float64) @
                 b.astype(np.float32).astype(np.float64))
        self.assertLessEqual(np.abs(got - exact).max() / np.abs(got).max(),
                             x.bound_rel)
        self.assertGreater(np.abs(got - exact).max(), 0.0)  # it IS relaxed

    def test_lockstep_sees_cancellation_at_the_site_that_owns_it(self):
        rows, cols = 4096, 2
        big = np.tile([[3e7, 1.0], [-3e7, 1.0]], (rows // 2, 1))
        plan = fx.Plan(fixtures.assemble(
            4 * (rows * cols + 8),
            train=[(10, 0, (0, 4 * rows * cols, N, N), (rows, cols, 0)),
                   (19, 0, (4 * rows * cols + 16, fx.f32_bits(1.0), N, N),
                    (2, 0, 0))]))
        ex = cn.LockstepExecutor(plan, 8)
        ex.mem.stage(0, big.astype("<f4").tobytes())
        ex.execute("train")
        self.assertEqual(sorted(ex.sites), ["reduce.rows"])  # fill: no site
        stat = ex.sites["reduce.rows"]
        self.assertLessEqual(stat["observed_rel"], stat["bound_rel"])
        self.assertGreater(stat["bound_rel"], 1.0)  # vacuous: refuse-worthy

    def test_the_flagged_contract_relaxes_only_what_the_plan_marks(self):
        # F2 (#130): under P3's contract the relaxed arm recomputes every
        # reduction site in lane-accumulated float32; under the flagged
        # contract it relaxes the kFlagRelaxed instructions alone, so an
        # unflagged reduction runs exactly as the reference does — the
        # shipped arithmetic, since every backend runs reductions exact.
        # One 1e8 among 4,095 ones: a float32 lane holding the 1e8 drops
        # every 1 added to it (half its spacing is 4), so the lane sum is
        # 1e8 + 7 * 512 where the exact sum rounds to 1e8 + 4,096.
        rows, cols = 4096, 2
        big = np.ones((rows, cols))
        big[0, 0] = 1e8
        plan = fx.Plan(fixtures.assemble(
            4 * (rows * cols + 8),
            train=[(10, 0, (0, 4 * rows * cols, N, N), (rows, cols, 0))]))
        self.assertEqual(cn.family_of(plan), cn.FAMILY)
        self.assertEqual(cn.flagged_sites(plan), [])

        def reduce_with(relax):
            ex = cn.RelaxedExecutor(plan, 8, relax=relax)
            ex.mem.stage(0, big.astype("<f4").tobytes())
            ex.execute("train")
            return np.frombuffer(ex.mem.export(), "<f4")[
                rows * cols:rows * cols + cols].copy()

        reference = fx.Executor(plan, fx.NumpyBackend("float64"))
        reference.mem.stage(0, big.astype("<f4").tobytes())
        reference.execute("train")
        exact = np.frombuffer(reference.mem.export(), "<f4")[
            rows * cols:rows * cols + cols]
        np.testing.assert_array_equal(reduce_with(cn.is_flagged), exact)
        self.assertFalse(np.array_equal(reduce_with(cn.is_site), exact))
        # A plan that marks a GEMM states the flagged contract over it.
        w = np.array([[1.0, -2.0], [0.5, 4.0]], np.float32)
        gemm = fx.Plan(fixtures.assemble(
            64, train=[(1, fx.FLAG_RELAXED, (0, fx.RODATA_BIT, 16, N),
                        (1, 2, 2))], rodata=w.tobytes()))
        self.assertEqual(cn.family_of(gemm), cn.FLAGGED_FAMILY)
        self.assertEqual(cn.flagged_sites(gemm), [cn.RELAXED_GEMM_SITE])


@unittest.skipIf(np is None, "NumPy not installed")
@unittest.skipUnless(fixtures.HAVE_CPP,
                     f"no built seeml tools under {fixtures.BUILD}")
class CertifyPlansTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="seeml-certify-test-")
        cls.dir = cls._tmp.name
        fixtures._write_models(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def compile(self, name, model, batch, *flags):
        out = os.path.join(self.dir, name)
        shutil.rmtree(out, ignore_errors=True)
        done = subprocess.run(
            [fixtures.COMPILER, "--source", os.path.join(self.dir, model),
             "--out", out, "--data-batch", str(batch), "--no-embed", *flags],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(done.returncode, 0, done.stdout)
        return out

    def test_certify_verify_pack_and_void(self):
        corpus = os.path.join(self.dir, "decoder_corpus.sds")
        pkg = self.compile("dec", "decoder.smf", 12, "--lr", "2e-3")
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "6", "--val-fraction", "0.25"])
        self.assertEqual(status, 0, log)
        cert_path = os.path.join(pkg, cn.CERT_FILE)
        with open(cert_path) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "granted")
        self.assertEqual(cert["contract"]["sites"], [
            "attn.dp", "attn.fwd", "rms_norm.bwd", "rms_norm.fwd",
            "softmax_rows.bwd", "softmax_xent.fwd"])
        for stat in cert["evidence"]["sites"].values():
            self.assertGreater(stat["instances"], 0)
            self.assertLessEqual(stat["observed_rel"], stat["bound_rel"])
            self.assertLess(stat["observed_rel"], 1e-5)
        free = cert["evidence"]["free_run"]
        self.assertLess(free["loss_max_rel_dev"], 1e-4)
        self.assertTrue(free["gate_improved_reference"])
        self.assertTrue(free["gate_improved_relaxed"])

        self.assertEqual(run(cn, ["verify", cert_path, "--plan", pkg,
                                  "--corpus", corpus])[0], 0)
        # A measured run (here: the exact C++ runtime itself) is inside the
        # certified tolerances.
        diff = os.path.join(self.dir, "diff.json")
        self.assertEqual(run(fx, ["diff", os.path.join(pkg, cn.PLAN_FILE),
                                  "--corpus", corpus, "--probe",
                                  fixtures.PROBE, "--steps", "2", "--report",
                                  diff])[0], 0)
        self.assertEqual(run(cn, ["verify", cert_path, "--plan", pkg,
                                  "--observed", diff])[0], 0)
        wrong = os.path.join(self.dir, "class.sds")
        status, log = run(cn, ["verify", cert_path, "--plan", pkg,
                               "--corpus", wrong])
        self.assertEqual(status, 3)
        self.assertIn("different corpus", log)

        # Packaging beside a valid certificate records it ...
        report = os.path.join(self.dir, "pack.json")
        self.assertEqual(run(pack_update, [pkg, "--report", report])[0], 0)
        with open(report) as f:
            self.assertEqual(json.load(f)["numerics_certificate"]["family"],
                             cn.FAMILY)
        # ... and a recompile voids it, loudly, at verify and at pack time.
        kept = os.path.join(self.dir, "kept.json")
        shutil.copy(cert_path, kept)
        pkg = self.compile("dec", "decoder.smf", 12, "--lr", "3e-3")
        shutil.copy(kept, os.path.join(pkg, cn.CERT_FILE))
        status, log = run(cn, ["verify", kept, "--plan", pkg])
        self.assertEqual(status, 3)
        self.assertIn("different plan", log)
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1)
        self.assertIn("does not vouch for this plan", log)

    def test_a_relaxed_plan_is_packaged_only_beside_its_certificate(self):
        # --precision certified-bf16 (v18): the packer refuses the plan bare,
        # the certifier prices the gemm.relaxed site (bf16 input rounding,
        # observed inside 2^-7 and its own bound), and the packer then
        # accepts it — but not an older certificate that never saw the site.
        corpus = os.path.join(self.dir, "class.sds")
        pkg = self.compile("mlp_relaxed", "mlp.smf", 8,
                           "--precision", "certified-bf16")
        plan = fx.Plan.load(os.path.join(pkg, cn.PLAN_FILE))
        self.assertGreater(plan.relaxed_gemms, 0)
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1)
        self.assertIn("relaxed GEMM", log)
        self.assertIn("no numerics certificate", log)
        # N17: neither the compiler's --build nor the package's own
        # build.sh links an uncertified relaxed plan — the script refuses
        # before compiling anything, so CXX=true drives it in milliseconds.
        done = subprocess.run(
            [fixtures.COMPILER, "--source", os.path.join(self.dir, "mlp.smf"),
             "--out", os.path.join(self.dir, "mlp_relaxed_build"),
             "--data-batch", "8", "--precision", "certified-bf16",
             "--build"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--build cannot certify", done.stdout)

        def build_sh():
            return subprocess.run(["sh", "build.sh"], cwd=pkg,
                                  env=dict(os.environ, CXX="true"),
                                  capture_output=True, text=True)

        done = build_sh()
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("no numerics_certificate.json", done.stderr)
        # A relaxed plan states the flagged contract (F2 #130): its loss
        # default is 1e-3, F2's acceptance line. On this toy MLP, four
        # steps of bf16 input rounding move the training loss further than
        # that, so the default refuses and the run states its own.
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4"])
        self.assertEqual(status, 3)
        self.assertIn("loss_max_rel_dev", log)
        self.assertIn("exceeds 1.0e-03", log)
        self.assertIn(f"contract {cn.FLAGGED_FAMILY} over 1 enforced site",
                      log)
        # Only the flagged instructions are enforced: an absurd tolerance on
        # the reduction sites, which every backend runs exact, refuses
        # nothing under this contract.
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", "--max-loss-deviation", "1e-2",
                               "--max-site-error", "1e-12"])
        self.assertEqual(status, 0, log)
        self.assertIn("evidence only", log)
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", "--max-loss-deviation", "1e-2"])
        self.assertEqual(status, 0, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "granted")
        self.assertEqual(cert["contract"]["family"], cn.FLAGGED_FAMILY)
        self.assertEqual(cert["contract"]["sites"], [cn.RELAXED_GEMM_SITE])
        self.assertGreater(len(cert["evidence"]["sites"]), 1)  # all measured
        self.assertEqual(cert["plan"]["relaxed_gemms"], plan.relaxed_gemms)
        site = cert["evidence"]["sites"][cn.RELAXED_GEMM_SITE]
        self.assertEqual(site["instances"], plan.relaxed_gemms * 4)
        self.assertGreater(site["observed_rel"], 1e-5)   # a real rounding
        self.assertLessEqual(site["observed_rel"], site["bound_rel"])
        self.assertLess(site["observed_rel"], 2.0 ** -7)
        self.assertEqual(cert["tolerances"]["relaxed_gemm_rel"], 2.0 ** -7)
        # ... broken down per opcode: the forward (NN) and the dX backward
        # (NT) products through the frozen weight.
        self.assertEqual(set(site["by_opcode"]), {"gemm.nn", "gemm.nt"})
        self.assertEqual(max(site["by_opcode"].values()), site["observed_rel"])
        report = os.path.join(self.dir, "pack_relaxed.json")
        status, log = run(pack_update, [pkg, "--report", report])
        self.assertEqual(status, 0, log)
        # The certificate was granted at a loss tolerance looser than its
        # contract's default (1e-2 against the flagged family's 1e-3): the
        # packer says so where the binary is built, and its report carries
        # the whole contract.
        self.assertIn("NOTE numerics certificate: loss_rel 1.0e-02 is looser "
                      "than the certify default 1.0e-03", log)
        with open(report) as f:
            packed = json.load(f)
        self.assertEqual(packed["relaxed_gemms"], plan.relaxed_gemms)
        self.assertEqual(packed["numerics_certificate"]["tolerances"],
                         cert["tolerances"])
        self.assertEqual(packed["numerics_certificate"]["evidence_steps"], 4)
        self.assertEqual(
            [n.split()[0] for n in
             packed["numerics_certificate"]["looser_than_defaults"]],
            ["loss_rel"])
        # Packed beside its certificate, the script gets past the
        # preflight; a refused certificate, or one for other plan bytes,
        # stops it again.
        self.assertEqual(build_sh().returncode, 0)
        cert_path = os.path.join(pkg, cn.CERT_FILE)
        with open(cert_path) as f:
            text = f.read()
        with open(cert_path, "w") as f:
            f.write(text.replace('"verdict": "granted"',
                                 '"verdict": "refused"'))
        done = build_sh()
        self.assertEqual(done.returncode, 1)
        self.assertIn("not a granted certificate", done.stderr)
        with open(cert_path, "w") as f:
            f.write(text.replace(cert["plan"]["sha256"], "0" * 64))
        done = build_sh()
        self.assertEqual(done.returncode, 1)
        self.assertIn("SHA-256 mismatch", done.stderr)
        with open(cert_path, "w") as f:
            f.write(text)
        # F72 / F42: a measured run keys its relaxed GEMMs under the
        # certificate's site name, names the plan it measured, and says how
        # many of those GEMMs the C++ backend ran on a relaxed kernel. On a
        # SEEML_ACCELERATE build of the probe every one of them runs on
        # SGEMM — the site measures SGEMM's f32 round-off against the exact
        # reference, far inside 2^-7 — and the certificate verifies; on the
        # reference build none does, the diff is exact against exact, and
        # verify refuses it: exact-vs-exact vouches for nothing.
        diff = os.path.join(self.dir, "diff_relaxed.json")
        self.assertEqual(run(fx, ["diff", os.path.join(pkg, cn.PLAN_FILE),
                                  "--corpus", corpus, "--probe",
                                  fixtures.PROBE, "--steps", "1",
                                  "--report", diff])[0], 0)
        with open(diff) as f:
            observed = json.load(f)
        self.assertIn(cn.RELAXED_GEMM_SITE, observed["opcodes"])
        site_stat = observed["opcodes"][cn.RELAXED_GEMM_SITE]
        self.assertLess(site_stat["worst_rel"], 1e-5)
        self.assertEqual(observed["plan_sha256"], cert["plan"]["sha256"])
        self.assertEqual(observed["plan_hash"], cert["plan"]["plan_hash"])
        self.assertEqual(observed["relaxed_instructions"], plan.relaxed_gemms)
        executed = observed["cpp_relaxed_executed"]
        self.assertIn(executed, (0, plan.relaxed_gemms))
        self.assertEqual(site_stat["cpp_relaxed_executed"], executed)
        status, log = run(cn, ["verify", cert_path, "--plan", pkg,
                               "--observed", diff])
        if executed:
            self.assertEqual(status, 0, log)
        else:
            self.assertEqual(status, 3, log)
            self.assertIn("on the exact kernels", log)
        # Both refusals, whatever this build ran: a report whose C++ arm
        # took the permission nowhere, and one whose reference arm modelled
        # the rounding too.

        def rewrite(**over):
            with open(diff) as f:
                r = json.load(f)
            r.update(over)
            with open(diff, "w") as f:
                json.dump(r, f)

        rewrite(cpp_relaxed_executed=0)
        status, log = run(cn, ["verify", cert_path, "--plan", pkg,
                               "--observed", diff])
        self.assertEqual(status, 3)
        self.assertIn(f"{plan.relaxed_gemms} relaxed GEMM(s) on the exact "
                      "kernels", log)
        rewrite(cpp_relaxed_executed=max(executed, 1), relaxed_modelled=True)
        status, log = run(cn, ["verify", cert_path, "--plan", pkg,
                               "--observed", diff])
        self.assertEqual(status, 3)
        self.assertIn("modelled the relaxed rounding", log)
        rewrite(cpp_relaxed_executed=max(executed, 1), relaxed_modelled=False)
        self.assertEqual(run(cn, ["verify", cert_path, "--plan", pkg,
                                  "--observed", diff])[0], 0)
        # A certificate over the exact plan does not cover the relaxed one.
        exact = self.compile("mlp_exact", "mlp.smf", 8)
        self.assertEqual(run(cn, ["certify", exact, "--corpus", corpus,
                                  "--steps", "4"])[0], 0)
        with open(os.path.join(exact, cn.CERT_FILE)) as f:
            other = json.load(f)
        other["plan"]["sha256"] = cert["plan"]["sha256"]
        other["digest"] = cn.self_digest(other)
        problems = cn.check_certificate(other, os.path.join(pkg, cn.PLAN_FILE))
        self.assertTrue(any("does not cover" in p for p in problems), problems)
        # And the exact plan's train program is the relaxed one's, minus
        # the bit: the format change is inert on the reference backend.
        a = fx.Plan.load(os.path.join(exact, cn.PLAN_FILE))
        for x, y in zip(a.sections["train"], plan.sections["train"]):
            self.assertEqual(x.opcode, y.opcode)
            self.assertEqual(x.flags, y.flags & ~fx.FLAG_RELAXED)
            self.assertEqual(x.src, y.src)

    def test_a_tolerance_the_plan_cannot_meet_is_a_refusal(self):
        pkg = self.compile("mlp", "mlp.smf", 8, "--clip-norm", "0.5")
        status, log = run(cn, ["certify", pkg, "--corpus",
                               os.path.join(self.dir, "class.sds"),
                               "--steps", "4", "--max-site-error", "1e-12"])
        self.assertEqual(status, 3, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["verdict"], "refused")
        self.assertTrue(any("exceeds 1.0e-12" in r for r in cert["reasons"]))
        self.assertEqual(run(cn, ["verify", os.path.join(pkg, cn.CERT_FILE),
                                  "--plan", pkg])[0], 3)
        status, log = run(pack_update, [pkg])  # a refusal packs nothing
        self.assertEqual(status, 1)
        self.assertIn("refusal", log)

    def test_every_loss_and_norm_family_certifies(self):
        for name, model, corpus, batch, flags in fixtures.MATRIX:
            if name not in ("mlp_mse_sgd", "mlp_xent_kl", "mlp_accum"):
                continue
            flags = [os.path.join(self.dir, f) if f.endswith(".smf") else f
                     for f in flags]
            pkg = self.compile(name, model, batch, *flags)
            with self.subTest(plan=name):
                status, log = run(cn, ["certify", pkg, "--corpus",
                                       os.path.join(self.dir, corpus),
                                       "--steps", "5"])
                self.assertEqual(status, 0, log)


if __name__ == "__main__":
    unittest.main()
