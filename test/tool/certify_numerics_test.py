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
            "evidence": {"steps": 20},
            "tolerances": {"site_rel": 1e-5, "loss_rel": 1e-4},
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
        self.assertIn("4 relaxed instruction(s) on the exact kernels",
                      exact[0])
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
        # Legal: looser than the defaults and at the ceilings, not above
        # them (F2 #206) — a NOTE where the binary is built, not a refusal.
        self.assertEqual(cn.check_certificate(loose, self.plan), [])
        self.assertEqual(cn.check_certificate(flagged, self.plan), [])
        odd = granted(self.plan, contract=dict(flagged["contract"],
                                               family="relaxed-something"))
        self.assertIn("unknown contract family",
                      cn.check_certificate(odd, self.plan)[0])

    def test_a_tolerance_above_its_ceiling_is_a_refusal(self):
        # F2 #206: `--max-*-error 1e30` was "granted" and packed with three
        # NOTE lines — a tolerance had only to be finite and positive. Each
        # now has a ceiling per contract family, the loosest value the
        # project ever shipped a certificate under; above it the document
        # is not a certificate, for check_certificate and so for certify,
        # verify and pack_update alike.
        self.assertEqual(cn.TOLERANCE_CEILINGS, {
            family: {"site_rel": 1e-4, "relaxed_gemm_rel": 2.0 ** -7,
                     "loss_rel": 1e-2} for family in cn.FAMILIES})
        for family in cn.FAMILIES:
            contract = {"family": family, "model_lanes": 8,
                        "unit_roundoff": cn.UNIT_ROUNDOFF,
                        "sites": ["attn.fwd"]}
            ceilings = cn.TOLERANCE_CEILINGS[family]
            for key, default in cn.FAMILY_DEFAULTS[family].items():
                self.assertLessEqual(default, ceilings[key])
            at = granted(self.plan, contract=contract,
                         tolerances=dict(ceilings))
            self.assertEqual(cn.check_certificate(at, self.plan), [])
            for key, ceiling in ceilings.items():
                for value in (ceiling * 1.001, ceiling * 10, 1.0, 1e30):
                    with self.subTest(family=family, key=key, value=value):
                        loose = granted(self.plan, contract=contract,
                                        tolerances=dict(ceilings,
                                                        **{key: value}))
                        problems = cn.check_certificate(loose, self.plan)
                        self.assertEqual(len(problems), 1, problems)
                        self.assertIn(f"tolerance {key} {value:.1e} is "
                                      f"looser than its ceiling "
                                      f"{ceiling:.1e}", problems[0])
                        self.assertEqual(cn.strength_problems(loose),
                                         problems)
        # The issue's certificate: all three at 1e30, one step.
        absurd = granted(self.plan,
                         tolerances={"site_rel": 1e30,
                                     "relaxed_gemm_rel": 1e30,
                                     "loss_rel": 1e30},
                         evidence={"steps": 1})
        problems = cn.check_certificate(absurd, self.plan)
        self.assertEqual(len(problems), 4, problems)
        self.assertEqual(sum("looser than its ceiling" in p
                             for p in problems), 3)
        self.assertIn("below the floor of 2", problems[3])
        # ... refused by verify,
        path = os.path.join(self._tmp.name, pack_update.CERTIFICATE_FILE)
        with open(path, "w") as f:
            json.dump(absurd, f)
        status, log = run(cn, ["verify", path, "--plan", self.plan])
        self.assertEqual(status, 3, log)
        self.assertEqual(log.count("NOT VERIFIED tolerance"), 3)
        self.assertIn("NOT VERIFIED the free run is 1 step(s)", log)
        # ... and by the packer, which used to print three NOTE lines; an
        # attestation an earlier pack left does not survive the refusal.
        attest = os.path.join(self._tmp.name, cn.ATTESTATION_FILE)
        with open(attest, "w") as f:
            f.write("stale\n")
        with self.assertRaisesRegex(pack_update.PackError,
                                    "looser than its ceiling"):
            pack_update.check_certificate(self._tmp.name, self.plan)
        self.assertFalse(os.path.exists(attest))

    def test_one_step_is_not_evidence(self):
        self.assertEqual(cn.MIN_EVIDENCE_STEPS, 2)
        ok = granted(self.plan, evidence={"steps": cn.MIN_EVIDENCE_STEPS})
        self.assertEqual(cn.check_certificate(ok, self.plan), [])
        for steps, says in ((1, "below the floor of 2"),
                            (0, "below the floor of 2"),
                            (-5, "below the floor of 2"),
                            (None, "records no evidence steps"),
                            (20.0, "records no evidence steps"),
                            ("20", "records no evidence steps"),
                            (True, "records no evidence steps")):
            with self.subTest(steps=steps):
                cert = granted(self.plan, evidence={"steps": steps})
                problems = cn.check_certificate(cert, self.plan)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(says, problems[0])
        bare = granted(self.plan, evidence={})
        self.assertIn("records no evidence steps",
                      cn.check_certificate(bare, self.plan)[0])
        # The command line holds the floor before any plan is read.
        for steps in ("1", "0", "-3"):
            status, log = run(cn, ["certify", self.plan, "--corpus", "c",
                                   f"--steps={steps}"])
            self.assertEqual(status, 2, log)
            self.assertIn("--steps must be at least 2", log)
            self.assertIn("one step is not evidence", log)

    def test_the_ledgers_certificates_are_inside_the_ceilings(self):
        # The ceilings and the floor were chosen to admit every certificate
        # the project has shipped (docs/ledger). The plans are not in the
        # ledger, so this holds the documents to the plan-independent half
        # of the check — and names the two issued at the ceilings.
        ledger = os.path.join(REPO, "docs", "ledger")
        found = {}
        for root, _, files in os.walk(ledger):
            for name in files:
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(root, name)) as f:
                    try:
                        doc = json.load(f)
                    except ValueError:
                        continue
                if isinstance(doc, dict) and doc.get("kind") == cn.CERT_KIND:
                    found[os.path.relpath(os.path.join(root, name),
                                          ledger)] = doc
        self.assertEqual(sorted(found), [
            "f2-2026-10-15/pkg_relaxed/numerics_certificate.json",
            "f2-2026-10-23/numerics_certificate.json",
            "f2-2026-10-23/numerics_certificate_defaults_refused.json",
            "f2-2026-10-30/numerics_certificate.json",
            "f3-2026-10-30/numerics_certificate.json"])
        at_ceiling = []
        for name, cert in sorted(found.items()):
            with self.subTest(certificate=name):
                self.assertEqual(cn.strength_problems(cert), [])
                self.assertEqual(cert["digest"], cn.self_digest(cert))
                ceilings = cn.TOLERANCE_CEILINGS[cert["contract"]["family"]]
                if (cert["tolerances"]["site_rel"] == ceilings["site_rel"]
                        and cert["tolerances"]["loss_rel"]
                        == ceilings["loss_rel"]
                        and cert["evidence"]["steps"]
                        == cn.MIN_EVIDENCE_STEPS):
                    at_ceiling.append(name)
        # SmolLM-135M, 2026-10-15 and 2026-10-23: site 1e-4, loss 1e-2,
        # two steps — the evidence the ceilings and the floor are read off.
        self.assertEqual(at_ceiling, [
            "f2-2026-10-15/pkg_relaxed/numerics_certificate.json",
            "f2-2026-10-23/numerics_certificate.json"])

    def test_the_attestation_is_one_exact_line(self):
        # F2 #206: what the package's build.sh checks in place of the JSON.
        cert_path = os.path.join(self._tmp.name, cn.CERT_FILE)
        self.assertEqual(cn.attestation_path(cert_path),
                         os.path.join(self._tmp.name, cn.ATTESTATION_FILE))
        self.assertEqual(cn.attestation_path("/x/other.cert"),
                         "/x/other.cert.attest")
        with open(cert_path, "w") as f:
            json.dump(granted(self.plan), f)
        with open(self.plan, "rb") as f:
            raw = f.read()
        at = fx.formats.PLAN_HEADER.offsets["plan_hash"]
        seal = f"{int.from_bytes(raw[at:at + 8], 'little'):016x}"
        self.assertEqual(cn.plan_seal(self.plan), seal)
        line = cn.attestation_line(cert_path, self.plan)
        self.assertEqual(
            line, "seeml-numerics-certificate v1 granted "
                  f"plan-sha256={cn.sha256_file(self.plan)} "
                  f"plan-seal={seal} "
                  f"certificate-sha256={cn.sha256_file(cert_path)}")
        self.assertRegex(line, r"\Aseeml-numerics-certificate v1 granted "
                               r"plan-sha256=[0-9a-f]{64} "
                               r"plan-seal=[0-9a-f]{16} "
                               r"certificate-sha256=[0-9a-f]{64}\Z")
        path = cn.write_attestation(cert_path, self.plan)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), line.encode("ascii") + b"\n")
        # The line names the certificate FILE: any other bytes there —
        # a verdict flipped, a key added — are another line.
        with open(cert_path, "a") as f:
            f.write(" ")
        self.assertNotEqual(cn.attestation_line(cert_path, self.plan), line)
        cn.remove_attestation(cert_path)
        self.assertFalse(os.path.exists(path))
        cn.remove_attestation(cert_path)  # absent is fine
        short = os.path.join(self._tmp.name, "short.seeu")
        with open(short, "wb") as f:
            f.write(raw[:at + 4])
        with self.assertRaisesRegex(fx.PlanError, "too short to carry"):
            cn.plan_seal(short)

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
        # The full check passed, so the packer settles the attestation the
        # package's build.sh trusts in its place (F2 #206) ...
        attest = os.path.join(pkg, cn.ATTESTATION_FILE)
        self.assertEqual(summary["attestation"], attest)
        self.assertEqual(summary["certified_plan_seal"],
                         cn.plan_seal(self.plan))
        with open(attest) as f:
            self.assertEqual(f.read(),
                             cn.attestation_line(path, self.plan) + "\n")
        with open(self.plan, "ab") as f:
            f.write(b"\0")
        with self.assertRaisesRegex(pack_update.PackError,
                                    "does not vouch for this plan"):
            pack_update.check_certificate(pkg, self.plan)
        # ... and takes it away again when the check fails.
        self.assertFalse(os.path.exists(attest))


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

    def test_both_arms_hold_the_staged_slot_rule(self):
        # CR2 #143 N03: the two executors run loops of their own, and each
        # names the program and opcode of a write into the staged input
        # slot as the oracle's executor does (assemble(64) stages [64, 68);
        # this reduction writes [60, 68)).
        plan = fx.Plan(fixtures.assemble(
            64, train=[(10, 0, (0, 60, N, N), (1, 2, 0))]))
        for ex in (cn.LockstepExecutor(plan, 8), cn.RelaxedExecutor(plan, 8)):
            with self.subTest(executor=type(ex).__name__):
                with self.assertRaisesRegex(
                        fx.PlanError, r"an instruction of the train program "
                        r"writes the staged input slot \(opcode 10\)"):
                    ex.execute("train")


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
        self.assertIn("relaxed instruction", log)
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
        # A refusal is written, and has no attestation (F2 #206): the
        # package's build.sh, which trusts that one line, cannot build
        # beside it — whatever the document says inside.
        cert_path = os.path.join(pkg, cn.CERT_FILE)
        attest_path = os.path.join(pkg, cn.ATTESTATION_FILE)
        self.assertTrue(os.path.exists(cert_path))
        self.assertFalse(os.path.exists(attest_path))
        done = build_sh()
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("no numerics_certificate.attest", done.stderr)
        # The issue's second hole: tolerances of 1e30 over one step were
        # "granted". One step is a bad command line; a tolerance above its
        # ceiling is refused before anything is measured, no certificate
        # is written, and the refusal already there is left as it was.
        with open(cert_path, "rb") as f:
            refusal = f.read()
        loose = ["--max-site-error", "1e30", "--max-relaxed-gemm-error",
                 "1e30", "--max-loss-deviation", "1e30"]
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "1", *loose])
        self.assertEqual(status, 2, log)
        self.assertIn("--steps must be at least 2", log)
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", *loose])
        self.assertEqual(status, 3, log)
        for key, ceiling in (("site_rel", "1.0e-04"),
                             ("relaxed_gemm_rel", "7.8e-03"),
                             ("loss_rel", "1.0e-02")):
            self.assertIn(f"REFUSED tolerance {key} 1.0e+30 is looser than "
                          f"its ceiling {ceiling}", log)
        self.assertIn("refused — no certificate written", log)
        self.assertNotIn("granted", log)
        with open(cert_path, "rb") as f:
            self.assertEqual(f.read(), refusal)
        self.assertFalse(os.path.exists(attest_path))
        # One above its ceiling is enough, whichever it is.
        for flag in ("--max-site-error=2e-4", "--max-loss-deviation=2e-2",
                     "--max-relaxed-gemm-error=1e-2"):
            status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                                   "--steps", "4", flag])
            self.assertEqual(status, 3, log)
            self.assertIn("is looser than its ceiling", log)
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
        # Granted — at the loss ceiling, 1e-2 — and attested: the one line
        # for exactly this plan and exactly this certificate file.
        self.assertIn(f"attested — wrote {attest_path}", log)
        with open(attest_path) as f:
            attested = f.read()
        self.assertEqual(attested, cn.attestation_line(
            cert_path, os.path.join(pkg, cn.PLAN_FILE)) + "\n")
        self.assertIn(f"plan-seal={plan.plan_hash:016x} ", attested)
        self.assertIn(f"plan-sha256={cert['plan']['sha256']} ", attested)
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
        self.assertEqual(packed["numerics_certificate"]["attestation"],
                         attest_path)
        self.assertEqual(
            packed["numerics_certificate"]["certified_plan_seal"],
            f"{plan.plan_hash:016x}")
        with open(attest_path) as f:
            self.assertEqual(f.read(), attested)  # the packer's is the same
        # Packed beside its certificate, the script gets past the gate —
        # and compiles the plan's seal into the runner, on the one unit
        # that loads the plan (F2 #206). CXX is a logger here.
        cxx = os.path.join(self.dir, "cxx.sh")
        cxx_log = os.path.join(self.dir, "cxx.log")
        with open(cxx, "w") as f:
            f.write(f'#!/bin/sh\necho "$*" >> "{cxx_log}"\n')
        os.chmod(cxx, 0o755)
        done = subprocess.run(["sh", "build.sh"], cwd=pkg,
                              env=dict(os.environ, CXX=cxx),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        with open(cxx_log) as f:
            compiles = f.read().splitlines()
        define = f"-DSEEML_CERTIFIED_PLAN_HASH=0x{plan.plan_hash:016x}ull"
        self.assertEqual([define in line for line in compiles],
                         ["-c update_main.cc " in line for line in compiles])
        self.assertEqual(sum(define in line for line in compiles), 1)
        # A refused certificate, a document that nests a granted one, or
        # one for other plan bytes stops it again — the JSON is another
        # file than the one attested, whatever it says. (Before F2 #206
        # the gate was two greps: the nested document below built.)
        self.assertEqual(build_sh().returncode, 0)
        with open(cert_path) as f:
            text = f.read()
        nested = json.dumps({
            "verdict": "refused", "plan": {"sha256": cert["plan"]["sha256"]},
            "evidence": {"earlier": json.loads(text)}}, indent=2)
        self.assertIn('"verdict": "granted"', nested)
        self.assertIn(f'"sha256": "{cert["plan"]["sha256"]}"', nested)
        for other in (text.replace('"verdict": "granted"',
                                   '"verdict": "refused"'),
                      nested,
                      text.replace(cert["plan"]["sha256"], "0" * 64)):
            with open(cert_path, "w") as f:
                f.write(other)
            done = build_sh()
            self.assertEqual(done.returncode, 1, done.stderr)
            self.assertIn("does not attest", done.stderr)
        with open(cert_path, "w") as f:
            f.write(text)
        self.assertEqual(build_sh().returncode, 0)
        # The attestation for another plan, or a line that is not the
        # whole file.
        for other in (attested.replace(f"{plan.plan_hash:016x}", "0" * 16),
                      attested.replace(cert["plan"]["sha256"], "0" * 64),
                      "seeml-numerics-certificate v1 refused\n" + attested,
                      attested + attested, ""):
            with open(attest_path, "w") as f:
                f.write(other)
            done = build_sh()
            self.assertEqual(done.returncode, 1, done.stderr)
            self.assertIn("does not attest", done.stderr)
        with open(attest_path, "w") as f:
            f.write(attested)
        self.assertEqual(build_sh().returncode, 0)
        # The issue's certificate, forged as the old certifier wrote it
        # (1e30 tolerances, a fresh digest): verify refuses it, the packer
        # refuses it — it printed NOTE lines before — and takes the
        # attestation away, so build.sh refuses it too.
        forged = json.loads(text)
        forged["tolerances"] = {key: 1e30 for key in forged["tolerances"]}
        forged["evidence"]["steps"] = 1
        forged["digest"] = cn.self_digest(forged)
        fx.write_report(cert_path, forged)
        status, log = run(cn, ["verify", cert_path, "--plan", pkg])
        self.assertEqual(status, 3, log)
        self.assertEqual(log.count("is looser than its ceiling"), 3)
        self.assertIn("below the floor of 2", log)
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1, log)
        self.assertIn("does not vouch for this plan", log)
        self.assertIn("tolerance loss_rel 1.0e+30 is looser than its "
                      "ceiling 1.0e-02", log)
        self.assertNotIn("NOTE", log)
        self.assertFalse(os.path.exists(attest_path))
        done = build_sh()
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("no numerics_certificate.attest", done.stderr)
        # The real certificate again: the packer re-attests it.
        with open(cert_path, "w") as f:
            f.write(text)
        self.assertEqual(run(pack_update, [pkg])[0], 0)
        with open(attest_path) as f:
            self.assertEqual(f.read(), attested)
        self.assertEqual(build_sh().returncode, 0)
        # A relaxed plan beside a build.sh without the device-side gate —
        # an exact plan's script, or one from an emitter before F2 #206 —
        # is not packed: its runner would carry no certified seal.
        script_path = os.path.join(pkg, "build.sh")
        with open(script_path) as f:
            script = f.read()
        self.assertIn(pack_update.CERTIFIED_HASH_DEFINE, script)
        with open(script_path, "w") as f:
            f.write(script.replace(" $CERT_FLAGS", "").replace(
                'CERT_FLAGS="' + pack_update.CERTIFIED_HASH_DEFINE + '"\n',
                ""))
        status, log = run(pack_update, [pkg])
        self.assertEqual(status, 1, log)
        self.assertIn("does not build the device-side certificate gate", log)
        with open(script_path, "w") as f:
            f.write(script)
        self.assertEqual(run(pack_update, [pkg])[0], 0)
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
        self.assertIn(f"{plan.relaxed_gemms} relaxed instruction(s) on the exact "
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

    def test_a_relaxed_masked_tiled_plan_certifies_the_masked_forward(self):
        # Plan v26 (#191 step 2): with the segment mask the tiled forward is
        # kAttnFwdTiledSeg, relaxed as the plain one is — and its own site,
        # attn.fwd.tiled.seg: the certifier prices it, the contract names it,
        # and a certificate that priced the unmasked forward instead does
        # not vouch for the plan.
        corpus = os.path.join(self.dir, "segdec", "corpus.sds")
        pkg = self.compile("dec_relaxed_tiled_seg",
                           os.path.join("segdec", "decoder.smf"), 12,
                           "--precision", "certified-bf16",
                           "--attention", "tiled", "--segment-mask", "on",
                           "--lora-rank", "4")
        plan = fx.Plan.load(os.path.join(pkg, cn.PLAN_FILE))
        self.assertGreater(plan.relaxed_instructions, plan.relaxed_gemms)
        attention = ["attn.dk.tiled", "attn.dq.tiled", "attn.dv.tiled",
                     "attn.fwd.tiled.seg"]
        self.assertEqual(cn.flagged_sites(plan),
                         attention + [cn.RELAXED_GEMM_SITE])
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", "--val-fraction", "0.25",
                               "--max-loss-deviation", "1e-2"])
        self.assertEqual(status, 0, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["contract"]["sites"],
                         attention + [cn.RELAXED_GEMM_SITE])
        site = cert["evidence"]["sites"]["attn.fwd.tiled.seg"]
        self.assertGreater(site["instances"], 0)
        self.assertLessEqual(site["observed_rel"], site["bound_rel"])
        self.assertEqual(cn.check_certificate(
            cert, os.path.join(pkg, cn.PLAN_FILE)), [])
        # A certificate naming the unmasked forward's site in its place.
        swapped = json.loads(json.dumps(cert))
        swapped["contract"]["sites"] = [
            "attn.fwd.tiled" if s == "attn.fwd.tiled.seg" else s
            for s in cert["contract"]["sites"]]
        self.assertTrue(any("attn.fwd.tiled.seg" in problem
                            for problem in cn.check_certificate(
                                swapped, os.path.join(pkg, cn.PLAN_FILE))))

    def test_a_relaxed_tiled_attention_plan_certifies_its_attention_sites(self):
        # Plan v20 (F3 #131): certified precision with tiled attention marks
        # the four tiled opcodes of training kFlagRelaxed beside the frozen
        # GEMMs. The flagged contract then enforces exactly those sites —
        # gemm.relaxed and each attn.*.tiled, which the certifier prices
        # for any summation order — and a certificate that left one of them
        # out does not vouch for the plan.
        corpus = os.path.join(self.dir, "decoder_corpus.sds")
        pkg = self.compile("dec_relaxed_tiled", "decoder.smf", 12,
                           "--precision", "certified-bf16",
                           "--attention", "tiled")
        plan = fx.Plan.load(os.path.join(pkg, cn.PLAN_FILE))
        # A plan without 8-bit moments is written at v26 (#192).
        self.assertEqual(plan.version,
                         fx.formats.SEEU_INT8_MOMENTS_VERSION - 1)
        self.assertGreater(plan.relaxed_instructions, plan.relaxed_gemms)
        attention = ["attn.dk.tiled", "attn.dq.tiled", "attn.dv.tiled",
                     "attn.fwd.tiled"]
        self.assertEqual(cn.flagged_sites(plan),
                         attention + [cn.RELAXED_GEMM_SITE])
        status, log = run(cn, ["certify", pkg, "--corpus", corpus,
                               "--steps", "4", "--val-fraction", "0.25",
                               "--max-loss-deviation", "1e-2"])
        self.assertEqual(status, 0, log)
        with open(os.path.join(pkg, cn.CERT_FILE)) as f:
            cert = json.load(f)
        self.assertEqual(cert["contract"]["family"], cn.FLAGGED_FAMILY)
        self.assertEqual(cert["contract"]["sites"],
                         attention + [cn.RELAXED_GEMM_SITE])
        for name in attention:
            site = cert["evidence"]["sites"][name]
            self.assertGreater(site["instances"], 0)
            self.assertLessEqual(site["observed_rel"], site["bound_rel"])
        self.assertEqual(cn.check_certificate(
            cert, os.path.join(pkg, cn.PLAN_FILE)), [])
        short = dict(cert, contract=dict(cert["contract"], sites=[
            s for s in cert["contract"]["sites"] if s != "attn.fwd.tiled"]))
        short["digest"] = cn.self_digest(short)
        problems = cn.check_certificate(short, os.path.join(pkg, cn.PLAN_FILE))
        self.assertTrue(any("attn.fwd.tiled site the certificate does not "
                            "cover" in p for p in problems), problems)
        # The packer accepts the plan beside its certificate.
        self.assertEqual(run(pack_update, [pkg])[0], 0)

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


    def test_the_8bit_moment_steps_fused_clip_is_a_certified_site(self):
        # #192 review: opcode 58 carries the fused clip as 18 does, so its
        # gradient norm is a site the certificate measures. It was missing
        # from FUSED_CLIP_SITES, and a clipped --moments int8 plan was
        # granted a certificate that never priced that reduction.
        for flags, site in (((), "adamw.step"),
                            (("--moments", "int8"), "adamw8.step")):
            with self.subTest(site=site):
                pkg = self.compile("clip8", "mlp.smf", 8, "--clip-norm",
                                   "0.5", *flags)
                status, log = run(cn, ["certify", pkg, "--corpus",
                                       os.path.join(self.dir, "class.sds"),
                                       "--steps", "4"])
                self.assertEqual(status, 0, log)
                with open(os.path.join(pkg, cn.CERT_FILE)) as f:
                    cert = json.load(f)
                self.assertIn(site, cert["contract"]["sites"])
                self.assertGreater(
                    cert["evidence"]["sites"][site]["instances"], 0)

if __name__ == "__main__":
    unittest.main()
