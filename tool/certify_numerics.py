#!/usr/bin/env python3
"""Certify a plan for relaxed reductions, or refuse to (P3, #77).

The runtime's bitwise-determinism contract makes every reduction accumulate
in float64, in a fixed chunk order — half the SIMD width and a convert per
element on every norm, loss, softmax and attention score. A relaxed-reduction
opcode family (vectorized float32 accumulation, reassociated within one
backend's fixed schedule) would buy that back, at the price of cross-schedule
bit-equality. That price is spent consciously or not at all:

    no certificate, no relaxed plan.

This tool writes the certificate. It runs the plan twice over ITS OWN
operands — the frozen weights in the plan, batches from the real corpus,
the adapters as they actually train — through tool/frontier_exec.py's
interpreter:

  reference  every reduction exact (float64), the contract of today;
  relaxed    the same interpreter with every reduction inside a reduction
             site recomputed in float32 over W independent lanes (strided
             sequential accumulators, combined by a fixed pairwise tree —
             the shape of a SIMD reduction), everything else identical.

and records three kinds of evidence:

  lockstep   per site opcode, on identical inputs: the worst observed
             output error, relative to the output tensor's scale;
  a priori   per reduction, the rounding-error bound that holds for ANY
             summation order of those operands (Higham, Accuracy and
             Stability of Numerical Algorithms, s.3.1 and s.4.2:
             |fl(sum) - sum| <= gamma_n * sum|x_i|, gamma_n = nu/(1 - nu)),
             so the certificate does not depend on the modelled schedule
             being the one a kernel author eventually picks;
  free run   both arms trained for --steps from the same initial state on
             the same batches: the loss-curve deviation, both validation
             losses, and whether the update gate (validation loss must
             improve) holds under either contract.

The certificate is GRANTED only if every enforced site's observed error is
within --max-site-error (the relaxed GEMM site: --max-relaxed-gemm-error),
the free-run deviations are within --max-loss-deviation, and the gate
verdict is the same in both arms. Which sites are enforced, what the
relaxed arm relaxes and the tolerance defaults follow the plan (F2 #130):
an exact plan states P3's contract (relaxed-reductions-f32: every
reduction site, all relaxed in the relaxed arm, loss 1e-4); a plan that
marks instructions kFlagRelaxed states the flagged contract
(relaxed-flagged-instructions: exactly those instructions, relaxed alone,
loss 1e-3 — F2's acceptance line), with every other site measured as
evidence and not enforced, since every backend runs it exact. It binds to the plan by
SHA-256 and by the plan's own sealed hash, and to the corpus by SHA-256; it
carries a digest of itself; given --source, it binds to the source model
by SHA-256 too. `verify` re-checks all of that against a plan, and,
given the measured deviations of a real relaxed run (--observed, a
frontier_exec.py diff report of the same plan bytes), that they are
inside the certified tolerances. tool/pack_update.py refuses to package
a plan beside a certificate that does not verify against it, and the
emitted build.sh of a relaxed plan refuses to build without one.

Usage:
    certify_numerics.py certify PLAN_OR_PACKAGE --corpus c.sds
        [--steps N] [--lanes W] [--seed S] [--val-fraction F]
        [--max-site-error F] [--max-loss-deviation F] [--out cert.json]
    certify_numerics.py verify cert.json --plan plan.seeu
        [--corpus c.sds] [--source model.smf] [--observed diff.json]

Exit codes: 0 granted / verified; 3 refused / not verified; 1 unusable
input; 2 bad arguments. Needs NumPy for `certify`; `verify` is standard
library only.
"""
import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_exec as fx  # noqa: E402

CERT_SCHEMA = 1
CERT_KIND = "seeml-numerics-certificate"
CERT_FILE = "numerics_certificate.json"
PLAN_FILE = "update_plan.seeu"
# The two contracts a certificate can state (F2 #130's decision,
# 2026-10-30). A certificate certifies the relaxations its plan takes:
#
#   FAMILY          an exact plan (no kFlagRelaxed): P3's contract — every
#                   reduction site of the plan, recomputed as a relaxed-
#                   reduction backend would run it, all of them enforced.
#   FLAGGED_FAMILY  a plan that marks instructions kFlagRelaxed (v18): the
#                   contract is exactly those instructions, as their
#                   certified kernels compute them. The free run's relaxed
#                   arm relaxes them and nothing else, so its deviation is
#                   the shipped arithmetic's; every other site is still
#                   measured, as evidence, but no backend runs it relaxed,
#                   so it is not enforced (the 2026-10-24 SmolLM-135M
#                   package was refused on the softmax cross-entropy site,
#                   a reduction every backend runs exact).
FAMILY = "relaxed-reductions-f32"
FLAGGED_FAMILY = "relaxed-flagged-instructions"
FAMILIES = (FAMILY, FLAGGED_FAMILY)
UNIT_ROUNDOFF = 2.0 ** -24  # float32, round to nearest
# `certify`'s default tolerances per contract, one table so
# tool/pack_update.py can say when a packaged certificate was granted under
# looser ones. The flagged family's loss tolerance is F2's own acceptance
# line (validation loss within 1e-3 relative of the exact run); 1e-4 is the
# reduction family's, which one bf16 rounding of a GEMM's input
# (2^-9 per element) does not meet.
FAMILY_DEFAULTS = {
    FAMILY: {"site_rel": 1e-5, "relaxed_gemm_rel": 2.0 ** -7,
             "loss_rel": 1e-4},
    FLAGGED_FAMILY: {"site_rel": 1e-5, "relaxed_gemm_rel": 2.0 ** -7,
                     "loss_rel": 1e-3},
}
# The relaxed GEMM family (plan v18, kFlagRelaxed): the activation input of
# a frozen-weight GEMM rounded to bfloat16 once (8 significant bits) before
# an f32-accumulating multiply against the exact stored weight — what the
# Metal tensor-op kernels compute. One site name for every such
# instruction, whatever its opcode; its own tolerance, since one bf16
# rounding is 2^-9 relative per element where the reductions above are
# 2^-24 per term.
RELAXED_GEMM_SITE = fx.RELAXED_SITE
BF16_UNIT_ROUNDOFF = 2.0 ** -9  # bfloat16, round to nearest

# The opcodes whose C++ kernels reduce in float64 today — the only places a
# relaxed family would change arithmetic (runtime/executor: attention.cc,
# normalization.cc, loss.cc, optimizer.cc, elementwise.cc). The GEMMs are
# float32 accumulations already and are not sites.
SITES = {
    10: "reduce.rows", 11: "softmax_xent.fwd", 13: "mse.fwd",
    15: "kl_distill.fwd", 26: "layer_norm.fwd", 27: "layer_norm.bwd",
    28: "clip.norm", 31: "rms_norm.fwd", 32: "rms_norm.bwd",
    35: "attn.fwd", 36: "attn.dp", 38: "softmax_rows.bwd",
    # The tiled family (v15) recomputes the same double reductions.
    47: "attn.fwd.tiled", 48: "attn.dq.tiled", 49: "attn.dk.tiled",
    50: "attn.dv.tiled",
}
# The optimizer steps reduce only when they carry a fused clip (plan v12,
# threshold bits in out[1]): then the gradient norm is theirs.
FUSED_CLIP_SITES = {17: "sgd.step", 18: "adamw.step"}
SITES.update(FUSED_CLIP_SITES)
assert all(fx.OPCODES[k] == v for k, v in SITES.items())


def is_site(ins):
    if ins.flags & fx.FLAG_RELAXED:
        return True
    return ins.opcode in SITES and (ins.opcode not in FUSED_CLIP_SITES or
                                    ins.out[1] != 0)


def site_name(ins):
    """The one gemm.relaxed site for a relaxed GEMM (F72); every other site
    instruction — a relaxed tiled attention one (v20) included — under its
    opcode's reduction site."""
    return fx.site_of(ins) if ins.flags & fx.FLAG_RELAXED else \
        SITES[ins.opcode]


def is_flagged(ins):
    """An instruction the plan marks kFlagRelaxed (v18): one a certified
    relaxed kernel may run, and the only kind the flagged contract
    relaxes."""
    return bool(ins.flags & fx.FLAG_RELAXED)


def family_of(plan):
    """The contract a certificate of `plan` states: the flagged family when
    its train or step program marks any instruction kFlagRelaxed (a GEMM,
    v18, or a tiled attention instruction, v20), P3's reduction family
    otherwise."""
    return FLAGGED_FAMILY if plan.relaxed_instructions else FAMILY


def flagged_sites(plan):
    """The site names of the instructions `plan` marks kFlagRelaxed — the
    flagged contract's enforced sites."""
    return sorted({site_name(i) for name in ("train", "step")
                   for i in plan.sections[name] if is_flagged(i)})


def gamma(n):
    """Higham's gamma_n for float32; inf where the bound is vacuous."""
    nu = n * UNIT_ROUNDOFF
    return nu / (1.0 - nu) if nu < 1.0 else math.inf


class LaneBackend(fx.NumpyBackend):
    """float64 everywhere except reductions, which run in float32 over
    `lanes` strided accumulators. Tracks, per reduction, the a-priori bound
    relative to the result's scale (worst over the current instruction)."""

    def __init__(self, lanes):
        super().__init__("float64")
        if lanes < 1 or lanes & (lanes - 1):
            raise fx.PlanError("--lanes must be a power of two")
        self.lanes, self.name = lanes, f"relaxed-f32x{lanes}"
        self.bound_rel = 0.0
        self._input_roundoff = 0.0  # set by relax_input for the next matmul

    relaxed_gemms = True  # the relaxed arm models the v18 GEMM family

    def relax_input(self, a):
        self._input_roundoff = BF16_UNIT_ROUNDOFF
        return fx.round_bf16(self.np, a)

    def _note(self, bound, result):
        scale = float(self.np.abs(result).max()) if result.size else 0.0
        if scale > 0.0:
            self.bound_rel = max(self.bound_rel,
                                 float(self.np.max(bound)) / scale)

    def _lane_sum(self, terms, axis):
        """sum over `axis` of float32 `terms`: `lanes` sequential float32
        accumulators over strided elements, then a pairwise tree."""
        np = self.np
        v = np.moveaxis(terms, axis, -1)
        n = v.shape[-1]
        pad = -n % self.lanes
        if pad:
            v = np.concatenate(
                [v, np.zeros(v.shape[:-1] + (pad,), np.float32)], -1)
        v = v.reshape(v.shape[:-1] + (-1, self.lanes))
        acc = np.cumsum(v, axis=-2, dtype=np.float32)[..., -1, :]
        while acc.shape[-1] > 1:
            acc = acc[..., 0::2] + acc[..., 1::2]
        return acc[..., 0]

    def sum(self, t, axis, keepdims=False):
        np = self.np
        terms = np.asarray(t).astype(np.float32)
        out = self._lane_sum(terms, axis)
        n = terms.shape[axis]
        self._note(gamma(n) * np.abs(terms.astype(np.float64)).sum(axis), out)
        out = out.astype(np.float64)
        return np.expand_dims(out, axis) if keepdims else out

    def mean(self, t, axis, keepdims=False):
        n = self.np.asarray(t).shape[axis]
        return self.sum(t, axis, keepdims) / n

    def matmul(self, a, b):
        """a @ b as float32 dot products, each lane-accumulated: the
        products are rounded once (hence gamma_{K+1}), then summed. The
        lanes are walked one k at a time — the same f32 additions in the
        same order as `_lane_sum` over the materialized products, without
        the [M, N, K] tensor that order would need (58 GB at a 49,152-wide
        LM head)."""
        np = self.np
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)
        k = a32.shape[-1]
        acc = np.zeros(a32.shape[:-1] + (b32.shape[-1], self.lanes),
                       np.float32)
        for i in range(k):
            acc[..., i % self.lanes] += a32[..., :, i, None] * b32[..., i, None, :]
        out = acc
        while out.shape[-1] > 1:
            out = out[..., 0::2] + out[..., 1::2]
        out = out[..., 0]
        b32 = np.swapaxes(b32, -1, -2)
        # A bf16-rounded input (v18) adds its unit roundoff to every term,
        # on top of the summation's gamma; the term is over the rounded
        # operand, which is within (1 + u) of the exact one.
        u_in, self._input_roundoff = self._input_roundoff, 0.0
        factor = gamma(k + 1) + u_in * (1.0 + gamma(k + 1))
        self._note(factor * (np.abs(a32.astype(np.float64)) @
                             np.abs(np.swapaxes(b32, -1, -2).astype(
                                 np.float64))), out)
        return out.astype(np.float64)


class DryMemory:
    """Reads through to a real Memory; captures writes instead of storing
    them — one instruction's relaxed result on the reference arm's inputs."""

    def __init__(self, mem):
        self.mem, self.writes = mem, []

    def read(self, ref, shape, kind="<f4"):
        return self.mem.read(ref, shape, kind)

    def frozen(self, *args, **kwargs):
        return self.mem.frozen(*args, **kwargs)

    def write(self, ref, tensor):
        np = self.mem.np
        self.writes.append((ref, np.asarray(tensor, np.float64).astype(
            np.float32).reshape(-1)))


class RelaxedExecutor(fx.Executor):
    """The relaxed arm: the instructions the contract relaxes through the
    lane backend — every site under P3's family, only the flagged ones
    under the flagged family (`relax`)."""

    def __init__(self, plan, lanes, source=None, relax=is_site):
        super().__init__(plan, fx.NumpyBackend("float64"), source)
        self.lane = LaneBackend(lanes)
        self.relax = relax

    def execute(self, section, scalars=None, on_instruction=None):
        scalars = scalars or self.step_scalars()
        for index, ins in enumerate(self.plan.sections[section]):
            x = self.lane if self.relax(ins) else self.x
            fx.INTERPRETER[ins.opcode](self.mem, x, ins, scalars)
            if on_instruction:
                on_instruction(index, ins)


class LockstepExecutor(fx.Executor):
    """The reference arm, measuring each site instruction's relaxed result
    against its exact one on the same inputs."""

    def __init__(self, plan, lanes, source=None):
        super().__init__(plan, fx.NumpyBackend("float64"), source)
        self.lane, self.sites = LaneBackend(lanes), {}

    def execute(self, section, scalars=None, on_instruction=None):
        np = self.x.np
        scalars = scalars or self.step_scalars()
        for ins in self.plan.sections[section]:
            op = fx.INTERPRETER[ins.opcode]
            if not is_site(ins):
                op(self.mem, self.x, ins, scalars)
                continue
            dry = DryMemory(self.mem)
            self.lane.bound_rel = 0.0
            op(dry, self.lane, ins, scalars)
            exact = []
            self.mem.recorder = lambda off, values: exact.append((off, values))
            op(self.mem, self.x, ins, scalars)
            self.mem.recorder = None
            stat = self.sites.setdefault(site_name(ins), {
                "instances": 0, "observed_rel": 0.0, "bound_rel": 0.0,
                "by_opcode": {}})
            stat["instances"] += 1
            stat["bound_rel"] = max(stat["bound_rel"], self.lane.bound_rel)
            # A site spans opcodes (the relaxed GEMM site covers every
            # kGemm* form): the certificate carries the worst relative
            # error per opcode as well as per site.
            opcode = fx.OPCODES[ins.opcode]
            by_opcode = stat["by_opcode"]
            by_opcode.setdefault(opcode, 0.0)
            for (_, got), (_, want) in zip(dry.writes, exact):
                scale = float(np.abs(want).max()) if want.size else 0.0
                if scale == 0.0:
                    continue
                worst = float(np.abs(got.astype(np.float64) -
                                     want.astype(np.float64)).max()) / scale
                stat["observed_rel"] = max(stat["observed_rel"], worst)
                by_opcode[opcode] = max(by_opcode[opcode], worst)


sha256_file = fx.sha256_file


def self_digest(cert):
    body = {k: v for k, v in cert.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def resolve_plan(path):
    return os.path.join(path, PLAN_FILE) if os.path.isdir(path) else path


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def cmd_certify(args):
    np = fx.require_numpy()
    plan_path = resolve_plan(args.target)
    plan = fx.Plan.load(plan_path)
    # The source model the eval program reads (v17): the certificate
    # binds to it by SHA-256 so `verify --source` can refuse another
    # file (review N35). Whether the file IS the model the plan was
    # compiled from is the plan header's ContentHash64, which the C++
    # probe checks on every eval section (a Python port would be a
    # second copy of the identity contract).
    source = ({"file": os.path.basename(args.source),
               "sha256": sha256_file(args.source)}
              if args.source else None)

    def feed():
        corpus = fx.Corpus(np, args.corpus)
        corpus.check(plan)
        val = None
        if args.val_fraction > 0.0:
            corpus, val = corpus.split_validation(args.val_fraction)
        corpus.enable_shuffle(args.seed)
        return corpus, val

    # The contract (F2 #130): what this plan relaxes decides which sites
    # are enforced, what the relaxed arm relaxes, and the defaults of any
    # tolerance not given on the command line.
    family = family_of(plan)
    defaults = FAMILY_DEFAULTS[family]
    site_tol = (defaults["site_rel"] if args.max_site_error is None
                else args.max_site_error)
    gemm_tol = (defaults["relaxed_gemm_rel"]
                if args.max_relaxed_gemm_error is None
                else args.max_relaxed_gemm_error)
    loss_tol = (defaults["loss_rel"] if args.max_loss_deviation is None
                else args.max_loss_deviation)
    relax = is_flagged if family == FLAGGED_FAMILY else is_site

    # Lockstep evidence rides the reference arm of the free run: the
    # operands every site sees are the ones real training produces.
    arms = {"reference": LockstepExecutor(plan, args.lanes, args.source),
            "relaxed": RelaxedExecutor(plan, args.lanes, args.source,
                                       relax)}
    runs = {}
    for name, ex in arms.items():
        corpus, val = feed()
        can_eval = val is not None and bool(plan.sections["eval"])
        run = {"loss_curve": []}
        if can_eval:
            run["val_initial"] = ex.evaluate(val)
        for _ in range(args.steps):
            run["loss_curve"].append(ex.train_step(corpus))
        if can_eval:
            run["val_final"] = ex.evaluate(val)
            run["gate_improved"] = run["val_final"] < run["val_initial"]
        else:  # the engine's fallback: the training-loss trend
            run["gate_improved"] = run["loss_curve"][-1] < run["loss_curve"][0]
        runs[name] = run

    ref, rlx = runs["reference"], runs["relaxed"]
    finite = all(math.isfinite(v) for r in runs.values()
                 for v in r["loss_curve"])
    free_run = {
        "loss_max_rel_dev": max(rel(a, b) for a, b in zip(
            rlx["loss_curve"], ref["loss_curve"])) if finite else math.inf,
        "loss_final_reference": ref["loss_curve"][-1],
        "loss_final_relaxed": rlx["loss_curve"][-1],
        "gate_improved_reference": ref["gate_improved"],
        "gate_improved_relaxed": rlx["gate_improved"],
    }
    if "val_final" in ref:
        free_run.update(val_initial=ref["val_initial"],
                        val_final_reference=ref["val_final"],
                        val_final_relaxed=rlx["val_final"],
                        val_rel_dev=rel(rlx["val_final"], ref["val_final"]))

    sites = arms["reference"].sites
    enforced = (sorted(sites) if family == FAMILY
                else [s for s in flagged_sites(plan) if s in sites])
    reasons = []
    if not sites:
        reasons.append("the plan has no reduction site to relax")
    elif family == FLAGGED_FAMILY and len(enforced) != len(
            flagged_sites(plan)):
        reasons.append("a relaxed instruction never ran in the free run, so "
                       "its site was not measured: raise --steps")
    for name in enforced:
        stat = sites[name]
        tolerance = gemm_tol if name == RELAXED_GEMM_SITE else site_tol
        if not stat["observed_rel"] <= tolerance:
            reasons.append(f"{name}: observed relative error "
                           f"{stat['observed_rel']:.3e} exceeds "
                           f"{tolerance:.1e}")
    for key in ("loss_max_rel_dev", "val_rel_dev"):
        if key in free_run and not free_run[key] <= loss_tol:
            reasons.append(f"{key} {free_run[key]:.3e} exceeds "
                           f"{loss_tol:.1e}")
    if ref["gate_improved"] != rlx["gate_improved"]:
        reasons.append("the update gate decides differently under the two "
                       "contracts")

    cert = {
        "schema": CERT_SCHEMA, "kind": CERT_KIND,
        "seeml_tool": "certify_numerics",
        "plan": {"file": os.path.basename(plan_path),
                 "sha256": sha256_file(plan_path),
                 "plan_hash": f"{plan.plan_hash:016x}",
                 "version": plan.version,
                 "relaxed_gemms": plan.relaxed_gemms,
                 "source_model_hash": f"{plan.source_model_hash:016x}"},
        "corpus": {"file": os.path.basename(args.corpus),
                   "sha256": sha256_file(args.corpus)},
        "source": source,
        # `sites` are the enforced ones; evidence.sites is every site
        # measured (under the flagged family, the reductions no backend
        # relaxes appear there only).
        "contract": {"family": family, "model_lanes": args.lanes,
                     "unit_roundoff": UNIT_ROUNDOFF,
                     "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
                     "sites": enforced},
        "evidence": {"steps": args.steps, "seed": args.seed,
                     "val_fraction": args.val_fraction,
                     "sites": {k: sites[k] for k in sorted(sites)},
                     "free_run": free_run},
        "tolerances": {"site_rel": site_tol,
                       "relaxed_gemm_rel": gemm_tol,
                       "loss_rel": loss_tol},
        "verdict": "refused" if reasons else "granted",
        "reasons": reasons,
    }
    cert["digest"] = self_digest(cert)

    print(f"certify_numerics: contract {family} over "
          f"{len(enforced)} enforced site(s)")
    for name, stat in cert["evidence"]["sites"].items():
        print(f"certify_numerics: {name:<18} {stat['instances']:>5} "
              f"instance(s)  observed {stat['observed_rel']:.2e}  "
              f"any-order bound {stat['bound_rel']:.2e}"
              + ("" if name in enforced else
                 "  (evidence only: no instruction runs it relaxed)"))
    print(f"certify_numerics: free run over {args.steps} step(s): loss "
          f"deviation {free_run['loss_max_rel_dev']:.2e}"
          + (f", validation {free_run['val_final_reference']:.6f} vs "
             f"{free_run['val_final_relaxed']:.6f} "
             f"({free_run['val_rel_dev']:.2e})" if "val_rel_dev" in free_run
             else "")
          + f"; gate improved: reference {ref['gate_improved']}, relaxed "
            f"{rlx['gate_improved']}")
    for reason in reasons:
        print(f"certify_numerics: REFUSED {reason}")
    out = args.out or os.path.join(os.path.dirname(plan_path) or ".",
                                   CERT_FILE)
    fx.write_report(out, cert)
    print(f"certify_numerics: {cert['verdict']} — wrote {out}")
    return 3 if reasons else 0


def check_certificate(cert, plan_path, corpus_path=None, source_path=None):
    """Every reason `cert` does not vouch for the plan at `plan_path`
    (empty = it does). Standard library only: tool/pack_update.py calls
    this on the build host's tier-0 Python."""
    problems = []
    if (not isinstance(cert, dict) or cert.get("kind") != CERT_KIND
            or cert.get("schema") != CERT_SCHEMA):
        return [f"not a schema-{CERT_SCHEMA} {CERT_KIND}"]
    try:
        if cert.get("digest") != self_digest(cert):
            problems.append("the certificate was edited after it was "
                            "written (digest mismatch)")
        if cert["plan"]["sha256"] != sha256_file(plan_path):
            problems.append("the certificate was issued for a different "
                            "plan (SHA-256 mismatch) — re-certify after "
                            "recompiling")
        if corpus_path and cert["corpus"]["sha256"] != sha256_file(
                corpus_path):
            problems.append("the certificate was issued over a different "
                            "corpus")
        if source_path:
            bound = cert.get("source")
            if not bound:
                problems.append("the certificate was issued without a "
                                "source model; re-certify with --source")
            elif bound["sha256"] != sha256_file(source_path):
                problems.append("the certificate was issued over a "
                                "different source model (SHA-256 "
                                "mismatch)")
        if cert["verdict"] != "granted":
            problems.append("the certificate is a refusal: " +
                            "; ".join(cert.get("reasons", [])))
        # A tolerance of zero (or NaN, or infinity) certifies nothing:
        # every site measures inside it or nothing does (review F41).
        for key, value in cert["tolerances"].items():
            if not (isinstance(value, (int, float)) and
                    math.isfinite(value) and value > 0.0):
                problems.append(f"tolerance {key} is not a positive "
                                f"finite number ({value!r})")
        if cert["contract"]["family"] not in FAMILIES:
            problems.append(f"unknown contract family "
                            f"{cert['contract']['family']!r}")
        # v18 / v20: a plan whose train / step programs carry relaxed
        # instructions is vouched for only by a certificate that priced
        # every one of their sites — gemm.relaxed for the GEMMs, each
        # attn.*.tiled site for the tiled attention family.
        with open(plan_path, "rb") as f:
            opcodes = fx.formats.relaxed_opcodes(f)
        needed = {}
        for op in opcodes:
            name = (RELAXED_GEMM_SITE
                    if op in fx.formats.RELAXED_GEMM_OPCODES
                    else fx.OPCODES.get(op, str(op)))
            needed[name] = needed.get(name, 0) + 1
        for name, count in sorted(needed.items()):
            if name not in cert["contract"]["sites"]:
                problems.append(f"the plan carries {count} relaxed "
                                f"instruction(s) at the {name} site the "
                                "certificate does not cover")
    except (KeyError, TypeError) as e:
        problems.append(f"malformed certificate ({e!r})")
    return problems


def tolerance_notes(cert):
    """Each certified tolerance looser than `certify`'s default for the
    certificate's contract family, as a sentence (empty: none is). Not a
    verification failure — the tolerances are the issuer's to set — but
    one the packager repeats where the binary is built, so the choice is
    read there rather than discovered when a row is measured (F2 #130)."""
    notes = []
    if not isinstance(cert, dict):
        return notes
    tolerances = cert.get("tolerances")
    contract = cert.get("contract")
    family = contract.get("family") if isinstance(contract, dict) else None
    for key, default in FAMILY_DEFAULTS.get(family,
                                            FAMILY_DEFAULTS[FAMILY]).items():
        value = (tolerances or {}).get(key)
        if isinstance(value, (int, float)) and value > default:
            notes.append(f"{key} {value:.1e} is looser than the certify "
                         f"default {default:.1e}")
    return notes


def check_observed(cert, observed):
    """A real run's measured deviations (a frontier_exec.py diff report)
    against the certified tolerances. The report must be of this plan,
    must line up with its reference (no structural mismatch) and agree
    with it outside the certified sites, must carry each opcode's relative
    error itself, and — when the certificate covers the relaxed GEMM site —
    must have measured the relaxed kernel: its C++ arm ran at least one
    relaxed GEMM on one and its reference arm ran exact (review F41, F42,
    F72)."""
    problems = []
    try:
        rtol = float(observed.get("rtol", 0.0))
        if not (math.isfinite(rtol) and rtol > 0.0):
            problems.append(f"the observed report's rtol is {rtol!r}: a "
                            "comparison at a non-positive tolerance measures "
                            "nothing")
        plan_sha, plan_hash = (observed.get("plan_sha256"),
                               observed.get("plan_hash"))
        if plan_sha is None and plan_hash is None:
            problems.append("the observed report does not name its plan (no "
                            "plan_sha256 / plan_hash): regenerate it with "
                            "the current frontier_exec.py diff")
        else:
            if plan_sha is not None and plan_sha != cert["plan"]["sha256"]:
                problems.append("the observed run was of a different plan "
                                "(SHA-256 mismatch)")
            if (plan_hash is not None and
                    plan_hash != cert["plan"]["plan_hash"]):
                problems.append("the observed run was of a different plan "
                                "(plan hash mismatch)")
        # Review F42, both halves. The certified site is measured only when
        # the C++ arm ran its relaxed GEMMs on a relaxed kernel — the probe
        # counts them per instruction, and the exact kernels measure f32
        # round-off, vouching for nothing (a Metal backend keeps skinny
        # shapes exact by design, so one is enough, not all) — AND the
        # reference arm ran exact: --relaxed models the bf16 rounding on
        # the reference too, so the two would agree on the model, not on
        # the kernel.
        # Every flagged-contract certificate covers relaxed instructions; an
        # older P3-family one did when it named the relaxed GEMM site.
        if (cert["contract"].get("family") == FLAGGED_FAMILY
                or RELAXED_GEMM_SITE in cert["contract"]["sites"]):
            if observed.get("relaxed_modelled", False):
                problems.append("the observed run modelled the relaxed "
                                "rounding on the reference side (--relaxed): "
                                "a certificate is verified against the exact "
                                "reference")
            executed = observed.get("cpp_relaxed_executed")
            if executed is None:
                problems.append("the observed report does not say whether "
                                "the C++ backend ran the relaxed kernels (no "
                                "cpp_relaxed_executed): regenerate it with "
                                "the current frontier_exec.py diff")
            elif not int(executed) > 0:
                problems.append(
                    "the observed run executed its "
                    f"{observed.get('relaxed_instructions', '?')} relaxed "
                    "instruction(s) on the exact kernels (C++ backend "
                    f"{observed.get('cpp_backend', '?')}): it measures "
                    "nothing about the certified relaxed sites; run the "
                    "diff on a backend that carries the relaxed kernels "
                    "(--cpp-backend metal, or a SEEML_ACCELERATE build of "
                    "seeml-plan-probe)")
        # A certified site may sit past the diff tolerance by design (the
        # relaxed GEMMs: 2^-7 against the diff's 1e-5), so `agrees`, which
        # judges every site at that tolerance, is not the test here. The
        # structural mismatches are (an older report without the list is
        # held to `agrees`); the per-site loop below holds every
        # uncertified site to the diff tolerance.
        if "mismatches" in observed:
            mismatches = observed["mismatches"]
            if mismatches:
                problems.append("the observed run does not line up with its "
                                f"reference ({len(mismatches)} structural "
                                f"mismatch(es)); the first: {mismatches[0]}")
        elif not observed.get("agrees", False):
            failures = observed.get("failures", [])
            problems.append("the observed run disagrees with its reference "
                            f"({len(failures)} failure(s)); the first: "
                            f"{failures[0] if failures else 'none listed'}")
        limits = cert["tolerances"]
        for name, stat in sorted(observed.get("opcodes", {}).items()):
            if name in cert["contract"]["sites"]:
                if "worst_rel" not in stat:
                    problems.append(f"{name}: the observed report carries no "
                                    "relative error (worst_rel); regenerate "
                                    "it with the current frontier_exec.py")
                    continue
                measured = float(stat["worst_rel"])
                limit = (limits.get("relaxed_gemm_rel", limits["site_rel"])
                         if name == RELAXED_GEMM_SITE else limits["site_rel"])
                if not measured <= limit:  # NaN fails
                    problems.append(f"{name}: measured relative error "
                                    f"{measured:.3e} exceeds the certified "
                                    f"{limit:.1e}")
            elif not float(stat.get("worst_ratio", "nan")) <= 1.0:
                problems.append(f"{name} is not a certified site and must "
                                "agree with the reference at the diff "
                                "tolerance")
    except (KeyError, TypeError, ValueError) as e:
        problems.append(f"malformed observed report ({e!r})")
    return problems


def cmd_verify(args):
    try:
        with open(args.certificate) as f:
            cert = json.load(f)
    except (OSError, ValueError) as e:
        raise fx.PlanError(f"cannot read {args.certificate}: {e}")
    problems = check_certificate(cert, resolve_plan(args.plan), args.corpus,
                                 args.source)
    if not problems and args.observed:
        try:
            with open(args.observed) as f:
                problems = check_observed(cert, json.load(f))
        except (OSError, ValueError) as e:
            raise fx.PlanError(f"cannot read {args.observed}: {e}")
    for problem in problems:
        print(f"certify_numerics: NOT VERIFIED {problem}")
    if not problems:
        print(f"certify_numerics: verified — {cert['contract']['family']} "
              f"over {len(cert['contract']['sites'])} site opcode(s), site "
              f"tolerance {cert['tolerances']['site_rel']:.1e}"
              + ("; bound to the source model" if args.source else "")
              + ("; the observed run is inside the certified "
                 "tolerances" if args.observed else ""))
    return 3 if problems else 0


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("certify")
    c.add_argument("target", help="a .seeu plan or an emitted package dir")
    c.add_argument("--corpus", required=True)
    c.add_argument("--steps", type=int, default=20)
    c.add_argument("--lanes", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--val-fraction", type=float, default=0.1)
    # The tolerance defaults depend on the plan's contract family
    # (FAMILY_DEFAULTS), so they are resolved once the plan is read.
    c.add_argument("--max-site-error", type=float, default=None,
                   help="tolerance for each enforced reduction site: "
                        "default 1e-5")
    c.add_argument("--max-relaxed-gemm-error", type=float, default=None,
                   help="tolerance for the v18 relaxed GEMM site (bf16 "
                        "input rounding): default 2^-7")
    c.add_argument("--source", metavar="SMF", default=None,
                   help="the source model (v17 plans whose eval program "
                        "reads it)")
    c.add_argument("--max-loss-deviation", type=float, default=None,
                   help="tolerance for the free run's loss and validation "
                        "deviation: default 1e-3 for a plan with relaxed "
                        "instructions, 1e-4 for an exact one")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_certify)
    v = sub.add_parser("verify")
    v.add_argument("certificate")
    v.add_argument("--plan", required=True)
    v.add_argument("--corpus", default=None)
    v.add_argument("--source", metavar="SMF", default=None,
                   help="the source model the certificate was issued "
                        "over (refused when it is another file)")
    v.add_argument("--observed", default=None)
    v.set_defaults(fn=cmd_verify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "certify":
        if args.steps < 1:
            print("certify_numerics: ERROR --steps must be positive",
                  file=sys.stderr)
            return 2
        for name in ("max_site_error", "max_relaxed_gemm_error",
                     "max_loss_deviation"):
            value = getattr(args, name)
            if value is None:  # the contract family's default
                continue
            if not (math.isfinite(value) and value > 0.0):
                print(f"certify_numerics: ERROR --{name.replace('_', '-')} "
                      f"must be positive and finite (got {value!r}): a "
                      "tolerance of zero certifies nothing",
                      file=sys.stderr)
                return 2
    try:
        return args.fn(args)
    except fx.PlanError as e:
        print(f"certify_numerics: ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
