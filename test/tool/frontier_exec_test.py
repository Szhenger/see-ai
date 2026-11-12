"""Tests for tool/frontier_exec.py, the frontier reference executor (P4, #78).

Three tiers, each skipping cleanly where its dependency is absent:

  stdlib   the .seeu reader, the f32 LR schedule, the feeder's RNG;
  numpy    the interpreter against ITSELF: every backward opcode against a
           central finite difference of its forward opcode, run through a
           hand-assembled plan — the oracle must not be trusted because it
           agrees with the C++ runtime, only if it is right on its own;
  C++      the differential suite: a fixture matrix compiled by
           seeml-update-compile (every loss, optimizer, storage precision,
           gradient accumulation, fused and unfused epilogues, both model
           families) and compared instruction by instruction against
           seeml-plan-probe. Needs the built tools: SEEML_BUILD_DIR, else
           ./build.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import contextlib
import io
import json
import math
import os
import re
import struct
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import frontier_exec as fx  # noqa: E402

try:
    import numpy as np
except ImportError:  # the torch-free CI tier
    np = None

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
PROBE = os.path.join(BUILD, "seeml-plan-probe")
HAVE_CPP = all(os.path.isfile(p) and os.access(p, os.X_OK)
               for p in (COMPILER, PROBE))


def assemble(arena_size, train=(), persistent=b"", rodata=b"", programs=None,
             **header):
    """A minimal .seeu blob around hand-written instructions that the
    device's load contract accepts (CR2 #143): the I/O slots it demands sit
    in an 8-byte trailer past the caller's `arena_size` bytes — the staged
    input slot one float at `arena_size`, the loss slot at `arena_size + 4`
    — and rodata starts 16-aligned. `programs` adds step / eval / merge
    streams (a step program brings the grad_accum_steps = 2 that the
    contract pairs with it). The plan hash is left zero: the Python reader
    does not seal, only the C++ probe does."""
    programs = {"train": list(train), **(programs or {})}
    fields = dict.fromkeys(fx._HEADER_FIELDS, 0)
    fields.update(magic=fx.SEEU_MAGIC, version=fx.SEEU_NEWEST,
                  arena_size=arena_size + 8, persistent_size=len(persistent),
                  persist_init_size=len(persistent), input_ref=arena_size,
                  input_floats=1, label_ref=fx.NULL_REF,
                  loss_ref=arena_size + 4, lr=1e-3, beta1=0.9, beta2=0.999,
                  eps=1e-8, batch=1,
                  grad_accum_steps=2 if programs.get("step") else 0)
    fields.update(header)
    body = b""
    for name in fx.SECTIONS:
        ops = programs.get(name, ())
        fields[name + "_instr_offset"] = fx._HEADER.size + len(body)
        fields[name + "_instr_count"] = len(ops)
        body += b"".join(fx._INSTRUCTION.pack(op, flags, imm[0] if imm else 0,
                                              *src, *out)
                         for op, flags, src, out, *imm in ops)
    fields["persist_init_offset"] = fx._HEADER.size + len(body)
    pad = -(fields["persist_init_offset"] + len(persistent)) % 16
    fields["rodata_offset"] = (fields["persist_init_offset"] +
                               len(persistent) + pad)
    fields["rodata_size"] = len(rodata)
    head = fx._HEADER.pack(*(fields[name] for name in fx._HEADER_FIELDS))
    return bytearray(head + body + persistent + bytes(pad) + rodata)


def with_header(blob, **fields):
    """`blob` with header fields rewritten after assembly — the fields
    assemble() derives from the layout included."""
    head = dict(zip(fx._HEADER_FIELDS, fx._HEADER.unpack_from(blob, 0)))
    head.update(fields)
    out = bytearray(blob)
    out[:fx._HEADER.size] = fx._HEADER.pack(
        *(head[name] for name in fx._HEADER_FIELDS))
    return out


N = fx.NULL_REF


class ReaderTest(unittest.TestCase):
    def test_header_round_trip_and_sections(self):
        blob = assemble(64, train=[(19, 0, (0, fx.f32_bits(2.5), N, N),
                                    (4, 0, 0))], persistent=b"\1" * 8)
        plan = fx.Plan(blob)
        self.assertEqual((plan.version, plan.arena_size, plan.input_ref,
                          plan.loss_ref), (fx.SEEU_NEWEST, 72, 64, 68))
        self.assertEqual(len(plan.sections["train"]), 1)
        self.assertEqual(plan.sections["train"][0].opcode, 19)
        self.assertEqual(plan.initial_arena()[:9], b"\1" * 8 + b"\0")
        self.assertEqual(plan.histogram(), {"fill": {"train": 1}})

    def test_refuses_what_it_cannot_prove(self):
        blob = assemble(64)
        for mutate, message in (
                (lambda b: b.__setitem__(slice(0, 4), b"NOPE"), "not a .seeu"),
                (lambda b: b.__setitem__(slice(4, 8), struct.pack("<I", 99)),
                 "version 99"),
                (lambda b: b.__delitem__(slice(100, None)), "truncated")):
            bad = bytearray(blob)
            mutate(bad)
            with self.assertRaisesRegex(fx.PlanError, message):
                fx.Plan(bad)
        with self.assertRaisesRegex(fx.PlanError,
                                    "plan section out of bounds"):
            fx.Plan(with_header(assemble(64), eval_instr_offset=10 ** 9,
                                eval_instr_count=1))

    def test_refuses_every_header_the_device_refuses(self):
        """CR2 #143: PlanHeaderViolation rule for rule, in the device's
        order and words — the N04 slot rules and the N05 rodata alignment
        the reader used to skip among them. Each case breaks one rule of a
        plan the device accepts; where a case breaks two, the first in the
        device's order is the one named."""
        good = assemble(64, rodata=bytes(16))
        fx.Plan(good)
        head = dict(zip(fx._HEADER_FIELDS, fx._HEADER.unpack_from(good, 0)))
        rodata = head["rodata_offset"]
        r, s = fx.RODATA_BIT, fx.SOURCE_BIT
        stepped = assemble(64, programs={"step": [(0, 0, (N, N, N, N),
                                                   (0, 0, 0))]})
        fx.Plan(stepped)
        cases = (
            (good, dict(train_instr_count=1 << 58),
             "plan section size overflows"),
            (good, dict(emit_count=1 << 61), "plan section size overflows"),
            # Misaligned AND past the file's end: alignment is checked first.
            (good, dict(rodata_offset=rodata + 4),
             "rodata section is not 16-byte aligned"),
            (good, dict(eval_instr_offset=10 ** 9, eval_instr_count=1),
             "plan section out of bounds"),
            (good, dict(emit_table_offset=len(good), emit_count=1),
             "plan section out of bounds"),
            (good, dict(rodata_size=len(good)), "plan section out of bounds"),
            (good, dict(version=10, gemm_tile_n=8),
             "plan carries GEMM tile fields below v11"),
            (good, dict(gemm_tile_k=6), "plan gemm_tile_k 6 is not a "
             "multiple of the kernel's 4-wide unroll"),
            (good, dict(grad_accum_steps=2),
             "plan step program presence disagrees with grad_accum_steps"),
            (stepped, dict(version=8),
             "pre-v9 plan carries gradient-accumulation fields"),
            (good, dict(batch=0), "plan batch size is zero"),
            (good, dict(input_floats=0), "plan input width is zero"),
            (good, dict(arena_size=(1 << 64) - 1), "arena size overflows"),
            (good, dict(persistent_size=73),
             "persistent segment exceeds arena"),
            (good, dict(lr=0.0),
             "plan header lr is not a finite value in its range"),
            (good, dict(input_ref=s | 64), "plan input slot out of bounds"),
            (good, dict(input_ref=r | 64), "plan input slot out of bounds"),
            (good, dict(input_ref=66), "plan input slot out of bounds"),
            (good, dict(input_ref=72), "plan input slot out of bounds"),
            (good, dict(input_floats=1 << 62),
             "plan input slot out of bounds"),
            (good, dict(label_kind=1, label_ref=r, label_bytes=4),
             "plan label slot out of bounds"),
            (good, dict(label_kind=1, label_ref=s, label_bytes=4),
             "plan label slot out of bounds"),
            (good, dict(label_kind=1, label_ref=2, label_bytes=4),
             "plan label slot out of bounds"),
            (good, dict(label_kind=1, label_ref=0, label_bytes=73),
             "plan label slot out of bounds"),
            (good, dict(loss_ref=fx.NULL_REF), "plan loss slot out of bounds"),
            (good, dict(loss_ref=s | 68), "plan loss slot out of bounds"),
            (good, dict(loss_ref=70), "plan loss slot out of bounds"),
            # The data contract (v22).
            (good, dict(version=21, tokenizer_hash=1),
             "plan carries data-contract fields below v22"),
            (good, dict(data_flags=16), "plan data_flags carries unknown bits"),
            # The metadata section (v23): the flag below its version, and
            # the flag over a plan that carries no section.
            (good, dict(version=22, data_flags=8),
             "plan carries a metadata section below v23"),
            (good, dict(data_flags=8), "plan metadata section"),
            (good, dict(data_flags=4),
             "plan header val_frac is not a finite value in its range"),
            (good, dict(data_flags=4, val_frac_bits=0x3F800000),  # 1.0
             "plan header val_frac is not a finite value in its range"),
            (good, dict(split_seed=5),
             "plan carries split fields without the split flag"),
            (good, dict(data_flags=2), "plan pins a corpus without its hash"))
        for base, fields, message in cases:
            with self.subTest(**{k: repr(v) for k, v in fields.items()}):
                with self.assertRaisesRegex(fx.PlanError, re.escape(message)):
                    fx.Plan(with_header(base, **fields))
        # What the rules leave alone: an empty rodata section's offset, the
        # label slot of a plan that stages no labels, a 16-aligned rodata
        # the compiler did not put on a page, and the persist image shorter
        # than the persistent segment (the engine copies it over zeros).
        fx.Plan(with_header(good, rodata_size=0, rodata_offset=rodata + 4))
        fx.Plan(with_header(good, label_kind=0, label_ref=r | 3))
        self.assertNotEqual(rodata % fx.formats.RODATA_ALIGNMENT, 0)
        short = fx.Plan(with_header(assemble(64, persistent=b"\7" * 8),
                                    persistent_size=16))
        self.assertEqual(bytes(short.initial_arena()[:17]),
                         b"\7" * 8 + bytes(9))

    def test_refuses_the_immediates_the_device_refuses(self):
        """CR1 #142's validator and load-contract rules, held by the reader
        too: a plan the device refuses neither trains nor diffs here."""
        bits, nan = fx.f32_bits, float("nan")

        def plan(*ops, rodata=b"", **header):
            return fx.Plan(assemble(64, train=list(ops), rodata=rodata,
                                    **header))

        def clip(t):
            return (28, 0, (0, t, N, N), (2, 0, 0))
        plan(clip(bits(1.0)))
        for t in (bits(0.0), bits(-1.0), bits(nan), bits(1.0) | 1 << 32):
            with self.assertRaisesRegex(fx.PlanError, "clip_norm threshold"):
                plan(clip(t))

        def kl(word):
            return (15, 0, (0, 16, 32, 48), (64, 1 << 32 | 2, word))
        plan(kl(bits(4.0) << 32 | bits(2.0)))
        plan(kl(bits(fx.formats.KL_TEMPERATURE_MIN)))
        plan(kl(bits(fx.formats.KL_TEMPERATURE_MAX)))
        for t in (0.0, -2.0, nan, 1e4, 1e-4):
            with self.assertRaisesRegex(fx.PlanError, "temperature"):
                plan(kl(bits(t)))
        with self.assertRaisesRegex(fx.PlanError, "pre-v8"):
            plan(kl(bits(4.0) << 32 | bits(2.0)), version=7)

        def q8(scale, flags=0):
            return (29, flags, (0, fx.RODATA_BIT, 16, scale), (1, 2, 2))
        plan(q8(bits(0.5)), rodata=bytes(16))
        for scale in (bits(0.0), bits(nan), bits(-0.5), N):
            with self.assertRaisesRegex(fx.PlanError, "dequant scale"):
                plan(q8(scale), rodata=bytes(16))
        col = fx.formats.FLAG_Q8_COL_SCALE
        good = bytes(16) + struct.pack("<2f", 0.5, 0.25)
        plan(q8(fx.RODATA_BIT | 16, col), rodata=good)
        with self.assertRaisesRegex(fx.PlanError, "column scale 1"):
            plan(q8(fx.RODATA_BIT | 16, col),
                 rodata=bytes(16) + struct.pack("<2f", 0.5, nan))
        with self.assertRaisesRegex(fx.PlanError, "outside the rodata"):
            plan(q8(fx.RODATA_BIT | 64, col), rodata=good)

        step = (17, 0, (0, 16, N, N), (4, bits(1.0), 0))
        plan(step)
        with self.assertRaisesRegex(fx.PlanError, "pre-v12"):
            plan(step, version=11)
        with self.assertRaisesRegex(fx.PlanError, "clip threshold"):
            plan((17, 0, (0, 16, N, N), (4, bits(nan), 0)))

        for name, value in (("lr", nan), ("lr", 0.0), ("beta1", 1.0),
                            ("beta2", nan), ("eps", 0.0),
                            ("weight_decay", -0.01), ("clip_norm", -1.0),
                            ("min_lr_factor", 1.5)):
            with self.assertRaisesRegex(fx.PlanError, f"header {name}"):
                plan(**{name: value})

        # The imm word: a norm forward's epsilon, an f32 GEMM's alpha, and
        # nothing anywhere else.
        def rms(eps):
            return (31, 0, (0, 16, 32, 48), (1 << 32 | 4, 0, 0), eps)
        plan(rms(bits(1e-6)))
        for eps in (bits(0.0) | 1 << 31, bits(-1e-6), bits(nan)):
            with self.assertRaisesRegex(fx.PlanError, "epsilon"):
                plan(rms(eps))

        def gemm(alpha, flags=0):
            return (1, flags, (0, 16, 32, N), (1, 1, 1), alpha)
        plan(gemm(bits(0.5)))
        for alpha in (bits(0.0) | 1 << 31, bits(nan)):
            with self.assertRaisesRegex(fx.PlanError, "GEMM alpha"):
                plan(gemm(alpha))
        with self.assertRaisesRegex(fx.PlanError, "relaxed"):
            plan(gemm(bits(0.5), fx.formats.FLAG_RELAXED))
        with self.assertRaisesRegex(fx.PlanError, "does not define"):
            plan(gemm(bits(0.5)), version=18)
        with self.assertRaisesRegex(fx.PlanError, "does not define"):
            plan((19, 0, (0, bits(2.5), N, N), (4, 0, 0), 1))

        # A fused map's immediates: finite where a scale stage reads one,
        # zero where none does.
        def fused(imm):
            return (46, 0, (0, N, N, 16), (4, 0x03, imm))
        plan(fused(bits(2.0)))
        with self.assertRaisesRegex(fx.PlanError, "fused immediate 0"):
            plan(fused(bits(nan)))
        with self.assertRaisesRegex(fx.PlanError, "fused immediate 1"):
            plan(fused(bits(2.0) << 32 | bits(2.0)))

    def test_every_opcode_of_the_abi_is_interpreted(self):
        # source/plan/instruction.h is the ABI; a new opcode there must fail
        # here until the interpreter learns it.
        with open(os.path.join(REPO, "source/plan/instruction.h")) as f:
            text = f.read()
        body = text[text.index("enum class OpCode"):text.index("};")]
        declared = sorted(int(line.split("=")[1].split(",")[0])
                          for line in body.splitlines()
                          if line.strip().startswith("k") and "=" in line)
        self.assertEqual(declared, sorted(fx.INTERPRETER))

    def test_cosine_schedule_matches_the_engine(self):
        plan = fx.Plan(assemble(64, lr_schedule=1, warmup_steps=4,
                                default_steps=24, min_lr_factor=0.1, lr=2e-3))
        base = fx.f32(2e-3)
        self.assertAlmostEqual(plan.effective_lr(2), base / 2, places=9)
        self.assertEqual(plan.effective_lr(4), base)       # end of warmup
        self.assertAlmostEqual(plan.effective_lr(14),       # cosine midpoint
                               base * (0.1 + 0.9 * 0.5), places=9)
        self.assertEqual(plan.effective_lr(24), fx.f32(base * fx.f32(0.1)))
        self.assertEqual(plan.effective_lr(500), plan.effective_lr(24))
        self.assertEqual(fx.Plan(assemble(64)).effective_lr(7), fx.f32(1e-3))

    def test_splitmix64_reference_vector(self):
        # The published splitmix64 stream for seed 1234567.
        state, out = 1234567, []
        for _ in range(3):
            state, z = fx._splitmix64(state)
            out.append(z)
        self.assertEqual(out, [6457827717110365317, 3203168211198807973,
                               9817491932198370423])


@unittest.skipIf(np is None, "NumPy not installed")
class InterpreterSelfCheck(unittest.TestCase):
    """Backward opcodes against finite differences of forward opcodes."""

    def run_ops(self, arena_floats, instructions, values):
        plan = fx.Plan(assemble(4 * arena_floats, train=instructions))
        ex = fx.Executor(plan, fx.NumpyBackend("float64"))
        for off, array in values.items():
            ex.mem.stage(4 * off, np.asarray(array, "<f4").tobytes())
        ex.execute("train")
        return np.frombuffer(ex.mem.export(), "<f4").astype(np.float64)

    def check_vjp(self, n_in, n_out, fwd, bwd, x0, extra=None, eps=1e-2,
                  tol=2e-3):
        """fwd(x_off, y_off) / bwd(dy_off, x_off, dx_off) build instructions;
        asserts dx == J^T dy with J from central differences in f32-safe
        steps. Layout: x | y | dy | dx | scratch."""
        rng = np.random.default_rng(7)
        dy = rng.standard_normal(n_out)
        x_off, y_off, dy_off, dx_off = 0, n_in, n_in + n_out, n_in + 2 * n_out
        size = 2 * n_in + 2 * n_out + 4096
        base = {x_off: x0, dy_off: dy}
        base.update(extra or {})

        def forward(x):
            vals = dict(base)
            vals[x_off] = x
            return self.run_ops(size, fwd(x_off, y_off), vals)[
                y_off:y_off + n_out]
        out = self.run_ops(size, fwd(x_off, y_off) + bwd(dy_off, x_off,
                                                         dx_off, y_off), base)
        dx = out[dx_off:dx_off + n_in]
        numeric = np.zeros(n_in)
        for i in range(n_in):
            step = np.zeros(n_in)
            step[i] = eps
            numeric[i] = np.dot(dy, forward(x0 + step) -
                                forward(x0 - step)) / (2 * eps)
        scale = max(np.abs(numeric).max(), 1e-6)
        self.assertLess(np.abs(dx - numeric).max() / scale, tol)

    SCRATCH = 3000  # float offset of side outputs (cached statistics, P)

    def test_activations(self):
        x0 = np.linspace(-2.5, 2.5, 12)
        for f, b in ((7, 8), (22, 23), (24, 25)):
            self.check_vjp(
                12, 12,
                lambda x, y, f=f: [(f, 0, (4 * x, 4 * y, N, N), (12, 0, 0))],
                lambda dy, x, dx, y, b=b: [(b, 0, (4 * dy, 4 * x, 4 * dx, N),
                                            (12, 0, 0))],
                x0 + 0.013)  # keep ReLU's kink off the difference stencil

    def test_gelu_backward_survives_saturation(self):
        out = self.run_ops(16, [(23, 0, (0, 8, 16, N), (2, 0, 0))],
                           {0: [1.0, 1.0], 2: [3e12, -3e12]})
        self.assertEqual(list(out[4:6]), [1.0, 0.0])

    def test_norms(self):
        rows, dim, s = 3, 8, self.SCRATCH
        rng = np.random.default_rng(3)
        x0 = rng.standard_normal(rows * dim)
        gamma = {s: rng.standard_normal(dim), s + dim: rng.standard_normal(dim)}
        nd = rows << 32 | dim
        self.check_vjp(
            rows * dim, rows * dim,
            lambda x, y: [(26, 0, (4 * x, 4 * s, 4 * (s + dim), 4 * y),
                           (nd, 4 * (s + 64), 4 * (s + 96)))],
            lambda dy, x, dx, y: [(27, 0, (4 * dy, 4 * x, 4 * s, 4 * dx),
                                   (4 * (s + 64), 4 * (s + 96), nd))],
            x0, gamma)
        self.check_vjp(
            rows * dim, rows * dim,
            lambda x, y: [(31, 0, (4 * x, 4 * s, 4 * y, 4 * (s + 64)),
                           (nd, 0, 0))],
            lambda dy, x, dx, y: [(32, 0, (4 * dy, 4 * x, 4 * s, 4 * dx),
                                   (4 * (s + 64), nd, 0))],
            x0, gamma)

    def test_rope_is_a_rotation_and_its_backward_the_transpose(self):
        b, s, h, d = 2, 5, 2, 6
        n = b * s * h * d
        dims = (b << 32 | s, h << 32 | d, fx.f32_bits(10000.0))
        x0 = np.random.default_rng(5).standard_normal(n)
        out = self.run_ops(3 * n, [(33, 0, (0, 4 * n, N, N), dims),
                                   (34, 0, (4 * n, 8 * n, N, N), dims)],
                           {0: x0})
        np.testing.assert_allclose(out[2 * n:3 * n], x0, atol=2e-6)
        y = out[n:2 * n].reshape(b, s, h, d)
        np.testing.assert_allclose(y[:, 0], x0.reshape(b, s, h, d)[:, 0],
                                   atol=1e-7)  # position 0: identity
        # pair (0, 1) of position 1 turns by exactly one radian
        a, c = x0.reshape(b, s, h, d)[0, 1, 0, :2]
        np.testing.assert_allclose(
            y[0, 1, 0, :2], [a * math.cos(1) - c * math.sin(1),
                             a * math.sin(1) + c * math.cos(1)], atol=1e-6)

    def test_attention_chain(self):
        b, s, h, d = 1, 4, 2, 4
        n, pn, sc = b * s * h * d, b * h * s * s, self.SCRATCH
        bs, hd = b << 32 | s, h << 32 | d
        rng = np.random.default_rng(11)
        k0, v0 = rng.standard_normal(n), rng.standard_normal(n)
        p, dp, ds = sc + 200, sc + 300, sc + 400

        def fwd_q(x, y):   # y = attention(q = x, k, v)
            return [(35, 0, (4 * x, 4 * sc, 4 * (sc + 64), 4 * y),
                     (4 * p, bs, hd))]

        def chain(dy):
            return [(36, 0, (4 * dy, 4 * (sc + 64), 4 * dp, N), (bs, hd, 0)),
                    (38, 0, (4 * p, 4 * dp, 4 * ds, N),
                     ((b * h * s) << 32 | s, 0, 0))]
        extra = {sc: k0, sc + 64: v0}
        self.check_vjp(n, n, fwd_q,
                       lambda dy, x, dx, y: chain(dy) + [
                           (39, 0, (4 * ds, 4 * sc, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)

        def fwd_k(x, y):
            return [(35, 0, (4 * sc, 4 * x, 4 * (sc + 64), 4 * y),
                     (4 * p, bs, hd))]
        self.check_vjp(n, n, fwd_k,
                       lambda dy, x, dx, y: chain(dy) + [
                           (40, 0, (4 * ds, 4 * sc, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)

        def fwd_v(x, y):
            return [(35, 0, (4 * sc, 4 * (sc + 64), 4 * x, 4 * y),
                     (4 * p, bs, hd))]
        self.check_vjp(n, n, fwd_v,
                       lambda dy, x, dx, y: [
                           (37, 0, (4 * p, 4 * dy, 4 * dx, N), (bs, hd, 0))],
                       rng.standard_normal(n), extra)
        out = self.run_ops(sc + 600, fwd_q(0, n), {0: k0, sc: k0, sc + 64: v0})
        probs = out[p:p + pn].reshape(b * h, s, s)
        np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-6)
        self.assertEqual(float(np.triu(probs, 1).max()), 0.0)  # causal

    def test_losses(self):
        rows, classes, sc = 3, 5, self.SCRATCH
        n = rows * classes
        rng = np.random.default_rng(13)
        labels = np.array([1, 4, 0], "<i4").view("<f4")
        seed = {sc: labels, sc + 8: [1.0]}

        def scalar(op, srcs, outs):
            return lambda x, y: [(op, 0, srcs(x, y), outs)]
        self.check_vjp(          # softmax cross-entropy
            n, 1,
            scalar(11, lambda x, y: (4 * x, 4 * sc, 4 * y, 4 * (sc + 16)),
                   (rows, classes, 0)),
            lambda dy, x, dx, y: [(12, 0, (4 * (sc + 16), 4 * sc, 4 * dy,
                                           4 * dx), (rows, classes, 0))],
            rng.standard_normal(n), seed, tol=5e-3)
        target = {sc: rng.standard_normal(n)}
        self.check_vjp(          # mean squared error
            n, 1,
            scalar(13, lambda x, y: (4 * x, 4 * sc, 4 * y, N), (n, 0, 0)),
            lambda dy, x, dx, y: [(14, 0, (4 * x, 4 * sc, 4 * dy, 4 * dx),
                                   (n, 0, 0))],
            rng.standard_normal(n), target)
        word = fx.f32_bits(4.0) << 32 | fx.f32_bits(2.0)  # T = 2, scale T^2
        nc = rows << 32 | classes
        self.check_vjp(          # KL distillation against a fixed teacher
            n, 1,
            scalar(15, lambda x, y: (4 * x, 4 * sc, 4 * y, 4 * (sc + 32)),
                   (4 * (sc + 64), nc, word)),
            lambda dy, x, dx, y: [(16, 0, (4 * (sc + 32), 4 * (sc + 64),
                                           4 * dy, 4 * dx), (nc, word, 0))],
            rng.standard_normal(n), target, tol=5e-3)

    def test_adamw_matches_the_textbook_recurrence(self):
        g = [0.5, -2.0, 0.0, 1e-9]
        out = self.run_ops(16, [(18, 0, (0, 16, 32, 48), (4, 0, 0))],
                           {0: [1.0, 1.0, 1.0, 1.0], 4: g})
        lr, eps, wd = 1e-3, 1e-8, 0.0
        for i, gi in enumerate(g):  # step 1: m_hat = g, v_hat = g^2
            want = 1.0 - lr * (gi / (abs(gi) + eps) + wd)
            self.assertAlmostEqual(out[i], want, places=6)

    def test_clip_norm(self):
        clip = [(28, 0, (0, fx.f32_bits(1.0), N, N), (2, 0, 0))]
        np.testing.assert_allclose(self.run_ops(4, clip, {0: [3.0, 4.0]})[:2],
                                   [0.6, 0.8], atol=1e-7)
        np.testing.assert_allclose(self.run_ops(4, clip, {0: [0.3, 0.4]})[:2],
                                   [0.3, 0.4], atol=1e-7)
        self.assertEqual(list(self.run_ops(4, clip, {0: [0.0, 0.0]})[:2]),
                         [0.0, 0.0])

    def test_the_opcodes_no_compiler_path_emits(self):
        ops = [(10, 0, (0, 24, N, N), (2, 3, 0)),    # reduce_rows -> [6, 9)
               (20, 0, (24, 36, N, N), (3, 0, 0)),   # copy        -> [9, 12)
               (0, 0, (N, N, N, N), (0, 0, 0))]      # nop
        out = self.run_ops(12, ops, {0: [1, 2, 3, 10, 20, 30]})
        self.assertEqual(list(out[6:12]), [11, 22, 33, 11, 22, 33])

    def test_tiled_probabilities_ignore_future_keys_of_any_size(self):
        # A future key's raw score far above the causal prefix's max must
        # not overflow into inf * 0 = NaN: the tiled backward's recomputed
        # P is exactly zero there, as the C++ kernels never touch it
        # (ultrareview on #125). f32 overflows past ~88, f64 past ~709.
        for x, big in ((fx.NumpyBackend("float32"), 200.0),
                       (fx.NumpyBackend(), 1000.0)):
            b, h, s, d = 1, 1, 4, 2
            q = np.zeros((b, h, s, d), x.compute)
            k = np.zeros((b, h, s, d), x.compute)
            q[..., 0, :] = 1.0
            k[..., 3, :] = big
            stats = np.zeros((b * h * s, 4), x.compute)
            stats[:, 1] = 1.0
            # exp(-1e30) underflowing to 0 is the point; overflow or an
            # invalid operation is the bug.
            with np.errstate(over="raise", invalid="raise"):
                p = fx._attn_probs(x, q, k, s, d, stats, b, h)
            self.assertFalse(np.isnan(p).any(), x.name)
            self.assertEqual(float(p[0, 0, 0, 3]), 0.0)
            self.assertEqual(float(p[0, 0, 0, 0]), 1.0)

    def test_quantized_and_bf16_weights_widen_exactly(self):
        q = np.array([[1, -2], [3, 127]], np.int8)
        h = np.array([[1.5, -0.25], [4.0, 1024.0]], np.float32)
        bf = (h.view(np.uint32) >> 16).astype("<u2")  # exact for these
        rodata = q.tobytes().ljust(16, b"\0") + bf.tobytes()
        ops = [(29, 0, (0, fx.RODATA_BIT, 16, fx.f32_bits(0.5)), (1, 2, 2)),
               (43, 0, (0, fx.RODATA_BIT | 16, 32, N), (1, 2, 2))]
        plan = fx.Plan(assemble(64, train=ops, rodata=rodata))
        ex = fx.Executor(plan, fx.NumpyBackend("float64"))
        ex.mem.stage(0, np.array([1.0, 2.0], "<f4").tobytes())
        ex.execute("train")
        out = np.frombuffer(ex.mem.export(), "<f4")
        np.testing.assert_allclose(out[4:6], [1, 2] @ (q * 0.5))
        np.testing.assert_allclose(out[8:10], [1, 2] @ h)

    def test_relaxed_gemms_round_the_activation_to_bf16_only_when_asked(self):
        # v18: kFlagRelaxed on a frozen-weight GEMM. The exact reference
        # ignores the bit; a backend in relaxed mode rounds A to bf16 once
        # (ties to even) and multiplies against the exact weight; the count
        # the packer gates on sees the train and step programs.
        self.assertEqual(fx.round_bf16(np, np.float32(1.00390625)), 1.0)
        self.assertEqual(fx.round_bf16(np, np.float32(1.01171875)), 1.015625)
        self.assertEqual(fx.round_bf16(np, np.float32(1.0)), 1.0)
        self.assertTrue(np.isnan(fx.round_bf16(np, np.float32("nan"))))
        self.assertEqual(fx.round_bf16(np, np.float64(3.0)).dtype, np.float64)
        w = np.array([[1.0, -2.0], [0.5, 4.0]], np.float32)
        a = np.array([1.00390625, 2.00390625], np.float32)  # not bf16-exact
        ops = [(1, fx.FLAG_RELAXED, (0, fx.RODATA_BIT, 16, N), (1, 2, 2)),
               (1, 0, (0, fx.RODATA_BIT, 32, N), (1, 2, 2))]
        blob = assemble(64, train=ops, rodata=w.tobytes())
        self.assertEqual(fx.formats.relaxed_gemm_count(blob), 1)
        plan = fx.Plan(blob)
        self.assertEqual(plan.relaxed_gemms, 1)
        for relaxed in (False, True):
            backend = fx.NumpyBackend("float64")
            backend.relaxed_gemms = relaxed
            ex = fx.Executor(plan, backend)
            ex.mem.stage(0, a.tobytes())
            ex.execute("train")
            out = np.frombuffer(ex.mem.export(), "<f4")
            expect_a = fx.round_bf16(np, a) if relaxed else a
            np.testing.assert_allclose(out[4:6], expect_a.astype(np.float64) @ w,
                                       rtol=1e-6)
            np.testing.assert_allclose(out[8:10], a.astype(np.float64) @ w,
                                       rtol=1e-6)  # the unflagged GEMM
        self.assertNotEqual(list(fx.round_bf16(np, a)), list(a))
        # The certificate's relaxed dX form folds the k-scale into A first.
        q = np.array([[3, -1], [2, 5]], np.int8)  # W as [N, K]
        s = np.array([0.5, 0.25], np.float32)     # one scale per k
        rodata = q.tobytes().ljust(16, b"\0") + s.tobytes()
        ops = [(30, fx.FLAG_RELAXED | fx.formats.FLAG_Q8_COL_SCALE,
                (0, fx.RODATA_BIT, 16, fx.RODATA_BIT | 16), (1, 2, 2))]
        plan = fx.Plan(assemble(64, train=ops, rodata=rodata))
        backend = fx.NumpyBackend("float64")
        backend.relaxed_gemms = True
        ex = fx.Executor(plan, backend)
        ex.mem.stage(0, a.tobytes())
        ex.execute("train")
        out = np.frombuffer(ex.mem.export(), "<f4")
        folded = fx.round_bf16(np, a * s).astype(np.float64)
        np.testing.assert_allclose(out[4:6], folded @ q.astype(np.float64).T,
                                   rtol=1e-6)

    def test_slot_memory_survives_arena_reuse(self):
        """SlotMemory (the GPU backends' model) under the aliasing the arena
        allocator really produces: overlapping writes, sub-range reads."""
        class Slots(fx.NumpyBackend):
            arena_views = False
        ops = [(19, 0, (0, fx.f32_bits(1.0), N, N), (8, 0, 0)),   # [0,8) = 1
               (19, 0, (16, fx.f32_bits(2.0), N, N), (4, 0, 0)),  # [4,8) = 2
               (5, 0, (0, 16, 32, N), (4, 0, 0)),   # [8,12) = [0,4)+[4,8)
               (9, 0, (8, 0, fx.f32_bits(3.0), N), (2, 0, 0))]    # [0,2)=3*[2,4)
        plan = fx.Plan(assemble(64, train=ops))
        outs = []
        for backend in (fx.NumpyBackend("float64"), Slots("float64")):
            ex = fx.Executor(plan, backend)
            ex.execute("train")
            outs.append(np.frombuffer(ex.mem.export(), "<f4").copy())
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[0][:12],
                                      [3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])


@unittest.skipIf(np is None, "NumPy not installed")
class StagedSlotTest(unittest.TestCase):
    """CR2 #143 N03: no instruction of any program writes the staged input
    or label slot — refused in the device's words on the bytes the
    instruction really writes, by both memory models."""

    class Slots(fx.NumpyBackend):
        arena_views = False

    @staticmethod
    def fill(off, n):
        return (19, 0, (off, fx.f32_bits(1.0), N, N), (n, 0, 0))

    def test_no_program_writes_the_staged_input_slot(self):
        # assemble(64): the input slot is [64, 68), the loss slot [68, 72).
        for backend in (fx.NumpyBackend, self.Slots):
            for program in fx.SECTIONS:
                with self.subTest(backend=backend.__name__, program=program):
                    plan = fx.Plan(assemble(
                        64, programs={program: [self.fill(60, 2)]}))
                    ex = fx.Executor(plan, backend("float64"))
                    with self.assertRaisesRegex(
                            fx.PlanError,
                            f"an instruction of the {program} program writes "
                            r"the staged input slot \(opcode 19\)"):
                        ex.execute(program)

    def test_the_label_slot_too_and_nothing_beside_them(self):
        for backend in (fx.NumpyBackend, self.Slots):
            with self.subTest(backend=backend.__name__):
                plan = fx.Plan(assemble(64, train=[self.fill(48, 1)],
                                        label_kind=1, label_ref=48,
                                        label_bytes=4))
                with self.assertRaisesRegex(fx.PlanError, "writes the staged "
                                            "label slot"):
                    fx.Executor(plan, backend("float64")).execute("train")
                # The bytes beside the slots, the loss slot (the loss
                # kernels write it), and the feeder's own staging are all
                # in order.
                plan = fx.Plan(assemble(
                    64, train=[self.fill(40, 2), self.fill(56, 2),
                               self.fill(68, 1)],
                    label_kind=1, label_ref=48, label_bytes=4))
                ex = fx.Executor(plan, backend("float64"))
                ex.mem.stage(48, struct.pack("<i", 3))
                ex.mem.stage(64, struct.pack("<f", 2.5))
                ex.execute("train")
                arena = ex.mem.export()
                self.assertEqual(struct.unpack_from("<i", arena, 48)[0], 3)
                self.assertEqual(struct.unpack_from("<f", arena, 64)[0], 2.5)
                self.assertEqual(struct.unpack_from("<f", arena, 68)[0], 1.0)
                self.assertEqual(struct.unpack_from("<f", arena, 60)[0], 1.0)


# --- The differential suite ----------------------------------------------------


def _write_models(out_dir):
    """Two feature models and a token decoder, NumPy only, covering every
    SMF operator: bias, the three activations, LayerNorm, a gated product
    and a residual add (MLP); RMSNorm, RoPE, attention, embedding (decoder)."""
    import export_model as em
    rng = np.random.default_rng(42)

    def mat(*shape):
        return np.ascontiguousarray(rng.standard_normal(shape) * 0.3, "<f4")

    def mlp(path, width, rich):
        b = em._SmfBuilder("x", 16)

        def tensor(name, arr):
            b.add_tensor(name, list(arr.shape), arr)
            return name
        b.add_op(em.OP_MATMUL, "mm0", ["x", tensor("w0", mat(16, width))], "z0")
        b.add_op(em.OP_ADDBIAS, "ab0", ["z0", tensor("b0", mat(width))], "zb0")
        b.add_op(em.OP_GELU if rich else em.OP_RELU, "a0", ["zb0"], "h0")
        prev = "h0"
        if rich:
            b.add_op(em.OP_LAYERNORM, "ln", ["h0", tensor("ln_g", mat(width) + 1),
                                             tensor("ln_b", mat(width))], "n")
            b.add_op(em.OP_MATMUL, "mmg", ["n", tensor("wg", mat(width, width))],
                     "g")
            b.add_op(em.OP_MATMUL, "mmu", ["n", tensor("wu", mat(width, width))],
                     "u")
            b.add_op(em.OP_SILU, "silu", ["g"], "s")
            b.add_op(em.OP_MUL, "gate", ["s", "u"], "m")
            b.add_op(em.OP_ADD, "res", ["m", "n"], "r")
            b.add_op(em.OP_MATMUL, "mm1", ["r", tensor("w1", mat(width, width))],
                     "z1")
            b.add_op(em.OP_ADDBIAS, "ab1", ["z1", tensor("b1", mat(width))],
                     "zb1")
            b.add_op(em.OP_RELU, "a1", ["zb1"], "h1")
            prev = "h1"
        b.add_op(em.OP_MATMUL, "mmo", [prev, tensor("wo", mat(width, 4))], "zo")
        b.add_op(em.OP_ADDBIAS, "abo", ["zo", tensor("bo", mat(4))], "logits")
        with open(path, "wb") as f:
            b.write(f)

    mlp(os.path.join(out_dir, "mlp.smf"), 24, rich=True)
    mlp(os.path.join(out_dir, "teacher.smf"), 32, rich=False)
    x = rng.standard_normal((96, 16)).astype(np.float32)
    y = (x[:, :4].sum(axis=1) > 0).astype(np.int32) + 2 * (x[:, 0] > 0)
    with contextlib.redirect_stdout(io.StringIO()):
        em.export_sds(x, y, os.path.join(out_dir, "class.sds"))
        em.export_sds(x, np.eye(4, dtype=np.float32)[y],
                      os.path.join(out_dir, "dense.sds"), label_kind=2)
        em.export_sds(x, None, os.path.join(out_dir, "none.sds"))
        em._demo_decoder(out_dir, vocab=40, dim=24, heads=4, seq=6,
                         blocks_n=2, samples=48, seed=1)
        # A masked token corpus (SDS v3, S3 #149): the decoder's geometry,
        # every record with at least one target, a prompt-shaped mask.
        rng = np.random.default_rng(5)
        ids = rng.integers(0, 40, (48, 7)).astype("<i4")
        mask = (rng.random((48, 6)) < 0.6).astype(np.uint8)
        mask[:, -1] = 1
        em.export_token_sds(ids, os.path.join(out_dir, "decoder_masked.sds"),
                            mask=mask)
        os.makedirs(os.path.join(out_dir, "eps6"), exist_ok=True)
        em._demo_decoder(os.path.join(out_dir, "eps6"), vocab=40, dim=24,
                         heads=4, seq=6, blocks_n=2, samples=48, seed=1,
                         norm_eps=1e-6)


# (name, model, corpus, data batch, compiler flags)
MATRIX = [
    ("mlp_xent_adamw", "mlp.smf", "class.sds", 8,
     ["--clip-norm", "0.5", "--lr-schedule", "cosine", "--warmup", "2",
      "--min-lr-factor", "0.1", "--steps", "8"]),
    # alpha / r pinned: the teeth test perturbs AdamW's eps on this plan,
    # which three steps only show while the second moments are small.
    ("mlp_unfused", "mlp.smf", "class.sds", 8,
     ["--no-fuse-epilogue", "--lora-scaling", "alpha_over_r"]),
    ("mlp_mse_sgd", "mlp.smf", "dense.sds", 8,
     ["--loss", "mse", "--optimizer", "sgd", "--weight-decay", "0.01",
      "--clip-norm", "0.5"]),                       # the fused SGD clip
    ("mlp_kl", "mlp.smf", "none.sds", 8,
     ["--loss", "kl", "--teacher", "teacher.smf", "--temperature", "2"]),
    ("mlp_xent_kl", "mlp.smf", "class.sds", 8,
     ["--loss", "xent+kl", "--teacher", "teacher.smf",
      "--distill-weight", "0.4"]),
    ("mlp_q8", "mlp.smf", "class.sds", 8, ["--quantize-base"]),
    ("mlp_bf16", "mlp.smf", "class.sds", 8, ["--bf16-base"]),
    ("mlp_accum", "mlp.smf", "class.sds", 8,
     ["--grad-accum", "3", "--clip-norm", "1.0"]),
    ("dec_f32", "decoder.smf", "decoder_corpus.sds", 12,
     ["--lora-rank", "4"]),
    ("dec_q8_accum", "decoder.smf", "decoder_corpus.sds", 12,
     ["--quantize-base", "--grad-accum", "2"]),
    ("dec_bf16", "decoder.smf", "decoder_corpus.sds", 12, ["--bf16-base"]),
    # The v11 program shapes plan v12 replaces, still compiled on request:
    # a standalone clip per tensor, RoPE angles recomputed in place.
    ("dec_v11_shape", "decoder.smf", "decoder_corpus.sds", 12,
     ["--clip-norm", "0.5", "--no-fuse-clip", "--no-rope-table",
      "--no-fuse-elementwise"]),
    ("mlp_unfused_maps", "mlp.smf", "class.sds", 8,
     ["--no-fuse-elementwise", "--no-fuse-epilogue"]),
    # A Qwen2-class RMSNorm epsilon (plan v16 imm word, P7 #96).
    ("dec_eps6", "eps6/decoder.smf", "eps6/decoder_corpus.sds", 12,
     ["--lora-rank", "4"]),
    # The tiled attention family (plan v15): the stats row, and every
    # probability recomputed by the backward passes.
    ("dec_tiled", "decoder.smf", "decoder_corpus.sds", 12,
     ["--attention", "tiled", "--lora-rank", "4"]),
    # The masked cross-entropy (S3 #149, plan v22): compiled over a masked
    # corpus by --data, which also pins the plan to it.
    ("dec_masked", "decoder.smf", "decoder_masked.sds", 12,
     ["--data", "decoder_masked.sds", "--lora-rank", "4"]),
]
# Opcodes no seeml-update-compile invocation produces today: kNop is a
# placeholder, kCopy a utility no pass selects, and kReduceRows the bias
# gradient, which needs-set pruning removes because LoRA freezes every
# bias. They are interpreted, and InterpreterSelfCheck runs them from a
# hand-assembled plan; when a compiler path starts emitting one, this set
# shrinks and the matrix grows.
NEVER_COMPILED = {"nop", "copy", "reduce.rows"}


@unittest.skipIf(np is None, "NumPy not installed")
@unittest.skipUnless(HAVE_CPP, f"no built seeml tools under {BUILD}")
class DifferentialSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="seeml-frontier-test-")
        cls.dir = cls._tmp.name
        _write_models(cls.dir)
        cls.plans = {}
        for name, model, corpus, batch, flags in MATRIX:
            out = os.path.join(cls.dir, name)
            flags = [os.path.join(cls.dir, f)
                     if f.endswith((".smf", ".sds")) else f for f in flags]
            done = subprocess.run(
                [COMPILER, "--source", os.path.join(cls.dir, model), "--out",
                 out, "--data-batch", str(batch), "--no-embed"] + flags,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if done.returncode != 0:
                raise AssertionError(f"{name}: compile failed\n{done.stdout}")
            cls.plans[name] = (os.path.join(out, "update_plan.seeu"),
                               os.path.join(cls.dir, corpus),
                               os.path.join(cls.dir, model))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def diff(self, name, *extra):
        plan, corpus, model = self.plans[name]
        report = os.path.join(self.dir, name + ".json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            # --source: a v17 narrow-storage plan's eval program reads the
            # shipped f32 weights from the model file (E12).
            status = fx.main(["diff", plan, "--corpus", corpus, "--probe",
                              PROBE, "--steps", "3", "--report", report,
                              "--source", model, *extra])
        with open(report) as f:
            return status, json.load(f), buf.getvalue()

    def test_the_matrix_agrees_and_covers_the_instruction_set(self):
        covered = set()
        for name, *_ in MATRIX:
            with self.subTest(plan=name):
                status, report, log = self.diff(name)
                self.assertEqual(status, 0, log)
                self.assertTrue(report["agrees"])
                for opcode, stat in report["opcodes"].items():
                    self.assertGreater(stat["extents"], 0, opcode)
                    self.assertLess(stat["worst_ratio"], 1.0)
                # F42: every report says how many relaxed GEMMs it compared
                # and how many the C++ backend ran on a relaxed kernel; the
                # matrix compiles no relaxed plan.
                self.assertEqual(report["relaxed_instructions"], 0)
                self.assertEqual(report["cpp_relaxed_executed"], 0)
                covered |= set(report["opcodes"])
        self.assertEqual(set(fx.OPCODES.values()) - covered, NEVER_COMPILED)

    def test_the_corpus_check_mirrors_the_feeder_contract(self):
        """CR2 #143 F47: Corpus.check holds the corpus to every rule the C++
        feeder contract holds it to — the geometry, the label kind and
        width, and every staged value's bound — so a corpus the device
        would refuse is refused here, not blamed on a kernel."""
        import export_model as em
        plan = fx.Plan.load(self.plans["mlp_unfused"][0])
        classes = plan.label_classes()
        d = plan.input_floats // plan.batch
        x = np.zeros((4, d), "<f4")

        def corpus(name, write):
            # Under its own prefix: the matrix's corpora share this directory.
            path = os.path.join(self.dir, name)
            write(path)
            return fx.Corpus(np, path)
        corpus("check_ok.sds", lambda p: em.export_sds(
            x, np.zeros(4, "<i4"), p)).check(plan)
        with self.assertRaisesRegex(fx.PlanError, "class label"):
            corpus("check_high.sds", lambda p: em.export_sds(
                x, np.array([0, 1, classes, 0], "<i4"), p)).check(plan)
        with self.assertRaisesRegex(fx.PlanError, "label kind"):
            corpus("check_dense.sds", lambda p: em.export_sds(
                x, np.zeros((4, 3), "<f4"), p, label_kind=2)).check(plan)
        with self.assertRaisesRegex(fx.PlanError, "input width"):
            corpus("check_narrow.sds", lambda p: em.export_sds(
                np.zeros((4, d + 1), "<f4"), np.zeros(4, "<i4"), p)).check(plan)
        token_plan = fx.Plan.load(self.plans["dec_f32"][0])
        s, vocab = token_plan.seq_len, token_plan.vocab()
        records = np.zeros((3, s + 1), "<i4")
        corpus("check_tok.sds", lambda p: em.export_token_sds(records, p)).check(
            token_plan)
        # The tokenizer identity (v22, S2 #148): a plan bound to a section
        # takes a corpus that names no tokenizer (SDS v1 / v2, an unbound
        # v3) or the same one, and refuses one that names another.
        with open(self.plans["dec_f32"][0], "rb") as f:
            blob = f.read()
        bound = fx.Plan(with_header(blob, tokenizer_hash=0xABCD))
        corpus("check_tok.sds", lambda p: em.export_token_sds(records, p)).check(
            bound)
        corpus("check_tok_same.sds", lambda p: em.export_token_sds(
            records, p, tokenizer_hash=0xABCD)).check(bound)
        with self.assertRaisesRegex(fx.PlanError, "another tokenizer"):
            corpus("check_tok_other.sds", lambda p: em.export_token_sds(
                records, p, tokenizer_hash=0x1234)).check(bound)
        # The corpus pin (v22): under the flag the corpus file's ContentHash64
        # must be the header's.
        from seeml import content_hash
        pin_path = os.path.join(self.dir, "check_pin.sds")
        em.export_token_sds(records, pin_path)
        with open(pin_path, "rb") as f:
            pin_hash = content_hash.content_hash64(f.read())
        pinned = fx.Plan(with_header(
            blob, data_flags=fx.formats.DATA_FLAG_CORPUS_PINNED,
            corpus_hash=pin_hash))
        fx.Corpus(np, pin_path).check(pinned)
        other = fx.Plan(with_header(
            blob, data_flags=fx.formats.DATA_FLAG_CORPUS_PINNED,
            corpus_hash=pin_hash ^ 1))
        with self.assertRaisesRegex(fx.PlanError, "another corpus"):
            fx.Corpus(np, pin_path).check(other)
        # The compiled split (v22, S3 #149): the literal fixture the C++
        # split_test pins (n = 10, seed = 1, 30 %), the hash over the
        # indices, the partition, and the contract rule.
        self.assertEqual(fx.held_out_indices(10, 1, 0.3), [1, 5, 7])
        self.assertEqual(fx.split_hash([1, 5, 7]), 0xe731eed086e2a95d)
        self.assertEqual(fx.held_out_count(501, 0.1), 50)
        self.assertEqual(fx.held_out_indices(1, 0, 0.5), [])
        ten = np.arange(10 * (s + 1), dtype="<i4").reshape(10, s + 1) % vocab
        ten_path = os.path.join(self.dir, "check_split.sds")
        em.export_token_sds(ten, ten_path)
        full = fx.Corpus(np, ten_path)
        train, held = full.split_held_out([1, 5, 7], 0xe731eed086e2a95d)
        self.assertEqual((train.num_samples, held.num_samples), (7, 3))
        self.assertTrue((held.tokens == ten[[1, 5, 7]]).all())
        self.assertTrue((train.tokens == ten[[0, 2, 3, 4, 6, 8, 9]]).all())
        split_plan = fx.Plan(with_header(
            blob, data_flags=fx.formats.DATA_FLAG_SPLIT, split_seed=1,
            val_frac_bits=struct.unpack("<I", struct.pack("<f", 0.3))[0],
            val_split_hash=0xe731eed086e2a95d))
        train.check(split_plan)
        held.check(split_plan)
        with self.assertRaisesRegex(fx.PlanError, "not split as the plan"):
            fx.Corpus(np, ten_path).check(split_plan)
        with self.assertRaisesRegex(fx.PlanError, "sorted, distinct"):
            fx.Corpus(np, ten_path).split_held_out([5, 1], 1)
        # corpus_for_plan prepares the corpus as the device does: the
        # compiled split recomputed (a --val-fraction refused), and the
        # WHOLE corpus checked — an id out of range in a held-out record
        # (5) is the feeder's refusal, as UpdateEngine::TrainImpl validates
        # the validation set too.
        train2, held2 = fx.corpus_for_plan(np, split_plan, ten_path)
        self.assertEqual((train2.num_samples, held2.num_samples), (7, 3))
        self.assertEqual(held2.split_hash, 0xe731eed086e2a95d)
        with self.assertRaisesRegex(fx.PlanError, "not a runner decision"):
            fx.corpus_for_plan(np, split_plan, ten_path, 0.25)
        bad = ten.copy()
        bad[5, 2] = vocab
        bad_path = os.path.join(self.dir, "check_split_bad.sds")
        em.export_token_sds(bad, bad_path)
        with self.assertRaisesRegex(fx.PlanError, "token id"):
            fx.corpus_for_plan(np, split_plan, bad_path)
        # The f32 fraction rule: 0.7 of ten records is 6 (0.7f widens below
        # 0.7), the count the C++ HeldOutCount and the plan header agree on.
        self.assertEqual(fx.held_out_count(10, 0.7), 6)
        records[1, 2] = vocab
        with self.assertRaisesRegex(fx.PlanError, "token id"):
            corpus("check_tokbad.sds", lambda p: em.export_token_sds(records, p)).check(
                token_plan)
        with self.assertRaisesRegex(fx.PlanError, "token ids"):
            corpus("check_tok2.sds", lambda p: em.export_token_sds(
                np.zeros((3, s + 1), "<i4"), p)).check(plan)

    def test_the_oracle_has_teeth(self):
        """One wrong constant in one opcode is caught, and blamed on it."""
        for opcode, broken in (
                (22, fx._unary(lambda x, t: 0.5 * t * (1.0 + x.tanh(
                    fx.GELU_C * (t + 0.045 * t * t * t))))),   # GELU's cubic
                (18, lambda m, x, ins, step: fx._adamw(
                    m, x, ins, dict(step, eps=2e-8))),):        # AdamW's eps
            original = fx.INTERPRETER[opcode]
            fx.INTERPRETER[opcode] = broken
            try:
                status, report, _ = self.diff("mlp_unfused")
            finally:
                fx.INTERPRETER[opcode] = original
            self.assertEqual(status, 1)
            blamed = {re.search(r"#\d+ (\S+)", line).group(1)
                      for line in report["failures"]}
            self.assertEqual(blamed, {fx.OPCODES[opcode]})

    def test_float32_backends_agree_too(self):
        for backend in ("numpy32", "torch", "mlx"):
            try:
                fx.make_backend(backend)
            except ImportError:
                continue
            for name in ("mlp_xent_kl", "dec_q8_accum"):
                with self.subTest(backend=backend, plan=name):
                    status, _, log = self.diff(name, "--backend", backend)
                    self.assertEqual(status, 0, log)

    def test_the_probe_is_strict_about_arguments_and_paths(self):
        plan, _, _ = self.plans["mlp_mse_sgd"]
        work = os.path.join(self.dir, "probe")
        os.makedirs(work, exist_ok=True)
        arena = os.path.join(work, "arena.in")
        with open(arena, "wb") as f:
            f.write(fx.Plan.load(plan).initial_arena())
        out = os.path.join(work, "arena.out")

        def probe(*argv):
            done = subprocess.run([PROBE, *argv], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
            return done.returncode, done.stderr
        base = ["--plan", plan, "--section", "merge", "--arena-in", arena]
        self.assertEqual(probe(*base, "--arena-out", out)[0], 0)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(out + ".tmp"))  # staged, then renamed
        done = subprocess.run([PROBE, *base, "--arena-out", out, "--profile",
                               "2"], stdout=subprocess.PIPE, text=True,
                              stderr=subprocess.DEVNULL, check=True)
        profile = json.loads(done.stdout)
        self.assertEqual(profile["executions"], 2)
        self.assertEqual(profile["relaxed_executions"], 0)  # an exact plan
        self.assertTrue(any(row["key"].startswith("op4 M")  # gemm.acc_nn
                            for row in profile["rows"]))

        for argv, code, message in (
                ((*base, "--arena-out", out, "--bogus"), 2, "unknown argument"),
                ((*base, "--arena-out"), 2, "missing its value"),
                ((*base, "--arena-out", out, "--section", "x"), 2, "--section"),
                ((*base, "--arena-out", out, "--time", "0"), 2, "--time"),
                ((*base, "--arena-out", out, "--time", "2", "--trace",
                  out + ".t"), 2, "mutually exclusive"),
                ((*base, "--arena-out", out, "--profile", "2", "--time", "2"),
                 2, "mutually exclusive"),
                ((*base, "--arena-out", out, "--profile", "0"), 2,
                 "--profile"),
                (("--plan", work, "--section", "merge", "--arena-in", arena,
                  "--arena-out", out), 1, "not a regular file"),
                ((*base, "--arena-out", os.path.join(work, "nowhere", "a")),
                 1, "does not exist"),
                ((*base, "--arena-out", work), 1, "not a regular file")):
            status, err = probe(*argv)
            self.assertEqual(status, code, err)
            self.assertIn(message, err)

        # An output that is a symlink is refused, never written through.
        victim = os.path.join(work, "victim")
        with open(victim, "w") as f:
            f.write("untouched")
        link = os.path.join(work, "link.out")
        os.symlink(victim, link)
        for flag in ("--arena-out", "--trace"):
            argv = list(base) + (["--arena-out", link] if flag == "--arena-out"
                                 else ["--arena-out", out, "--trace", link])
            status, err = probe(*argv)
            self.assertEqual(status, 1, err)
            self.assertIn("symlink is refused", err)
        with open(victim) as f:
            self.assertEqual(f.read(), "untouched")

        # A run that fails leaves no output behind to be mistaken for one.
        os.remove(out)
        with open(arena, "ab") as f:
            f.write(b"\0")  # no longer the plan's arena size
        status, err = probe(*base, "--arena-out", out, "--trace", out + ".t")
        self.assertEqual(status, 1)
        self.assertIn("the plan declares", err)
        self.assertEqual(sorted(os.listdir(work)),
                         ["arena.in", "link.out", "victim"])

    def test_run_reproduces_the_emitted_package(self):
        """The whole update — tail split, per-epoch shuffle, schedule, G
        micro-batches, evaluation — against the real model_update binary:
        same loss curve, same validation losses."""
        out = os.path.join(self.dir, "e2e")
        done = subprocess.run(
            [COMPILER, "--source", os.path.join(self.dir, "decoder.smf"),
             "--out", out, "--data-batch", "12", "--grad-accum", "2",
             "--lr-schedule", "cosine", "--warmup", "3", "--steps", "25",
             "--build"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        binary = os.path.join(out, "model_update")
        if done.returncode != 0 or not os.path.exists(binary):
            # A missing toolchain is a skip; a package that does not build
            # under the toolchain that built the compiler is a failure.
            if shutil.which(os.environ.get("CXX", "c++")) is None:
                self.skipTest("no C++ toolchain to build the emitted package")
            self.fail(f"the emitted package did not build:\n{done.stdout[-3000:]}")
        corpus = os.path.join(self.dir, "decoder_corpus.sds")
        # 0.25 splits 12 validation records (whole batches of 2); 0.2
        # splits 9, so the final eval batch wraps and both sides must score
        # its real rows alone (G13, #97).
        for frac in ("0.25", "0.2"):
            with self.subTest(val_fraction=frac):
                self._run_matches_package(out, binary, corpus, frac)

    def _run_matches_package(self, out, binary, corpus, frac):
        log = os.path.join(out, "loss.csv")
        trained = subprocess.run(
            [binary, "--model", os.path.join(self.dir, "decoder.smf"),
             "--data", corpus, "--out", os.path.join(out, "updated.smf"),
             "--steps", "25", "--seed", "11", "--val-frac", frac,
             # The last state, as `frontier_exec run` trains it: the
             # best-state gate (E9) would commit an earlier one.
             "--eval-every", "0",
             "--loss-log", log, "--force"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        self.assertEqual(trained.returncode, 0, trained.stdout)
        before, after = re.search(r"validation loss (\S+) -> (\S+)",
                                  trained.stdout).groups()
        with open(log) as f:
            cpp = [float(line.split(",")[1]) for line in f.read().split()[1:]]
        report = os.path.join(out, "run.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fx.main([
                "run", os.path.join(out, "update_plan.seeu"), "--corpus",
                corpus, "--steps", "25", "--seed", "11", "--val-fraction",
                frac, "--log-every", "0", "--report", report]), 0)
        with open(report) as f:
            ran = json.load(f)
        self.assertEqual(len(cpp), 25)
        np.testing.assert_allclose(ran["loss_curve"], cpp, rtol=2e-5)
        self.assertAlmostEqual(ran["val_initial_loss"], float(before), 4)
        self.assertAlmostEqual(ran["val_final_loss"], float(after), 4)

    def test_run_trains_and_price_reports(self):
        plan, corpus, _ = self.plans["dec_f32"]
        report = os.path.join(self.dir, "run.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(fx.main(["run", plan, "--corpus", corpus,
                                      "--steps", "30", "--report", report,
                                      "--log-every", "0"]), 0)
        with open(report) as f:
            ran = json.load(f)
        self.assertLess(ran["loss_curve"][-1], ran["loss_curve"][0])
        self.assertLess(ran["val_final_loss"], ran["val_initial_loss"])

        with contextlib.redirect_stdout(buf):
            self.assertEqual(fx.main([
                "price", plan, "--backends", "numpy32", "--probe", PROBE,
                "--steps-lo", "2",
                "--steps-hi", "4", "--repeats", "1", "--peak-gflops", "100",
                "--report", report]), 0)
        with open(report) as f:
            priced = json.load(f)
        for entry in (priced["frontier"]["numpy32"], priced["cpp"]["cpu"]):
            self.assertTrue(entry["available"])
            for key in ("step_ms", "rows_per_s", "tokens_per_s", "it_per_s",
                        "gemm_gflops", "mfu"):
                self.assertGreater(entry[key], 0)
        self.assertIn("speedup_vs_cpp_cpu", priced["frontier"]["numpy32"])


@unittest.skipIf(np is None, "NumPy not installed")
class DifferTest(unittest.TestCase):
    def test_a_write_outside_the_declared_extents_is_a_mismatch(self):
        """CR2 #143 N33: the per-extent comparison never looked outside the
        extents the validator declared written, so a kernel writing past
        them agreed. A changed byte outside every declared extent is a
        structural mismatch, named by its first byte."""
        differ = fx.Differ(np, 1e-6, 1e-6)
        before = bytes(64)
        after = bytearray(before)
        after[10] = 1   # inside the declared extent at [8, 12)
        after[40] = 1   # outside every declared extent
        trace = [(0, 5, [(8, np.zeros(1, "<f4"))], 0)]
        differ.outside("step 1 train", before, bytes(after), trace)
        self.assertEqual(len(differ.mismatches), 1)
        self.assertIn("outside its declared write extents (first at byte 40)",
                      differ.mismatches[0])
        clean = fx.Differ(np, 1e-6, 1e-6)
        clean.outside("step 1 train", before, bytes(after),
                      trace + [(1, 5, [(40, np.zeros(1, "<f4"))], 0)])
        self.assertEqual(clean.mismatches, [])


if __name__ == "__main__":
    unittest.main()
