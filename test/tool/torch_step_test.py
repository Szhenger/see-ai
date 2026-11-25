"""The compiled train step next to PyTorch (review 2026-11-25).

Everything else in the tree that checks the compiler's differentiation is
the compiler's own: the system suite's finite differences read the loss the
plan computes, and the NumPy oracle (frontier_exec.py) replays the plan the
compiler emitted, instruction by instruction — it proves the kernels, not
the calculus that chose them. This suite holds the product to a framework
that shares none of its code: a model is written here, in torch, in
float64; the same weights are exported and compiled with LoRA adapters; the
plan's train program runs for a few steps on one staged batch
(seeml-plan-probe: the executor alone, no feeder, no gate); and the
adapters it leaves are compared with what `torch.autograd` and
`torch.optim` leave from the same start.

What is read from the plan is read as bytes, through `seeml-seeu-dump
--json`: the I/O slots, the persistent segment's initial image (the
adapters as the compiler initialized them), and — from the merge program,
whose every delta is `scale * A @ B` — where each adapter's two factors
live and what its scale is. No opcode is interpreted here.

Tier 2 (torch). The e2e CI job installs torch and fails if this suite
skips there."""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tool"))

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILE = os.path.join(BUILD, "seeml-update-compile")
PROBE = os.path.join(BUILD, "seeml-plan-probe")
DUMP = os.path.join(BUILD, "seeml-seeu-dump")
GEMM_ACC_NN = 4  # the merge program's delta: C += alpha * A @ B

# Agreement, relative. The compiled step is f32 and torch here is f64, so
# the floor is f32 rounding through the graph: measured on an Apple M5,
# 3e-8 on the loss, 6e-7 on a gradient, 5e-5 on a parameter's movement
# after a few steps. The bounds leave two decades for other compilers and
# libms and are still far below what a wrong rule costs — a missing term or
# a wrong scale moves a gradient by its own size.
LOSS_RTOL = 1e-5
GRAD_RTOL = 1e-4
STEP_RTOL = 2e-3


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


class CompiledPlan:
    """A compiled package's plan, as the bytes the device would load."""

    def __init__(self, directory, smf, flags):
        self.dir = directory
        pkg = os.path.join(directory, "pkg")
        done = subprocess.run(
            [COMPILE, "--source", smf, "--out", pkg + "/"] + flags,
            capture_output=True, text=True)
        if done.returncode != 0:
            raise AssertionError("compile failed: " + done.stderr[-2000:])
        self.path = os.path.join(pkg, "update_plan.seeu")
        dump = subprocess.run([DUMP, self.path, "--json"], check=True,
                              capture_output=True, text=True)
        self.doc = json.loads(dump.stdout)
        self.header = self.doc["header"]
        with open(self.path, "rb") as f:
            self.blob = f.read()

    def initial_arena(self):
        """The engine's arena at load: zeros under the persistent image."""
        h = self.header
        arena = bytearray(h["arena_size"])
        o, n = h["persist_init_offset"], h["persist_init_size"]
        arena[:n] = self.blob[o:o + n]
        return arena

    def adapters(self):
        """(A offset, B offset, scale) per emitted delta, in the order of
        the weights in the model file."""
        by_delta = {}
        for ins in self.doc["sections"]["merge"]:
            if ins["opcode"] == GEMM_ACC_NN:
                by_delta[ins["in"][2]] = (ins["in"][0], ins["in"][1],
                                          bits_f32(ins["in"][3]))
        emit = sorted(self.doc["emit"], key=lambda e: e["smf_data_offset"])
        return [by_delta[e["arena_offset"]] for e in emit]

    def train(self, arena, step, lr, backend="cpu"):
        """One execution of the train program over `arena`; the arena it
        leaves."""
        a_in = os.path.join(self.dir, "arena_in.bin")
        a_out = os.path.join(self.dir, "arena_out.bin")
        with open(a_in, "wb") as f:
            f.write(arena)
        done = subprocess.run(
            [PROBE, "--plan", self.path, "--section", "train", "--arena-in",
             a_in, "--arena-out", a_out, "--lr-bits", hex(f32_bits(lr)),
             "--step", str(step), "--backend", backend],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise AssertionError("probe failed: " + done.stderr[-2000:])
        with open(a_out, "rb") as f:
            return bytearray(f.read())


def read_f32(buf, offset, shape):
    count = int(np.prod(shape))
    return np.frombuffer(bytes(buf[offset:offset + 4 * count]),
                         "<f4").reshape(shape).astype(np.float64)


def t64(a):
    return torch.tensor(np.asarray(a, np.float64))


class Case:
    """One model in two forms: the SMF the compiler reads and the torch
    forward of the same weights, with `weights` the adapted matmul weights
    ([in, out]) in file order."""

    def stage(self, arena, header):
        raise NotImplementedError

    def loss(self, effective):
        raise NotImplementedError


class DecoderCase(Case):
    """A two-block token decoder: embedding, RMSNorm with a learned gain,
    interleaved RoPE, causal attention, SwiGLU, cross-entropy on the next
    token — the graph `--hf` imports produce."""

    def __init__(self, directory, seed=3):
        import export_model as em
        self.em = em
        self.vocab, self.dim, self.heads, self.seq = 41, 32, 4, 8
        self.batch, self.ffn, n_blocks = 4, 48, 2
        rng = np.random.default_rng(seed)

        def mat(r, c, s=0.15):
            return (rng.standard_normal((r, c)) * s).astype(np.float32)

        def gain():
            return (1.0 + 0.2 * rng.standard_normal(self.dim)).astype(
                np.float32)

        D, Fn = self.dim, self.ffn
        self.blocks = [{
            "ln1_g": gain(), "wq": mat(D, D), "wk": mat(D, D),
            "wv": mat(D, D), "wo": mat(D, D), "ln2_g": gain(),
            "w_gate": mat(D, Fn), "w_up": mat(D, Fn), "w_down": mat(Fn, D),
        } for _ in range(n_blocks)]
        self.head = {"lnf_g": gain(), "w_head": mat(D, self.vocab)}
        self.emb = mat(self.vocab, D, 0.5)
        self.smf = os.path.join(directory, "decoder.smf")
        em.export_token_decoder_smf(self.emb, self.blocks, self.head,
                                    self.smf, seq_len=self.seq,
                                    num_heads=self.heads)
        self.names = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
        self.weights = [b[k] for b in self.blocks for k in self.names]
        self.weights.append(self.head["w_head"])
        records = rng.integers(0, self.vocab, (self.batch, self.seq + 1))
        self.tokens = records[:, :-1].reshape(-1).astype("<i4")
        self.labels = records[:, 1:].reshape(-1).astype("<i4")
        self.data_batch = self.batch * self.seq

    def stage(self, arena, header):
        for ref, data in ((header["input_ref"], self.tokens),
                          (header["label_ref"], self.labels)):
            arena[ref:ref + data.nbytes] = data.tobytes()

    def loss(self, effective):
        B, S, H, D = self.batch, self.seq, self.heads, self.dim
        d = D // H
        eps = float(self.em.DEFAULT_NORM_EPS)

        def rms(v, g):
            return v / torch.sqrt((v * v).mean(-1, keepdim=True) + eps) * t64(g)

        freq = float(self.em.DEFAULT_ROPE_BASE) ** (
            -2.0 * torch.arange(d // 2, dtype=torch.float64) / d)
        theta = torch.arange(S, dtype=torch.float64)[:, None] * freq[None]
        cs, sn = theta.cos()[None, :, None, :], theta.sin()[None, :, None, :]

        def rope(v):
            v = v.reshape(B, S, H, d)
            a, b = v[..., 0::2], v[..., 1::2]
            return torch.stack((a * cs - b * sn, a * sn + b * cs),
                               -1).reshape(B, S, D)

        def split(v):
            return v.reshape(B, S, H, d).transpose(1, 2)

        x = t64(self.emb)[torch.tensor(self.tokens.astype(np.int64))]
        x = x.reshape(B, S, D)
        future = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
        for i, blk in enumerate(self.blocks):
            wq, wk, wv, wo, wg, wu, wd = effective[7 * i:7 * i + 7]
            n1 = rms(x, blk["ln1_g"])
            q, k, v = rope(n1 @ wq), rope(n1 @ wk), n1 @ wv
            scores = (split(q) @ split(k).transpose(-1, -2)) / d ** 0.5
            p = torch.softmax(scores.masked_fill(future, float("-inf")), -1)
            x = x + (p @ split(v)).transpose(1, 2).reshape(B, S, D) @ wo
            n2 = rms(x, blk["ln2_g"])
            x = x + (F.silu(n2 @ wg) * (n2 @ wu)) @ wd
        logits = rms(x, self.head["lnf_g"]) @ effective[-1]
        return F.cross_entropy(logits.reshape(-1, self.vocab),
                               torch.tensor(self.labels.astype(np.int64)))


class MlpCase(Case):
    """A feed-forward classifier over every operator the decoder does not
    use: AddBias, LayerNorm, GELU (tanh form), ReLU — and SiLU again."""

    def __init__(self, directory, seed=5):
        import export_model as em
        import torch.nn as nn
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(12, 24), nn.GELU(approximate="tanh"), nn.LayerNorm(24),
            nn.Linear(24, 20), nn.ReLU(), nn.Linear(20, 16), nn.SiLU(),
            nn.Linear(16, 7))
        with torch.no_grad():  # a LayerNorm that is not the identity's
            self.model[2].weight.add_(0.3 * torch.randn(24))
            self.model[2].bias.add_(0.3 * torch.randn(24))
        self.smf = os.path.join(directory, "mlp.smf")
        em.export_smf(self.model, self.smf)
        self.linears = [m for m in self.model if isinstance(m, nn.Linear)]
        self.weights = [m.weight.detach().numpy().T.copy()
                        for m in self.linears]
        rng = np.random.default_rng(seed)
        self.data_batch = 16
        self.x = rng.standard_normal((self.data_batch, 12)).astype("<f4")
        self.y = rng.integers(0, 7, self.data_batch).astype("<i4")

    def stage(self, arena, header):
        for ref, data in ((header["input_ref"], self.x),
                          (header["label_ref"], self.y)):
            arena[ref:ref + data.nbytes] = data.tobytes()

    def loss(self, effective):
        import torch.nn as nn
        h, weights = t64(self.x), iter(effective)
        for m in self.model:
            if isinstance(m, nn.Linear):
                h = h @ next(weights) + t64(m.bias.detach().numpy())
            elif isinstance(m, nn.LayerNorm):
                h = F.layer_norm(h, (h.shape[-1],),
                                 t64(m.weight.detach().numpy()),
                                 t64(m.bias.detach().numpy()), m.eps)
            elif isinstance(m, nn.GELU):
                h = F.gelu(h, approximate="tanh")
            elif isinstance(m, nn.ReLU):
                h = F.relu(h)
            else:
                h = F.silu(h)
        return F.cross_entropy(h, torch.tensor(self.y.astype(np.int64)))


def skip_reason():
    if np is None:
        return "NumPy not installed"
    if torch is None:
        return "PyTorch not installed"
    for tool in (COMPILE, PROBE, DUMP):
        if not os.path.isfile(tool):
            return "no built " + os.path.basename(tool)
    return None


@unittest.skipIf(skip_reason(), skip_reason())
class TorchStepTest(unittest.TestCase):
    RANK, ALPHA = 4, 8.0

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="seeml-torch-step-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def run_case(self, case, optimizer, lr, steps, extra=(), backend="cpu"):
        flags = ["--data-batch", str(case.data_batch), "--loss", "xent",
                 "--lora-rank", str(self.RANK), "--lora-alpha",
                 str(self.ALPHA), "--optimizer", optimizer, "--lr", repr(lr),
                 "--lr-schedule", "const", "--weight-decay",
                 "0.01" if optimizer == "adamw" else "0", "--steps",
                 str(steps)] + list(extra)
        plan = CompiledPlan(self.dir, case.smf, flags)
        adapters = plan.adapters()
        self.assertEqual(len(adapters), len(case.weights))
        start = plan.initial_arena()

        def factors(arena):
            out = []
            for w, (a_off, b_off, _) in zip(case.weights, adapters):
                k, m = w.shape
                out.append((read_f32(arena, a_off, (k, self.RANK)),
                            read_f32(arena, b_off, (self.RANK, m))))
            return out

        # The compiled steps.
        arena, first_loss, after_one = start, None, None
        for step in range(1, steps + 1):
            staged = bytearray(arena)
            case.stage(staged, plan.header)
            arena = plan.train(staged, step, lr, backend)
            if step == 1:
                ref = plan.header["loss_ref"]
                first_loss = struct.unpack_from("<f", arena, ref)[0]
                after_one = factors(arena)
        compiled = factors(arena)

        # The same steps in torch, from the same adapters.
        initial = factors(start)
        A = [t64(a).requires_grad_() for a, _ in initial]
        B = [t64(b).requires_grad_() for _, b in initial]
        scales = [s for _, _, s in adapters]
        self.assertTrue(all(abs(s - self.ALPHA / self.RANK) < 1e-6
                            for s in scales), scales)

        def loss():
            return case.loss([t64(w) + s * a @ b for w, s, a, b in
                              zip(case.weights, scales, A, B)])

        params = A + B
        opt = (torch.optim.SGD(params, lr=lr) if optimizer == "sgd" else
               torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0.01))
        grads = None
        for step in range(steps):
            opt.zero_grad()
            value = loss()
            value.backward()
            if step == 0:
                torch_loss = value.item()
                grads = [(a.grad.numpy().copy(), b.grad.numpy().copy())
                         for a, b in zip(A, B)]
            opt.step()

        self.assertLess(abs(first_loss - torch_loss),
                        LOSS_RTOL * abs(torch_loss))
        nonzero = 0
        for i in range(len(adapters)):
            for f in (0, 1):
                name = "adapter %d factor %s" % (i, "AB"[f])
                want = (A, B)[f][i].detach().numpy()
                moved = np.abs(want - initial[i][f]).max()
                # Every factor trains by the last step (LoRA starts one
                # factor of each pair at zero, so the other's first
                # gradient is zero in both systems).
                self.assertGreater(moved, 0.0, name)
                self.assertLess(np.abs(compiled[i][f] - want).max(),
                                STEP_RTOL * moved, name)
                if optimizer != "sgd":
                    continue
                # Under SGD with no decay the first step IS the gradient.
                g = grads[i][f]
                got = (initial[i][f] - after_one[i][f]) / lr
                if np.abs(g).max() == 0.0:
                    self.assertEqual(np.abs(got).max(), 0.0, name)
                    continue
                nonzero += 1
                self.assertLess(np.abs(got - g).max(),
                                GRAD_RTOL * np.abs(g).max(), name)
        if optimizer == "sgd":
            self.assertEqual(nonzero, len(adapters))

    def test_decoder_sgd_gradients_and_steps_match_autograd(self):
        self.run_case(DecoderCase(self.dir), "sgd", 0.5, 3)

    def test_decoder_adamw_steps_match_torch_optim(self):
        self.run_case(DecoderCase(self.dir), "adamw", 0.01, 5)

    def test_decoder_tiled_attention_matches_autograd(self):
        # The tiled family recomputes the probabilities in the backward;
        # the function, and so the reference, is the cached family's.
        self.run_case(DecoderCase(self.dir), "sgd", 0.5, 3,
                      extra=["--attention", "tiled"])

    def test_decoder_unfused_plan_matches_autograd(self):
        # The plan before the fusion passes: the same calculus through
        # separate instructions.
        self.run_case(DecoderCase(self.dir), "sgd", 0.5, 3,
                      extra=["--no-fuse-epilogue", "--no-fuse-addend",
                             "--no-fuse-elementwise", "--no-rope-table"])

    def test_mlp_sgd_gradients_and_steps_match_autograd(self):
        self.run_case(MlpCase(self.dir), "sgd", 0.2, 3)

    def test_mlp_adamw_steps_match_torch_optim(self):
        self.run_case(MlpCase(self.dir), "adamw", 0.01, 5)

    @unittest.skipIf(sys.platform != "darwin", "the Metal backend is Apple's")
    def test_decoder_on_metal_matches_autograd(self):
        try:
            self.run_case(DecoderCase(self.dir), "adamw", 0.01, 5,
                          backend="metal")
        except AssertionError as e:
            # Only a host without the backend is a skip (metal_backend.mm,
            # metal_backend_stub.cc): any other Metal failure is the test's.
            if "probe failed" in str(e) and (
                    "no Metal device" in str(e) or
                    "Metal backend is not built" in str(e)):
                self.skipTest("no Metal device: " + str(e)[-200:])
            raise


if __name__ == "__main__":
    unittest.main()
