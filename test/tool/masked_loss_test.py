"""The masked cross-entropy (S3 #149, opcodes 51 / 52) against PyTorch.

The oracle's kernels (tool/frontier_exec.py, the reference the C++ CPU and
Metal kernels are diffed against instruction by instruction) are held to
`torch.nn.functional.cross_entropy(..., ignore_index=-1)` and its autograd
gradient on random logits with ignored rows, to 1e-5 (tier 2: needs torch);
tier 0 checks the two kernels against each other and the plain pair.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tool"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
import frontier_exec as fx  # noqa: E402


class FakeMemory:
    """The oracle kernels' memory: refs are plain keys into a dict."""

    def __init__(self, backend, tensors):
        self.x, self.t = backend, dict(tensors)

    def read(self, ref, shape, dtype="<f4"):
        return self.t[ref].reshape(shape)

    def write(self, ref, value):
        self.t[ref] = np.asarray(value)


class Ins:
    def __init__(self, src, out):
        self.src, self.out = src, out


@unittest.skipIf(np is None, "NumPy not installed")
class MaskedLossTest(unittest.TestCase):
    def setUp(self):
        self.x = fx.NumpyBackend()
        rng = np.random.default_rng(7)
        self.n, self.c = 8, 5
        self.logits = rng.standard_normal((self.n, self.c)).astype(np.float64)
        labels = rng.integers(0, self.c, self.n).astype("<i4")
        labels[[1, 4, 6]] = -1
        self.labels = labels
        self.seed = np.array([1.5])

    def run_pair(self, fwd, bwd, labels):
        m = FakeMemory(self.x, {"logits": self.logits, "labels": labels,
                                "seed": self.seed})
        fwd(m, self.x, Ins(["logits", "labels", "loss", "probs"],
                           [self.n, self.c]), 0)
        bwd(m, self.x, Ins(["probs", "labels", "seed", "dlogits"],
                           [self.n, self.c]), 0)
        return float(m.t["loss"].reshape(-1)[0]), m.t["dlogits"]

    def test_the_masked_pair_is_the_plain_pair_over_the_target_rows(self):
        loss, d = self.run_pair(fx._xent_masked_fwd, fx._xent_masked_bwd,
                                self.labels)
        keep = self.labels >= 0
        self.logits_kept = self.logits[keep]
        m = FakeMemory(self.x, {"logits": self.logits[keep],
                                "labels": self.labels[keep], "seed": self.seed})
        k = int(keep.sum())
        fx._xent_fwd(m, self.x, Ins(["logits", "labels", "loss", "probs"],
                                    [k, self.c]), 0)
        fx._xent_bwd(m, self.x, Ins(["probs", "labels", "seed", "dlogits"],
                                    [k, self.c]), 0)
        self.assertAlmostEqual(loss, float(m.t["loss"].reshape(-1)[0]), 12)
        np.testing.assert_allclose(d[keep], m.t["dlogits"], rtol=0, atol=1e-12)
        self.assertTrue((d[~keep] == 0).all())
        # No targets at all: loss 0, gradient 0.
        none = np.full(self.n, -1, "<i4")
        loss0, d0 = self.run_pair(fx._xent_masked_fwd, fx._xent_masked_bwd, none)
        self.assertEqual(loss0, 0.0)
        self.assertTrue((d0 == 0).all())
        # No mask: identical to the plain pair.
        full = np.abs(self.labels).astype("<i4")
        lm, dm = self.run_pair(fx._xent_masked_fwd, fx._xent_masked_bwd, full)
        lp, dp = self.run_pair(fx._xent_fwd, fx._xent_bwd, full)
        self.assertEqual(lm, lp)
        np.testing.assert_array_equal(dm, dp)

    def test_the_masked_pair_matches_pytorch_ignore_index(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("tier 2: needs torch")
        loss, d = self.run_pair(fx._xent_masked_fwd, fx._xent_masked_bwd,
                                self.labels)
        logits = torch.tensor(self.logits, dtype=torch.float64,
                              requires_grad=True)
        labels = torch.tensor(self.labels.astype(np.int64))
        ref = F.cross_entropy(logits, labels, ignore_index=-1)
        (grad,) = torch.autograd.grad(ref * float(self.seed[0]), logits)
        self.assertAlmostEqual(loss, float(ref), 5)
        np.testing.assert_allclose(d, grad.numpy(), rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
