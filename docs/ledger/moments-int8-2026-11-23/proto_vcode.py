"""Prototype (#192): which v code? AdamW on a small LoRA-shaped regression
with f32 moments vs block-wise int8 moments under three v codes. Reports the
final loss and how often a v code dequantizes to 0 while m does not (the
elements that then step by m / eps)."""
import numpy as np

B = 256
F32 = np.float32


def q_m(m):
    out = np.zeros_like(m, dtype=np.int8)
    sc = np.zeros((m.size + B - 1) // B, F32)
    for b in range(sc.size):
        blk = m[b * B:(b + 1) * B]
        a = F32(np.max(np.abs(blk)))
        sc[b] = a
        if a > 0:
            d = F32(a / F32(127))
            out[b * B:(b + 1) * B] = np.clip(np.rint(blk / d), -127, 127)
    return out, sc


def dq_m(q, sc):
    out = np.zeros(q.size, F32)
    for b in range(sc.size):
        d = F32(sc[b] / F32(127))
        out[b * B:(b + 1) * B] = q[b * B:(b + 1) * B].astype(F32) * d
    return out


K = 8
F = np.array([2.0 ** (-r / K) for r in range(K)], F32)
H = np.array([2.0 ** (1 - (j + 0.5) / K) for j in range(K)], F32)


def q_v(v, code):
    out = np.zeros(v.size, np.uint8)
    sc = np.zeros((v.size + B - 1) // B, F32)
    for b in range(sc.size):
        blk = v[b * B:(b + 1) * B]
        a = F32(np.max(blk))
        sc[b] = a
        if not a > 0:
            continue
        if code == "linear":
            q = np.clip(np.rint(blk / F32(a / F32(255))), 0, 255)
        elif code == "sqrt":
            s = np.sqrt(blk / a)
            q = np.clip(np.rint(s * F32(255)), 0, 255)
        else:  # log2, K levels per octave, nonzero never maps to 0
            x = (blk / a).astype(F32)
            q = np.zeros(blk.size)
            nz = blk > 0
            mant, ex = np.frexp(x[nz])  # x = mant * 2^ex, mant in [0.5, 1)
            y = (mant * 2).astype(F32)  # [1, 2)
            E = ex - 1
            cnt = (y[:, None] < H[None, :]).sum(1)
            n = np.minimum(-8 * (E + 1) + cnt, 254)
            q[nz] = 255 - n
        out[b * B:(b + 1) * B] = q
    return out, sc


def dq_v(q, sc, code):
    out = np.zeros(q.size, F32)
    for b in range(sc.size):
        a = sc[b]
        qq = q[b * B:(b + 1) * B].astype(np.int64)
        if code == "linear":
            out[b * B:(b + 1) * B] = qq.astype(F32) * F32(a / F32(255))
        elif code == "sqrt":
            s = qq.astype(F32) * F32(1 / 255)
            out[b * B:(b + 1) * B] = (s * s) * a
        else:
            n = 255 - qq
            val = np.where(qq == 0, F32(0),
                           a * np.ldexp(F[n % K], -(n // K)).astype(F32))
            out[b * B:(b + 1) * B] = val
    return out


def run(code, seed, steps=600, lr=2e-3):
    rng = np.random.default_rng(seed)
    d_in, d_out, r, n = 64, 48, 8, 512
    W0 = rng.standard_normal((d_in, d_out)).astype(F32) * 0.3
    X = rng.standard_normal((n, d_in)).astype(F32)
    # target: a low-rank change with a heavy-tailed column scale, so the
    # gradient magnitudes span decades inside a block
    U = rng.standard_normal((d_in, r)).astype(F32)
    V = (rng.standard_normal((r, d_out)) *
         np.exp(rng.standard_normal(d_out) * 1.5)).astype(F32)
    Y = X @ (W0 + 0.1 * U @ V)
    A = (rng.standard_normal((d_in, r)) * 0.02).astype(F32)
    Bm = np.zeros((r, d_out), F32)
    params = [A, Bm]
    state = [None, None]
    zero_v_nonzero_m = 0
    for t in range(1, steps + 1):
        P = X @ (W0 + A @ Bm)
        E = (P - Y) / n
        loss = float((E * E).sum() * n)
        gA = (X.T @ (2 * E)) @ Bm.T
        gB = A.T @ (X.T @ (2 * E))
        for i, (p, g) in enumerate(zip(params, [gA, gB])):
            g = g.reshape(-1).astype(F32)
            if state[i] is None:
                if code == "f32":
                    state[i] = (np.zeros_like(g), np.zeros_like(g))
                else:
                    state[i] = (q_m(np.zeros_like(g)), q_v(np.zeros_like(g), code))
            if code == "f32":
                m0, v0 = state[i]
            else:
                (mq, ms), (vq, vs) = state[i]
                m0, v0 = dq_m(mq, ms), dq_v(vq, vs, code)
                zero_v_nonzero_m += int(np.sum((v0 == 0) & (m0 != 0)))
            m1 = F32(0.9) * m0 + F32(0.1) * g
            v1 = F32(0.999) * v0 + F32(0.001) * g * g
            mh = m1 / F32(1 - 0.9 ** t)
            vh = v1 / F32(1 - 0.999 ** t)
            flat = p.reshape(-1)
            flat -= F32(lr) * (mh / (np.sqrt(vh) + F32(1e-8)))
            if code == "f32":
                state[i] = (m1, v1)
            else:
                state[i] = (q_m(m1), q_v(v1, code))
    return loss, zero_v_nonzero_m


for seed in (0, 1):
    for code in ("f32", "linear", "sqrt", "log"):
        loss, bad = run(code, seed)
        print(f"seed {seed} {code:7s} final loss {loss:.6f}  v=0 & m!=0: {bad}")
