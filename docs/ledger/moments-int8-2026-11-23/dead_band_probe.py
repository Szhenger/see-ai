"""#192: the 8-bit moments' dead band, and what the dithered code does to it.

One block of 256 AdamW moments in f32 arithmetic, coded after every step
and compared with uncoded f32 moments over the same gradients. Five codings:

  first      the rule first measured here: m linear in 127 levels a side of
             the block's absmax, v the eight-levels-an-octave code, both
             rounded to the NEAREST level (reimplemented below);
  linear     the same m code with the unbiased dither (round up with the
             probability of the fraction of the way to the upper level) —
             v coded by the oracle;
  nearest    the shipped levels (m and v both on the eight-levels-an-octave
             code) rounded to the nearest level in the log domain: a finer
             m code, the first rule's rounding;
  kronecker  the shipped code with a dither that is regular in the step —
             the element's hashed phase plus the step times 0x9E3779B9,
             mod 2^32 — in place of an independent hash every step;
  shipped    tool/frontier_exec.py's moment_quantize.

Element 0 keeps gradient 1 and so holds the block's scales; the other 255
are the starved ones every number of the first table is about.

  starved  0.3 for 10 steps, then N(0, 1e-3) noise           (m's dead band)
  fall     0.3 for 200 steps, then exactly zero              (v must decay)
  step     0.01 for half the steps, then 0.06, no weight decay
           (m and v must both rise)
  nudge    0.010 for half the steps, then 0.013, no weight decay
           (a change under half a level of any code)

The second table is the common case, a block with no large neighbour: 256
gradients N(0, 0.1) + 0.05 p0 (test/runtime/executor/kernels_test.cc's
ManyStepsTrackF32AdamW, lr 0.01), and how far the coded run's parameters
end from the f32 run's, over what the f32 run moved.

    python3 dead_band_probe.py REPO [STEPS]
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1] + "/tool")
import frontier_exec as fx  # noqa: E402

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
f32 = np.float32
B1, B2, EPS = f32(0.9), f32(0.999), f32(1e-8)
N = 256
SALT = 8192  # the m codes' operand offset: any constant
hash_dither = fx.moment_dither


def nearest_level(x, levels):
    """Each x > 0 on the nearest of `levels` in the log domain."""
    with np.errstate(divide="ignore"):
        at = np.abs(np.log(levels)[None, :] - np.log(x)[:, None]).argmin(1)
    return np.where(x > 0, levels[at], 0).astype(f32)


def first_quantize(m, v, step):
    """Round to nearest: m in 127 linear levels, v in the log code."""
    amax, vmax = np.abs(m).max(), v.max()
    d = f32(amax / f32(127.0))
    qm = np.clip(np.rint(m / d), -127, 127) * d if amax > 0 else m * 0
    levels = fx.moment_level_value(np, np.arange(255), np.full(255, vmax, f32))
    return qm.astype(f32), nearest_level(v, levels)


def linear_quantize(m, v, step):
    """The linear m code with the unbiased dither; the oracle's v."""
    amax = np.abs(m).max()
    d = f32(amax / f32(127.0))
    x = (m / d).astype(f32)
    low = np.floor(x)
    cut = ((x - low) * f32(1 << 24)).astype(np.uint32)
    u = fx.moment_dither(np, step, SALT, fx.formats.MOMENT_DITHER_M, N)
    qm = ((low + (u < cut)) * d).astype(f32)
    return qm, shipped_quantize(m, v, step)[1]


def nearest_quantize(m, v, step):
    """The shipped levels of both codes, rounded to nearest."""
    amax, vmax = np.abs(m).max(), v.max()
    m_levels = fx.moment_level_value(np, np.arange(127),
                                     np.full(127, amax, f32))
    v_levels = fx.moment_level_value(np, np.arange(255),
                                     np.full(255, vmax, f32))
    return (np.sign(m) * nearest_level(np.abs(m), m_levels)).astype(f32), \
        nearest_level(v, v_levels)


def kronecker_dither(np_, step, salt, stream, count):
    """A dither regular in the step: phase(element) + step * golden."""
    mask = np.uint64(0xFFFFFFFF)
    phase = hash_dither(np_, 0, salt, stream, count).astype(np.uint64) << \
        np.uint64(8)
    turn = np.uint64((step * 0x9E3779B9) & 0xFFFFFFFF)
    return (((phase + turn) & mask) >> np.uint64(8)).astype(np.uint32)


def kronecker_quantize(m, v, step):
    fx.moment_dither = kronecker_dither
    try:
        return shipped_quantize(m, v, step)
    finally:
        fx.moment_dither = hash_dither


def shipped_quantize(m, v, step):
    return fx.moment_dequantize(np, *fx.moment_quantize(np, m, v, step, SALT))


CODINGS = (("first", first_quantize), ("linear", linear_quantize),
           ("nearest", nearest_quantize), ("kronecker", kronecker_quantize),
           ("shipped", shipped_quantize))


def gradient(scenario, t, rng):
    g = np.empty(N, f32)
    if scenario == "starved":
        g[:] = 0.3 if t <= 10 else rng.normal(0, 1e-3, N)
    elif scenario == "fall":
        g[:] = 0.3 if t <= 200 else 0.0
    elif scenario == "step":
        g[:] = 0.01 if t <= STEPS // 2 else 0.06
    else:
        g[:] = 0.010 if t <= STEPS // 2 else 0.013
    g[0] = 1.0
    return g


def adamw(p, m, v, g, t, wd, lr):
    m = (B1 * m + (1 - B1) * g).astype(f32)
    v = (B2 * v + (1 - B2) * g * g).astype(f32)
    update = (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + EPS)
    return (p - lr * (update + wd * p)).astype(f32), m, v, update


def run(scenario, quantize, wd):
    rng = np.random.default_rng(0)
    lr = f32(1e-3)
    p, m, v = (np.zeros(N, f32) for _ in range(3))
    p8, m8, v8 = (np.zeros(N, f32) for _ in range(3))
    err = ref = 0.0
    for t in range(1, STEPS + 1):
        g = gradient(scenario, t, rng)
        p, m, v, u = adamw(p, m, v, g, t, wd, lr)
        p8, m1, v1, u8 = adamw(p8, m8, v8, g, t, wd, lr)
        m8, v8 = quantize(m1, v1, t)
        if t > 10:
            err += float(((u8[1:] - u[1:]) ** 2).mean())
            ref += float((u[1:] ** 2).mean())
    s = slice(1, None)
    with np.errstate(divide="ignore"):
        scatter = np.exp(np.sqrt((np.log(v8[s] / v[s]) ** 2).mean()))
    return (np.abs(p[s]).mean(), np.abs(p8[s]).mean(),
            np.abs(p8[s] - p[s]).mean(), (err / ref) ** 0.5,
            np.abs(m[s]).mean(), np.abs(m8[s]).mean(),
            v[s].mean(), v8[s].mean(), scatter)


def even(quantize, steps):
    """apart / moved of a block with no large neighbour."""
    rng = np.random.default_rng(57)
    start = rng.normal(0, 1, N).astype(f32)
    lr, wd = f32(0.01), f32(0.01)
    p, p8 = start.copy(), start.copy()
    m, v, m8, v8 = (np.zeros(N, f32) for _ in range(4))
    for t in range(1, steps + 1):
        g = (rng.normal(0, 0.1, N) + 0.05 * start).astype(f32)
        p, m, v, _ = adamw(p, m, v, g, t, wd, lr)
        p8, m1, v1, _ = adamw(p8, m8, v8, g, t, wd, lr)
        m8, v8 = quantize(m1, v1, t)
    return float(np.abs(p8 - p).sum() / np.abs(p - start).sum())


print(f"{STEPS} steps, lr 1e-3, the 255 starved elements; update error is "
      "the RMS of (8-bit update - f32 update) over the RMS of the f32 "
      "update, steps 11 on; v scatter is the RMS factor between an "
      "element's coded v and its f32 v")
for scenario, wd in (("starved", f32(0.01)), ("fall", f32(0.01)),
                     ("step", f32(0.0)), ("nudge", f32(0.0))):
    for name, quantize in CODINGS:
        r = run(scenario, quantize, wd)
        print(f"{scenario:8s} {name:9s} mean |p| f32 {r[0]:.4f} 8-bit "
              f"{r[1]:.4f}  mean |p8 - p32| {r[2]:.4f}  update error "
              f"{r[3]:6.3f}  mean |m| f32 {r[4]:.3g} 8-bit {r[5]:.3g}  "
              f"mean v f32 {r[6]:.3g} 8-bit {r[7]:.3g}  v scatter "
              f"x{r[8]:.2f}")
print("a block with no large neighbour, lr 0.01: |p8 - p32| summed over "
      "|p32 - p0| summed")
for name, quantize in CODINGS:
    print(f"even     {name:9s} " + "  ".join(
        f"{steps} steps {even(quantize, steps):.4f}"
        for steps in (50, 500, 3000)))
