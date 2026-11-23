"""#192: the v code's loss effect on a toy fixture, through the oracle.

Runs one --moments int8 plan with tool/frontier_exec.py's interpreter
(numpy32) three times: with the shipped logarithmic v code, and with the
oracle's quantizer swapped for a linear v code and a sqrt-domain linear v
code (m stays the shipped code in all three; when vcode_toy.txt was written
that was signed linear and round-to-nearest — see the README note in
docs/benchmarks.md); then the f32-moment plan. Prints
each run's final held-out loss and how many coded elements dequantize v to
0 while m is not 0 — the elements that then step by m / eps.

    python3 vcode_toy.py REPO PLAN_F32 PLAN_INT8 MODEL.smf CORPUS.sds
"""
import sys

sys.path.insert(0, sys.argv[1] + "/tool")
import frontier_exec as fx  # noqa: E402
import numpy as np  # noqa: E402

REPO, PLAN32, PLAN8, MODEL, CORPUS = sys.argv[1:6]
B = fx.formats.MOMENT_BLOCK
shipped_q, shipped_dq = fx.moment_quantize, fx.moment_dequantize
counter = {"bad": 0}


def variant(kind):
    def quantize(np_, m1, v1, step, salt):
        qm, _, scales = shipped_q(np_, m1, np.zeros_like(v1), step, salt)
        n = v1.size
        blocks = (n + B - 1) // B
        vp = np.zeros(blocks * B, np.float32)
        vp[:n] = v1
        vp = vp.reshape(blocks, B)
        vmax = vp.max(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.nan_to_num(vp / vmax[:, None]).astype(np.float32)
        x = np.sqrt(x) if kind == "sqrt" else x
        qv = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
        scales[blocks:] = vmax
        return qm, qv.reshape(-1)[:n], scales

    def dequantize(np_, mq, vq, scales):
        m0, _ = shipped_dq(np_, mq, np.zeros_like(vq), scales)
        blocks = scales.size // 2
        blk = np.arange(mq.size) // B
        x = vq.astype(np.float32) / np.float32(255)
        x = x * x if kind == "sqrt" else x
        v0 = (x * scales[blocks:][blk]).astype(np.float32)
        counter["bad"] += int(((v0 == 0) & (m0 != 0)).sum())
        return m0, v0
    return quantize, dequantize


def counting(np_, mq, vq, scales):
    m0, v0 = shipped_dq(np_, mq, vq, scales)
    counter["bad"] += int(((v0 == 0) & (m0 != 0)).sum())
    return m0, v0


def run(plan, label):
    counter["bad"] = 0
    out = []
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        fx.main(["run", plan, "--corpus", CORPUS, "--steps", "300",
                 "--seed", "1", "--val-fraction", "0.25", "--backend",
                 "numpy32", "--source", MODEL])
    for line in buf.getvalue().splitlines():
        if "val loss" in line and "after" in line:
            out.append(line.split("val loss")[1].strip())
    print(f"{label:28s} {out[-1] if out else '?':28s} v=0 & m!=0: "
          f"{counter['bad']}")


run(PLAN32, "f32 moments")
fx.moment_dequantize = counting
run(PLAN8, "int8, v log2 (shipped)")
for kind in ("sqrt", "linear"):
    fx.moment_quantize, fx.moment_dequantize = variant(kind)
    run(PLAN8, f"int8, v {kind}-linear")
