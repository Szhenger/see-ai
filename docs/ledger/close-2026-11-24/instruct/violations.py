"""#218 acceptance 4, on a genuine instruct checkpoint: over the corpus the
compiler wrote, count the response positions whose segment start lies after
the first position of their own prompt. Starts are computed as the
kSegmentStarts kernel computes them, once with the id the plan carries and
once with the model's EOS (the scan before the fix). The prompt of a
response position is found from the ChatML structure alone: the turn before
the one the position is in."""
import struct
import sys

sds, plan_sep, eos, im_start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = open(sds, "rb").read()
magic, version, n, S, label_kind, input_kind, label_dim = struct.unpack_from("<IIQQIIQ", d, 0)
tok_hash, flags, _, _ = struct.unpack_from("<QIIQ", d, 40)
assert version == 3 and input_kind == 1 and flags & 1, (version, input_kind, flags)
rec = 4 * (S + 1) + ((S + 3) & ~3)
assert 64 + n * rec == len(d), (len(d), n, rec)
def starts(tokens, sep):
    out, start = [], 0
    for t, tok in enumerate(tokens):
        out.append(start)
        if tok == sep: start = t + 1
    return out
tot = {"responses": 0, "plan": 0, "eos": 0}
windows_multi = 0
for r in range(n):
    off = 64 + r * rec
    ids = struct.unpack_from(f"<{S + 1}i", d, off)
    mask = d[off + 4 * (S + 1): off + 4 * (S + 1) + S]
    tokens = ids[:S]
    windows_multi += plan_sep in tokens[:-1]
    sp, se = starts(tokens, plan_sep), starts(tokens, eos)
    turns = []  # positions of <|im_start|>
    for p in range(S):
        if tokens[p] == im_start: turns.append(p)
        # mask[p] says the label of position p (token p + 1) is a target:
        # position p attends as part of the response turn.
        if not mask[p]: continue
        prompt_first = turns[-2] if len(turns) >= 2 else 0
        tot["responses"] += 1
        tot["plan"] += sp[p] > prompt_first
        tot["eos"] += se[p] > prompt_first
print(f"{n} records of {S} positions, {windows_multi} holding more than one document")
print(f"response positions: {tot['responses']}")
print(f"start after its own prompt, starts from the plan's id {plan_sep}: {tot['plan']}")
print(f"start after its own prompt, starts from the EOS {eos}: {tot['eos']}")
