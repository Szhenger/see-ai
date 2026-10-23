#!/usr/bin/env python3
"""Gate a seeml-bench run against stored baselines (stdlib only).

Usage:
    python3 bench_compare.py baseline.json current.json
        [--max-regression F] [--epoch-baseline epoch.json]
        [--max-epoch-regression F] [--state gate_state.json] [--promote]
        [--sha COMMIT] [--across-hosts] [--seed-epoch]
    python3 bench_compare.py --host-slug current.json
    python3 bench_compare.py --cache-key current.json

Compares the Tier A throughput metric — rows_per_s, per fixture per thread
width — and exits 1 if any pair regressed by more than --max-regression
(rows_per_s is MLX-LM's "Tokens/sec" under its own definition — every row
of a SeeAI batch is a loss target — and schema-2 reports also carry it as
tokens_per_s / samples_per_s; the gate keeps the schema-1 key so a baseline
stored before the rename still compares)
(default 0.10, the >10% gate of docs/benchmarks.md). A missing baseline
file exits 0 with a note: the first run of a new host seeds the baseline
rather than failing it. Fixtures or thread widths present on only one side
are reported and skipped — adding a fixture must not fail the night it
lands — but a comparison with NO overlapping keys is an error (exit 1),
not a pass: after a key rename the gate would otherwise be blind forever
while printing "no regression".

The rolling baseline (last green night) catches step changes; a slow
compounding drift of <10%/night would never trip it. --epoch-baseline names
a second, pinned report that is never rolled forward — the gate also diffs
against it (--max-epoch-regression, default 0.15) so drift accumulates
against a fixed reference. A missing epoch file is a note, not a failure.
The nightly keeps its epochs in git (docs/ledger/bench-epochs/<slug>.json),
where nothing evicts them: an actions cache unused for seven days is
evicted, and an epoch that re-seeds after every lapse is a reference that
rolls forward. This tool therefore writes a missing epoch only when asked
(--seed-epoch, on a green gate), never as a side effect of --promote.

Everything is medians-of-medians upstream, so a >10% delta is signal, not
noise — on ONE machine. The nightly gate compares machines of one runner
class, which differ by 15-35% on identical work: one fast runner sets a
high-water baseline and every ordinary runner after it fails (Nightly
#20-#23, and 09-13/09-14 on the commit that was green on 09-12), while a
real regression is indistinguishable from the noise. Two defences (P5, #79):

Releases. Reports carry `seeml_version`. A rolling baseline measured on a
different release is refused (exit 1) unless --across-releases is passed: a
release is a deliberate re-seed, never a silent comparison. The pinned epoch
baseline spans releases by design, so there the change is a NOTE.

Calibration. A schema-4 report carries `calibration`: the rate of a frozen
single-threaded reference kernel that shares no code with the runtime,
timed before and after the fixtures. Each Tier A delta is taken after
dividing the current rows/s by (current calibration / reference
calibration), so a uniformly slow or fast runner cancels and a slow SeeAI
kernel does not. A reference that predates schema 4 compares un-normalized,
with a note (the migration pattern of the rows_per_s key above); a schema-4
report WITHOUT a usable calibration block, two different calibration
kernels, or a ratio outside [1/3, 3] (not the same host class) are errors,
never silent raw comparisons.

Host model. Calibration normalizes a runner's speed within one CPU model,
not the ratio between models: the nightly's hosted runners rotate across
six models, and from 2026-10-08 to 10-22 every WARN and FAIL was measured
against a baseline from another model (an Intel Xeon 8573C, 8370C or
6973P-C, or an AMD EPYC 9V45, against an AMD EPYC 7763's or 9V74's
night; on the 8573C the decoder fixtures read 0.63-0.73x of a 7763's
rows/s while its calibration kernel read 1.0-1.15x) — P8 (#165). So the
gate compares like with like a second way: a report names the host it
measured on (host_arch: the CPU model and its physical core count; the
ISA, cores and caches when the kernel names no model) and a baseline from
another host is refused (exit 1) unless --across-hosts is passed. Two
spellings of one model (case, punctuation) are one host: hosts compare as
slugs, the cache key's own form. `--host-slug REPORT` prints that slug
and `--cache-key REPORT` the rolling cache's whole key segment (host,
release, kernel policy, calibration kernel, joined by "--", which no slug
contains), so the workflow keeps one rolling baseline and gate state per
host and measuring setup. A report without host_arch or a host_key
(schema < 3) compares with a NOTE.

Two-consecutive-red. With --state PATH a key that regresses for the first
time is a WARNING (exit 0; a `::warning::` annotation under GitHub Actions)
and is recorded; the gate fails only when the same key regresses against
the same reference on the next run too. A key that recovers is forgotten.
Without --state every red fails at once, as before. --promote makes this
tool the one that rolls the baseline forward: the current report replaces
the baseline only on a fully green gate (or when no baseline exists), so a
warned run is never what the second night is compared against; a missing
--epoch-baseline is seeded under the same condition when --seed-epoch is
also given, and never rewritten. The state carries the commit the run
measured (--sha, else GITHUB_SHA), and a key that was red on the previous
run and is compared and green now is announced as CLEARED with both
commits: cleared at the same commit is the runner's variance on record,
cleared at a later one is a fix. The workflow saves the state after every
gate, failed ones included, so "the previous run" is the previous night.
A red key the current report does not measure (a fixture dropped, a
thread width removed) is not cleared: it stays red in the state, fails
the night it is measured red again, and while it is carried the baseline
is not rolled forward (the report would drop the key's only reference).

Schema-3 reports name the kernel policy they ran under (the CPU GEMM
tiles the compiler wrote into every plan); two reports with different
policies are never compared — that delta is what tool/autotune.py
measures, not a regression — and the gate exits 1 telling you to re-seed.
"""
import argparse
import json
import os
import re
import sys


def tier_a(report):
    """{(fixture, threads, backend): rows_per_s} from a seeml-bench JSON
    object. The backend joins the key (per-backend determinism doctrine:
    a metal row is never compared against a cpu row); reports that predate
    the field are cpu, the only backend that existed."""
    out = {}
    backend = report.get("backend", "cpu")
    for name, fx in report.get("fixtures", {}).items():
        for t, m in fx.get("threads", {}).items():
            out[(name, t, backend)] = float(m["rows_per_s"])
    return out


def kernel_policy(report):
    """The kernel policy a run measured under: None for the runtime's
    compiled-in defaults (every report that predates schema 3, and every
    schema-3 report whose policy source is "default"), else the explicit
    (gemm_tile_k, gemm_tile_n). A tuned run (tool/autotune.py's table, or
    --gemm-tiles) is never gated against a default baseline, and vice
    versa: the delta would be the tiling, not the commit."""
    kp = report.get("kernel_policy")
    if not kp or kp.get("source", "default") == "default":
        return None
    return (int(kp["gemm_tile_k"]), int(kp["gemm_tile_n"]))


class GateError(Exception):
    """A comparison the gate refuses to make (fail closed, exit 1)."""


CALIBRATION_SCHEMA = 4      # the first seeml-bench schema that calibrates
MAX_CALIBRATION_RATIO = 3.0  # beyond this the two hosts are not one class


def calibration(report, path):
    """(kernel, rows_per_s) of a report's calibration block, or None for a
    report that predates schema 4. A schema-4 report whose block is absent
    or unusable is a GateError: the harness that wrote it always
    calibrates, so its absence is a truncated or edited report."""
    block = report.get("calibration")
    if int(report.get("schema", 1)) < CALIBRATION_SCHEMA and block is None:
        return None
    try:
        kernel, rate = str(block["kernel"]), float(block["calib_rows_per_s"])
    except (TypeError, KeyError, ValueError):
        raise GateError(f"{path} is a schema-{report.get('schema')} report "
                        "without a usable calibration block")
    if not kernel or not rate > 0.0 or rate == float("inf"):
        raise GateError(f"{path} carries a non-positive or non-finite "
                        "calibration rate")
    return kernel, rate


UNKNOWN_MODELS = ("", "unknown")


def _host_fields(report):
    """(isa, model, cores, l1d, l2) of the CPU a report measured on, or
    None. host_arch (schema >= 3) is the measuring host as detected. The
    host_key is the kernel-policy LOOKUP key — it names the --target-host
    when one was given — so it is read only for a report without
    host_arch."""
    arch = report.get("host_arch")
    if isinstance(arch, dict):
        return (str(arch.get("isa") or ""), str(arch.get("cpu_model") or ""),
                arch.get("physical_cores"), arch.get("l1d_bytes"),
                arch.get("l2_bytes"))
    key = report.get("host_key")
    if not key:
        return None
    parts = str(key).split(";")
    if len(parts) == 1:
        return ("", parts[0], None, None, None)
    kv = dict(f.split("=", 1) for f in parts[2:] if "=" in f)
    return (parts[0], parts[1], kv.get("cores"), kv.get("l1d"), kv.get("l2"))


def host_model(report):
    """The host a report measured on, as one string: "<model>, <N> cores",
    or, when the kernel names no model (Linux arm64 prints none), "<isa>
    CPU (model unknown), <N> cores, L1d <bytes>, L2 <bytes>", so two such
    hosts differ by what does distinguish them. None for a report from
    before host_arch and host_key (schema < 3)."""
    fields = _host_fields(report)
    if fields is None:
        return None
    isa, model, cores, l1d, l2 = fields
    model = model.strip()
    cores_s = f", {cores} cores" if cores not in (None, "") else ""
    if model.lower() not in UNKNOWN_MODELS:
        return model + cores_s
    return (f"{isa or 'unknown'} CPU (model unknown){cores_s}, "
            f"L1d {l1d}, L2 {l2}")


def host_slug(model):
    """The model as a cache-key slug: lower case, every run of other
    characters one dash ("amd-epyc-7763-64-core-processor")."""
    return re.sub(r"[^a-z0-9]+", "-", (model or "unknown host").lower()).strip("-")


def cache_key(report):
    """The rolling cache's key segment for a report: the host, the
    release, the kernel policy and the calibration kernel, each a slug,
    joined by "--" (a slug never contains two dashes in a row, so no
    segment is a prefix of another host's). A change in any of the last
    three is a re-seed by design — the gate refuses that comparison — so
    keying by them re-seeds every host at once instead of failing each
    host in turn until a reseed dispatch happens to land on it."""
    policy = kernel_policy(report)
    calib = report.get("calibration") or {}
    parts = [report.get("seeml_version") or "no release",
             ("default tiles" if policy is None
              else f"tiles {policy[0]} {policy[1]}"),
             calib.get("kernel") or "uncalibrated"]
    # The host segment is --host-slug's own spelling (a report without a
    # host is "unknown-host" in both), so the key and the epoch name agree.
    return "--".join([host_slug(host_model(report))]
                     + [host_slug(str(p)) for p in parts])


class Report:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.path = path
        self.tier_a = tier_a(raw)
        self.policy = kernel_policy(raw)
        self.calibration = calibration(raw, path)
        self.version = raw.get("seeml_version")
        self.host = host_model(raw)


def runner_ratio(label, ref, cur):
    """current-runner speed over the reference runner's, from the two
    calibration rates; 1.0 (un-normalized, with a note) when either side
    predates calibration."""
    if ref.calibration is None or cur.calibration is None:
        old = label if ref.calibration is None else "current report"
        print(f"bench_compare: NOTE the {old} predates schema "
              f"{CALIBRATION_SCHEMA} (no calibration) — comparing "
              f"un-normalized against the {label}")
        return 1.0
    (ref_kernel, ref_rate), (cur_kernel, cur_rate) = (ref.calibration,
                                                      cur.calibration)
    if ref_kernel != cur_kernel:
        raise GateError(f"the {label} calibrated with '{ref_kernel}' but the "
                        f"current report with '{cur_kernel}'; the rates are "
                        "not comparable — re-seed the baseline deliberately")
    ratio = cur_rate / ref_rate
    if not 1.0 / MAX_CALIBRATION_RATIO <= ratio <= MAX_CALIBRATION_RATIO:
        raise GateError(f"the current runner calibrates at {ratio:.2f}x the "
                        f"{label}'s — not the same host class; re-seed the "
                        "baseline deliberately")
    return ratio


def key_name(key):
    return f"{key[0]}@{key[1]}t/{key[2]}"


def compare(label, ref, cur, max_regression):
    """Prints per-key lines; returns ({key_name: line} of regressed keys,
    {key_name of every key compared}). Deltas are runner-normalized (see
    runner_ratio)."""
    base, now = ref.tier_a, cur.tier_a
    ratio = runner_ratio(label, ref, cur)
    shared = [k for k in base if k in now and base[k] > 0]
    if ratio != 1.0 or (ref.calibration and cur.calibration):
        # The median raw key ratio is printed beside the calibration ratio
        # as a diagnostic only: normalizing by it would hide a regression
        # that slows every fixture at once.
        raw = sorted(now[k] / base[k] for k in shared)
        mid = (f", median raw key ratio {raw[len(raw) // 2]:.3f}x"
               if raw else "")
        print(f"bench_compare: NOTE runner calibrates at {ratio:.3f}x the "
              f"{label}'s{mid}; deltas below are normalized")
    failures, skipped, compared = {}, [], set()
    for key, base_v in sorted(base.items()):
        if key not in now:
            skipped.append(f"{key_name(key)}: in {label} only")
            continue
        compared.add(key_name(key))
        cur_v = now[key]
        norm_v = cur_v / ratio
        delta = (norm_v - base_v) / base_v if base_v > 0 else 0.0
        shown = (f"{cur_v:.0f}" if ratio == 1.0
                 else f"{cur_v:.0f} (normalized {norm_v:.0f})")
        line = (f"{key_name(key)}: {base_v:.0f} -> {shown} rows/s "
                f"({delta:+.1%}) vs {label}")
        if delta < -max_regression:
            failures[key_name(key)] = line
        else:
            print(f"bench_compare: OK   {line}")
    for key in sorted(now.keys() - base.keys()):
        skipped.append(f"{key_name(key)}: new, not in {label}")
    for s in skipped:
        print(f"bench_compare: SKIP {s}")
    return failures, compared


STATE_SCHEMA = 2  # 2: carries the commit the reds were measured at


def load_state(path):
    """({label: [key_name, ...]}, sha) — the keys that were red on the
    previous run and the commit it measured (None for a schema-1 state).
    A missing file is an empty state; an unreadable one is an error (a
    gate that forgets its reds silently never fails)."""
    if path is None or not os.path.exists(path):
        return {}, None
    try:
        with open(path) as f:
            state = json.load(f)
        if state.get("schema") not in (1, STATE_SCHEMA):
            raise ValueError(f"schema {state.get('schema')!r}")
        return ({label: list(keys) for label, keys in state["red"].items()},
                state.get("sha"))
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        raise GateError(f"cannot read the gate state {path}: {e}")


def write_atomic(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def judge(label, failures, compared, previous, use_state, shas=(None, None)):
    """Splits one reference's regressed keys into (fatal, warned) lines and
    prints them. Without --state every red is fatal. A key red on the
    previous run, compared now and green is announced with both commits
    (shas = (previous, current)): the same commit twice is the runner's
    variance on record, a later one is a fix. A previous red this run did
    not compare is neither cleared nor announced."""
    fatal, warned = [], []
    for name in sorted(previous.get(label, [])):
        if name in compared and name not in failures:
            prev_sha, cur_sha = (s or "an unrecorded commit" for s in shas)
            print(f"bench_compare: CLEARED {name} vs {label} (red at "
                  f"{prev_sha} on the previous run, green at {cur_sha})")
    for name, line in sorted(failures.items()):
        if not use_state or name in previous.get(label, []):
            fatal.append(line)
            suffix = " (second consecutive red)" if use_state else ""
            print(f"bench_compare: FAIL {line}{suffix}")
        else:
            warned.append(line)
            print(f"bench_compare: WARN {line} (first red — fails if it "
                  "repeats next run)")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::warning title=bench gate, first red::{line}")
    return fatal, warned


def main():
    # --host-slug and --cache-key stand alone. They are read first so the
    # two positionals stay required: optional positionals stop argparse
    # from reading "BASELINE --option CURRENT" before Python 3.12.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--host-slug", metavar="REPORT", default=None)
    pre.add_argument("--cache-key", metavar="REPORT", default=None)
    early, _ = pre.parse_known_args()
    for path, render in ((early.host_slug, lambda r: host_slug(host_model(r))),
                         (early.cache_key, cache_key)):
        if path is not None:
            with open(path) as f:
                print(render(json.load(f)))
            return 0
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("current")
    p.add_argument("--max-regression", type=float, default=0.10,
                   help="fractional Tier A loss vs the rolling baseline "
                        "that fails the gate (0.10)")
    p.add_argument("--epoch-baseline", metavar="PATH", default=None,
                   help="pinned report that is never rolled forward; "
                        "guards against compounding drift")
    p.add_argument("--max-epoch-regression", type=float, default=0.15,
                   help="fractional Tier A loss vs the epoch baseline that "
                        "fails the gate (0.15)")
    p.add_argument("--state", metavar="PATH", default=None,
                   help="gate state carried between runs: a key's first red "
                        "is a warning, its second consecutive red fails")
    p.add_argument("--promote", action="store_true",
                   help="on a fully green gate (or a missing baseline) "
                        "replace the baseline with the current report")
    p.add_argument("--seed-epoch", action="store_true",
                   help="with --promote, also write a missing "
                        "--epoch-baseline from the current report (the "
                        "nightly keeps its epochs in git and never passes "
                        "this)")
    p.add_argument("--across-releases", action="store_true",
                   help="compare against a rolling baseline measured on a "
                        "different seeml_version (refused by default)")
    p.add_argument("--across-hosts", action="store_true",
                   help="compare against baselines measured on a different "
                        "CPU model (refused by default: calibration does "
                        "not transfer between models, P8 #165)")
    p.add_argument("--sha", metavar="COMMIT", default=None,
                   help="the commit the current report measured, recorded "
                        "in the state (default: $GITHUB_SHA)")
    p.add_argument("--host-slug", metavar="REPORT", default=None,
                   help="print REPORT's host as a slug and exit (the "
                        "workflow names the committed epoch by it)")
    p.add_argument("--cache-key", metavar="REPORT", default=None,
                   help="print REPORT's rolling-cache key segment (host, "
                        "release, kernel policy, calibration kernel) and "
                        "exit")
    args = p.parse_args()
    try:
        return gate(args)
    except GateError as e:
        print(f"bench_compare: ERROR {e}")
        return 1


def gate(args):
    cur = Report(args.current)
    if not cur.tier_a:
        raise GateError(f"current report {args.current} carries no Tier A "
                        "metrics")

    def check_policy(label, ref):
        if ref.policy == cur.policy:
            return
        # Fail closed: a delta between two tilings is a tuning result, not a
        # regression signal — re-seed the baseline with the policy in use.
        def name(p):
            return "the runtime defaults" if p is None else f"GEMM tiles {p}"
        raise GateError(f"the {label} was measured with {name(ref.policy)} "
                        f"but the current report with {name(cur.policy)}; "
                        "the gate compares like with like — re-seed the "
                        "baseline deliberately")

    def check_release(label, ref):
        """A baseline from another release must not compare silently (#86).
        The rolling baseline fails closed — a release is a deliberate
        re-seed, like a policy change. The pinned epoch baseline exists to
        span releases, so there the change is announced, not refused."""
        if ref.version == cur.version:
            return
        def name(v):
            return "no seeml_version" if v is None else f"seeml {v}"
        said = (f"the {label} was measured on {name(ref.version)}, the "
                f"current report on {name(cur.version)}")
        if label == "baseline" and not args.across_releases:
            raise GateError(f"{said}; a delta across releases is not a "
                            "nightly regression signal — re-seed the "
                            "baseline deliberately, or pass "
                            "--across-releases to compare anyway")
        print(f"bench_compare: NOTE {said}")

    def check_host(label, ref):
        """A baseline from another host must not compare at all (P8,
        #165): the calibration kernel reads 0.99-1.15x between an AMD EPYC
        7763 and an Intel Xeon 8573C while the decoder fixtures read
        0.63-0.73x, so the normalized delta is the microarchitecture, not
        the commit. Both anchors fail closed; the workflow keys them by
        host so each runner meets its own. Hosts compare as slugs, the
        cache key's form, so two spellings of one model are one host and
        a slug that keys a cache always passes its own check. A report
        without a host predates the field and compares with a note."""
        if ref.host is None and cur.host is None:
            return
        if (ref.host is not None and cur.host is not None
                and host_slug(ref.host) == host_slug(cur.host)):
            return
        if ref.host is None or cur.host is None:
            print(f"bench_compare: NOTE the {label if ref.host is None else 'current report'} "
                  "predates host_key (no CPU model) — comparing across hosts")
            return
        said = (f"the {label} was measured on '{ref.host}' but the current "
                f"report on '{cur.host}'")
        if not args.across_hosts:
            raise GateError(f"{said}; calibration does not transfer between "
                            "CPU models (P8 #165) — the nightly keys its "
                            "baselines by model, so this is a wrong cache "
                            "key or a hand-run comparison; pass "
                            "--across-hosts to compare anyway")
        print(f"bench_compare: NOTE {said}; comparing across hosts as asked")

    previous, previous_sha = load_state(args.state)
    use_state = args.state is not None
    sha = args.sha or os.environ.get("GITHUB_SHA") or None
    red, fatal, warned, unmeasured = {}, [], [], []

    def against(label, path, max_regression, what):
        ref = Report(path)
        check_release(label, ref)
        check_policy(label, ref)
        check_host(label, ref)
        short = label.split()[0]
        failures, compared = compare(short, ref, cur, max_regression)
        if not compared:
            # Fail closed: zero overlap means the gate measured nothing.
            raise GateError("no Tier A metric is present in both the "
                            f"{label} and the current report — the gate "
                            "would be blind; re-seed the baseline "
                            "deliberately if the keys were renamed")
        f, w = judge(short, failures, compared, previous, use_state,
                     (previous_sha, sha))
        if f:
            print(f"bench_compare: {len(f)} Tier A metric(s) regressed more "
                  f"than {max_regression:.0%} vs {what}")
        # A red this run did not measure stays red: nothing showed it green.
        carried = [n for n in previous.get(short, []) if n not in compared]
        for n in carried:
            print(f"bench_compare: NOTE {n} was red vs {short} on the "
                  "previous run and is not measured now; it stays red")
        red[short] = sorted(set(failures) | set(carried))
        fatal.extend(f)
        warned.extend(w)
        if short == "baseline":
            unmeasured.extend(carried)

    seeded = not os.path.exists(args.baseline)
    if seeded:
        print(f"bench_compare: no baseline at {args.baseline} — "
              "this run seeds it")
    else:
        against("baseline", args.baseline, args.max_regression,
                "the rolling baseline")

    if args.epoch_baseline is not None:
        if not os.path.exists(args.epoch_baseline):
            seeding = args.seed_epoch and args.promote
            how = ("this run seeds it on a green gate" if seeding
                   else "no epoch comparison; pin one by committing a "
                        "green night's report there")
            print(f"bench_compare: NOTE no epoch baseline at "
                  f"{args.epoch_baseline} — {how}")
            if os.environ.get("GITHUB_ACTIONS") == "true" and not seeding:
                print(f"::notice title=bench gate, no epoch::no epoch "
                      f"baseline at {args.epoch_baseline}; this host's "
                      "drift is unguarded until one is committed")
        else:
            against("epoch baseline", args.epoch_baseline,
                    args.max_epoch_regression,
                    "the pinned epoch baseline (compounding drift)")

    if use_state:
        write_atomic(args.state, json.dumps(
            {"schema": STATE_SCHEMA, "red": red, "sha": sha,
             "host": cur.host}, indent=2, sort_keys=True) + "\n")
    if fatal:
        return 1
    if warned:
        print(f"bench_compare: {len(warned)} Tier A metric(s) red for the "
              "first time — warning only; the baseline is not rolled "
              "forward")
        return 0
    print("bench_compare: no Tier A regression")
    if args.promote and unmeasured:
        # Promoting a report that lacks a red key would drop the only
        # reference that can show it still slow.
        print(f"bench_compare: {len(unmeasured)} red key(s) not measured "
              "tonight — the baseline is not rolled forward (re-seed it if "
              "they were removed deliberately)")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            # A green exit that never rolls forward is a stall: say so in
            # the run summary, not only in the log.
            print("::warning title=bench gate, baseline held::"
                  f"{', '.join(sorted(unmeasured))} red and not measured; "
                  "the rolling baseline stays frozen until they are "
                  "measured or the baseline is re-seeded")
        return 0
    if args.promote:
        with open(args.current) as f:
            text = f.read()
        write_atomic(args.baseline, text)
        print(f"bench_compare: promoted {args.current} to {args.baseline}")
        if (args.seed_epoch and args.epoch_baseline is not None
                and not os.path.exists(args.epoch_baseline)):
            write_atomic(args.epoch_baseline, text)
            print(f"bench_compare: seeded the epoch {args.epoch_baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
