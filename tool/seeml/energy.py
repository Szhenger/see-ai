"""Energy per token on macOS (F6 #134, design item 2): the SoC's energy
counters over an interval, read through IOReport — the library behind
`powermetrics` — without privileges, by an unprivileged process.

What it measures, exactly. IOReport's "Energy Model" channel group holds
cumulative energy counters per SoC domain (CPU, GPU, ANE, DRAM, ...). A
meter samples them at start and at stop and reports each domain's delta in
joules. Two honest limits, both stated in every report:

- the counters are *whole-SoC*: every process on the machine is in them,
  so a measurement means something on a quiet machine, and the rows say
  `attribution: whole-SoC`;
- what an unprivileged reader sees is the OS's choice: on macOS 26 the
  GPU's counter advances with every sample, while the CPU, DRAM and ANE
  counters are published in bursts — nothing for seconds, then the whole
  backlog at once (747 J of CPU energy landed in one 8-second window, none
  in the next; `sudo powermetrics` sees them continuously). An interval
  reading of a burst domain is not a measurement, so the meter keeps every
  raw delta in `joules` but counts only the continuous domains as `scope`
  (`available` follows it) and lists the others under `burst`. A GPU-only
  scope is what a Metal row can show here; a row that computes on the CPU
  has no energy number to show and must not borrow the GPU's idle draw
  (the harness and the bench attribute by the row's compute).

The harness (frontier_run.py) and the bench take the same regression the
step time takes: the energy of the lo and hi runs, divided by the extra
steps, is the energy per step with the lifecycle removed, and per token
from there. Elsewhere — Linux, a VM without the library or the channel
group, an OS that hides them — the meter is unavailable and says why; it
never raises. Standard library only (ctypes): tier 0.
"""
import ctypes
import ctypes.util
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

GROUP = "Energy Model"
LIBRARIES: Sequence[str] = ("/usr/lib/libIOReport.dylib",
                            "/System/Library/PrivateFrameworks/IOReport.framework/IOReport")
_UNITS: Dict[str, float] = {"nJ": 1e-9, "uJ": 1e-6, "µJ": 1e-6, "mJ": 1e-3, "J": 1.0}
_UTF8 = 0x08000100
# The domains an unprivileged reader sees advance with every sample on
# macOS 26: the ones an interval reading can measure.
CONTINUOUS: Sequence[str] = ("GPU Energy",)
Reading = Dict[str, Any]


class EnergyMeter:
    """start() samples the counters; stop() samples again and returns the
    reading: {"available": bool, "attribution": "whole-SoC", "scope":
    [domains that moved], "joules": {domain: J}, "reason": str}."""

    def __init__(self, libraries: Sequence[str] = LIBRARIES) -> None:
        self.reason: Optional[str] = None
        self._lib: Any = None  # the IOReport library, a ctypes.CDLL
        self._cf: Any = None   # CoreFoundation
        self._sub: Any = None
        self._channels: Any = None
        self._key: Any = None
        self._start: Any = None
        if sys.platform != "darwin":
            self.reason = f"no IOReport on {sys.platform}"
            return
        try:
            self._open(libraries)
        except Exception as e:  # a missing symbol, a refused subscription
            self.reason = f"IOReport unavailable: {e}"
            self._lib = None

    # --- the IOReport binding ------------------------------------------------
    def _open(self, libraries: Sequence[str]) -> None:
        # The library lives in the dyld shared cache, not as a file on disk
        # (os.path.exists is false for it): the loader is the only test.
        lib: Any = None
        for path in libraries:
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libIOReport not found")
        cf: Any = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        self._cf = cf
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                 ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                          ctypes.c_long, ctypes.c_uint32]
        cf.CFDictionaryCreateMutableCopy.restype = ctypes.c_void_p
        cf.CFDictionaryCreateMutableCopy.argtypes = [ctypes.c_void_p, ctypes.c_long,
                                                     ctypes.c_void_p]
        cf.CFDictionaryGetValue.restype = ctypes.c_void_p
        cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        lib.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
        lib.IOReportCopyChannelsInGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.c_uint64, ctypes.c_uint64,
                                                    ctypes.c_uint64]
        lib.IOReportCreateSubscription.restype = ctypes.c_void_p
        lib.IOReportCreateSubscription.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.POINTER(ctypes.c_void_p),
                                                   ctypes.c_uint64, ctypes.c_void_p]
        lib.IOReportCreateSamples.restype = ctypes.c_void_p
        lib.IOReportCreateSamples.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p]
        lib.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
        lib.IOReportCreateSamplesDelta.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p]
        for name in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
            getattr(lib, name).restype = ctypes.c_void_p
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        lib.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        lib.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int]
        group = cf.CFStringCreateWithCString(None, GROUP.encode(), _UTF8)
        channels = lib.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
        cf.CFRelease(group)
        if not channels:
            raise RuntimeError(f"no '{GROUP}' channel group")
        wanted = cf.CFDictionaryCreateMutableCopy(None, 0, channels)
        subscribed = ctypes.c_void_p()
        sub = lib.IOReportCreateSubscription(None, wanted, ctypes.byref(subscribed),
                                             0, None)
        if not sub or not subscribed.value:
            raise RuntimeError("subscription refused")
        self._lib, self._sub, self._channels = lib, sub, subscribed.value
        self._key = cf.CFStringCreateWithCString(None, b"IOReportChannels", _UTF8)

    def _cfstr(self, ref: Any) -> str:
        buf = ctypes.create_string_buffer(256)
        if ref and self._cf.CFStringGetCString(ref, buf, 256, _UTF8):
            return str(buf.value.decode("utf-8", "replace"))
        return "?"

    def _sample(self) -> Any:
        return self._lib.IOReportCreateSamples(self._sub, self._channels, None)

    def _release(self, ref: Any) -> None:
        if ref:  # CFRelease(NULL) aborts the process
            self._cf.CFRelease(ref)

    # --- the interval --------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._lib is not None

    def start(self) -> "EnergyMeter":
        if self._lib is not None:
            self._release(self._start)  # a start never stopped
            self._start = self._sample()
        return self

    def stop(self) -> Reading:
        out: Reading = {"available": False, "attribution": "whole-SoC", "scope": [],
                        "burst": [], "joules": {}, "reason": self.reason}
        if self._lib is None or self._start is None:
            out["reason"] = out["reason"] or "meter not started"
            return out
        start, self._start = self._start, None
        end = self._sample()
        # A NULL sample or delta (IOReport declined) is an unavailable
        # reading, never a NULL handed on to CoreFoundation.
        delta = (self._lib.IOReportCreateSamplesDelta(start, end, None)
                 if end else None)
        rows = self._cf.CFDictionaryGetValue(delta, self._key) if delta else None
        joules: Dict[str, float] = {}
        for i in range(int(self._cf.CFArrayGetCount(rows)) if rows else 0):
            row = self._cf.CFArrayGetValueAtIndex(rows, i)
            value = int(self._lib.IOReportSimpleGetIntegerValue(row, 0))
            if value <= 0:
                continue
            name = self._cfstr(self._lib.IOReportChannelGetChannelName(row))
            unit = self._cfstr(self._lib.IOReportChannelGetUnitLabel(row))
            if unit not in _UNITS:
                continue
            joules[name] = joules.get(name, 0.0) + value * _UNITS[unit]
        for ref in (delta, end, start):
            self._release(ref)
        out["joules"] = joules
        out["scope"] = [d for d in CONTINUOUS if d in joules]
        out["burst"] = sorted(d for d in joules if d not in CONTINUOUS)
        out["available"] = bool(out["scope"])
        if delta is None:
            out["reason"] = "IOReport returned no sample"
        elif not out["scope"]:
            out["reason"] = ("no continuously published Energy Model counter "
                             "moved for this process (an unprivileged reader "
                             "sees the GPU's advance per sample on macOS 26; "
                             "the CPU, DRAM and ANE counters arrive in bursts, "
                             "and need root to read continuously)")
        return out


def domain_joules(reading: Mapping[str, Any],
                  names: Sequence[str] = CONTINUOUS) -> Optional[float]:
    """The joules of the first named domain the reading holds, else None."""
    joules: Mapping[str, float] = reading.get("joules", {})
    for name in names:
        if name in joules:
            return float(joules[name])
    return None


def per_step_from_slope(lo_reading: Mapping[str, Any], hi_reading: Mapping[str, Any],
                        lo_steps: int, hi_steps: int,
                        domain: str = "GPU Energy") -> Optional[float]:
    """Energy per step by the same regression the step time takes: the hi
    run's joules less the lo run's, over the extra steps — the lifecycle
    (load, the evaluations) cancels. None where either reading lacks the
    domain or the slope is not positive."""
    lo = lo_reading.get("joules", {}).get(domain)
    hi = hi_reading.get("joules", {}).get(domain)
    if lo is None or hi is None or hi_steps <= lo_steps:
        return None
    slope = (float(hi) - float(lo)) / (hi_steps - lo_steps)
    return slope if slope > 0 else None
