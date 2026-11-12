"""ContentHash64 (source/identity/hash.h), mirrored in Python.

The model-identity hash: 8-lane striped FNV-1a over 1 MiB chunks laid out
by ParallelFor's chunk geometry, the per-chunk digests folded in chunk
order, then the byte count — bitwise the C++ function at any thread count.
The exporter uses it for the tokenizer section's identity (SDS v3
`tokenizer_hash`, plan v22), the one place the Python plane has to produce
a hash the device will compare, so it is held to the C++ by test
(test/tool/tokenizer_test.py reads `seeml-tokenize --info`).

Pure Python: about a second per megabyte, which is the size of a tokenizer
section; it is not the way to hash a model file.
"""
from typing import Any, Iterable, List

_MASK = (1 << 64) - 1
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
CONTENT_HASH_LANES = 8
CONTENT_HASH_CHUNK = 1 << 20
MAX_PARALLEL_CHUNKS = 256


def _mix_word(state: int, word: int) -> int:
    """FnvMixWord: fold a 64-bit word into the state, low byte first."""
    for b in range(8):
        state ^= (word >> (8 * b)) & 0xFF
        state = (state * FNV_PRIME) & _MASK
    return state


def striped_fnv1a64(data: bytes) -> int:
    """StripedFnv1a64: the 8-lane interleaved kernel over one span."""
    lanes = [(FNV_OFFSET_BASIS ^ ((0x9E3779B97F4A7C15 * (lane + 1)) & _MASK))
             for lane in range(CONTENT_HASH_LANES)]
    n = len(data)
    full = n - n % CONTENT_HASH_LANES
    for lane in range(CONTENT_HASH_LANES):
        h = lanes[lane]
        for b in data[lane:full:CONTENT_HASH_LANES]:
            h = ((h ^ b) * FNV_PRIME) & _MASK
        lanes[lane] = h
    for lane, i in enumerate(range(full, n)):
        lanes[lane] = ((lanes[lane] ^ data[i]) * FNV_PRIME) & _MASK
    h = FNV_OFFSET_BASIS
    for lane in lanes:
        h = _mix_word(h, lane)
    return _mix_word(h, n)


def _chunk_grain(n: int, grain: int) -> int:
    """ParallelChunkGrain: the grain, or n / 256 rounded up when larger."""
    cap = (n + MAX_PARALLEL_CHUNKS - 1) // MAX_PARALLEL_CHUNKS
    return cap if grain < cap else grain


def _chunks(n: int) -> Iterable[range]:
    g = _chunk_grain(n, CONTENT_HASH_CHUNK)
    count = (n + g - 1) // g if n else 0
    for c in range(count):
        yield range(c * g, min((c + 1) * g, n))


def content_hash64(data: bytes) -> int:
    """ContentHash64 of `data`: the in-order fold of the chunk digests,
    then the size."""
    partials = [striped_fnv1a64(data[r.start:r.stop]) for r in _chunks(len(data))]
    if not partials:
        partials = [striped_fnv1a64(b"")]
    h = FNV_OFFSET_BASIS
    for p in partials:
        h = _mix_word(h, p)
    return _mix_word(h, len(data))


def content_hash64_np(np: Any, data: Any) -> int:
    """ContentHash64 of `data` (bytes, or any uint8 buffer such as a
    memory-mapped file), with NumPy: the same digest as content_hash64,
    computed across every chunk's eight lanes at once — a 500 MB model file
    in seconds where the pure-Python fold takes minutes. The exporter uses
    it to bind its report to the SMF it wrote (S4 #150)."""
    buf = np.frombuffer(data, dtype=np.uint8) if not hasattr(data, "dtype") \
        else np.asarray(data, dtype=np.uint8).reshape(-1)
    n = int(buf.size)
    prime = np.uint64(FNV_PRIME)
    lanes0 = np.array([(FNV_OFFSET_BASIS ^ ((0x9E3779B97F4A7C15 * (lane + 1)) & _MASK))
                       for lane in range(CONTENT_HASH_LANES)], dtype=np.uint64)

    def kernel(block: Any) -> List[int]:
        """StripedFnv1a64 over each row of `block` ([chunks, g] uint8)."""
        chunks, g = block.shape
        full = g - g % CONTENT_HASH_LANES
        state = np.repeat(lanes0[None, :], chunks, axis=0)
        if full:
            cols = block[:, :full].reshape(chunks, full // CONTENT_HASH_LANES,
                                           CONTENT_HASH_LANES)
            with np.errstate(over="ignore"):
                for i in range(full // CONTENT_HASH_LANES):
                    state = (state ^ cols[:, i, :].astype(np.uint64)) * prime
        with np.errstate(over="ignore"):
            for lane, i in enumerate(range(full, g)):
                state[:, lane] = (state[:, lane] ^ block[:, i].astype(np.uint64)) * prime
        out = []
        for row in state:
            h = FNV_OFFSET_BASIS
            for lane in row:
                h = _mix_word(h, int(lane))
            out.append(_mix_word(h, g))
        return out

    g = _chunk_grain(n, CONTENT_HASH_CHUNK)
    whole = n // g if n else 0
    partials = kernel(buf[:whole * g].reshape(whole, g)) if whole else []
    if n - whole * g or not n:
        partials += kernel(buf[whole * g:].reshape(1, n - whole * g))
    h = FNV_OFFSET_BASIS
    for p in partials:
        h = _mix_word(h, p)
    return _mix_word(h, n)
