#ifndef SEEML_SOURCE_CONTRACT_BOUNDS_H_
#define SEEML_SOURCE_CONTRACT_BOUNDS_H_

#include <cstdint>

// =============================================================================
// Overflow-safe bounds arithmetic for file-supplied offsets and sizes — the
// one way plan bounds math is written, on both planes (CR2 #143 design item
// 1 moved it here from runtime/verifier/, which re-exports it).
// =============================================================================

namespace seeml::update {

/// Overflow-checked u64 multiply for file-supplied offsets and sizes.
inline bool MulOk(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

/// True iff [off, off + bytes) lies inside [0, size) without wrapping.
inline bool RangeOk(uint64_t off, uint64_t bytes, uint64_t size) {
  return off <= size && bytes <= size - off;
}

/// Saturating u64 arithmetic for counts that must not wrap (the
/// accountant's bounds, a plan's work): UINT64_MAX reads "at least this".
inline uint64_t SatAdd(uint64_t a, uint64_t b) {
  return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}
inline uint64_t SatMul(uint64_t a, uint64_t b) {
  uint64_t out = 0;
  return MulOk(a, b, &out) ? out : UINT64_MAX;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_CONTRACT_BOUNDS_H_
