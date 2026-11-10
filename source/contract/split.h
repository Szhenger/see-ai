#ifndef SEEML_SOURCE_CONTRACT_SPLIT_H_
#define SEEML_SOURCE_CONTRACT_SPLIT_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include "source/identity/hash.h"

// =============================================================================
// The compiled held-out split (S3 #149, plan v22 kDataFlagSplit).
//
// The validation set is a decision the compiler makes, not the device: a
// seeded partition of the corpus's records, recorded in the plan header as
// (split_seed, val_frac_bits, val_split_hash) and recomputed on the device
// from the same corpus by the same function — so the gate scores the set
// the compile named, and a corpus that does not yield it (another file, or
// a file edited since) is refused before a step runs. A function, not a
// list: the plan carries three words whatever the corpus size.
//
// The partition: a Fisher–Yates permutation of [0, n) driven by splitmix64
// from the seed (the shuffle's generator, in its own domain), the first
// HeldOutCount(n, frac) positions are held out, sorted ascending. Pure,
// host-independent integer arithmetic; the Python plane mirrors it
// (tool/frontier_exec.py) and the two are held together by literal
// fixtures on both sides.
// =============================================================================

namespace seeml::update {

/// The split's own splitmix64 stream (the shuffle's generator, seeded in a
/// domain of its own so the same --seed never correlates the two).
inline uint64_t SplitMix64Next(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline constexpr uint64_t kSplitDomain = 0x53504C49545F3031ULL;  // "SPLIT_01"

/// Records held out of n at fraction frac: floor(n * frac), at least one,
/// at most n - 1 — Dataset::SplitValidation's rule, over the f32 fraction
/// the plan header carries (val_frac_bits). The positional split takes the
/// runner's parsed double, so for a fraction that is not exact in f32
/// (0.7, 0.3, ...) the two counts can differ by one on the same corpus;
/// the compiled split is the one the plan names, and the device recomputes
/// exactly this function.
inline uint64_t HeldOutCount(uint64_t n, float frac) {
  if (n < 2) return 0;
  uint64_t k = static_cast<uint64_t>(static_cast<double>(n) *
                                     static_cast<double>(frac));
  if (k == 0) k = 1;
  if (k >= n) k = n - 1;
  return k;
}

/// The held-out record indices of a corpus of n records, sorted ascending.
/// Empty when n < 2 (no split is possible).
inline std::vector<uint64_t> HeldOutIndices(uint64_t n, uint64_t seed,
                                            float frac) {
  const uint64_t k = HeldOutCount(n, frac);
  std::vector<uint64_t> held;
  if (k == 0) return held;
  std::vector<uint64_t> perm(n);
  for (uint64_t i = 0; i < n; ++i) perm[i] = i;
  uint64_t state = seed ^ kSplitDomain;
  state = SplitMix64Next(&state) | 1ULL;
  for (uint64_t i = n - 1; i > 0; --i) {
    const uint64_t j = SplitMix64Next(&state) % (i + 1);
    std::swap(perm[i], perm[j]);
  }
  held.assign(perm.begin(), perm.begin() + static_cast<ptrdiff_t>(k));
  std::sort(held.begin(), held.end());
  return held;
}

/// The split's identity: ContentHash64 over the held-out indices as
/// little-endian u64 words. Two compiles of the same corpus and seed agree
/// on it; the device recomputes it and compares.
inline uint64_t SplitHash(const std::vector<uint64_t>& held) {
  static_assert(std::endian::native == std::endian::little,
                "the split hash is over little-endian words");
  return ContentHash64(reinterpret_cast<const uint8_t*>(held.data()),
                       held.size() * sizeof(uint64_t));
}

inline uint32_t ValFracBits(float frac) { return std::bit_cast<uint32_t>(frac); }

inline float ValFracOf(uint32_t bits) { return std::bit_cast<float>(bits); }

}  // namespace seeml::update

#endif  // SEEML_SOURCE_CONTRACT_SPLIT_H_
