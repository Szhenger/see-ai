#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_SEGMENT_MASK_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_SEGMENT_MASK_H_

#include <cstdint>
#include <expected>
#include <string>

#include "compiler/frontend/representation/block.h"

// =============================================================================
// The attention segment mask (#191, plan v25). A packed window holds several
// documents joined by EOS; without a mask a position attends to the tail of
// the previous, unrelated document. Segments are contiguous, so the mask is
// one start per position — one past the last EOS strictly before it — which
// one sc_low.segment_starts op computes from the token input, and every
// sc_high.attention op reads as its fourth operand (keys before the row's
// start masked as the causal ones are). It runs before the primal
// snapshot, so the eval program masks alike, and before autodiff, whose
// attention rule reads q, k and v alone: the cached backward consumes the
// masked probabilities and needs nothing new, and the tiled backward
// (plan v26) reads each row's start from the stats row its forward wrote.
// Both families are masked, and compute the same bits masked as unmasked
// families do.
// =============================================================================

namespace seeml::update {

class SegmentMasker {
 public:
  SegmentMasker(seeml::sir::Value* tokens, int32_t eos)
      : tokens_(tokens), eos_(eos) {}
  /// The number of attention ops masked.
  [[nodiscard]] std::expected<size_t, std::string> Run(
      seeml::sir::Block& block);

 private:
  seeml::sir::Value* tokens_;
  int32_t eos_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_SEGMENT_MASK_H_
