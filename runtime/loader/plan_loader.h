#ifndef SEEML_RUNTIME_LOADER_PLAN_LOADER_H_
#define SEEML_RUNTIME_LOADER_PLAN_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "source/plan/metadata.h"
#include "source/plan/update_types.h"

// =============================================================================
// The loader (S7 #153, seeai.md §6) — the security boundary. From the plan
// bytes to a decoded, proven plan and its one arena: the seal hash, the
// version window, the plan contract (the header against the blob), the
// metadata section, the instruction streams, the executor contract (every
// operand bounds-proven; the staged slots, the label and token bindings, the
// segment starts), the class-width and vocabulary scans the feeder enforces,
// and the arena the resource contract sized. Pure: nothing here touches an
// engine; UpdateEngine commits the result only after every rule has passed.
// =============================================================================

namespace seeml::update_rt {

/// How the eval loss excludes a final batch's wrapped duplicates (G13):
/// exactly for a lone cross-entropy (a = probs, b = labels, width =
/// classes) or MSE (a = pred, b = target, width = elements per row), by
/// real-row weight for a composite loss.
struct EvalTail {
  enum Kind { kWeighted, kXent, kMse, kXentMasked } kind = kWeighted;
  uint64_t a = seeml::update::kNullRef;
  uint64_t b = seeml::update::kNullRef;
  uint64_t width = 0;
};

/// A plan proven and decoded: its header and metadata, the four programs and
/// the emit table, and the facts the scans recovered.
struct DecodedPlan {
  seeml::update::PlanHeader header{};
  seeml::update::DecodedMetadata metadata;
  std::vector<seeml::update::UpdateInstruction> train, merge, eval, step;
  std::vector<seeml::update::EmitEntry> emit;
  uint64_t num_classes = 0;  // the narrowest softmax width, 0 = no class loss
  uint64_t vocab_bound = 0;  // the narrowest embedding table, 0 = none
  uint64_t eval_probs_ref = seeml::update::kNullRef;
  uint64_t eval_softmax_rows = 0;
  uint64_t eval_softmax_cols = 0;
  EvalTail eval_tail;
  uint64_t source_required = 0;  // bytes of the source model the programs read
  bool train_reads_source = false;
  bool merge_reads_source = false;
};

/// Every container and executor rule over `plan`'s bytes, then the decode.
[[nodiscard]] std::expected<DecodedPlan, std::string> DecodePlan(
    const uint8_t* plan, size_t plan_size);

/// The one allocation of the update: the arena the header sized (bounded by
/// the address space and the host's memory first), zeroed, the persistent
/// segment's initial image copied in. The caller owns `data` (std::free).
struct Arena {
  uint8_t* data = nullptr;
  size_t bytes = 0;
};
[[nodiscard]] std::expected<Arena, std::string> AllocateArena(
    const seeml::update::PlanHeader& header, const uint8_t* plan);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_LOADER_PLAN_LOADER_H_
