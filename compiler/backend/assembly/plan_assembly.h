#ifndef SEEML_COMPILER_BACKEND_ASSEMBLY_PLAN_ASSEMBLY_H_
#define SEEML_COMPILER_BACKEND_ASSEMBLY_PLAN_ASSEMBLY_H_

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/analysis/algebra/lora_grafter.h"
#include "compiler/analysis/algebra/merge_builder.h"
#include "compiler/backend/allocation/arena_binder.h"
#include "compiler/frontend/computation/graph_build.h"
#include "compiler/frontend/representation/sir.h"
#include "source/plan/update_types.h"

// =============================================================================
// Plan assembly (S6 #152): the last backend stage, lifted out of the driver.
// The lowered programs, the arena binding and the header words the driver
// decided become the .seeu blob:
//
//   SeedPersistentImage  the persistent segment's initial bytes — zeros, and
//                        a counter-based randn for every adapter A
//   BuildEmitTable       where each merged delta lands in the source model
//   LayOutPlan           every section's offset and size, into the header
//   AssemblePlan         the blob, each frozen weight written exactly once
//   SealPlan             the integrity hash, patched into the header
//
// Pure functions of their arguments: the same inputs give the same bytes at
// any thread count.
// =============================================================================

namespace seeml::update {

/// The four instruction streams of a plan. `step` is empty unless the plan
/// accumulates gradients.
struct PlanPrograms {
  std::vector<UpdateInstruction> train;
  std::vector<UpdateInstruction> merge;
  std::vector<UpdateInstruction> eval;
  std::vector<UpdateInstruction> step;
};

/// The persistent segment's initial image. Every randn element is a pure
/// function of (seed, element index): counter-based splitmix64 (the same
/// generator the feeder's shuffler uses) feeding Box-Muller. That lets
/// ParallelFor split *within* a tensor — the common case is one large
/// adapter tensor, which a per-tensor stream would generate serially —
/// while the image stays bit-identical at any thread count. It also removes
/// the last implementation-defined numeric in the plan:
/// std::normal_distribution's algorithm varies across standard libraries.
std::vector<uint8_t> SeedPersistentImage(const ArenaBinding& binding);

/// Delta arena ranges -> source-file byte ranges, one entry per merge
/// output, in the merge program's order.
[[nodiscard]] std::expected<std::vector<EmitEntry>, std::string> BuildEmitTable(
    const MergeProgram& merge, const std::vector<GraftedAdapter>& adapters,
    const GraphBuild& build,
    const std::unordered_map<const seeml::sir::Value*, uint64_t>& merge_bound);

/// Writes every section's offset, count and size into `header` and returns
/// the blob's size: the programs in train / merge / eval / step order, the
/// rodata on a kSeeuRodataAlignment boundary (a GPU backend wraps the
/// frozen weights zero-copy), the persistent image, the emit table, and
/// padding to a page multiple.
uint64_t LayOutPlan(PlanHeader& header, const PlanPrograms& programs,
                    const ArenaBinding& binding, uint64_t persist_init_bytes,
                    uint64_t emit_count);

/// Assembles the blob of `size` bytes by APPENDING into one exact
/// reservation (E2, #81): each frozen weight is written exactly once, from
/// the loaded model straight to its place in the plan; only alignment gaps
/// (and the int8 / bf16 packs, computed in place) are zeroed. With
/// `consume`, each source payload is released after its last pack, so the
/// weights stay resident about once. The hash field is left zero for
/// SealPlan.
std::vector<uint8_t> AssemblePlan(const PlanHeader& header,
                                  const PlanPrograms& programs,
                                  const ArenaBinding& binding,
                                  const std::vector<uint8_t>& persist_init,
                                  const std::vector<EmitEntry>& emit_table,
                                  uint64_t size, bool consume);

/// The integrity seal: hashes the blob with the hash field zeroed (as
/// AssemblePlan leaves it) and patches the result in. Initialize()
/// re-derives and compares via the same PlanSelfHash.
void SealPlan(std::vector<uint8_t>& plan);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_ASSEMBLY_PLAN_ASSEMBLY_H_
