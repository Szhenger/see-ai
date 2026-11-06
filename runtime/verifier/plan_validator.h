#ifndef SEEML_RUNTIME_VERIFIER_PLAN_VALIDATOR_H_
#define SEEML_RUNTIME_VERIFIER_PLAN_VALIDATOR_H_

#include <cstdint>
#include <expected>
#include <string>

#include "source/contract/bounds.h"
#include "source/plan/update_types.h"

// =============================================================================
// Load-time validation of a .seeu plan's instruction streams.
//
// Every operand ref of every instruction is checked against the address
// space it targets (arena or rodata) before the first step runs — after
// this, the engine's Execute() dispatch can trust the programs blindly.
// The overflow-safe helpers are exposed for the engine's own section and
// I/O-slot checks, so plan bounds math is written exactly one way.
// =============================================================================

namespace seeml::update_rt {

// The overflow-safe bounds helpers live in source/contract/bounds.h,
// shared with the compiler (CR2 #143 design item 1); re-exported here
// so the runtime's call sites name them as they always have.
using seeml::update::MulOk;
using seeml::update::RangeOk;

/// Checks that every ref operand of `ins` lies fully inside its address space
/// (arena or rodata), with byte extents derived from the instruction's dims
/// the same way the kernels derive their loop bounds, and that writes only
/// target the mutable arena. Rejects unknown opcodes, which Execute() would
/// silently skip, and polices the flags word against `plan_version`: below
/// kSeeuFlagsVersion every instruction must carry flags == 0; from it on,
/// only the defined epilogue bits are accepted, only on the opcodes they
/// are defined for (kGemmNN: bias + activation, with the bias ref in in[3]
/// joining the bounds and overlap discipline; kGemmNNQ8: activation only —
/// its in[3] is the dequant scale). An unknown bit is a hard error, never a
/// silent skip.
/// `allow_source`: the instruction may read the source model file — the
/// eval program's from v17, every program's from v21 (an adapter-only
/// plan, F6 #134); the caller decides from the program and the version.
[[nodiscard]] std::expected<void, std::string> ValidateInstruction(
    const seeml::update::UpdateInstruction& ins, uint64_t arena_size,
    uint64_t rodata_size, uint32_t plan_version, bool allow_source = false);

/// One operand's byte extent as the validator derived it — the same
/// arithmetic the kernels use for their loop bounds.
struct OperandExtent {
  uint64_t off = 0;
  uint64_t bytes = 0;
  bool write = false;
  bool rodata = false;
  bool source = false;  // v17: the bound source model file (read-only)
};

/// The bound a source ref is checked against at load, before any file is
/// bound: large enough for any model, small enough that offset arithmetic
/// cannot overflow. The real file's size is checked at BindSourceModel
/// against the extents the eval program proved.
inline constexpr uint64_t kSourceSpaceLimit = 1ULL << 48;

/// Every operand extent of one instruction (at most 8).
struct InstructionExtents {
  OperandExtent ranges[8];
  size_t count = 0;
};

/// ValidateInstruction, additionally returning the operand extents it
/// proved. A backend that defers work (a GPU command buffer) uses these to
/// decide when a CPU-resident instruction must wait for pending GPU work
/// — the read/write ranges are exactly the ones the bounds proof covers,
/// so dependency tracking can never be looser than validation.
[[nodiscard]] std::expected<InstructionExtents, std::string>
DescribeInstruction(const seeml::update::UpdateInstruction& ins,
                    uint64_t arena_size, uint64_t rodata_size,
                    uint32_t plan_version, bool allow_source = false);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_VERIFIER_PLAN_VALIDATOR_H_
