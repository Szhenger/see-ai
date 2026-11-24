#include "runtime/loader/contract.h"

#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "runtime/diagnostics/executing/error.h"
#include "runtime/diagnostics/feeding/error.h"
#include "runtime/diagnostics/persisting/error.h"
#include "runtime/diagnostics/validating/error.h"
#include "runtime/verifier/plan_validator.h"
#include "source/contract/plan_contract.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Every unit registered across the four runtime diagnostics process
/// modules. A message that cannot be attributed to one of these escaped
/// the partition.
constexpr std::string_view kRegisteredUnits[] = {
    diag::feeding::kDataset,          diag::feeding::kBatchPipeline,
    diag::validating::kPlanValidator, diag::executing::kEngine,
    diag::persisting::kDurableIo,     diag::persisting::kCheckpoint,
};

}  // namespace

bool WellFormedDiagnostic(std::string_view message) {
  for (std::string_view unit : kRegisteredUnits) {
    if (message.size() > unit.size() + 2 && message.starts_with(unit) &&
        message.substr(unit.size(), 2) == ": ")
      return true;
  }
  return false;
}

std::expected<void, std::string> VerifyPlanContract(
    const up::PlanHeader& header, uint64_t plan_size) {
  // The rules are the shared, pure statement of the load contract
  // (source/contract/plan_contract.h); the engine reports a violation
  // under its own unit.
  if (auto why = up::PlanHeaderViolation(header, plan_size))
    return diag::executing::Error(*why);
  return {};
}

namespace {

/// Which of the four programs an instruction belongs to, as the executor
/// contract's rules need it.
struct ProgramRole {
  const char* name;
  /// The eval and merge programs: no relaxed instruction and no optimizer
  /// step may appear in either.
  bool scores_or_merges;
  /// The eval program may read the source model file from v17; every
  /// program may from v21 (an adapter-only plan's frozen weights live
  /// there, F6 #134). Reads only: the validator refuses a source ref as a
  /// destination in any program.
  bool allow_source;
};

bool Overlaps(uint64_t off, uint64_t bytes, uint64_t ref, uint64_t ref_bytes) {
  const uint64_t slot = up::RefOffset(ref);
  return ref != up::kNullRef && ref_bytes != 0 && bytes != 0 &&
         off < slot + ref_bytes && slot < off + bytes;
}

/// The staged input and label slots are the feeder's (CR2 #143 N03): the
/// provenance proofs hold only while no program writes them, so no
/// instruction of any program may.
std::expected<void, std::string> StagedSlotsUnwritten(
    const up::UpdateInstruction& ins, const InstructionExtents& extents,
    const up::PlanHeader& header, const ProgramRole& role) {
  for (size_t t = 0; t < extents.count; ++t) {
    const OperandExtent& r = extents.ranges[t];
    if (!r.write || r.rodata || r.source) continue;
    if (Overlaps(r.off, r.bytes, header.input_ref,
                 header.input_floats * sizeof(float)))
      return diag::executing::Error(
          std::string("an instruction of the ") + role.name +
          " program writes the staged input slot (opcode " +
          std::to_string(ins.opcode) + ")");
    if (header.label_kind != 0 &&
        Overlaps(r.off, r.bytes, header.label_ref, header.label_bytes))
      return diag::executing::Error(
          std::string("an instruction of the ") + role.name +
          " program writes the staged label slot (opcode " +
          std::to_string(ins.opcode) + ")");
  }
  return {};
}

/// Label provenance for the class-indexed kernels. The softmax pair
/// indexes probability rows with raw i32 labels (probs[n*C + labels[n]]) —
/// the one place a validated instruction dereferences data-dependent
/// offsets. That is only safe because the feeder contract proves every
/// STAGED dataset label < the softmax width. The proof covers exactly the
/// label slot's `batch` staged entries, so a softmax whose label operand
/// is any other range, or whose row count exceeds the staged batch, or
/// that appears in a plan not declaring class labels, would read
/// unvalidated bytes as indices — an out-of-bounds write through the
/// backward kernel, from a plan that passed every range check. Bind all
/// three here.
///
/// The masked loss (S3 #149): the masked opcodes appear exactly when the
/// header carries kDataFlagMaskedLoss — the flag is what lets the feeder
/// stage -1 labels, and a plain kernel would index a row with -1.
/// `saw_masked` is set when `ins` is one.
std::expected<void, std::string> SoftmaxLabelsAreStaged(
    const up::UpdateInstruction& ins, const up::PlanHeader& header,
    bool* saw_masked) {
  const auto op = static_cast<up::OpCode>(ins.opcode);
  const bool masked_op = op == up::OpCode::kSoftmaxXEntMaskedFwd ||
                         op == up::OpCode::kSoftmaxXEntMaskedBwd;
  const bool plain_op =
      op == up::OpCode::kSoftmaxXEntFwd || op == up::OpCode::kSoftmaxXEntBwd;
  if (masked_op) {
    if ((header.data_flags & up::kDataFlagMaskedLoss) == 0)
      return diag::executing::Error(
          "masked cross-entropy in a plan without the masked-loss flag");
    *saw_masked = true;
  } else if (plain_op && (header.data_flags & up::kDataFlagMaskedLoss) != 0) {
    return diag::executing::Error(
        "plain cross-entropy in a plan that carries the masked-loss flag");
  }
  if (plain_op || masked_op) {
    if (header.label_kind != 1)
      return diag::executing::Error(
          "softmax cross-entropy in a plan without class labels");
    if (ins.in[1] != header.label_ref)
      return diag::executing::Error(
          "softmax labels are not the plan's staged label slot");
    if (ins.out[0] != header.batch)
      return diag::executing::Error(
          "softmax row count disagrees with the plan batch");
  }
  return {};
}

/// Embedding gather is the SECOND data-dependent-offset dereference: it
/// indexes table rows with raw i32 tokens (table[tokens[t]]). The feeder
/// contract proves only the STAGED ids < vocab, so — exactly as for the
/// softmax labels — its token operand must be the staged input slot, its
/// row count the staged batch, and the plan must declare token input. Any
/// other range would gather with unvalidated arena bytes read as indices:
/// an out-of-bounds read from a range-valid plan.
std::expected<void, std::string> EmbeddingTokensAreStaged(
    const up::UpdateInstruction& ins, const up::PlanHeader& header) {
  if (static_cast<up::OpCode>(ins.opcode) != up::OpCode::kEmbedFwd) return {};
  if (header.input_kind != 1)
    return diag::executing::Error(
        "embedding gather in a plan without token input");
  if (ins.in[0] != header.input_ref)
    return diag::executing::Error(
        "embedding tokens are not the plan's staged input slot");
  if (ins.out[0] != header.batch)
    return diag::executing::Error(
        "embedding row count disagrees with the plan batch");
  return {};
}

/// The segment starts (v25, #191; review 2026-11-16 D1): kAttnFwdSeg uses
/// each start as a loop bound, so its starts operand must be the very range
/// a kSegmentStarts of the same program wrote — whose values are positions
/// by construction — with no instruction writing it in between. One per
/// program: the ranges written by kSegmentStarts and still intact.
class LiveSegmentStarts {
 public:
  /// The rule for `ins`, then what `ins` does to the live ranges.
  std::expected<void, std::string> Step(const up::UpdateInstruction& ins,
                                        const InstructionExtents& extents) {
    const auto op = static_cast<up::OpCode>(ins.opcode);
    // v26: the tiled forward too.
    if (op == up::OpCode::kAttnFwdSeg || op == up::OpCode::kAttnFwdTiledSeg) {
      bool bound = false;
      for (const Range& s : live_) bound |= s.ref == ins.out[2];
      if (!bound)
        return diag::executing::Error(
            "segment-masked attention reads starts no segment_starts of the "
            "same program wrote (or one written over since)");
    }
    // Any write over a live starts range retires it; kSegmentStarts then
    // (re)establishes the range it writes.
    for (size_t t = 0; t < extents.count; ++t) {
      const OperandExtent& r = extents.ranges[t];
      if (!r.write || r.rodata || r.source) continue;
      std::erase_if(live_, [&](const Range& s) {
        return r.off < s.off + s.bytes && s.off < r.off + r.bytes;
      });
    }
    if (op == up::OpCode::kSegmentStarts) {
      const uint64_t positions =
          up::HighWord(ins.out[0]) * up::LowWord(ins.out[0]);
      live_.push_back(
          {ins.in[1], up::RefOffset(ins.in[1]), positions * sizeof(int32_t)});
    }
    return {};
  }

 private:
  struct Range {
    uint64_t ref, off, bytes;
  };
  std::vector<Range> live_;
};

/// An optimizer step is the update, whatever its moments are stored as: it
/// belongs to the train and step programs, as the certificate's bit does.
/// One stepping in the eval program would move the adapters while scoring
/// them — the engine evaluates between steps on the word that the eval
/// program writes transients only — and one in the merge program would move
/// them while their deltas are materialized. No compiler has emitted
/// either; the f32 steps are held to the rule the 8-bit step arrived with.
///
/// The 8-bit moment step (v27, #192) is the update too, and its codes and
/// scales are optimizer state — the persistent prefix the checkpoint
/// carries and no step's transients reuse. A plan whose moments sat in the
/// transient segment would train on whatever a forward last left there.
std::expected<void, std::string> OptimizerStepInItsPlace(
    const up::UpdateInstruction& ins, const up::PlanHeader& header,
    const ProgramRole& role) {
  const auto op = static_cast<up::OpCode>(ins.opcode);
  if ((op == up::OpCode::kSgdStep || op == up::OpCode::kAdamWStep) &&
      role.scores_or_merges)
    return diag::executing::Error(
        "an optimizer step outside the train / step programs (opcode " +
        std::to_string(ins.opcode) + ")");
  if (op != up::OpCode::kAdamW8Step) return {};
  if (role.scores_or_merges)
    return diag::executing::Error(
        "an 8-bit moment step outside the train / step programs");
  const uint64_t count = ins.out[0];
  const uint64_t scales = up::MomentBlocks(count) * 2 * sizeof(float);
  for (const auto& [ref, bytes] :
       {std::pair{ins.in[2], count}, std::pair{ins.in[3], count},
        std::pair{ins.out[2], scales}})
    if (!RangeOk(up::RefOffset(ref), bytes, header.persistent_size))
      return diag::executing::Error(
          "an 8-bit moment step's codes or scales lie outside the "
          "persistent segment");
  return {};
}

/// The column scales of a quantized GEMM (v17, kFlagQ8ColScale) are a
/// rodata vector the kernels multiply in blindly — N floats in the forward
/// (NN), K in the dX GEMM (NT): each is a finite positive float (CR1 #142
/// F33). The validator proved the vector in bounds.
std::expected<void, std::string> Q8ColumnScalesArePositive(
    const up::UpdateInstruction& ins, std::span<const uint8_t> rodata) {
  const auto op = static_cast<up::OpCode>(ins.opcode);
  if ((op != up::OpCode::kGemmNNQ8 && op != up::OpCode::kGemmNTQ8) ||
      !(ins.flags & up::kFlagQ8ColScale))
    return {};
  const uint64_t count = op == up::OpCode::kGemmNTQ8 ? ins.out[2] : ins.out[1];
  const uint8_t* scales = rodata.data() + up::RefOffset(ins.in[3]);
  for (uint64_t i = 0; i < count; ++i) {
    float s;
    std::memcpy(&s, scales + i * sizeof(float), sizeof(s));
    if (!std::isfinite(s) || s <= 0.0f)
      return diag::executing::Error("quantized GEMM column scale " +
                                    std::to_string(i) +
                                    " is not a finite positive float (opcode " +
                                    std::to_string(ins.opcode) + ")");
  }
  return {};
}

/// Every per-instruction rule of the executor contract over one program,
/// in the order a plan breaking several is reported.
std::expected<void, std::string> VerifyProgram(
    std::span<const up::UpdateInstruction> program, const ProgramRole& role,
    std::span<const uint8_t> rodata, const up::PlanHeader& header,
    bool* saw_masked) {
  LiveSegmentStarts starts;
  for (const up::UpdateInstruction& ins : program) {
    const auto extents =
        DescribeInstruction(ins, header.arena_size, header.rodata_size,
                            header.version, role.allow_source);
    if (!extents) return std::unexpected(extents.error());
    if (auto r = StagedSlotsUnwritten(ins, *extents, header, role); !r)
      return r;
    // The relaxed bit (v18 GEMMs, v20 tiled attention) belongs to the
    // train and step programs: the eval program scores the model that
    // ships in exact arithmetic, and the merge program has no
    // instruction a certificate covers.
    if ((ins.flags & up::kFlagRelaxed) && role.scores_or_merges)
      return diag::executing::Error(
          "a relaxed instruction outside the train / step programs "
          "(opcode " +
          std::to_string(ins.opcode) + ")");
    if (auto r = SoftmaxLabelsAreStaged(ins, header, saw_masked); !r) return r;
    if (auto r = EmbeddingTokensAreStaged(ins, header); !r) return r;
    if (auto r = starts.Step(ins, *extents); !r) return r;
    if (auto r = OptimizerStepInItsPlace(ins, header, role); !r) return r;
    if (auto r = Q8ColumnScalesArePositive(ins, rodata); !r) return r;
  }
  return {};
}

}  // namespace

std::expected<void, std::string> VerifyExecutorContract(
    std::span<const up::UpdateInstruction> train,
    std::span<const up::UpdateInstruction> merge,
    std::span<const up::UpdateInstruction> eval,
    std::span<const up::UpdateInstruction> step,
    std::span<const up::EmitEntry> emit_table, std::span<const uint8_t> rodata,
    const up::PlanHeader& header) {
  // The rodata handed in is the section the header describes: the column
  // scales are read from it at validator-proven offsets.
  if (rodata.size() != header.rodata_size)
    return diag::executing::Error(
        "rodata section handed to the executor contract disagrees with the "
        "header");
  const bool source_anywhere = header.version >= up::kSeeuSourceWeightsVersion;
  bool saw_masked = false;
  for (const auto& [program, role] :
       {std::pair{train, ProgramRole{"train", false, source_anywhere}},
        std::pair{merge, ProgramRole{"merge", true, source_anywhere}},
        std::pair{eval, ProgramRole{"eval", true, true}},
        std::pair{step, ProgramRole{"step", false, source_anywhere}}})
    if (auto r = VerifyProgram(program, role, rodata, header, &saw_masked); !r)
      return r;

  if ((header.data_flags & up::kDataFlagMaskedLoss) != 0 && !saw_masked)
    return diag::executing::Error(
        "plan carries the masked-loss flag but no masked cross-entropy");

  // The emit table's arena side is fixed at compile time; its file side is
  // validated against the actual model at commit. Commit reads the delta
  // through a float*, so the arena offset must be element-aligned too.
  for (const up::EmitEntry& e : emit_table) {
    if (!RangeOk(e.arena_offset, e.byte_size, header.arena_size))
      return diag::executing::Error("emit entry outside the arena");
    if (e.arena_offset % sizeof(float) != 0 || e.byte_size % sizeof(float) != 0)
      return diag::executing::Error("emit entry misaligned");
  }
  return {};
}

std::expected<void, std::string> VerifyRelaxedPermit(
    const up::PlanHeader& header, std::span<const up::UpdateInstruction> train,
    std::span<const up::UpdateInstruction> merge,
    std::span<const up::UpdateInstruction> eval,
    std::span<const up::UpdateInstruction> step, RelaxedPlanPermit permit) {
  // Every program is counted, not only the two the executor contract lets
  // carry the bit: the rule is "no relaxed instruction runs uncertified",
  // and it does not lean on another rule's order.
  uint64_t relaxed = 0;
  for (const auto program : {train, merge, eval, step})
    for (const up::UpdateInstruction& ins : program)
      if (ins.flags & up::kFlagRelaxed) ++relaxed;
  if (relaxed == 0 || permit.uncertified_for_measurement()) return {};
  const uint64_t certified = permit.certified_plan_hash();
  if (certified != 0 && certified == header.plan_hash) return {};
  // The certificate is checked where the binary is built (the package's
  // build.sh, tool/pack_update.py); what reaches the device is the seal of
  // the plan it was checked for. A plan swapped beside the binary or named
  // by --plan is another plan, whatever it was certified under elsewhere.
  return diag::executing::Error(
      "the plan carries " + std::to_string(relaxed) +
      " relaxed instruction(s) and this package was built with " +
      (certified == 0 ? "no numerics certificate for it"
                      : "a certificate for another plan") +
      ": certify the plan (tool/certify_numerics.py) and rebuild the package "
      "beside the certificate, or recompile with --precision f32");
}

std::expected<void, std::string> VerifyFeederContract(
    const up::PlanHeader& header, Dataset& data, uint64_t num_classes,
    uint64_t vocab_bound) {
  // The geometry half is the shared, pure rule (source/contract/); the
  // value half reads the records.
  const up::CorpusShape shape{
      data.input_kind(),     data.input_dim(),
      data.label_kind(),     data.label_bytes_per_sample(),
      data.tokenizer_hash(), data.masked(),
      data.content_hash(),   data.split_hash()};
  if (auto why = up::CorpusGeometryViolation(header, shape))
    return diag::executing::Error(*why);
  if (header.input_kind == 1) {
    // Token ids index embedding rows regardless of the loss, so they are
    // always bounded — by the narrowest table, and (when there is a class
    // loss) the narrowest softmax width too.
    uint64_t bound = vocab_bound;
    if (num_classes != 0 && (bound == 0 || num_classes < bound))
      bound = num_classes;
    return data.ValidateClassLabels(bound);
  }
  if (header.label_kind == 1)
    if (auto r = data.ValidateClassLabels(num_classes); !r)
      return std::unexpected(r.error());
  return {};
}

}  // namespace seeml::update_rt
