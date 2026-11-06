#include "runtime/dispatcher/contract.h"

#include <cmath>
#include <cstring>
#include <string>

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
    diag::feeding::kDataset,       diag::feeding::kBatchPipeline,
    diag::validating::kPlanValidator, diag::executing::kEngine,
    diag::persisting::kDurableIo,  diag::persisting::kCheckpoint,
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

std::expected<void, std::string> VerifyExecutorContract(
    std::span<const up::UpdateInstruction> train,
    std::span<const up::UpdateInstruction> merge,
    std::span<const up::UpdateInstruction> eval,
    std::span<const up::UpdateInstruction> step,
    std::span<const up::EmitEntry> emit_table, std::span<const uint8_t> rodata,
    const up::PlanHeader& header) {
  // The rodata handed in is the section the header describes: the column
  // scales below are read from it at validator-proven offsets.
  if (rodata.size() != header.rodata_size)
    return diag::executing::Error(
        "rodata section handed to the executor contract disagrees with the "
        "header");
  for (const auto* program : {&train, &merge, &eval, &step})
    for (const up::UpdateInstruction& ins : *program) {
      // The eval program may read the source model file from v17; every
      // program may from v21 (an adapter-only plan's frozen weights live
      // there, F6 #134). Reads only: the validator refuses a source ref as
      // a destination in any program.
      const bool allow_source =
          program == &eval ||
          header.version >= up::kSeeuSourceWeightsVersion;
      const auto extents = DescribeInstruction(
          ins, header.arena_size, header.rodata_size, header.version,
          allow_source);
      if (!extents) return std::unexpected(extents.error());
      // The staged input and label slots are the feeder's (CR2 #143 N03):
      // the provenance proofs below hold only while no program writes
      // them, so no instruction of any program may.
      {
        auto overlaps = [](uint64_t off, uint64_t bytes, uint64_t ref,
                           uint64_t ref_bytes) {
          const uint64_t slot = up::RefOffset(ref);
          return ref != up::kNullRef && ref_bytes != 0 && bytes != 0 &&
                 off < slot + ref_bytes && slot < off + bytes;
        };
        const char* const name = program == &train   ? "train"
                                 : program == &merge ? "merge"
                                 : program == &eval  ? "eval"
                                                     : "step";
        for (size_t t = 0; t < extents->count; ++t) {
          const OperandExtent& r = extents->ranges[t];
          if (!r.write || r.rodata || r.source) continue;
          if (overlaps(r.off, r.bytes, header.input_ref,
                       header.input_floats * sizeof(float)))
            return diag::executing::Error(
                std::string("an instruction of the ") + name +
                " program writes the staged input slot (opcode " +
                std::to_string(ins.opcode) + ")");
          if (header.label_kind != 0 &&
              overlaps(r.off, r.bytes, header.label_ref, header.label_bytes))
            return diag::executing::Error(
                std::string("an instruction of the ") + name +
                " program writes the staged label slot (opcode " +
                std::to_string(ins.opcode) + ")");
        }
      }
      // The relaxed bit (v18 GEMMs, v20 tiled attention) belongs to the
      // train and step programs: the eval program scores the model that
      // ships in exact arithmetic, and the merge program has no
      // instruction a certificate covers.
      if ((ins.flags & up::kFlagRelaxed) &&
          (program == &eval || program == &merge))
        return diag::executing::Error(
            "a relaxed instruction outside the train / step programs "
            "(opcode " + std::to_string(ins.opcode) + ")");
      // Label provenance for the class-indexed kernels. The softmax pair
      // indexes probability rows with raw i32 labels
      // (probs[n*C + labels[n]]) — the one place a validated instruction
      // dereferences data-dependent offsets. That is only safe because the
      // feeder contract proves every STAGED dataset label < the softmax
      // width. The proof covers exactly the label slot's `batch` staged
      // entries, so a softmax whose label operand is any other range, or
      // whose row count exceeds the staged batch, or that appears in a
      // plan not declaring class labels, would read unvalidated bytes as
      // indices — an out-of-bounds write through the backward kernel, from
      // a plan that passed every range check. Bind all three here.
      const auto op = static_cast<up::OpCode>(ins.opcode);
      if (op == up::OpCode::kSoftmaxXEntFwd ||
          op == up::OpCode::kSoftmaxXEntBwd) {
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
      // Embedding gather is the SECOND data-dependent-offset dereference:
      // it indexes table rows with raw i32 tokens (table[tokens[t]]). The
      // feeder contract proves only the STAGED ids < vocab, so — exactly as
      // for the softmax labels — its token operand must be the staged input
      // slot, its row count the staged batch, and the plan must declare
      // token input. Any other range would gather with unvalidated arena
      // bytes read as indices: an out-of-bounds read from a range-valid plan.
      if (op == up::OpCode::kEmbedFwd) {
        if (header.input_kind != 1)
          return diag::executing::Error(
              "embedding gather in a plan without token input");
        if (ins.in[0] != header.input_ref)
          return diag::executing::Error(
              "embedding tokens are not the plan's staged input slot");
        if (ins.out[0] != header.batch)
          return diag::executing::Error(
              "embedding row count disagrees with the plan batch");
      }
      // The column scales of a quantized GEMM (v17, kFlagQ8ColScale) are a
      // rodata vector the kernels multiply in blindly — N floats in the
      // forward (NN), K in the dX GEMM (NT): each is a finite positive
      // float (CR1 #142 F33). The validator proved the vector in bounds.
      if ((op == up::OpCode::kGemmNNQ8 || op == up::OpCode::kGemmNTQ8) &&
          (ins.flags & up::kFlagQ8ColScale)) {
        const uint64_t count =
            op == up::OpCode::kGemmNTQ8 ? ins.out[2] : ins.out[1];
        const uint8_t* scales = rodata.data() + up::RefOffset(ins.in[3]);
        for (uint64_t i = 0; i < count; ++i) {
          float s;
          std::memcpy(&s, scales + i * sizeof(float), sizeof(s));
          if (!std::isfinite(s) || s <= 0.0f)
            return diag::executing::Error(
                "quantized GEMM column scale " + std::to_string(i) +
                " is not a finite positive float (opcode " +
                std::to_string(ins.opcode) + ")");
        }
      }
    }

  // The emit table's arena side is fixed at compile time; its file side is
  // validated against the actual model at commit. Commit reads the delta
  // through a float*, so the arena offset must be element-aligned too.
  for (const up::EmitEntry& e : emit_table) {
    if (!RangeOk(e.arena_offset, e.byte_size, header.arena_size))
      return diag::executing::Error("emit entry outside the arena");
    if (e.arena_offset % sizeof(float) != 0 ||
        e.byte_size % sizeof(float) != 0)
      return diag::executing::Error("emit entry misaligned");
  }
  return {};
}

std::expected<void, std::string> VerifyFeederContract(
    const up::PlanHeader& header, Dataset& data, uint64_t num_classes,
    uint64_t vocab_bound) {
  // The geometry half is the shared, pure rule (source/contract/); the
  // value half reads the records.
  const up::CorpusShape shape{data.input_kind(), data.input_dim(),
                              data.label_kind(),
                              data.label_bytes_per_sample()};
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
