#include "runtime/loader/plan_loader.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "runtime/diagnostics/executing/error.h"
#include "runtime/executor/backend.h"
#include "runtime/loader/contract.h"
#include "runtime/verifier/plan_validator.h"
#include "source/identity/hash.h"
#include "source/platform/memory.h"

namespace seeml::update_rt {

namespace up = seeml::update;

std::expected<DecodedPlan, std::string> DecodePlan(const uint8_t* plan,
                                                   size_t plan_size) {
  // Everything below validates into locals; the engine commits them only
  // after every contract has passed. A failed re-Load must leave the
  // previous plan fully usable.
  up::PlanHeader header{};
  if (plan_size < sizeof(header))
    return diag::executing::Error("plan smaller than its header");
  std::memcpy(&header, plan, sizeof(header));
  if (header.magic != up::kSeeuMagic)
    return diag::executing::Error("bad plan magic");
  // Version negotiation (schema.h): additive format bumps stay readable,
  // semantic breaks raise the floor, unknown-future formats are rejected.
  // Checked on the freshly decoded local header — the header_ member still
  // holds whatever plan was loaded before this call.
  if (header.version < up::kSeeuOldestReadable ||
      header.version > up::kSeeuVersion)
    return diag::executing::Error(
        "unsupported plan version " + std::to_string(header.version) +
        " (this runtime reads v" + std::to_string(up::kSeeuOldestReadable) +
        "..v" + std::to_string(up::kSeeuVersion) + ")");

  // Integrity: the plan hashes over itself with the hash field zeroed.
  // A flipped bit anywhere — header, instructions, frozen weights — fails
  // here instead of surfacing as silent numerical garbage on-device.
  {
    constexpr size_t kHashAt = offsetof(up::PlanHeader, plan_hash);
    if (up::PlanSelfHash(plan, plan_size, kHashAt) != header.plan_hash)
      return diag::executing::Error(
          "plan hash mismatch — the .seeu blob is corrupt");
  }

  // Plan boundary: the header must be self-consistent with the blob (and
  // its I/O slots with the arena) before any section is decoded. The
  // contract also proves every section size overflow-free, so the byte
  // counts below are plain multiplies.
  if (auto ok = VerifyPlanContract(header, plan_size); !ok)
    return std::unexpected(ok.error());
  // The metadata section (v23): the manifest words and the ledger, held to
  // the plan they sit in before anything reads them.
  up::DecodedMetadata metadata;
  if (auto why = up::PlanMetadataViolation(header, plan, plan_size, &metadata))
    return diag::executing::Error(*why);
  const uint64_t train_bytes =
      header.train_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t merge_bytes =
      header.merge_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t eval_bytes =
      header.eval_instr_count * sizeof(up::UpdateInstruction);
  const uint64_t emit_bytes = header.emit_count * sizeof(up::EmitEntry);

  // Decode the instruction streams once; per-step execution touches only the
  // decoded vectors and the arena. An empty section (the step program of a
  // plan that does not accumulate) decodes to an empty vector whose data()
  // is null — memcpy's arguments are declared non-null even for a zero
  // length, so the copy is skipped, not called.
  auto decode = [&](auto& out, uint64_t offset, uint64_t bytes) {
    if (bytes) std::memcpy(out.data(), plan + offset, bytes);
  };
  std::vector<up::UpdateInstruction> train(header.train_instr_count);
  decode(train, header.train_instr_offset, train_bytes);
  std::vector<up::UpdateInstruction> merge(header.merge_instr_count);
  decode(merge, header.merge_instr_offset, merge_bytes);
  std::vector<up::UpdateInstruction> eval(header.eval_instr_count);
  decode(eval, header.eval_instr_offset, eval_bytes);
  std::vector<up::UpdateInstruction> step(header.step_instr_count);
  decode(step, header.step_instr_offset,
         header.step_instr_count * sizeof(up::UpdateInstruction));
  std::vector<up::EmitEntry> emit(header.emit_count);
  decode(emit, header.emit_table_offset, emit_bytes);

  // Executor boundary: every operand ref of every instruction is
  // bounds-proven by the validator and every emit entry targets the arena —
  // after this, Execute() dispatches the programs blindly.
  if (auto ok = VerifyExecutorContract(
          train, merge, eval, step, emit,
          std::span<const uint8_t>(plan + header.rodata_offset,
                                   header.rodata_size),
          header);
      !ok)
    return std::unexpected(ok.error());

  // Softmax class-width discipline (the kernels index rows of this width
  // with raw dataset labels). Every program must agree on the forward
  // width — train and eval are the label-consuming ones, and a softmax in
  // the merge or step program is a crafted plan's, whose narrower rows the
  // labels bounded by another program's width would index past (CR2
  // #143 F07: the check walked train and eval alone) — and a class-label
  // plan with no softmax at all would leave labels entirely unvalidated,
  // so it is rejected. Scanned on the freshly decoded local streams and
  // the local header — the members still describe whatever plan was loaded
  // before this call, and nothing may be committed until every contract
  // has passed.
  uint64_t fwd_classes = 0;
  for (const auto* program : {&train, &merge, &eval, &step})
    for (const up::UpdateInstruction& ins : *program)
      if (static_cast<up::OpCode>(ins.opcode) ==
              up::OpCode::kSoftmaxXEntFwd ||
          static_cast<up::OpCode>(ins.opcode) ==
              up::OpCode::kSoftmaxXEntMaskedFwd) {
        if (fwd_classes != 0 && fwd_classes != ins.out[1])
          return diag::executing::Error(
              "plan softmax class widths disagree across programs");
        fwd_classes = ins.out[1];
      }
  if (header.label_kind == 1 && fwd_classes == 0)
    return diag::executing::Error(
        "class-label plan carries no softmax cross-entropy — its labels "
        "could never be validated");

  // The accuracy metric reads the probabilities the eval program's softmax
  // already materializes (its in[3] write); no extra compute or arena space
  // is spent. Plans without a class-label softmax report loss only.
  // Scanned on the freshly decoded local stream — like everything here,
  // committed to members only after every contract has passed.
  uint64_t eval_probs_ref = up::kNullRef;
  uint64_t eval_softmax_rows = 0;
  uint64_t eval_softmax_cols = 0;
  // The eval program's loss instruction, when the loss slot is exactly one
  // row-separable loss (G13, #97): the wrapped duplicates of a final
  // partial batch can then be excluded from the reported loss exactly, as
  // they already are from accuracy. A composite loss (xent + KL, pure KL)
  // is not row-separable from what the program keeps; it is weighted.
  EvalTail eval_tail;
  // The source model extent the programs read (v17: the eval program's
  // f32 weights; v21: every program's, in an adapter-only plan): the file
  // bound at BindSourceModel must cover it. The contract proved every
  // source ref against kSourceSpaceLimit; this recovers the real
  // requirement.
  uint64_t source_required = 0;
  bool train_reads_source = false, merge_reads_source = false;
  for (const auto* program : {&train, &merge, &eval, &step})
    for (const up::UpdateInstruction& ins : *program)
      if (auto ex = DescribeInstruction(ins, header.arena_size,
                                        header.rodata_size, header.version,
                                        /*allow_source=*/true))
        for (size_t i = 0; i < ex->count; ++i)
          if (ex->ranges[i].source) {
            source_required =
                std::max(source_required,
                         ex->ranges[i].off + ex->ranges[i].bytes);
            if (program == &train || program == &step)
              train_reads_source = true;
            else if (program == &merge)
              merge_reads_source = true;
          }
  for (const up::UpdateInstruction& ins : eval) {
    const auto op = static_cast<up::OpCode>(ins.opcode);
    if (op == up::OpCode::kSoftmaxXEntFwd ||
        op == up::OpCode::kSoftmaxXEntMaskedFwd) {
      eval_probs_ref = ins.in[3];
      eval_softmax_rows = ins.out[0];
      eval_softmax_cols = ins.out[1];
      if (ins.in[2] == header.loss_ref && ins.out[0] == header.batch)
        eval_tail = {op == up::OpCode::kSoftmaxXEntFwd ? EvalTail::kXent
                                                        : EvalTail::kXentMasked,
                     ins.in[3], ins.in[1], ins.out[1]};
    } else if (op == up::OpCode::kMseFwd && ins.in[2] == header.loss_ref &&
               header.batch > 0 && ins.out[0] % header.batch == 0) {
      eval_tail = {EvalTail::kMse, ins.in[0], ins.in[1],
                   ins.out[0] / header.batch};
    }
  }

  // Class count for validating class-index labels at Train() time. The raw
  // dataset labels feed every softmax kernel in every program, so the check
  // must hold for the narrowest width anywhere — not just the train
  // program's last softmax, which would leave a narrower eval softmax
  // indexing past its validated rows.
  uint64_t num_classes = 0;
  // Every program, the step program included (CR2 #143 F07): a softmax
  // anywhere narrows the bound the feeder enforces on every staged label.
  for (const auto* program : {&train, &merge, &eval, &step})
    for (const up::UpdateInstruction& ins : *program) {
      const auto op = static_cast<up::OpCode>(ins.opcode);
      if (op != up::OpCode::kSoftmaxXEntFwd &&
          op != up::OpCode::kSoftmaxXEntBwd &&
          op != up::OpCode::kSoftmaxXEntMaskedFwd &&
          op != up::OpCode::kSoftmaxXEntMaskedBwd)
        continue;
      const uint64_t classes = ins.out[1];
      if (classes != 0 && (num_classes == 0 || classes < num_classes))
        num_classes = classes;
    }

  // Token-native plans (v7): the vocab bound plays the role class labels'
  // softmax width plays — every staged token indexes embedding rows, so the
  // narrowest table anywhere is the runtime bound the feeder contract will
  // enforce, and a token plan with NO embedding would leave the ids
  // entirely unvalidated (and meaningless), so it is rejected. Sequence
  // geometry must divide the batch: staging serves whole records.
  uint64_t vocab_bound = 0;
  for (const auto* program : {&train, &merge, &eval, &step})
    for (const up::UpdateInstruction& ins : *program)
      if (static_cast<up::OpCode>(ins.opcode) == up::OpCode::kEmbedFwd) {
        const uint64_t vocab = up::HighWord(ins.out[1]);
        if (vocab != 0 && (vocab_bound == 0 || vocab < vocab_bound))
          vocab_bound = vocab;
      }
  if (header.input_kind > 1)
    return diag::executing::Error("unknown plan input kind " +
                                  std::to_string(header.input_kind));
  if (header.input_kind == 1) {
    if (vocab_bound == 0)
      return diag::executing::Error(
          "token plan carries no embedding — its token ids could never be "
          "validated");
    if (header.seq_len == 0 || header.batch % header.seq_len != 0)
      return diag::executing::Error(
          "token plan sequence length must be nonzero and divide the batch");
  }

  DecodedPlan out;
  out.header = header;
  out.metadata = std::move(metadata);
  out.train = std::move(train);
  out.merge = std::move(merge);
  out.eval = std::move(eval);
  out.step = std::move(step);
  out.emit = std::move(emit);
  out.num_classes = num_classes;
  out.vocab_bound = vocab_bound;
  out.eval_probs_ref = eval_probs_ref;
  out.eval_softmax_rows = eval_softmax_rows;
  out.eval_softmax_cols = eval_softmax_cols;
  out.eval_tail = eval_tail;
  out.source_required = source_required;
  out.train_reads_source = train_reads_source;
  out.merge_reads_source = merge_reads_source;
  return out;
}

std::expected<Arena, std::string> AllocateArena(const up::PlanHeader& header,
                                                const uint8_t* plan) {
  // The single allocation of the update: the pre-planned arena. Its size was
  // known at compile time — the device's resource contract.
  // Bound before rounding: on a 32-bit host a u64 arena_size past SIZE_MAX
  // would otherwise truncate into a small allocation the validator's
  // (u64) bounds proof does not cover — the feeder guards the same class.
  // Page-aligned (kArenaAlignment) so a GPU backend can wrap it as a
  // zero-copy shared buffer; on the CPU path the alignment changes the
  // allocation's address and nothing else.
  if (header.arena_size > SIZE_MAX - kArenaAlignment)
    return diag::executing::Error(
        "plan arena of " + std::to_string(header.arena_size) +
        " bytes exceeds this host's address space");
  // An arena the host cannot hold is refused before the allocation is
  // attempted: the executing fuzzer (CR3 #144 F22) reached aligned_alloc
  // with a header asking for exabytes — an abort under the sanitizer, and
  // on an overcommitting host a lazy success that dies at the first touch.
  // A host that reports no memory (an unknown platform) is bounded by the
  // allocation's own failure, as before.
  if (const uint64_t host = up::DetectHostMemoryBytes();
      host != 0 && header.arena_size > host)
    return diag::executing::Error(
        "plan arena of " + std::to_string(header.arena_size) +
        " bytes exceeds this host's memory (" + std::to_string(host) +
        " bytes)");
  const size_t arena_bytes =
      (static_cast<size_t>(header.arena_size) + (kArenaAlignment - 1)) &
      ~size_t{kArenaAlignment - 1};
  uint8_t* arena = static_cast<uint8_t*>(
      std::aligned_alloc(kArenaAlignment, arena_bytes));
  if (!arena) return diag::executing::Error("arena allocation failed");
  std::memset(arena, 0, arena_bytes);
  std::memcpy(arena, plan + header.persist_init_offset,
              header.persist_init_size);

  return Arena{arena, arena_bytes};
}

}  // namespace seeml::update_rt
