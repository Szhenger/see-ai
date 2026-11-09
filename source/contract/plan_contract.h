#ifndef SEEML_SOURCE_CONTRACT_PLAN_CONTRACT_H_
#define SEEML_SOURCE_CONTRACT_PLAN_CONTRACT_H_

#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "source/contract/bounds.h"
#include "source/plan/update_types.h"

// =============================================================================
// The plan's geometry contract as pure functions (CR2 #143, design item 1):
// over the plan header and the plan's size, and over the plan header and a
// corpus's header-level shape. No engine, no dataset, no diagnostics unit:
// each returns the violated rule's reason, or nothing, and each caller
// reports it under its own unit — the runtime's loader and feeder
// (runtime/dispatcher/contract.cc), the compiler's backend contract on the
// plan it has just emitted (compiler/driver/contract.cc), and the plan probe
// through the runtime's wrapper. One statement of the rules, so no copy of
// them can skip one: the review's F07, N03-N05 and F58 were each a copy
// that did.
// =============================================================================

namespace seeml::update {

namespace contract_detail {
inline std::optional<std::string> Violation(std::string why) { return why; }
}  // namespace contract_detail

/// The header's section table, version gates, arena relations, I/O slots
/// and optimizer hyperparameters are self-consistent with a plan blob of
/// `plan_size` bytes — the load contract. The first violated rule's reason.
inline std::optional<std::string> PlanHeaderViolation(
    const PlanHeader& header, uint64_t plan_size) {
  uint64_t train_bytes = 0, merge_bytes = 0, eval_bytes = 0, emit_bytes = 0,
           step_bytes = 0;
  if (!MulOk(header.train_instr_count, sizeof(UpdateInstruction),
             &train_bytes) ||
      !MulOk(header.merge_instr_count, sizeof(UpdateInstruction),
             &merge_bytes) ||
      !MulOk(header.eval_instr_count, sizeof(UpdateInstruction),
             &eval_bytes) ||
      !MulOk(header.step_instr_count, sizeof(UpdateInstruction),
             &step_bytes) ||
      !MulOk(header.emit_count, sizeof(EmitEntry), &emit_bytes))
    return contract_detail::Violation("plan section size overflows");
  // The rodata section's offset is proven aligned (CR2 #143 N05): every
  // rodata ref's offset is relative to it, so a 4-aligned ref lands
  // misaligned in memory when the section does not — the compiler puts
  // it on a page boundary; kSeeuRodataOffsetAlignment (16 bytes) is what
  // every kernel's vector path may assume at most.
  if (header.rodata_size != 0 &&
      header.rodata_offset % kSeeuRodataOffsetAlignment != 0)
    return contract_detail::Violation(
        "rodata section is not " +
        std::to_string(kSeeuRodataOffsetAlignment) + "-byte aligned");
  if (!RangeOk(header.train_instr_offset, train_bytes, plan_size) ||
      !RangeOk(header.merge_instr_offset, merge_bytes, plan_size) ||
      !RangeOk(header.eval_instr_offset, eval_bytes, plan_size) ||
      !RangeOk(header.step_instr_offset, step_bytes, plan_size) ||
      !RangeOk(header.rodata_offset, header.rodata_size, plan_size) ||
      !RangeOk(header.persist_init_offset, header.persist_init_size,
               plan_size) ||
      !RangeOk(header.emit_table_offset, emit_bytes, plan_size))
    return contract_detail::Violation("plan section out of bounds");

  // GEMM tiles (v11): a throughput knob the kernels accept only as a
  // positive multiple of their 4-wide unroll (K) and positive (N); zero is
  // the runtime default. Below v11 the words are padding and must be zero.
  if (header.version < kSeeuGemmTilesVersion &&
      (header.gemm_tile_k != 0 || header.gemm_tile_n != 0))
    return contract_detail::Violation(
        "plan carries GEMM tile fields below v" +
        std::to_string(kSeeuGemmTilesVersion));
  if (header.gemm_tile_k % 4 != 0)
    return contract_detail::Violation(
        "plan gemm_tile_k " + std::to_string(header.gemm_tile_k) +
        " is not a multiple of the kernel's 4-wide unroll");

  // Gradient accumulation (v9): a step program exists exactly when the
  // plan accumulates more than one micro-batch — a lone step program would
  // never run, and an accumulating plan without one would never step.
  if ((header.grad_accum_steps > 1) != (header.step_instr_count > 0))
    return contract_detail::Violation(
        "plan step program presence disagrees with grad_accum_steps");
  if (header.version < kSeeuGradAccumVersion &&
      (header.grad_accum_steps > 1 || header.step_instr_count > 0))
    return contract_detail::Violation(
        "pre-v" + std::to_string(kSeeuGradAccumVersion) +
        " plan carries gradient-accumulation fields");

  // The engine chunks Evaluate passes by dividing sample counts by the
  // batch, and the feeder stages batch x input width every step — zero for
  // either is a malformed plan, not a degenerate-but-runnable one (a zero
  // batch reaches an integer division no later guard checks: batch = 0
  // with input_floats = 0 would pass every slot check below).
  if (header.batch == 0)
    return contract_detail::Violation("plan batch size is zero");
  if (header.input_floats == 0)
    return contract_detail::Violation("plan input width is zero");

  // The arena is the target of the persistent image, the checkpoints, and
  // every arena ref — its size must dominate all of them.
  if (header.arena_size > UINT64_MAX - 63)
    return contract_detail::Violation("arena size overflows");
  if (header.persist_init_size > header.arena_size ||
      header.persistent_size > header.arena_size)
    return contract_detail::Violation("persistent segment exceeds arena");

  // The optimizer hyperparameters (CR1 #142): the step kernels consume them
  // blindly, so each is finite and within the range the compiler accepts
  // and the CLI enforces — lr positive; weight decay and the clip norm
  // non-negative (0 = off); the cosine floor in [0, 1]; the moment decays
  // in [0, 1) and eps positive (a decay of 1 or an eps of 0 divides by
  // zero in the first step). A NaN passes every comparison a kernel makes,
  // so a NaN here trained on garbage without a word.
  {
    struct Rule {
      const char* name;
      float value;
      bool in_range;
    };
    const Rule rules[] = {
        {"lr", header.lr, header.lr > 0.0f},
        {"beta1", header.beta1, header.beta1 >= 0.0f && header.beta1 < 1.0f},
        {"beta2", header.beta2, header.beta2 >= 0.0f && header.beta2 < 1.0f},
        {"eps", header.eps, header.eps > 0.0f},
        {"weight_decay", header.weight_decay, header.weight_decay >= 0.0f},
        {"clip_norm", header.clip_norm, header.clip_norm >= 0.0f},
        {"min_lr_factor", header.min_lr_factor,
         header.min_lr_factor >= 0.0f && header.min_lr_factor <= 1.0f},
    };
    for (const Rule& r : rules)
      if (!std::isfinite(r.value) || !r.in_range)
        return contract_detail::Violation(std::string("plan header ") + r.name +
                                      " is not a finite value in its range");
  }

  // The header's I/O slots are written by the data feeder each step and
  // read through typed pointers, so they must be in bounds and element-
  // aligned (labels are i32 or f32 — 4 bytes either way).
  // The slots live in the arena: a rodata or a source ref there would be
  // read through the engine's arena pointer and written by the feeder.
  uint64_t input_bytes = 0;
  if (IsRodataRef(header.input_ref) || IsSourceRef(header.input_ref) ||
      RefOffset(header.input_ref) % sizeof(float) != 0 ||
      !MulOk(header.input_floats, sizeof(float), &input_bytes) ||
      !RangeOk(RefOffset(header.input_ref), input_bytes,
               header.arena_size))
    return contract_detail::Violation("plan input slot out of bounds");
  if (header.label_kind != 0 &&
      (IsRodataRef(header.label_ref) ||
       IsSourceRef(header.label_ref) ||
       RefOffset(header.label_ref) % sizeof(float) != 0 ||
       !RangeOk(RefOffset(header.label_ref), header.label_bytes,
                header.arena_size)))
    return contract_detail::Violation("plan label slot out of bounds");
  if (IsRodataRef(header.loss_ref) || IsSourceRef(header.loss_ref) ||
      RefOffset(header.loss_ref) % sizeof(float) != 0 ||
      !RangeOk(RefOffset(header.loss_ref), sizeof(float),
               header.arena_size))
    return contract_detail::Violation("plan loss slot out of bounds");

  // The data contract (v22, S2 #148 / S3 #149): below v22 the words are
  // padding and must be zero; above, the flags carry no unknown bit, a
  // split carries a finite fraction in (0, 1) and only a split carries
  // split fields, and a pinned corpus names its hash.
  const bool data_tail = header.tokenizer_hash != 0 || header.corpus_hash != 0 ||
                         header.split_seed != 0 ||
                         header.val_split_hash != 0 ||
                         header.val_frac_bits != 0 || header.data_flags != 0;
  if (header.version < kSeeuDataContractVersion && data_tail)
    return contract_detail::Violation(
        "plan carries data-contract fields below v" +
        std::to_string(kSeeuDataContractVersion));
  if (header.data_flags & ~kDataFlagsMask)
    return contract_detail::Violation("plan data_flags carries unknown bits");
  if (header.data_flags & kDataFlagSplit) {
    const float frac = std::bit_cast<float>(header.val_frac_bits);
    if (!std::isfinite(frac) || !(frac > 0.0f) || !(frac < 1.0f))
      return contract_detail::Violation(
          "plan header val_frac is not a finite value in its range");
  } else if (header.val_frac_bits != 0 || header.split_seed != 0 ||
             header.val_split_hash != 0) {
    return contract_detail::Violation(
        "plan carries split fields without the split flag");
  }
  if ((header.data_flags & kDataFlagCorpusPinned) && header.corpus_hash == 0)
    return contract_detail::Violation("plan pins a corpus without its hash");
  return std::nullopt;
}

/// What the feeder knows of a corpus before it reads a record.
struct CorpusShape {
  uint32_t input_kind = 0;  // 0 feature rows, 1 token records
  uint64_t input_dim = 0;   // features per row, or tokens per record
  uint32_t label_kind = 0;  // 0 none, 1 class index, 2 dense vector
  uint64_t label_bytes_per_sample = 0;
  // SDS v3 (S2 #148 / S3 #149): the tokenizer the records were encoded by
  // (0 = the corpus names none), and whether each record carries a
  // per-position loss mask (labels then stage -1 at ignored positions).
  uint64_t tokenizer_hash = 0;
  bool masked = false;
  // The corpus identity (ContentHash64 of the SDS file; 0 = built in
  // memory), checked only when the plan pins one (kDataFlagCorpusPinned).
  uint64_t corpus_hash = 0;
};

/// The corpus's shape fits the plan's I/O slots — the feeder contract's
/// geometry half. (Its value half — every class label and token id below
/// the plan's widths — reads the records and stays with the feeder.)
inline std::optional<std::string> CorpusGeometryViolation(
    const PlanHeader& header, const CorpusShape& corpus) {
  using contract_detail::Violation;
  // The tokenizer identity (v22): a plan bound to a tokenizer section runs
  // only on a corpus encoded by that section — the same vocabulary, merges
  // and special tokens, or the ids mean other strings. A corpus that names
  // no tokenizer (SDS v1 / v2, or an unbound v3) makes no claim and is
  // taken on its geometry alone; one that names another is refused.
  if (header.tokenizer_hash != 0 && corpus.tokenizer_hash != 0 &&
      corpus.tokenizer_hash != header.tokenizer_hash)
    return Violation(
        "the dataset was tokenized by another tokenizer than the plan's "
        "model (tokenizer hash mismatch)");
  // The corpus pin (v22, S2 #148): a plan compiled over a corpus (--data)
  // runs on that corpus — the records the accountant counted and the split
  // (S3) was drawn from — and refuses any other, even of the same shape.
  if ((header.data_flags & kDataFlagCorpusPinned) != 0 &&
      corpus.corpus_hash != header.corpus_hash)
    return Violation(
        "the plan is pinned to another corpus (corpus hash mismatch)");
  // The loss mask (v22): a masked corpus stages -1 labels the masked
  // cross-entropy ignores; a plan compiled for the plain loss would train
  // on them as class -1, and a masked plan over a plain corpus would read
  // every position as a target the compile did not price.
  const bool plan_masked = (header.data_flags & kDataFlagMaskedLoss) != 0;
  if (corpus.masked && !plan_masked)
    return Violation(
        "the dataset carries a loss mask, but the compiled plan takes every "
        "position as a target");
  if (plan_masked && !corpus.masked)
    return Violation(
        "the compiled plan takes a loss mask the dataset does not carry");
  if (header.input_kind == 1) {
    // Token plan: the corpus must be token records of exactly seq_len ids
    // (staging serves batch / seq_len whole records into batch i32 rows).
    if (corpus.input_kind != 1)
      return Violation(
          "the compiled plan takes token ids, but the dataset carries "
          "feature rows");
    if (corpus.input_dim != header.seq_len)
      return Violation(
          "dataset sequence length does not match the compiled plan");
    if (header.input_floats != header.batch)
      return Violation(
          "token plan input slot does not hold one id per batch row");
    // A token plan takes class labels (next-token ids) or none at all
    // (distillation). Only the class-label case stages a label slot to
    // check; a dense-label token plan is a nonsense configuration.
    if (header.label_kind == 1) {
      uint64_t batch_label_bytes = 0;
      if (!MulOk(header.batch, sizeof(int32_t), &batch_label_bytes) ||
          batch_label_bytes != header.label_bytes)
        return Violation(
            "token plan label slot does not hold one id per batch row");
    } else if (header.label_kind != 0) {
      return Violation(
          "token plans take class labels or none, not dense labels");
    }
    return std::nullopt;
  }
  if (corpus.input_kind != 0)
    return Violation(
        "the compiled plan takes feature rows, but the dataset carries "
        "token ids");
  uint64_t expected_floats = 0;
  if (!MulOk(header.batch, corpus.input_dim, &expected_floats) ||
      expected_floats != header.input_floats)
    return Violation("dataset input width does not match the compiled plan");
  if (header.label_kind != 0) {
    if (corpus.label_kind != header.label_kind)
      return Violation("dataset label kind does not match the compiled plan");
    // Same kind is not enough: batch staging copies the dataset's per-sample
    // width into the plan's fixed label slot, so the widths must agree too.
    uint64_t batch_label_bytes = 0;
    if (!MulOk(header.batch, corpus.label_bytes_per_sample,
               &batch_label_bytes) ||
        batch_label_bytes != header.label_bytes)
      return Violation("dataset label width does not match the compiled plan");
  }
  return std::nullopt;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_CONTRACT_PLAN_CONTRACT_H_
