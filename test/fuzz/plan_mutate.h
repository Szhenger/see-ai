#ifndef SEEML_TEST_FUZZ_PLAN_MUTATE_H_
#define SEEML_TEST_FUZZ_PLAN_MUTATE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>

#include "source/contract/bounds.h"
#include "source/plan/update_types.h"
#include "test/fuzz/fuzz_reach.h"

// =============================================================================
// A plan-aware edit for the fuzz harness (CR3 #208).
//
// A plan is a header, four instruction streams and megabytes of weights. A
// byte-level mutator spends its edits where the bytes are — the weights —
// and the validator's operand rules, which stand between a hostile plan and
// an out-of-bounds kernel, see almost none of them: with one bounds rule
// removed from the validator, ninety seconds of libFuzzer over the seed
// corpus (28,179 units, 2026-11-24) found nothing. MutatePlanOperand puts
// one edit where those rules read: one field of one instruction of one
// program takes a value a bounds check must refuse or a kernel must survive.
//
// It is a pure function of the input and four random words, so the replay
// suite runs it in every build; the libFuzzer build calls it from
// LLVMFuzzerCustomMutator (test/fuzz/binary_formats.cc) for three in four edits
// of a plan-arm input and leaves the rest to libFuzzer's own mutator.
// =============================================================================

namespace seeml::fuzz {

/// Rewrites one operand field of one instruction of `input`, a framed
/// plan-arm unit (the selector byte, then the plan). False, and `input`
/// untouched, when the bytes after the selector are not a plan header
/// naming at least one instruction stream inside the input.
inline bool MutatePlanOperand(std::span<uint8_t> input, uint64_t r0,
                              uint64_t r1, uint64_t r2, uint64_t r3) {
  using seeml::update::MulOk;
  using seeml::update::PlanHeader;
  using seeml::update::RangeOk;
  using seeml::update::UpdateInstruction;
  if (input.size() < 1 + sizeof(PlanHeader)) return false;
  const std::span<uint8_t> plan = input.subspan(1);
  PlanHeader header;
  std::memcpy(&header, plan.data(), sizeof(header));
  struct Stream {
    uint64_t offset, count;
  };
  const Stream named[] = {{header.train_instr_offset, header.train_instr_count},
                          {header.merge_instr_offset, header.merge_instr_count},
                          {header.eval_instr_offset, header.eval_instr_count},
                          {header.step_instr_offset, header.step_instr_count}};
  Stream streams[std::size(named)];
  size_t live = 0;
  for (const Stream& s : named) {
    uint64_t bytes = 0;
    if (s.count != 0 && MulOk(s.count, sizeof(UpdateInstruction), &bytes) &&
        RangeOk(s.offset, bytes, plan.size()))
      streams[live++] = s;
  }
  if (live == 0) return false;
  const Stream& stream = streams[r0 % live];
  // The instruction: an opcode drawn evenly from those the stream holds,
  // then one of its instances. Drawn evenly over instructions, a plan's one
  // bias add or one loss is an edit in hundreds — and each opcode has its
  // own operand rule.
  const uint8_t* const first = plan.data() + stream.offset;
  const auto opcode_at = [first](uint64_t i) {
    uint16_t opcode;
    std::memcpy(&opcode, first + i * sizeof(UpdateInstruction), sizeof(opcode));
    return opcode;
  };
  uint16_t present[64];
  size_t distinct = 0;
  for (uint64_t i = 0; i < stream.count && distinct < std::size(present); ++i) {
    const uint16_t opcode = opcode_at(i);
    bool seen = false;
    for (size_t k = 0; k < distinct; ++k) seen |= present[k] == opcode;
    if (!seen) present[distinct++] = opcode;
  }
  const uint16_t chosen = present[r1 % distinct];
  uint64_t instances = 0;
  for (uint64_t i = 0; i < stream.count; ++i)
    instances += opcode_at(i) == chosen;
  uint64_t index = 0;
  for (uint64_t i = 0, nth = (r1 >> 16) % instances; i < stream.count; ++i)
    if (opcode_at(i) == chosen && nth-- == 0) {
      index = i;
      break;
    }
  uint8_t* const at =
      plan.data() + stream.offset + index * sizeof(UpdateInstruction);
  UpdateInstruction ins;
  std::memcpy(&ins, at, sizeof(ins));

  // The field: a ref, a dimension word, the flags, the immediate or — one
  // edit in sixteen — the opcode itself.
  const uint64_t which = r2 % 16;
  uint64_t value = which < 8     ? ins.in[which % 4]
                   : which < 14  ? ins.out[which % 3]
                   : which == 14 ? ins.flags
                                 : ins.opcode;
  static constexpr uint64_t kInteresting[] = {
      0,          1,          2,          3,          4,          8,
      64,         255,        256,        4095,       4096,       65535,
      65536,      0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1ull << 32, 1ull << 40,
      1ull << 62, ~0ull >> 1, 1ull << 63, ~0ull};
  const uint64_t pick = r3 >> 8;
  switch (r3 % 6) {
    case 0:  // an interesting value
      value = kInteresting[pick % std::size(kInteresting)];
      break;
    case 1:  // a little more, in floats
      value += (1 + pick % 64) * sizeof(float);
      break;
    case 2:  // a little less
      value -= (1 + pick % 64) * sizeof(float);
      break;
    case 3:  // the arena's end, give or take: where a range stops fitting
      value = header.arena_size - sizeof(float) * (pick % 64) +
              sizeof(float) * ((pick >> 8) % 64);
      break;
    case 4:  // far outside every mapping, the address-space bits kept
      value =
          (value & (seeml::update::kRodataBit | seeml::update::kSourceBit)) |
          (1ull << (32 + pick % 24));
      break;
    default:  // one bit
      value ^= 1ull << (pick % 64);
      break;
  }
  if (which < 8)
    ins.in[which % 4] = value;
  else if (which < 14)
    ins.out[which % 3] = value;
  else if (which == 14)
    ins.flags = static_cast<uint16_t>(value);
  else
    ins.opcode = static_cast<uint16_t>(value);
  std::memcpy(at, &ins, sizeof(ins));
  return true;
}

}  // namespace seeml::fuzz

#endif  // SEEML_TEST_FUZZ_PLAN_MUTATE_H_
