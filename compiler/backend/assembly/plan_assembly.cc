#include "compiler/backend/assembly/plan_assembly.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#include "compiler/diagnostics/generating/error.h"
#include "source/contract/split.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace generating = seeml::diag::generating;

namespace {

float CounterRandn(uint64_t seed, uint64_t index) {
  // Stride the counter by the two draws each element consumes: with a
  // stride of one, element j's second draw and element j+1's first draw
  // would mix the identical counter, coupling every sample's magnitude
  // to its neighbor's angle (u1_{j+1} == u2_j + 2^-53).
  uint64_t s = seed + 0x9E3779B97F4A7C15ULL * (2 * index + 1);
  auto next = [&s]() { return SplitMix64Next(&s); };  // splitmix64
  // u1 in (0, 1] keeps the log finite; u2 in [0, 1).
  const double u1 = (static_cast<double>(next() >> 11) + 1.0) * 0x1.0p-53;
  const double u2 = static_cast<double>(next() >> 11) * 0x1.0p-53;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                            std::cos(kTwoPi * u2));
}

uint64_t PageUp(uint64_t off) {
  return (off + kSeeuRodataAlignment - 1) & ~(kSeeuRodataAlignment - 1);
}

}  // namespace

std::vector<uint8_t> SeedPersistentImage(const ArenaBinding& binding) {
  std::vector<uint8_t> persist_init(binding.persistent_size, 0);
  for (const ParamInit& p : binding.params) {
    if (p.init != "randn") continue;  // zeros already
    auto* dst = reinterpret_cast<float*>(persist_init.data() + p.offset);
    const uint64_t count = static_cast<uint64_t>(p.value->shape().volume());
    ParallelFor(count, 4096, [&](size_t begin, size_t end, size_t) {
      for (size_t j = begin; j < end; ++j)
        dst[j] = CounterRandn(p.seed, j) * p.std;
    });
  }
  return persist_init;
}

std::expected<std::vector<EmitEntry>, std::string> BuildEmitTable(
    const MergeProgram& merge, const std::vector<GraftedAdapter>& adapters,
    const GraphBuild& build,
    const std::unordered_map<const sir::Value*, uint64_t>& merge_bound) {
  std::vector<EmitEntry> emit_table;
  for (const auto& [delta, adapter_index] : merge.outputs) {
    const GraftedAdapter& adapter = adapters[adapter_index];
    auto src = build.weight_sources.find(adapter.frozen_weight);
    if (src == build.weight_sources.end())
      return generating::Error(generating::kDriver,
                               "adapter weight lacks SMF source");
    emit_table.push_back({.smf_data_offset = src->second->data_offset,
                          .byte_size = src->second->byte_size,
                          .arena_offset = RefOffset(merge_bound.at(delta))});
  }
  return emit_table;
}

uint64_t LayOutPlan(PlanHeader& header, const PlanPrograms& programs,
                    const ArenaBinding& binding, uint64_t persist_init_bytes,
                    uint64_t emit_count) {
  uint64_t off = AlignUp(sizeof(PlanHeader));
  header.train_instr_offset = off;
  header.train_instr_count = programs.train.size();
  off += programs.train.size() * sizeof(UpdateInstruction);
  header.merge_instr_offset = off;
  header.merge_instr_count = programs.merge.size();
  off += programs.merge.size() * sizeof(UpdateInstruction);
  header.eval_instr_offset = off;
  header.eval_instr_count = programs.eval.size();
  off += programs.eval.size() * sizeof(UpdateInstruction);
  header.step_instr_offset = off;
  header.step_instr_count = programs.step.size();
  off += programs.step.size() * sizeof(UpdateInstruction);
  // rodata on a page boundary (schema.h kSeeuRodataAlignment): with the
  // plan itself page-aligned, a GPU backend wraps the frozen weights
  // zero-copy instead of duplicating them.
  off = PageUp(off);
  header.rodata_offset = off;
  header.rodata_size = binding.rodata_size;
  off = AlignUp(off + binding.rodata_size);
  header.persist_init_offset = off;
  header.persist_init_size = persist_init_bytes;
  off = AlignUp(off + persist_init_bytes);
  header.emit_table_offset = off;
  header.emit_count = emit_count;
  off += emit_count * sizeof(EmitEntry);
  // Pad the blob to a page multiple: a zero-copy wrap of the rodata section
  // rounds its length up to whole pages, and those pages must be the plan's.
  return PageUp(off);
}

std::vector<uint8_t> AssemblePlan(const PlanHeader& header,
                                  const PlanPrograms& programs,
                                  const ArenaBinding& binding,
                                  const std::vector<uint8_t>& persist_init,
                                  const std::vector<EmitEntry>& emit_table,
                                  uint64_t size, bool consume) {
  std::vector<uint8_t> plan;
  plan.reserve(size);
  auto pad_to = [&](uint64_t at) { plan.resize(at, 0); };
  auto append = [&](uint64_t at, const void* src, size_t n) {
    pad_to(at);
    const auto* p = static_cast<const uint8_t*>(src);
    plan.insert(plan.end(), p, p + n);
  };
  auto append_program = [&](uint64_t at,
                            const std::vector<UpdateInstruction>& program) {
    append(at, program.data(), program.size() * sizeof(UpdateInstruction));
  };
  append(0, &header, sizeof(header));
  append_program(header.train_instr_offset, programs.train);
  append_program(header.merge_instr_offset, programs.merge);
  append_program(header.eval_instr_offset, programs.eval);
  if (!programs.step.empty())
    append_program(header.step_instr_offset, programs.step);

  // A consuming compile releases each source payload after its last pack:
  // resident weights stay ~one copy (the model shrinking as the plan
  // grows) instead of two. Tied weights pack once per SIR value, so the
  // release waits for the last of them.
  std::unordered_map<const SmfTensor*, size_t> packs_left;
  if (consume)
    for (const RodataPack& pack : binding.rodata_packs)
      ++packs_left[pack.source];
  for (const RodataPack& pack : binding.rodata_packs) {
    const uint64_t at = header.rodata_offset + pack.offset;
    if (pack.storage == RodataPack::Storage::kF32) {
      append(at, pack.source->data.data(), pack.bytes);
    } else {
      pad_to(at + pack.bytes);  // the (smaller) packed form, in place
      PackRodata(pack, plan.data() + at);
    }
    if (consume && --packs_left[pack.source] == 0) {
      // The consuming compile owns these models: they are not const objects.
      auto& payload = const_cast<SmfTensor*>(pack.source)->data;
      SmfBytes().swap(payload);
    }
  }
  if (!persist_init.empty())
    append(header.persist_init_offset, persist_init.data(),
           persist_init.size());
  if (!emit_table.empty())
    append(header.emit_table_offset, emit_table.data(),
           emit_table.size() * sizeof(EmitEntry));
  pad_to(size);
  return plan;
}

void SealPlan(std::vector<uint8_t>& plan) {
  const uint64_t plan_hash =
      PlanSelfHash(plan.data(), plan.size(), offsetof(PlanHeader, plan_hash));
  std::memcpy(plan.data() + offsetof(PlanHeader, plan_hash), &plan_hash,
              sizeof(plan_hash));
}

}  // namespace seeml::update
