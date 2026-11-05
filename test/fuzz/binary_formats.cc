// =============================================================================
// libFuzzer harness over the three attacker/corruption-facing parsers:
//
//   [0] LoadSmf                 — the model container reader
//   [1] UpdateEngine::Initialize — the .seeu plan loader + validator
//   [2] Dataset::LoadFromFile    — the corpus reader
//   [3] LoadSmf + UpdateCompiler::Compile — the semantic layer past the
//       loader (sema, parser, passes, driver), which consumes
//       attacker-controlled loader-ACCEPTED models and would otherwise be
//       fuzz-blind
//
// The first input byte selects the parser; the rest is the file image. The
// contract under test: arbitrary bytes may be REJECTED but must never crash,
// overflow, or over-read — the runtime trusts these validators completely
// (Execute() runs unchecked after Initialize() accepts a plan).
//
// Build: cmake -DSEEML_FUZZ=ON (clang), then run seeml_fuzz_formats.
// =============================================================================

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

#include "source/plan/update_types.h"
#include "compiler/driver/update_compiler.h"
#include "runtime/pipeline/dataset.h"
#include "runtime/dispatcher/update_engine.h"
#include "source/identity/hash.h"
#include "compiler/frontend/ingressor/model_reader.h"

namespace {

std::string TempPath(const char* tag) {
  return std::string("/tmp/seeml_fuzz_") + tag + "_" +
         std::to_string(::getpid());
}

void WriteBytes(const std::string& path, const uint8_t* data, size_t size) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  if (size) std::fwrite(data, 1, size, f);
  std::fclose(f);
}

}  // namespace

namespace {
// The bounds an accepted plan must meet before the harness executes it.
constexpr uint64_t kFuzzArenaCap = 64ull << 20;       // 64 MiB
constexpr uint64_t kFuzzInstructionCap = 4096;
constexpr uint64_t kFuzzGemmWorkCap = 100000000ull;  // M*N*K summed

bool IsGemm(uint16_t opcode) {
  switch (static_cast<seeml::update::OpCode>(opcode)) {
    case seeml::update::OpCode::kGemmNN:
    case seeml::update::OpCode::kGemmNT:
    case seeml::update::OpCode::kGemmTN:
    case seeml::update::OpCode::kGemmAccNN:
    case seeml::update::OpCode::kGemmNNQ8:
    case seeml::update::OpCode::kGemmNTQ8:
    case seeml::update::OpCode::kGemmNNBF16:
    case seeml::update::OpCode::kGemmNTBF16:
      return true;
    default:
      return false;
  }
}

// One training step of an accepted plan on a synthesized corpus; false when
// the plan is outside the harness's bounds or no corpus fits its header.
bool ExecuteOnce(seeml::update_rt::UpdateEngine& engine,
                 const std::vector<uint8_t>& plan) {
  seeml::update::PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  if (h.arena_size > kFuzzArenaCap) return false;
  const uint64_t instructions = h.train_instr_count + h.merge_instr_count +
                                h.eval_instr_count + h.step_instr_count;
  if (instructions > kFuzzInstructionCap) return false;
  // The loader proved every section in bounds; the GEMM work is summed
  // with saturation, so a fuzzed dimension cannot wrap the cap away.
  uint64_t work = 0;
  const struct { uint64_t off, count; } programs[] = {
      {h.train_instr_offset, h.train_instr_count},
      {h.merge_instr_offset, h.merge_instr_count},
      {h.eval_instr_offset, h.eval_instr_count},
      {h.step_instr_offset, h.step_instr_count}};
  for (const auto& p : programs)
    for (uint64_t i = 0; i < p.count; ++i) {
      seeml::update::UpdateInstruction ins;
      std::memcpy(&ins, plan.data() + p.off + i * sizeof(ins), sizeof(ins));
      if (!IsGemm(ins.opcode)) continue;
      const long double w = static_cast<long double>(ins.out[0]) *
                            static_cast<long double>(ins.out[1]) *
                            static_cast<long double>(ins.out[2]);
      if (w > static_cast<long double>(kFuzzGemmWorkCap)) return false;
      work += static_cast<uint64_t>(w);
      if (work > kFuzzGemmWorkCap) return false;
    }
  if (h.batch == 0) return false;
  std::expected<seeml::update_rt::Dataset, std::string> data =
      std::unexpected(std::string("no corpus"));
  if (h.input_kind == 1) {
    if (h.seq_len == 0 || h.batch % h.seq_len != 0) return false;
    const uint64_t records = h.batch / h.seq_len;
    std::vector<int32_t> tokens(records * (h.seq_len + 1), 0);
    data = seeml::update_rt::Dataset::FromTokens(std::move(tokens), records,
                                                 h.seq_len);
  } else {
    if (h.input_floats % h.batch != 0) return false;
    const uint64_t input_dim = h.input_floats / h.batch;
    std::vector<float> inputs(h.input_floats, 0.0f);
    std::vector<uint8_t> labels;
    uint64_t label_dim = 0;
    if (h.label_kind == 1) {
      if (h.label_bytes != h.batch * sizeof(int32_t)) return false;
      labels.assign(h.label_bytes, 0);
    } else if (h.label_kind == 2) {
      if (h.label_bytes % (h.batch * sizeof(float)) != 0) return false;
      label_dim = h.label_bytes / (h.batch * sizeof(float));
      labels.assign(h.label_bytes, 0);
    } else if (h.label_kind != 0) {
      return false;
    }
    data = seeml::update_rt::Dataset::FromMemory(
        std::move(inputs), std::move(labels), h.batch, input_dim,
        h.label_kind, label_dim);
  }
  if (!data) return false;
  seeml::update_rt::TrainOptions options;
  options.log_every = 0;
  auto r = engine.Train(*data, 1, options);
  (void)r;
  return true;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1) return 0;
  const uint8_t selector = data[0] % 4;
  const uint8_t* body = data + 1;
  const size_t body_size = size - 1;

  switch (selector) {
    case 0: {
      const std::string path = TempPath("smf");
      WriteBytes(path, body, body_size);
      auto r = seeml::update::LoadSmf(path);
      (void)r;
      std::remove(path.c_str());
      break;
    }
    case 1: {
      // The integrity hash would reject virtually every mutated input at the
      // first gate and starve the deeper validators of coverage. The hash
      // exists to catch corruption, not adversaries — so the harness seals
      // the blob the same way the compiler does, letting the fuzzer attack
      // the section/instruction/bounds validation underneath.
      std::vector<uint8_t> plan(body, body + body_size);
      if (plan.size() >= sizeof(seeml::update::PlanHeader)) {
        constexpr size_t kHashAt =
            offsetof(seeml::update::PlanHeader, plan_hash);
        // PlanSelfHash treats the field as zero itself — the runtime's exact
        // seal (v4+); a serial Fnv1a64 here would fail the load-time hash
        // gate on every input and starve the deeper validators of coverage.
        const uint64_t h =
            seeml::update::PlanSelfHash(plan.data(), plan.size(), kHashAt);
        std::memcpy(plan.data() + kHashAt, &h, sizeof(h));
      }
      // Bounded before the load as well as before the step: the load
      // allocates the header's arena, and a header asking for more than
      // the harness's cap is not loaded — the engine refuses an arena
      // beyond the host's memory on its own (a unit test holds it to
      // that), and anything between the cap and that bound would only
      // exhaust the fuzzing host.
      if (plan.size() >= sizeof(seeml::update::PlanHeader)) {
        seeml::update::PlanHeader h;
        std::memcpy(&h, plan.data(), sizeof(h));
        if (h.arena_size > kFuzzArenaCap) break;
      }
      seeml::update_rt::UpdateEngine engine;
      if (!engine.LoadFromMemory(plan.data(), plan.size())) break;
      // An accepted plan is EXECUTED (CR3 #144 F22): one training step on a
      // corpus synthesized from the header's geometry — zero features or
      // token ids, zero labels, every id below any vocabulary — so the
      // kernels run over the fuzzed instruction stream under the
      // sanitizer, not only the validators. Bounded first: the arena, the
      // instruction count and the GEMM work, so no input turns into a
      // minutes-long step that libFuzzer would report as a hang.
      if (!ExecuteOnce(engine, plan)) break;
      break;
    }
    case 2: {
      const std::string path = TempPath("sds");
      WriteBytes(path, body, body_size);
      auto r = seeml::update_rt::Dataset::LoadFromFile(path);
      (void)r;
      std::remove(path.c_str());
      break;
    }
    case 3: {
      // The contract under test: a loader-accepted model may be REFUSED by
      // sema/parser/driver but must never crash them. Tiny batch and budget
      // keep per-input compile cost bounded.
      const std::string path = TempPath("smfc");
      WriteBytes(path, body, body_size);
      if (auto model = seeml::update::LoadSmf(path)) {
        seeml::update::UpdateConfig config;
        config.batch = 2;
        config.memory_budget_bytes = 64u << 20;
        auto compiled = seeml::update::UpdateCompiler(config).Compile(*model);
        (void)compiled;
      }
      std::remove(path.c_str());
      break;
    }
  }
  return 0;
}
