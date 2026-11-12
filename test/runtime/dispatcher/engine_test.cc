// =============================================================================
// Engine tests: the subsystem usage contracts the engine enforces at each
// boundary (plan / executor / feeder), the diagnostics contract at the
// engine's Train boundary, and the runtime unit registry.
// =============================================================================

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>

#include "compiler/driver/update_compiler.h"
#include "runtime/dispatcher/contract.h"
#include "source/contract/plan_contract.h"
#include "runtime/dispatcher/update_engine.h"
#include "source/identity/hash.h"
#include "source/plan/update_types.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update_rt;
namespace up = seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeClassificationData;
using seeml::testing::MakeMlp;

constexpr int64_t kInDim = 6;
constexpr int64_t kHidden = 10;
constexpr int64_t kOutDim = 3;
constexpr int64_t kBatch = 4;

up::PlanHeader HeaderOf(const up::CompiledUpdate& compiled) {
  up::PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));
  return h;
}

// =============================================================================
// The diagnostics contract
// =============================================================================

TEST(EngineContract, RecognizesEveryRegisteredUnit) {
  EXPECT_TRUE(WellFormedDiagnostic("Dataset: bad magic in 'c.sds'"));
  EXPECT_TRUE(WellFormedDiagnostic("PlanValidator: unknown opcode 9"));
  EXPECT_TRUE(WellFormedDiagnostic("UpdateEngine: no plan loaded"));
  EXPECT_TRUE(WellFormedDiagnostic("DurableIO: cannot write '/x'"));
  EXPECT_TRUE(WellFormedDiagnostic("Checkpoint: payload is corrupt"));
}

TEST(EngineContract, RejectsUnattributableDiagnostics) {
  EXPECT_FALSE(WellFormedDiagnostic("something went wrong"));
  EXPECT_FALSE(WellFormedDiagnostic("Frobnicator: unknown unit"));
  EXPECT_FALSE(WellFormedDiagnostic("Dataset"));
  EXPECT_FALSE(WellFormedDiagnostic("Dataset: "));
  EXPECT_FALSE(WellFormedDiagnostic(""));
}

// =============================================================================
// The plan boundary
// =============================================================================

TEST(EngineContract, PlanContractAcceptsACompiledPlanAndCatchesCorruption) {
  up::SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  ASSERT_OK_AND_ASSIGN(
      up::CompiledUpdate compiled,
      up::UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  const up::PlanHeader good = HeaderOf(compiled);
  EXPECT_TRUE(VerifyPlanContract(good, compiled.plan.size()).has_value());

  up::PlanHeader oversized = good;
  oversized.persistent_size = oversized.arena_size + 1;
  const auto r_seg = VerifyPlanContract(oversized, compiled.plan.size());
  ASSERT_FALSE(r_seg.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r_seg.error()));
  EXPECT_NE(r_seg.error().find("persistent segment exceeds arena"),
            std::string::npos);

  up::PlanHeader escaped = good;
  escaped.train_instr_offset = compiled.plan.size();
  EXPECT_FALSE(
      VerifyPlanContract(escaped, compiled.plan.size()).has_value());
}

TEST(EngineContract, HeaderHyperparametersAreFiniteAndInRange) {
  // CR1 #142: the step kernels consume the header's floats blindly, so the
  // load refuses any that is not finite and within the range the compiler
  // accepts — by field name.
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.batch = 4;
  header.input_floats = 16;
  header.arena_size = 4096;
  header.input_ref = up::MakeArenaRef(0);
  header.loss_ref = up::MakeArenaRef(256);
  constexpr uint64_t kPlanSize = 65536;
  ASSERT_OK(VerifyPlanContract(header, kPlanSize));
  const float nan = std::bit_cast<float>(0x7FC00000u);
  struct Case {
    const char* name;
    void (*set)(up::PlanHeader&, float);
    std::vector<float> bad;
    std::vector<float> good;
  };
  const Case cases[] = {
      {"lr", [](up::PlanHeader& h, float v) { h.lr = v; },
       {0.0f, -1e-3f, nan, INFINITY}, {1e-3f, 1.0f}},
      {"beta1", [](up::PlanHeader& h, float v) { h.beta1 = v; },
       {1.0f, -0.1f, nan}, {0.0f, 0.9f}},
      {"beta2", [](up::PlanHeader& h, float v) { h.beta2 = v; },
       {1.0f, 1.5f, nan}, {0.999f}},
      {"eps", [](up::PlanHeader& h, float v) { h.eps = v; },
       {0.0f, -1e-8f, nan}, {1e-8f}},
      {"weight_decay", [](up::PlanHeader& h, float v) { h.weight_decay = v; },
       {-0.01f, nan, INFINITY}, {0.0f, 0.01f}},
      {"clip_norm", [](up::PlanHeader& h, float v) { h.clip_norm = v; },
       {-1.0f, nan, INFINITY}, {0.0f, 1.0f}},
      {"min_lr_factor",
       [](up::PlanHeader& h, float v) { h.min_lr_factor = v; },
       {-0.1f, 1.5f, nan}, {0.0f, 1.0f}},
  };
  for (const Case& c : cases) {
    for (const float v : c.bad) {
      up::PlanHeader h = header;
      c.set(h, v);
      EXPECT_ERROR_CONTAINS(VerifyPlanContract(h, kPlanSize), c.name);
    }
    for (const float v : c.good) {
      up::PlanHeader h = header;
      c.set(h, v);
      EXPECT_OK(VerifyPlanContract(h, kPlanSize));
    }
  }
}

// =============================================================================
// The executor boundary
// =============================================================================

TEST(EngineContract, ExecutorContractSpeaksAsThePlanValidator) {
  up::UpdateInstruction bogus{};
  bogus.opcode = 0xFFFF;
  up::PlanHeader header{};
  header.arena_size = 1024;

  const auto r = VerifyExecutorContract(std::span(&bogus, 1), {}, {}, {}, {}, {},
                                        header);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_EQ(r.error().find("PlanValidator: "), 0u);
}

// =============================================================================
// The feeder boundary
// =============================================================================

TEST(EngineContract, PlanContractTiesTheStepSectionToAccumulation) {
  namespace up = seeml::update;
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.batch = 4;
  header.input_floats = 16;
  header.arena_size = 4096;
  header.input_ref = up::MakeArenaRef(0);
  header.loss_ref = up::MakeArenaRef(256);
  constexpr uint64_t kPlanSize = 65536;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  // Accumulation without a step program: nothing would ever step.
  header.grad_accum_steps = 4;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "step program presence");
  // A step program without accumulation: it would never run.
  header.grad_accum_steps = 1;
  header.step_instr_offset = 1024;
  header.step_instr_count = 2;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "step program presence");
  // Both present: accepted; the section must lie inside the blob.
  header.grad_accum_steps = 4;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.step_instr_offset = kPlanSize - 64;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "out of bounds");
  // A pre-v9 plan cannot carry the fields at all.
  header.step_instr_offset = 1024;
  header.version = up::kSeeuGradAccumVersion - 1;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "gradient-accumulation fields");
}

TEST(EngineContract, PlanContractProvesTheGemmTileFields) {
  // v11: the tiles are a throughput knob the kernels accept only with the
  // K tile on the 4-wide unroll; zero is the runtime default; below v11
  // the words are padding and must be zero.
  namespace up = seeml::update;
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.batch = 4;
  header.input_floats = 16;
  header.arena_size = 4096;
  header.input_ref = up::MakeArenaRef(0);
  header.loss_ref = up::MakeArenaRef(256);
  constexpr uint64_t kPlanSize = 65536;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.gemm_tile_k = 128;
  header.gemm_tile_n = 512;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.gemm_tile_k = 6;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "4-wide unroll");
  header.gemm_tile_k = 0;  // one default, one set: fine
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.version = up::kSeeuGemmTilesVersion - 1;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "GEMM tile fields below v11");
  header.gemm_tile_n = 0;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
}

TEST(EngineContract, PlanContractProvesTheDataContractWords) {
  // v22 (S2 #148 / S3 #149): the forty bytes past the v21 header are the
  // data contract — zero below v22, no unknown flag, a split only with a
  // fraction in (0, 1) and split fields only under the flag, a pin only
  // with its hash.
  namespace up = seeml::update;
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.batch = 4;
  header.input_floats = 16;
  header.arena_size = 4096;
  header.input_ref = up::MakeArenaRef(0);
  header.loss_ref = up::MakeArenaRef(256);
  constexpr uint64_t kPlanSize = 65536;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.tokenizer_hash = 0x1234;
  header.corpus_hash = 0x5678;
  header.data_flags = up::kDataFlagCorpusPinned | up::kDataFlagSplit;
  header.split_seed = 7;
  header.val_frac_bits = std::bit_cast<uint32_t>(0.1f);
  header.val_split_hash = 99;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
  header.version = up::kSeeuDataContractVersion - 1;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "data-contract fields below v22");
  header.version = up::kSeeuVersion;
  header.data_flags |= 1u << 7;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "unknown bits");
  header.data_flags = up::kDataFlagCorpusPinned | up::kDataFlagSplit;
  header.val_frac_bits = std::bit_cast<uint32_t>(1.0f);
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "val_frac is not a finite value in its range");
  header.val_frac_bits = std::bit_cast<uint32_t>(0.1f);
  header.data_flags = up::kDataFlagCorpusPinned;  // split fields, no flag
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "split fields without the split flag");
  header.split_seed = 0;
  header.val_frac_bits = 0;
  header.val_split_hash = 0;
  header.corpus_hash = 0;
  EXPECT_ERROR_CONTAINS(VerifyPlanContract(header, kPlanSize),
                        "pins a corpus without its hash");
  header.data_flags = 0;
  EXPECT_OK(VerifyPlanContract(header, kPlanSize));
}

TEST(EngineContract, FeederContractBindsTheTokenizerAndTheMask) {
  // A plan bound to a tokenizer runs only on a corpus encoded by it; a
  // masked corpus needs a plan that takes the mask, and the reverse.
  namespace up = seeml::update;
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.input_kind = 1;
  header.seq_len = 4;
  header.batch = 8;
  header.input_floats = 8;
  header.label_kind = 1;
  header.label_bytes = 32;
  up::CorpusShape corpus{1, 4, 1, 16, 0, false};
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  // A corpus that names no tokenizer makes no claim; one that names
  // another is refused; the same one passes.
  header.tokenizer_hash = 0xABCD;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  corpus.tokenizer_hash = 0x1234;
  EXPECT_STR_CONTAINS(up::CorpusGeometryViolation(header, corpus).value_or(""),
                      "another tokenizer");
  corpus.tokenizer_hash = 0xABCD;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  corpus.masked = true;
  EXPECT_STR_CONTAINS(up::CorpusGeometryViolation(header, corpus).value_or(""),
                      "carries a loss mask");
  header.data_flags = up::kDataFlagMaskedLoss;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  corpus.masked = false;
  EXPECT_STR_CONTAINS(up::CorpusGeometryViolation(header, corpus).value_or(""),
                      "loss mask the dataset does not carry");
  // The corpus pin: enforced only under the flag, then exactly.
  header.data_flags = 0;
  header.corpus_hash = 0x77;
  corpus.corpus_hash = 0x78;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  header.data_flags = up::kDataFlagCorpusPinned;
  EXPECT_STR_CONTAINS(up::CorpusGeometryViolation(header, corpus).value_or(""),
                      "pinned to another corpus");
  corpus.corpus_hash = 0x77;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  // The compiled split: the fed set must carry the plan's split hash.
  header.data_flags = up::kDataFlagSplit;
  header.val_split_hash = 0x99;
  EXPECT_STR_CONTAINS(up::CorpusGeometryViolation(header, corpus).value_or(""),
                      "not split as the plan compiled it");
  corpus.split_hash = 0x99;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
  header.data_flags = 0;
  corpus.split_hash = 0x98;
  EXPECT_FALSE(up::CorpusGeometryViolation(header, corpus).has_value());
}

TEST(EngineContract, ExecutorContractBindsEmbeddingTokensToTheStagedSlot) {
  // Regression: kEmbedFwd gathers table[tokens[t]] — a data-dependent
  // offset. Its token operand MUST be the staged input slot (like the
  // softmax labels), or a range-valid hostile plan gathers with arena
  // bytes read as indices.
  namespace up = seeml::update;
  up::PlanHeader header{};
  header.version = up::kSeeuVersion;
  header.input_kind = 1;
  header.batch = 4;
  header.arena_size = 4096;
  header.rodata_size = 1024;
  header.input_ref = up::MakeArenaRef(0);
  const std::vector<uint8_t> rodata(header.rodata_size);  // the table's bytes
  up::UpdateInstruction emb{};
  emb.opcode = static_cast<uint16_t>(up::OpCode::kEmbedFwd);
  emb.in[0] = up::MakeArenaRef(0);   // the staged input slot
  emb.in[1] = up::MakeRodataRef(0);  // table [8, 4]
  emb.in[2] = up::MakeArenaRef(256);
  emb.out[0] = 4;
  emb.out[1] = (uint64_t{8} << 32) | 4;
  std::vector<up::UpdateInstruction> train = {emb};
  EXPECT_OK(VerifyExecutorContract(train, {}, {}, {}, {}, rodata, header));
  // Tokens read from anywhere but the staged slot: rejected.
  train[0].in[0] = up::MakeArenaRef(512);
  EXPECT_ERROR_CONTAINS(VerifyExecutorContract(train, {}, {}, {}, {}, rodata, header),
                        "staged input slot");
  // Embedding in a plan that does not declare token input: rejected.
  train[0].in[0] = up::MakeArenaRef(0);
  header.input_kind = 0;
  EXPECT_ERROR_CONTAINS(VerifyExecutorContract(train, {}, {}, {}, {}, rodata, header),
                        "without token input");
}

TEST(EngineContract, FeederContractMatchesCorpusToPlanGeometry) {
  up::SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 2);
  ASSERT_OK_AND_ASSIGN(
      up::CompiledUpdate compiled,
      up::UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  const up::PlanHeader header = HeaderOf(compiled);

  ASSERT_OK_AND_ASSIGN(Dataset fits,
                       MakeClassificationData(32, kInDim, 3));
  EXPECT_TRUE(VerifyFeederContract(header, fits, kOutDim, 0).has_value());

  ASSERT_OK_AND_ASSIGN(Dataset wrong_width,
                       MakeClassificationData(32, kInDim + 1, 3));
  const auto r = VerifyFeederContract(header, wrong_width, kOutDim, 0);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_NE(r.error().find("input width does not match"), std::string::npos);
}

// =============================================================================
// The engine end to end
// =============================================================================

TEST(Engine, ErrorsCrossingTheTrainBoundaryAreWellFormed) {
  ASSERT_OK_AND_ASSIGN(Dataset data, MakeClassificationData(16, kInDim, 5));
  UpdateEngine engine;  // no plan loaded
  const auto r = engine.Train(data, 1);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  EXPECT_EQ(r.error(), "UpdateEngine: no plan loaded");
}


// =============================================================================
// CR2 (#143): the coverage table — every program x every check, as refusals.
// A check missing from a program is a failing row here, not a review finding
// (F07 was one: the step program sat outside the width scans).
// =============================================================================

constexpr size_t kPlanHashAt = offsetof(up::PlanHeader, plan_hash);
constexpr int kPrograms = 4;  // train, merge, eval, step
const char* const kProgramName[kPrograms] = {"train", "merge", "eval", "step"};

// A compiled plan opened for surgery: the header as a struct, the programs
// addressable, and Seal() recomputing the self-hash the loader verifies.
struct OpenPlan {
  std::vector<uint8_t> bytes;
  up::PlanHeader h{};
  explicit OpenPlan(const std::vector<uint8_t>& plan) : bytes(plan) {
    std::memcpy(&h, bytes.data(), sizeof(h));
  }
  uint64_t Offset(int program) const {
    return program == 0   ? h.train_instr_offset
           : program == 1 ? h.merge_instr_offset
           : program == 2 ? h.eval_instr_offset
                          : h.step_instr_offset;
  }
  uint64_t Count(int program) const {
    return program == 0   ? h.train_instr_count
           : program == 1 ? h.merge_instr_count
           : program == 2 ? h.eval_instr_count
                          : h.step_instr_count;
  }
  void Replace(int program, size_t index, const up::UpdateInstruction& ins) {
    std::memcpy(bytes.data() + Offset(program) + index * sizeof(ins), &ins,
                sizeof(ins));
  }
  std::vector<uint8_t> Sealed() {
    std::memcpy(bytes.data(), &h, sizeof(h));
    const uint64_t hash = up::PlanSelfHash(bytes.data(), bytes.size(), kPlanHashAt);
    std::memcpy(bytes.data() + kPlanHashAt, &hash, sizeof(hash));
    return bytes;
  }
  // An arena range of `bytes` at the arena's tail that touches no staged slot.
  uint64_t Scratch(uint64_t bytes, uint64_t slot) const {
    const uint64_t off = (h.arena_size - bytes - 64 * slot) & ~uint64_t{15};
    auto overlaps = [&](uint64_t ref, uint64_t ref_bytes) {
      const uint64_t s = up::RefOffset(ref);
      return ref != up::kNullRef && off < s + ref_bytes && s < off + bytes;
    };
    if (overlaps(h.input_ref, h.input_floats * sizeof(float)) ||
        overlaps(h.label_ref, h.label_bytes) || overlaps(h.loss_ref, 4))
      return UINT64_MAX;
    return off;
  }
};

// Both fixtures accumulate gradients: the step program exists only then
// (without accumulation the optimizer lives in the train program), and
// the table's step rows need somewhere to plant their instruction.
up::UpdateConfig AccumulatingConfig(int64_t batch) {
  up::UpdateConfig c = BaseConfig(batch);
  c.grad_accum_steps = 2;
  return c;
}
std::vector<uint8_t> MlpPlan() {
  up::SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  auto compiled = up::UpdateCompiler(AccumulatingConfig(kBatch)).Compile(model);
  return compiled ? compiled->plan : std::vector<uint8_t>{};
}
constexpr int64_t kTokVocab = 16, kTokDim = 8, kTokSeq = 8, kTokBatch = 16;
std::vector<uint8_t> TokenPlan() {
  up::SmfModel model = seeml::testing::MakeTinyTokenDecoder(
      kTokVocab, kTokDim, /*heads=*/2, kTokSeq, /*ffn=*/16, 41);
  auto compiled = up::UpdateCompiler(AccumulatingConfig(kTokBatch)).Compile(model);
  return compiled ? compiled->plan : std::vector<uint8_t>{};
}
TrainOptions QuietOptions() {
  TrainOptions o;
  o.log_every = 0;
  return o;
}

TEST(CoverageTable, TheClassWidthScanCoversEveryProgram) {
  // F07: a softmax of width 1 planted in any program disagrees with the
  // plan's width, and the load refuses it — before, the merge and step
  // programs were outside both the agreement check and the scan that
  // bounds the staged labels, so a step-program softmax's rows could be
  // indexed past by labels another program's width admitted.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  for (int p = 0; p < kPrograms; ++p) {
    OpenPlan open(plan);
    ASSERT_TRUE(open.Count(p) > 0);
    const uint64_t logits = open.Scratch(kBatch * 4, 0);
    const uint64_t loss = open.Scratch(16, 1);
    const uint64_t probs = open.Scratch(kBatch * 4, 2);
    ASSERT_TRUE(logits != UINT64_MAX && loss != UINT64_MAX && probs != UINT64_MAX);
    up::UpdateInstruction sm{};
    sm.opcode = static_cast<uint16_t>(up::OpCode::kSoftmaxXEntFwd);
    sm.in[0] = up::MakeArenaRef(logits);
    sm.in[1] = open.h.label_ref;
    sm.in[2] = up::MakeArenaRef(loss);
    sm.in[3] = up::MakeArenaRef(probs);
    sm.out[0] = open.h.batch;
    sm.out[1] = 1;  // narrower than the plan's 3 classes
    open.Replace(p, 0, sm);
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    const auto loaded = engine.LoadFromMemory(edited.data(), edited.size());
    std::printf("  [ %-5s ] class width: %s\n", kProgramName[p],
                loaded.has_value() ? "NOT checked" : loaded.error().c_str());
    ASSERT_FALSE(loaded.has_value());
    EXPECT_STR_CONTAINS(loaded.error(), "class widths");
  }
}

TEST(CoverageTable, TheMaskedLossFlagAndTheMaskedOpcodesAgree) {
  // S3 #149: the masked cross-entropy appears exactly when the header
  // carries kDataFlagMaskedLoss — a masked opcode without the flag, a
  // plain one under it, and the flag with no masked opcode are refused,
  // in every program.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  for (int p = 0; p < kPrograms; ++p) {
    OpenPlan open(plan);
    ASSERT_TRUE(open.Count(p) > 0);
    const uint64_t logits = open.Scratch(kBatch * 3 * 4, 0);
    const uint64_t loss = open.Scratch(16, 1);
    const uint64_t probs = open.Scratch(kBatch * 3 * 4, 2);
    ASSERT_TRUE(logits != UINT64_MAX && loss != UINT64_MAX && probs != UINT64_MAX);
    up::UpdateInstruction sm{};
    sm.opcode = static_cast<uint16_t>(up::OpCode::kSoftmaxXEntMaskedFwd);
    sm.in[0] = up::MakeArenaRef(logits);
    sm.in[1] = open.h.label_ref;
    sm.in[2] = up::MakeArenaRef(loss);
    sm.in[3] = up::MakeArenaRef(probs);
    sm.out[0] = open.h.batch;
    sm.out[1] = 3;
    open.Replace(p, 0, sm);
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    const auto loaded = engine.LoadFromMemory(edited.data(), edited.size());
    ASSERT_FALSE(loaded.has_value());
    EXPECT_STR_CONTAINS(loaded.error(), "without the masked-loss flag");
  }
  {
    // The flag alone: the plain cross-entropy the plan carries is refused.
    OpenPlan open(plan);
    open.h.data_flags |= up::kDataFlagMaskedLoss;
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    const auto loaded = engine.LoadFromMemory(edited.data(), edited.size());
    ASSERT_FALSE(loaded.has_value());
    EXPECT_STR_CONTAINS(loaded.error(), "carries the masked-loss flag");
  }
  {
    // A masked plan the compiler built loads with the flag. A mask marks
    // the positions of a token record, so the plan is a token plan: the
    // driver refuses --masked-loss on a feature model (no corpus could
    // feed it — the loader refuses a masked feature SDS).
    up::SmfModel model = seeml::testing::MakeTinyTokenDecoder(
        kTokVocab, kTokDim, /*heads=*/2, kTokSeq, /*ffn=*/16, 41);
    up::UpdateConfig cfg = AccumulatingConfig(kTokBatch);
    cfg.masked_loss = true;
    auto compiled = up::UpdateCompiler(cfg).Compile(model);
    ASSERT_OK(compiled);
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(compiled->plan.data(), compiled->plan.size()));
    EXPECT_NE(engine.header().data_flags & up::kDataFlagMaskedLoss, 0u);
    up::UpdateConfig feature_cfg = AccumulatingConfig(kBatch);
    feature_cfg.masked_loss = true;
    EXPECT_ERROR_CONTAINS(
        up::UpdateCompiler(feature_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 1)),
        "token models alone");
  }
}

TEST(CoverageTable, TheMetadataSectionIsProvenBeforeItIsRead) {
  // Plan v23: the loader holds the metadata section to the plan it sits in
  // (PlanMetadataViolation) — every rule, by mutation of a compiled plan.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  const OpenPlan base(plan);
  ASSERT_TRUE((base.h.data_flags & up::kDataFlagMetadata) != 0);
  const uint64_t off = up::PlanMetadataOffset(base.h);
  auto load = [](OpenPlan& open) {
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    return engine.LoadFromMemory(edited.data(), edited.size());
  };
  {
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
    EXPECT_TRUE(engine.metadata().present);
    EXPECT_NE(engine.metadata().meta.recipe_hash, 0u);
    EXPECT_TRUE(!engine.metadata().ledger.empty());
  }
  struct Edit {
    uint64_t at;      // byte offset inside the section
    uint8_t value;
    const char* says;
  };
  const uint64_t entry0 = sizeof(up::PlanMetadata);
  const Edit edits[] = {
      {offsetof(up::PlanMetadata, magic), 0x00, "bad magic"},
      {offsetof(up::PlanMetadata, format), 0x09, "unsupported plan metadata format"},
      {offsetof(up::PlanMetadata, bytes), 0x01, "size disagrees with its ledger"},
      {offsetof(up::PlanMetadata, ledger_count), 0x7F, "ledger is too long"},
      {offsetof(up::PlanMetadata, reserved0), 0x01, "reserved plan metadata words"},
      {offsetof(up::PlanMetadata, reserved1), 0x01, "reserved plan metadata words"},
      {offsetof(up::PlanMetadata, export_flags), 0x08, "unknown bits"},
      {offsetof(up::PlanMetadata, export_flags), 0x02, "without a report"},
      {offsetof(up::PlanMetadata, export_flags), 0x01, "without its hash"},
      {offsetof(up::PlanMetadata, export_report_hash), 0x01, "without a report"},
      {offsetof(up::PlanMetadata, tokenizer_parity_ids), 0x01,
       "parity ids without the parity flag"},
      {offsetof(up::PlanMetadata, logits_parity_max_delta) + 2, 0x80,
       "logits-parity figure"},
      {entry0 + 0, 0x00, "not NUL-padded text"},             // an empty name
      {entry0 + 31, 'x', "not NUL-padded text"},             // no terminator
      {entry0 + 1, 0x07, "not NUL-padded text"},             // a control byte
      {entry0 + offsetof(up::LedgerEntry, arithmetic), 0x02,
       "unknown arithmetic"},
      {entry0 + offsetof(up::LedgerEntry, instructions), 0x01,
       "an exact ledger entry counts changed instructions"},
      {entry0 + offsetof(up::LedgerEntry, arithmetic), 0x01,
       "a reordered ledger entry changes no instruction"},
  };
  for (const Edit& e : edits) {
    OpenPlan open(plan);
    open.bytes[off + e.at] = e.value;
    const auto loaded = load(open);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_STR_CONTAINS(loaded.error(), e.says);
  }
  {
    // The ledger against the programs: an instruction flagged relaxed that
    // no ledger line accounts for.
    OpenPlan open(plan);
    up::UpdateInstruction ins;
    std::memcpy(&ins, open.bytes.data() + open.Offset(0), sizeof(ins));
    ins.flags |= up::kFlagRelaxed;
    open.Replace(0, 0, ins);
    const auto loaded = load(open);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_STR_CONTAINS(loaded.error(), "the ledger accounts for 0 instruction(s)");
  }
  {
    // The flag below its version; and a plan without the flag is read as
    // every plan before v23 was (the section's bytes are then padding).
    OpenPlan old(plan);
    old.h.version = up::kSeeuMetadataVersion - 1;
    const auto refused = load(old);
    ASSERT_FALSE(refused.has_value());
    EXPECT_STR_CONTAINS(refused.error(), "metadata section below v23");
    OpenPlan bare(plan);
    bare.h.data_flags &= ~up::kDataFlagMetadata;
    const std::vector<uint8_t> edited = bare.Sealed();
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(edited.data(), edited.size()));
    EXPECT_FALSE(engine.metadata().present);
  }
  {
    // A section that would run past the blob.
    OpenPlan open(plan);
    open.bytes.resize(off + 32);
    open.h.data_flags |= up::kDataFlagMetadata;
    const auto loaded = load(open);
    ASSERT_FALSE(loaded.has_value());
  }
}

TEST(CoverageTable, TheVocabularyScanCoversEveryProgram) {
  // F07's twin for token plans: an embedding gather over a one-row table
  // planted in each program must bound every staged token id.
  const std::vector<uint8_t> plan = TokenPlan();
  ASSERT_FALSE(plan.empty());
  for (int p = 0; p < kPrograms; ++p) {
    OpenPlan open(plan);
    ASSERT_TRUE(open.Count(p) > 0);
    ASSERT_TRUE(open.h.rodata_size >= uint64_t{kTokDim} * sizeof(float));
    const uint64_t out = open.Scratch(uint64_t{kTokBatch} * kTokDim * sizeof(float), 0);
    ASSERT_TRUE(out != UINT64_MAX);
    up::UpdateInstruction emb{};
    emb.opcode = static_cast<uint16_t>(up::OpCode::kEmbedFwd);
    emb.in[0] = open.h.input_ref;
    emb.in[1] = up::MakeRodataRef(0);
    emb.in[2] = up::MakeArenaRef(out);
    emb.out[0] = open.h.batch;
    emb.out[1] = (uint64_t{1} << 32) | uint64_t{kTokDim};  // V = 1
    open.Replace(p, 0, emb);
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    ASSERT_OK(engine.LoadFromMemory(edited.data(), edited.size()));
    ASSERT_OK_AND_ASSIGN(Dataset data,
                         seeml::testing::MakeTokenCorpus(10, kTokSeq, kTokVocab, 43));
    const auto r = engine.Train(data, 1, QuietOptions());
    std::printf("  [ %-5s ] vocabulary: %s\n", kProgramName[p],
                r.has_value() ? "NOT scanned" : "refused");
    EXPECT_FALSE(r.has_value());
  }
}

TEST(CoverageTable, NoProgramWritesTheStagedSlots) {
  // N03: the feeder's proofs hold only while the staged input and label
  // slots stay the feeder's; a write to either from any program is refused
  // at load.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  for (int p = 0; p < kPrograms; ++p) {
    for (int slot = 0; slot < 2; ++slot) {
      OpenPlan open(plan);
      ASSERT_TRUE(open.Count(p) > 0);
      const uint64_t count = slot == 0 ? open.h.input_floats
                                       : open.h.label_bytes / sizeof(float);
      const uint64_t src = open.Scratch(count * sizeof(float), 0);
      ASSERT_TRUE(src != UINT64_MAX);
      up::UpdateInstruction add{};
      add.opcode = static_cast<uint16_t>(up::OpCode::kAddEW);
      add.in[0] = up::MakeArenaRef(src);
      add.in[1] = up::MakeArenaRef(src);
      add.in[2] = slot == 0 ? open.h.input_ref : open.h.label_ref;
      add.out[0] = count;
      open.Replace(p, 0, add);
      const std::vector<uint8_t> edited = open.Sealed();
      UpdateEngine engine;
      const auto r = engine.LoadFromMemory(edited.data(), edited.size());
      std::printf("  [ %-5s ] %s slot write: %s\n", kProgramName[p],
                  slot ? "label" : "input", r.has_value() ? "ACCEPTED" : "refused");
      ASSERT_FALSE(r.has_value());
      EXPECT_STR_CONTAINS(r.error(), slot == 0 ? "staged input slot" : "staged label slot");
    }
  }
}

TEST(CoverageTable, NoProgramWritesThroughASourceRef) {
  // The validator's rule, per program: a source ref is read-only everywhere.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  for (int p = 0; p < kPrograms; ++p) {
    OpenPlan open(plan);
    ASSERT_TRUE(open.Count(p) > 0);
    const uint64_t src = open.Scratch(64, 0);
    ASSERT_TRUE(src != UINT64_MAX);
    up::UpdateInstruction add{};
    add.opcode = static_cast<uint16_t>(up::OpCode::kAddEW);
    add.in[0] = up::MakeArenaRef(src);
    add.in[1] = up::MakeArenaRef(src);
    add.in[2] = up::MakeSourceRef(0);
    add.out[0] = 16;
    open.Replace(p, 0, add);
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    const auto r = engine.LoadFromMemory(edited.data(), edited.size());
    std::printf("  [ %-5s ] write through a source ref: %s\n", kProgramName[p],
                r.has_value() ? "ACCEPTED" : "refused");
    EXPECT_FALSE(r.has_value());
  }
}

TEST(CoverageTable, TheHeaderSlotsAndTheRodataSectionAreProven) {
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  {
    // N04: a header slot carrying the source bit would bypass the feeder's
    // validated labels.
    OpenPlan open(plan);
    open.h.label_ref = up::MakeSourceRef(up::RefOffset(open.h.label_ref));
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    EXPECT_ERROR(engine.LoadFromMemory(edited.data(), edited.size()));
  }
  {
    OpenPlan open(plan);
    open.h.input_ref = up::MakeRodataRef(0);
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    EXPECT_ERROR(engine.LoadFromMemory(edited.data(), edited.size()));
  }
  {
    // N05: the rodata section's own offset, which every rodata ref is
    // relative to.
    OpenPlan open(plan);
    ASSERT_TRUE(open.h.rodata_size > 0);
    open.h.rodata_offset += 4;
    const std::vector<uint8_t> edited = open.Sealed();
    UpdateEngine engine;
    const auto r = engine.LoadFromMemory(edited.data(), edited.size());
    ASSERT_FALSE(r.has_value());
    EXPECT_STR_CONTAINS(r.error(), "rodata section");
  }
}

TEST(CoverageTable, TheFeederRefusesTheWrongCorpus) {
  // Label kind, label width and the token bound — the dataset side of the
  // table, one row each against the plans above.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_FALSE(plan.empty());
  UpdateEngine engine;
  ASSERT_OK(engine.LoadFromMemory(plan.data(), plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset dense,
                       seeml::testing::MakeRegressionData(64, kInDim, 3, 7));
  EXPECT_ERROR(engine.Train(dense, 1, QuietOptions()));  // label kind
  ASSERT_OK_AND_ASSIGN(Dataset tokens,
                       seeml::testing::MakeTokenCorpus(10, kTokSeq, kTokVocab, 43));
  EXPECT_ERROR(engine.Train(tokens, 1, QuietOptions()));  // input kind
  const std::vector<uint8_t> token_plan = TokenPlan();
  ASSERT_FALSE(token_plan.empty());
  UpdateEngine token_engine;
  ASSERT_OK(token_engine.LoadFromMemory(token_plan.data(), token_plan.size()));
  ASSERT_OK_AND_ASSIGN(Dataset wide,
                       seeml::testing::MakeTokenCorpus(10, kTokSeq, 2 * kTokVocab, 43));
  EXPECT_ERROR(token_engine.Train(wide, 1, QuietOptions()));  // token bound
}

TEST(CoverageTable, QuantizedColumnScalesAreFinitePositive) {
  // F33: the column scales a q8 GEMM reads from rodata are multiplied in
  // blindly; a NaN or a zero among them is refused at load, by index. A
  // 2x2x2 q8 GEMM planted as train[0]: A and C in scratch, B (4 int8) at
  // rodata 16, its two column scales at rodata 0.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_TRUE(!plan.empty());
  OpenPlan open(plan);
  ASSERT_TRUE(open.h.rodata_size >= 64);
  const uint64_t a = open.Scratch(64, 0), c = open.Scratch(64, 1);
  ASSERT_TRUE(a != UINT64_MAX && c != UINT64_MAX);
  up::UpdateInstruction g{};
  g.opcode = static_cast<uint16_t>(up::OpCode::kGemmNNQ8);
  g.flags = up::kFlagQ8ColScale;
  g.in[0] = up::MakeArenaRef(a);
  g.in[1] = up::MakeRodataRef(16);
  g.in[2] = up::MakeArenaRef(c);
  g.in[3] = up::MakeRodataRef(0);
  g.out[0] = 2;
  g.out[1] = 2;
  g.out[2] = 2;
  open.Replace(0, 0, g);
  auto load_with = [&](float s0, float s1) {
    OpenPlan o = open;
    const float scales[2] = {s0, s1};
    std::memcpy(o.bytes.data() + o.h.rodata_offset, scales, sizeof(scales));
    const std::vector<uint8_t> sealed = o.Sealed();
    UpdateEngine engine;
    return engine.LoadFromMemory(sealed.data(), sealed.size());
  };
  const float nan = std::bit_cast<float>(0x7FC00000u);
  ASSERT_OK(load_with(1.0f, 0.5f));
  EXPECT_ERROR_CONTAINS(load_with(1.0f, nan), "column scale 1");
  EXPECT_ERROR_CONTAINS(load_with(0.0f, 1.0f), "column scale 0");
  EXPECT_ERROR_CONTAINS(load_with(INFINITY, 1.0f), "column scale 0");
}

TEST(CoverageTable, AnArenaBeyondTheHostsMemoryIsRefusedBeforeAllocation) {
  // The executing fuzzer's first catch (CR3 #144 F22): a header asking for
  // an arena of exabytes passed every contract (the arena size is bounded
  // against overflow alone) and reached aligned_alloc. Refused by name,
  // against the host's memory, before the allocation.
  const std::vector<uint8_t> plan = MlpPlan();
  ASSERT_TRUE(!plan.empty());
  OpenPlan open(plan);
  open.h.arena_size = uint64_t{1} << 62;
  const std::vector<uint8_t> sealed = open.Sealed();
  UpdateEngine engine;
  EXPECT_ERROR_CONTAINS(engine.LoadFromMemory(sealed.data(), sealed.size()),
                        "exceeds this host's memory");
}

}  // namespace
