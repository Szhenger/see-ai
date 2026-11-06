// =============================================================================
// source/statistics tests (S0 #146): a screen renders one row table four
// ways — console, report JSON, golden block, and (through seeml-abi) the
// published schema — so each rendering is held here to the table; the plan's
// work is counted from its own instructions with seeml-bench's GEMM unit
// and the attention model FLOPs beside it.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "source/plan/update_types.h"
#include "source/statistics/plan_work.h"
#include "source/statistics/process.h"
#include "source/statistics/screen.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;

constexpr StatColumn kCols[] = {
    {"name", StatKind::kText, ""},
    {"ops", StatKind::kCount, "ops"},
    {"ms", StatKind::kReal, "ms", kStatVolatile, 3},
};
constexpr StatTable kTables[] = {
    {"passes", "passes", "each pass's ops after it and its time", kCols},
};
constexpr StatRow kRows[] = {
    {"ops", "operators", StatKind::kCount, "ops", "SMF ops in the model file"},
    {"weight_bytes", "weights", StatKind::kBytes, "B",
     "the constant tensors' payload"},
    {"ratio", "estimate / exact", StatKind::kReal, "", "the accountant's bound",
     0, 4},
    {"attention", "attention", StatKind::kText, "", "the family chosen"},
    {"detail", "detail only", StatKind::kCount, "", "a verbose row",
     kStatVerbose},
    {"wall_ms", "time", StatKind::kReal, "ms", "the phase's wall time",
     kStatVolatile, 2},
    {"vocab", "vocabulary", StatKind::kCount, "tokens", "embedding rows",
     kStatOptional},
};
constexpr ScreenSpec kSpec = {"frontend", "what the compiler read", kRows,
                              kTables};

Screen Filled() {
  Screen s(kSpec);
  s.Set("ops", 107);
  s.Set("weight_bytes", uint64_t{163287104});
  s.Set("ratio", 0.88136);  // 0.88135 has no exact binary form: 0.8813
  s.Set("attention", "cached");
  s.Set("detail", size_t{7});
  s.Set("wall_ms", 1.25);
  s.AddRow("passes", {std::string("dce"), uint64_t{594}, 0.5});
  s.AddRow("passes", {std::string("lower"), uint64_t{381}, 2.0});
  return s;
}

bool Has(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

TEST(Screen, TheConsoleLineCarriesTheValueItsUnitAndItsDerivation) {
  const Screen s = Filled();
  const std::string text = s.Text(/*verbose=*/false);
  EXPECT_TRUE(text.starts_with("frontend — what the compiler read\n"));
  EXPECT_STR_CONTAINS(text, "107 ops");
  EXPECT_STR_CONTAINS(text, "SMF ops in the model file");
  EXPECT_STR_CONTAINS(text, "163,287,104 B (155.7 MiB)");
  EXPECT_STR_CONTAINS(text, "0.8814");  // the row's four decimals
  EXPECT_STR_CONTAINS(text, "cached");
  // Verbose rows and the detail tables stay off the default screen; the
  // unset optional row is skipped, not printed as a placeholder.
  EXPECT_FALSE(Has(text, "detail only"));
  EXPECT_FALSE(Has(text, "dce"));
  EXPECT_FALSE(Has(text, "vocabulary"));
  const std::string verbose = s.Text(/*verbose=*/true);
  EXPECT_STR_CONTAINS(verbose, "detail only");
  EXPECT_STR_CONTAINS(verbose, "passes — each pass's ops after it");
  EXPECT_STR_CONTAINS(verbose, "dce");
  EXPECT_STR_CONTAINS(verbose, "594 ops");
}

TEST(Screen, TheReportObjectCarriesEveryValueAndOmitsTheUnset) {
  const Screen s = Filled();
  const std::string json = s.Json(2);
  EXPECT_STR_CONTAINS(json, "\"ops\": 107");
  EXPECT_STR_CONTAINS(json, "\"weight_bytes\": 163287104");
  EXPECT_STR_CONTAINS(json, "\"attention\": \"cached\"");
  EXPECT_STR_CONTAINS(json, "\"detail\": 7");  // verbose rows are reported
  EXPECT_STR_CONTAINS(json, "\"wall_ms\": 1.25");
  EXPECT_STR_CONTAINS(json, "\"passes\": [{\"name\": \"dce\", \"ops\": 594, \"ms\": 0.5}");
  EXPECT_FALSE(Has(json, "vocab"));
  EXPECT_TRUE(json.starts_with("{\n    \"ops\""));  // nested two deep
  EXPECT_TRUE(json.ends_with("\n  }"));
  // A real that is not finite is null, never a token JSON cannot parse.
  Screen t(kSpec);
  t.Set("ratio", std::numeric_limits<double>::quiet_NaN());
  EXPECT_STR_CONTAINS(t.Json(0), "\"ratio\": null");
  // Text is escaped.
  Screen u(kSpec);
  u.Set("attention", "a \"quoted\"\nline");
  EXPECT_STR_CONTAINS(u.Json(0), "\"a \\\"quoted\\\"\\u000aline\"");
}

TEST(Screen, TheGoldenBlockLeavesOutWhatVariesWithTheHost) {
  const std::string golden = Filled().Golden();
  EXPECT_STR_CONTAINS(golden, "[frontend]\n");
  EXPECT_STR_CONTAINS(golden, "ops = 107 ops\n");
  EXPECT_STR_CONTAINS(golden, "detail = 7\n");  // verbose rows are pinned
  EXPECT_STR_CONTAINS(golden, "passes[0] = dce 594 ops\n");  // no ms column
  EXPECT_STR_CONTAINS(golden, "passes[1] = lower 381 ops\n");
  EXPECT_FALSE(Has(golden, "wall_ms"));
  EXPECT_FALSE(Has(golden, " ms"));
}

TEST(Screen, ProblemsNameEveryDepartureFromTheSpec) {
  EXPECT_TRUE(Filled().Problems().empty());
  Screen s(kSpec);
  s.Set("opss", 1);              // a misspelled key
  s.Set("ops", 1.5);             // a real into a count
  s.Set("ratio", 2);             // a count into a real
  s.Set("weight_bytes", -1);     // a negative count
  s.AddRow("passes", {std::string("dce")});  // a short row
  s.AddRow("nopes", {});
  const std::vector<std::string> p = s.Problems();
  std::string all;
  for (const auto& line : p) all += line + "\n";
  EXPECT_STR_CONTAINS(all, "unknown row 'opss'");
  EXPECT_STR_CONTAINS(all, "row 'ops' set with the wrong kind (it is count)");
  EXPECT_STR_CONTAINS(all, "row 'ratio' set with the wrong kind (it is real)");
  EXPECT_STR_CONTAINS(all, "negative count for 'weight_bytes'");
  EXPECT_STR_CONTAINS(all, "table 'passes' row has 1 cells, not 3");
  EXPECT_STR_CONTAINS(all, "unknown table 'nopes'");
  // Every non-optional row is unset here; the optional one is not reported.
  EXPECT_STR_CONTAINS(all, "unset row 'attention'");
  EXPECT_FALSE(Has(all, "unset row 'vocab'"));
}

TEST(Screen, CountsAndBytesReadTheSameOnEveryHost) {
  using screen_detail::Bytes;
  using screen_detail::Grouped;
  EXPECT_EQ(Grouped(0), std::string("0"));
  EXPECT_EQ(Grouped(999), std::string("999"));
  EXPECT_EQ(Grouped(1000), std::string("1,000"));
  EXPECT_EQ(Grouped(1576898688), std::string("1,576,898,688"));
  EXPECT_EQ(Bytes(1023), std::string("1,023 B"));
  EXPECT_EQ(Bytes(1024), std::string("1,024 B (1.0 KiB)"));
  EXPECT_EQ(Bytes(1536), std::string("1,536 B (1.5 KiB)"));
  EXPECT_EQ(Bytes(1048575), std::string("1,048,575 B (1024.0 KiB)"));
  EXPECT_EQ(Bytes(1576898688), std::string("1,576,898,688 B (1.5 GiB)"));
  EXPECT_EQ(Bytes(UINT64_MAX).substr(0, 5), std::string("18,44"));
}

// --- The plan's work ------------------------------------------------------------

UpdateInstruction Ins(OpCode op, uint64_t o0, uint64_t o1, uint64_t o2) {
  UpdateInstruction ins;
  ins.opcode = static_cast<uint16_t>(op);
  ins.out[0] = o0;
  ins.out[1] = o1;
  ins.out[2] = o2;
  return ins;
}

TEST(PlanWork, GemmFlopsAreTwoMNKOverTheEightGemmOpcodes) {
  const std::vector<UpdateInstruction> program = {
      Ins(OpCode::kGemmNN, 2, 3, 4),     Ins(OpCode::kGemmNT, 2, 3, 4),
      Ins(OpCode::kGemmTN, 2, 3, 4),     Ins(OpCode::kGemmAccNN, 2, 3, 4),
      Ins(OpCode::kGemmNNQ8, 2, 3, 4),   Ins(OpCode::kGemmNTQ8, 2, 3, 4),
      Ins(OpCode::kGemmNNBF16, 2, 3, 4), Ins(OpCode::kGemmNTBF16, 2, 3, 4),
      Ins(OpCode::kAddEW, 1000, 0, 0),   Ins(OpCode::kReluFwd, 1000, 0, 0)};
  EXPECT_EQ(GemmFlops(program), 8u * 2 * 2 * 3 * 4);
  // Saturating, never wrapping, on a hostile instruction.
  const std::vector<UpdateInstruction> huge = {
      Ins(OpCode::kGemmNN, UINT64_MAX, 2, 2)};
  EXPECT_EQ(GemmFlops(huge), UINT64_MAX);
}

TEST(PlanWork, AttentionFlopsCountTheModelProductsInBothFamilies) {
  const uint64_t B = 2, S = 16, H = 4, d = 8;
  const uint64_t macs = B * H * S * S * d;
  const uint64_t bs = B << 32 | S, hd = H << 32 | d;
  const std::vector<UpdateInstruction> cached = {
      Ins(OpCode::kAttnFwd, 0, bs, hd), Ins(OpCode::kAttnDP, bs, hd, 0),
      Ins(OpCode::kAttnDV, bs, hd, 0),  Ins(OpCode::kSoftmaxRowsBwd, 0, 0, 0),
      Ins(OpCode::kAttnDQ, bs, hd, 0),  Ins(OpCode::kAttnDK, bs, hd, 0)};
  const uint64_t packed = PackAttnGeometry(B, S, H, d);
  const std::vector<UpdateInstruction> tiled = {
      Ins(OpCode::kAttnFwdTiled, 0, bs, hd),
      Ins(OpCode::kAttnDQTiled, 0, 0, packed),
      Ins(OpCode::kAttnDKTiled, 0, 0, packed),
      Ins(OpCode::kAttnDVTiled, 0, 0, packed)};
  // 4·B·H·S²·d forward + 8·B·H·S²·d backward, either family.
  EXPECT_EQ(AttentionFlops(cached), 12 * macs);
  EXPECT_EQ(AttentionFlops(tiled), 12 * macs);
  EXPECT_EQ(AttentionFlops({cached.data(), 1}), 4 * macs);
  EXPECT_EQ(GemmFlops(cached), 0u);  // never folded into the GEMM unit
}

TEST(PlanWork, AStepIsTheMicroBatchesOfTheTrainProgramPlusTheStepProgram) {
  SmfModel model = seeml::testing::MakeMlp(6, 10, 3, 1);
  UpdateConfig config = seeml::testing::BaseConfig(4);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain, UpdateCompiler(config).Compile(model));
  config.grad_accum_steps = 3;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate accum, UpdateCompiler(config).Compile(model));
  auto train_of = [](const CompiledUpdate& c) {
    PlanHeader h;
    std::memcpy(&h, c.plan.data(), sizeof(h));
    std::vector<UpdateInstruction> v(h.train_instr_count);
    std::memcpy(v.data(), c.plan.data() + h.train_instr_offset,
                v.size() * sizeof(UpdateInstruction));
    return v;
  };
  const PlanWork p = MeasurePlanWork(plain.plan);
  EXPECT_EQ(p.micro_batches, 1u);
  EXPECT_TRUE(p.train_gemm_flops > 0);
  EXPECT_EQ(p.train_gemm_flops, GemmFlops(train_of(plain)));
  EXPECT_EQ(p.GemmFlopsPerStep(), p.train_gemm_flops + p.step_gemm_flops);
  EXPECT_EQ(p.train_attention_flops, 0u);  // an MLP has no attention
  const PlanWork a = MeasurePlanWork(accum.plan);
  EXPECT_EQ(a.micro_batches, 3u);
  EXPECT_EQ(a.GemmFlopsPerStep(), 3 * a.train_gemm_flops + a.step_gemm_flops);
  EXPECT_EQ(a.FlopsPerStep(), a.GemmFlopsPerStep() + a.AttentionFlopsPerStep());
  // A blob whose sections do not lie inside it measures as nothing.
  std::vector<uint8_t> cut(plain.plan.begin(), plain.plan.begin() + 512);
  EXPECT_EQ(MeasurePlanWork(cut).train_gemm_flops, 0u);
  EXPECT_EQ(MeasurePlanWork(std::vector<uint8_t>(8)).train_gemm_flops, 0u);
}

TEST(PlanWork, TheBenchFixturesKeepTheirRecordedGemmFlops) {
  // The October nightly ledger (docs/ledger/p8-2026-10-22/nightly/*/
  // bench.json) recorded these per-step GEMM FLOPs; the shared function is
  // that definition, so the bench's baselines and utilization figures keep
  // their unit. Compiled as seeml-bench compiles its fixtures.
  struct Case {
    const char* name;
    SmfModel model;
    uint64_t flops;
  };
  Case cases[] = {
      {"mlp_64x512x3", seeml::testing::MakeMlpStack(64, 512, 3, 4, 101),
       74258432},
      {"mlp_128x1024x2", seeml::testing::MakeMlpStack(128, 1024, 2, 4, 103),
       149559296},
  };
  for (Case& c : cases) {
    UpdateConfig config = seeml::testing::BaseConfig(32);  // seeml-bench's
    config.lora.rank = 8;                                   // fixture config
    config.lora.alpha = 16.0f;
    auto compiled = UpdateCompiler(config).Compile(c.model);
    ASSERT_OK(compiled);
    if (MeasurePlanWork(compiled->plan).train_gemm_flops != c.flops)
      ADD_FAILURE(std::string(c.name) + ": " +
                  std::to_string(MeasurePlanWork(compiled->plan).train_gemm_flops) +
                  " GEMM FLOPs, the ledger recorded " + std::to_string(c.flops));
  }
}

TEST(Process, PeakRssIsTheProcessesOwnInBytes) {
  // At least the pages this test touches; at most absurd. On Linux
  // ru_maxrss is in kilobytes — reading it as bytes would be 1024x low.
  std::vector<uint8_t> touched(64u << 20, 1);
  uint64_t sum = 0;
  for (size_t i = 0; i < touched.size(); i += 4096) sum += touched[i];
  EXPECT_TRUE(sum > 0);
#if !defined(_WIN32)
  const uint64_t rss = PeakRssBytes();
  EXPECT_TRUE(rss >= (64u << 20));
  EXPECT_TRUE(rss < (uint64_t{1} << 40));
#endif
}

}  // namespace
