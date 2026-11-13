// =============================================================================
// The compiler's statistics screens (S0 #146): every screen complete for
// every fixture family, every number the plan's own (the backend screen's
// counts and bytes against CompiledUpdate, its work against the plan's
// instructions), the optional rows present exactly when they apply, and the
// default console inside the 60-line budget.
// =============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/driver/statistics.h"
#include "compiler/driver/update_compiler.h"
#include "source/plan/update_types.h"
#include "source/statistics/plan_work.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;
using seeml::testing::MakeTinyTokenDecoder;

uint64_t U(const Screen& s, const char* key) {
  const StatValue* v = s.Get(key);
  return v && std::holds_alternative<uint64_t>(*v) ? std::get<uint64_t>(*v)
                                                    : UINT64_MAX;
}
double R(const Screen& s, const char* key) {
  const StatValue* v = s.Get(key);
  return v && std::holds_alternative<double>(*v) ? std::get<double>(*v) : -1.0;
}
std::string T(const Screen& s, const char* key) {
  const StatValue* v = s.Get(key);
  return v && std::holds_alternative<std::string>(*v)
             ? std::get<std::string>(*v)
             : std::string("<unset>");
}
void ExpectComplete(const CompileScreens& screens, const char* fixture) {
  for (const Screen* s : {&screens.frontend, &screens.analysis,
                          &screens.backend, &screens.runtime})
    for (const std::string& problem : s->Problems())
      ADD_FAILURE(std::string(fixture) + ": " + s->spec().name + ": " +
                  problem);
}
size_t Lines(const std::string& text) {
  size_t n = 0;
  for (const char c : text) n += c == '\n';
  return n;
}

TEST(CompileScreens, EveryScreenIsCompleteForEveryFixtureFamily) {
  struct Case {
    const char* name;
    SmfModel model;
    UpdateConfig config;
    SmfModel teacher;
    bool with_teacher = false;
  };
  std::vector<Case> cases;
  cases.push_back({"mlp", MakeMlp(6, 10, 3, 1), BaseConfig(4), {}});
  {
    UpdateConfig c = BaseConfig(4);
    c.grad_accum_steps = 2;
    cases.push_back({"mlp, accumulating", MakeMlp(6, 10, 3, 2), c, {}});
  }
  {
    UpdateConfig c = BaseConfig(4);
    c.quantize_base = true;
    cases.push_back({"mlp, int8 base", MakeMlp(6, 10, 3, 3), c, {}});
  }
  {
    UpdateConfig c = BaseConfig(4);
    c.loss = LossKind::kKLDistill;
    cases.push_back({"mlp, distilled", MakeMlp(6, 10, 3, 4), c,
                     MakeMlp(6, 14, 3, 5), true});
  }
  cases.push_back({"token decoder", MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41),
                   BaseConfig(16), {}});
  for (Case& c : cases) {
    auto compiled = UpdateCompiler(c.config).Compile(
        c.model, c.with_teacher ? &c.teacher : nullptr);
    ASSERT_OK(compiled);
    ExpectComplete(compiled->screens, c.name);
  }
}

TEST(CompileScreens, TheWorkAndTheLedgerAreThePlansOwn) {
  // S5 #151 / S6 #152: the analysis screen's forward / backward / optimizer
  // work is the unit seeml-bench and the runtime screen report (GEMM + attention
  // FLOPs of the plan's own programs), the GEMM census sums to the train
  // program's GEMM work, the ledger is the plan's metadata, and the GPU
  // prediction is the same work over the GPU's peak.
  SmfModel model = MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41);
  UpdateConfig config = BaseConfig(16);
  config.grad_accum_steps = 2;
  config.peak_gflops = 100.0;
  config.gpu_peak_gflops = 400.0;
  config.precision = Precision::kCertifiedBf16;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate c, UpdateCompiler(config).Compile(model));
  const CompileScreens& s = c.screens;
  PlanHeader h;
  std::memcpy(&h, c.plan.data(), sizeof(h));
  auto program = [&](uint64_t off, uint64_t count) {
    return std::span<const UpdateInstruction>(
        reinterpret_cast<const UpdateInstruction*>(c.plan.data() + off), count);
  };
  auto flops = [](std::span<const UpdateInstruction> p) {
    return GemmFlops(p) + AttentionFlops(p);
  };
  const auto train = program(h.train_instr_offset, h.train_instr_count);
  const auto eval = program(h.eval_instr_offset, h.eval_instr_count);
  const auto step = program(h.step_instr_offset, h.step_instr_count);
  EXPECT_EQ(U(s.analysis, "flops_forward"), flops(eval));
  EXPECT_EQ(U(s.analysis, "flops_forward") + U(s.analysis, "flops_backward"),
            flops(train));
  EXPECT_EQ(U(s.analysis, "flops_optimizer"), flops(step));
  EXPECT_TRUE(U(s.analysis, "flops_backward") > U(s.analysis, "flops_forward"));
  EXPECT_TRUE(U(s.analysis, "adjoint_ops") > 0);
  EXPECT_TRUE(U(s.analysis, "saved_activation_bytes") > 0);
  EXPECT_TRUE(U(s.analysis, "saved_activation_bytes") <= c.transient_size);
  // The ledger: the plan's own section, and its reordered count the relaxed
  // instructions the compile flagged.
  DecodedMetadata meta;
  ASSERT_FALSE(PlanMetadataViolation(h, c.plan.data(), c.plan.size(), &meta)
                   .has_value());
  EXPECT_EQ(U(s.analysis, "reordered_instructions"),
            c.relaxed_gemms + c.relaxed_attention);
  EXPECT_TRUE(c.relaxed_gemms > 0);
  const std::string analysis = s.analysis.Json(0);
  for (const LedgerEntry& e : meta.ledger)
    EXPECT_STR_CONTAINS(analysis, std::string("\"pass\": \"") + e.pass + "\"");
  EXPECT_STR_CONTAINS(analysis, "\"arithmetic\": \"reordered\"");
  // The census sums to the train program's GEMM work; the largest tensors
  // are arena values.
  uint64_t census = 0;
  for (const GemmShape& g : GemmCensus(train)) census += g.flops;
  EXPECT_EQ(census, GemmFlops(train));
  const std::string backend = s.backend.Json(0);
  EXPECT_STR_CONTAINS(backend, "\"gemm_census\"");
  EXPECT_STR_CONTAINS(backend, "\"largest_tensors\"");
  EXPECT_EQ(U(s.backend, "io_bytes"), c.io_size);
  EXPECT_EQ(U(s.backend, "persistent_bytes") + U(s.backend, "io_bytes") +
                U(s.backend, "transient_bytes"),
            c.arena_size);
  // The GPU half: the same work over the GPU's stated peak.
  const PlanWork work = MeasurePlanWork(c.plan);
  EXPECT_EQ(R(s.backend, "gpu_peak_gflops"), 400.0);
  EXPECT_EQ(R(s.backend, "predicted_step_gpu_ms"),
            static_cast<double>(work.FlopsPerStep()) / (400.0 * 1e6));
  // The manifest's recipe word, on the frontend screen.
  char hex[24];
  std::snprintf(hex, sizeof(hex), "%016llx",
                static_cast<unsigned long long>(meta.meta.recipe_hash));
  EXPECT_EQ(T(s.frontend, "recipe_hash"), std::string(hex));
}

TEST(CompileScreens, TheActivationsSavedAreTheLoweredPlans) {
  // S5 #151: a forward value the backward recomputes is not kept for it, so
  // the row is taken on the block that is lowered — after rematerialization,
  // not as autodiff left it.
  SmfModel model = MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41);
  UpdateConfig config = BaseConfig(16);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(config).Compile(model));
  config.remat = RematKind::kOn;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate remat,
                       UpdateCompiler(config).Compile(model));
  ASSERT_TRUE(remat.remat_recomputed_values > 0);
  EXPECT_TRUE(U(remat.screens.analysis, "saved_activation_bytes") <
              U(plain.screens.analysis, "saved_activation_bytes"));
  EXPECT_TRUE(U(remat.screens.analysis, "saved_activation_bytes") <=
              remat.transient_size);
}

TEST(CompileScreens, TheTargetIsTheOneThePolicyWasReadFor) {
  // S6 #152: a cross-compile (--target-host) names the key its kernel policy
  // came from and claims no GPU; a native compile names this host; the
  // plan's bytes are the same either way.
  SmfModel model = MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41);
  UpdateConfig config = BaseConfig(16);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate native, UpdateCompiler(config).Compile(model));
  EXPECT_TRUE(T(native.screens.backend, "target")
                  .starts_with(HostKey(DetectHostArch()) + "; GPU "));
  config.target_host = "apple_m9_10p6e";
  ASSERT_OK_AND_ASSIGN(CompiledUpdate cross, UpdateCompiler(config).Compile(model));
  EXPECT_EQ(T(cross.screens.backend, "target"),
            std::string("apple_m9_10p6e (--target-host); GPU not described"));
  EXPECT_TRUE(native.plan == cross.plan);
}

TEST(CompileScreens, TheNumbersAreThePlansOwn) {
  SmfModel model = MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41);
  UpdateConfig config = BaseConfig(16);
  config.peak_gflops = 100.0;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate c, UpdateCompiler(config).Compile(model));
  const CompileScreens& s = c.screens;
  // backend: the counts and bytes CompiledUpdate carries.
  EXPECT_EQ(U(s.backend, "train_instructions"), c.train_instruction_count);
  EXPECT_EQ(U(s.backend, "eval_instructions"), c.eval_instruction_count);
  EXPECT_EQ(U(s.backend, "merge_instructions"), c.merge_instruction_count);
  EXPECT_EQ(U(s.backend, "arena_bytes"), c.arena_size);
  EXPECT_EQ(U(s.backend, "rodata_bytes"), c.rodata_size);
  EXPECT_EQ(U(s.backend, "plan_bytes"), c.plan.size());
  EXPECT_EQ(U(s.backend, "device_bytes"), c.arena_size + c.plan.size());
  EXPECT_EQ(U(s.runtime, "device_bytes"), c.arena_size + c.plan.size());
  // ... and the work its instructions do, with the prediction it implies.
  const PlanWork work = MeasurePlanWork(c.plan);
  EXPECT_TRUE(work.train_attention_flops > 0);  // a decoder attends
  EXPECT_EQ(U(s.backend, "gemm_flops"), work.GemmFlopsPerStep());
  EXPECT_EQ(U(s.backend, "attention_flops"), work.AttentionFlopsPerStep());
  EXPECT_EQ(U(s.runtime, "flops_per_step"), work.FlopsPerStep());
  const double predicted =
      static_cast<double>(work.FlopsPerStep()) / (100.0 * 1e6);
  EXPECT_EQ(R(s.backend, "predicted_step_ms"), predicted);
  EXPECT_EQ(R(s.runtime, "predicted_step_ms"), predicted);
  EXPECT_EQ(R(s.backend, "peak_gflops"), 100.0);
  // frontend: the model's own tensors.
  uint64_t weight_bytes = 0, weights = 0;
  for (const SmfTensor& t : model.tensors)
    if (t.is_const) {
      ++weights;
      weight_bytes += t.data.size();
    }
  EXPECT_EQ(U(s.frontend, "weight_tensors"), weights);
  EXPECT_EQ(U(s.frontend, "weight_bytes"), weight_bytes);
  EXPECT_EQ(U(s.frontend, "parameters"), weight_bytes / 4);
  EXPECT_EQ(U(s.frontend, "vocabulary"), 16u);
  EXPECT_EQ(T(s.frontend, "input"), std::string("token ids, 8 per sequence"));
  EXPECT_EQ(U(s.frontend, "footprint_total"), c.estimate_bytes);
  // analysis: the adapters' own geometry, every timed pass.
  uint64_t trainable = 0;
  for (const AdapterDebugInfo& a : c.adapters)
    trainable += static_cast<uint64_t>(a.r * (a.k + a.m));
  EXPECT_EQ(U(s.analysis, "adapters"), c.adapters.size());
  EXPECT_EQ(U(s.analysis, "trainable_parameters"), trainable);
  EXPECT_EQ(U(s.analysis, "gradients"), 2 * c.adapters.size());
  EXPECT_EQ(U(s.analysis, "passes"), c.pass_timings.size());
  EXPECT_EQ(U(s.analysis, "rope_tables"), 1u);
  EXPECT_TRUE(T(s.analysis, "attention").starts_with("cached, "));
  EXPECT_EQ(T(s.analysis, "narrow_weights"), std::string("none: f32"));
  // The report object carries the tables too.
  const std::string json = s.analysis.Json(0);
  EXPECT_STR_CONTAINS(json, "\"passes\": [{\"name\": \"ingress\"");
  EXPECT_STR_CONTAINS(json, "\"adapter_sites\": [{\"weight\": ");
}

TEST(CompileScreens, OptionalRowsAppearExactlyWhenTheyApply) {
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(BaseConfig(4)).Compile(MakeMlp(6, 10, 3, 1)));
  EXPECT_TRUE(plain.screens.frontend.Get("vocabulary") == nullptr);
  EXPECT_TRUE(plain.screens.frontend.Get("teacher_ops") == nullptr);
  EXPECT_TRUE(plain.screens.backend.Get("step_instructions") == nullptr);
  EXPECT_TRUE(plain.screens.backend.Get("predicted_step_ms") == nullptr);
  EXPECT_TRUE(plain.screens.runtime.Get("measured_step_ms") == nullptr);
  EXPECT_EQ(T(plain.screens.analysis, "attention"), std::string("none"));
  EXPECT_EQ(T(plain.screens.frontend, "input"),
            std::string("6 features per row"));
  UpdateConfig accum = BaseConfig(4);
  accum.grad_accum_steps = 2;
  accum.quantize_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a,
                       UpdateCompiler(accum).Compile(MakeMlp(6, 10, 3, 1)));
  EXPECT_EQ(U(a.screens.backend, "step_instructions"),
            a.step_instruction_count);
  EXPECT_TRUE(T(a.screens.analysis, "narrow_weights").ends_with(" int8"));
  // Work per step counts every micro-batch.
  EXPECT_EQ(U(a.screens.backend, "gemm_flops"),
            MeasurePlanWork(a.plan).GemmFlopsPerStep());
  EXPECT_EQ(MeasurePlanWork(a.plan).micro_batches, 2u);
}

TEST(CompileScreens, TheDefaultConsoleFitsTheSixtyLineBudget) {
  // The issue's budget for a compile: four screens, at most 60 lines by
  // default. The rows are aggregates, so the count does not grow with the
  // model; --verbose adds the verbose rows and the per-pass and per-adapter
  // tables.
  UpdateConfig config = BaseConfig(16);
  config.peak_gflops = 100.0;
  config.grad_accum_steps = 2;
  ASSERT_OK_AND_ASSIGN(
      CompiledUpdate c,
      UpdateCompiler(config).Compile(MakeTinyTokenDecoder(16, 8, 2, 8, 16, 41)));
  const std::string text = RenderCompileScreens(c.screens, false);
  const std::string verbose = RenderCompileScreens(c.screens, true);
  EXPECT_TRUE(Lines(text) <= 60);
  EXPECT_TRUE(Lines(verbose) > Lines(text) + c.pass_timings.size());
  for (const char* heading :
       {"frontend — ", "analysis — ", "backend — ", "runtime — "})
    EXPECT_STR_CONTAINS(text, heading);
  EXPECT_STR_CONTAINS(text, "predicted step");
  if (Lines(text) > 60)
    ADD_FAILURE("the default screens take " + std::to_string(Lines(text)) +
                " lines:\n" + text);
}

// --- Golden statistic blocks (S0 #146 acceptance) ------------------------------
// The four screens' rows that are neither timings, hashes nor host facts,
// for the MLP fixtures and the dec_wide shape (tool/frontier_run.py SHAPES:
// vocab 4096, width 512, 8 heads, ffn 1024, 4 blocks, 256 tokens; compiled
// with the nightly's flags), committed under test/fixtures/golden/screens/.
// A pass that changes what it does changes a line here. Integer-derived
// throughout, so the same text on every toolchain. To accept a deliberate
// change: SEEML_UPDATE_GOLDEN=1 ./build/seeml_screens_test, then review the
// diff of test/fixtures/golden/screens/ like any other change.

std::string GoldenOf(const CompileScreens& s) {
  return s.frontend.Golden() + s.analysis.Golden() + s.backend.Golden() +
         s.runtime.Golden();
}

void ExpectGolden(const std::string& fixture, const std::string& actual) {
  const std::string path = seeml::testing::RepoRoot() +
                           "/test/fixtures/golden/screens/" + fixture + ".txt";
  const char* update = std::getenv("SEEML_UPDATE_GOLDEN");
  if (update && *update && std::string(update) != "0") {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << actual;
    out.close();
    if (out.fail()) ADD_FAILURE("cannot write " + path);
    std::printf("  [ note ] wrote %s\n", path.c_str());
    return;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    ADD_FAILURE("no golden block at " + path +
                " (SEEML_UPDATE_GOLDEN=1 writes it)");
    return;
  }
  const std::string expected((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  if (expected == actual) return;
  // Name the first line that differs: the row a pass changed.
  size_t e = 0, a = 0, line = 1;
  while (true) {
    const size_t en = expected.find('\n', e), an = actual.find('\n', a);
    const std::string el = expected.substr(e, en == std::string::npos ? std::string::npos : en - e);
    const std::string al = actual.substr(a, an == std::string::npos ? std::string::npos : an - a);
    if (el != al || en == std::string::npos || an == std::string::npos) {
      ADD_FAILURE(fixture + ": golden block differs at line " +
                  std::to_string(line) + ":\n    golden: " + el +
                  "\n    now:    " + al +
                  "\n  (a deliberate change: SEEML_UPDATE_GOLDEN=1, then review "
                  "the diff)");
      return;
    }
    e = en + 1;
    a = an + 1;
    ++line;
  }
}

TEST(GoldenScreens, TheMlpFixtures) {
  UpdateConfig plain = BaseConfig(4);
  plain.peak_gflops = 100.0;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate c,
                       UpdateCompiler(plain).Compile(MakeMlp(6, 10, 3, 1)));
  ExpectGolden("mlp", GoldenOf(c.screens));
  UpdateConfig narrow = plain;
  narrow.quantize_base = true;
  narrow.grad_accum_steps = 2;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate n,
                       UpdateCompiler(narrow).Compile(MakeMlp(6, 10, 3, 1)));
  ExpectGolden("mlp_int8_accumulating", GoldenOf(n.screens));
}

TEST(GoldenScreens, TheDecWideShape) {
  UpdateConfig config = BaseConfig(512);  // the nightly's dec_wide flags
  config.lora.rank = 16;
  config.lora.alpha = 32.0f;
  config.optimizer.kind = OptimizerKind::kAdamW;
  config.optimizer.lr = 1e-4f;
  config.optimizer.lr_schedule = LrSchedule::kConstant;
  config.default_steps = 300;
  config.peak_gflops = 3500.0;
  SmfModel model = seeml::testing::MakeTokenDecoderStack(4096, 512, 8, 256,
                                                         1024, 4, 7);
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_OK_AND_ASSIGN(CompiledUpdate c, UpdateCompiler(config).Compile(model));
  std::printf("  [ note ] dec_wide compiled in %.0f ms\n",
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count());
  ExpectGolden("dec_wide", GoldenOf(c.screens));
}

}  // namespace
