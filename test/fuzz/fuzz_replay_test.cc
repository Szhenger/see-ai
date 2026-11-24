// =============================================================================
// fuzz/ replay suite (CR3 #208): the fuzz harness's entry point, run in
// every build over the checked-in seed corpus (test/fuzz/seeds/, written by
// test/fuzz/make_seeds.py), with what each seed must REACH asserted from
// the harness's own counters (fuzz_reach.h).
//
// Why it exists: the pull-request fuzz job ran for months without ever
// executing an accepted plan, and passed — nothing said what the fuzzer
// reached. Here every seed names the arm it selects and the counters it
// must move (a plan the loader accepted, a training step that ran, the
// family's own opcodes executed, a compile completed, a tokenizer section
// loaded, a window ingested, a recipe applied, a report read), so a seed
// that goes stale — a plan-version bump, a contract change — is a red
// suite that names the seed and the regenerate command, not a fuzzer that
// quietly covers refusals only. It also holds the harness itself: the
// re-seal, the refusal of malformed inputs, and a deterministic mutation
// pass over every seed (under ASan/UBSan and TSan in those jobs).
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "source/plan/update_types.h"
#include "test/framework/seetest.h"
#include "test/fuzz/fuzz_reach.h"
#include "test/fuzz/plan_mutate.h"
#include "test/support/builders.h"

namespace {

using seeml::fuzz::Reach;
using seeml::fuzz::TheReach;
using seeml::update::OpCode;
using seeml::update::PlanHeader;

const char* const kRegenerate =
    " — regenerate the corpus with `python3 test/fuzz/make_seeds.py` (after "
    "a build) if the format moved";

std::string SeedDir() {
  return seeml::testing::RepoRoot() + "/test/fuzz/seeds";
}

std::vector<uint8_t> ReadSeed(const std::string& name) {
  std::ifstream f(SeedDir() + "/" + name, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/// What one input reached: the counters from zero.
Reach Run(const std::vector<uint8_t>& input) {
  TheReach() = {};
  LLVMFuzzerTestOneInput(input.data(), input.size());
  return TheReach();
}

// --- What a seed must reach --------------------------------------------------

struct Counter {
  const char* what;
  uint64_t Reach::* field;
};

constexpr Counter kSmfLoaded{"the model reader accepted", &Reach::smf_loaded};
constexpr Counter kSdsLoaded{"the corpus reader accepted", &Reach::sds_loaded};
constexpr Counter kPlanAccepted{"a plan was accepted by the loader",
                                &Reach::plans_accepted};
constexpr Counter kExecuted{"ExecuteOnce ran instructions",
                            &Reach::instructions_executed};
constexpr Counter kTrained{"a train step ran", &Reach::train_steps};
constexpr Counter kEvaluated{"the eval program ran", &Reach::evals};
constexpr Counter kMerged{"the merge program ran", &Reach::merges};
constexpr Counter kRelaxed{"a relaxed instruction was executed",
                           &Reach::relaxed_executed};
constexpr Counter kCompiled{"a compile completed", &Reach::compiles_completed};
constexpr Counter kCompiledAccepted{"the loader accepted the compiled plan",
                                    &Reach::compiled_plans_accepted};
constexpr Counter kCorpusCompiled{"the compile took a corpus",
                                  &Reach::corpora_compiled};
constexpr Counter kTokenizer{"the tokenizer section loaded",
                             &Reach::tokenizers_loaded};
constexpr Counter kEncoded{"the text was encoded", &Reach::texts_encoded};
constexpr Counter kJsonl{"the JSONL parsed", &Reach::jsonl_parsed};
constexpr Counter kWindows{"ingest produced a window", &Reach::ingest_windows};
constexpr Counter kIngestSds{"the corpus reader took the ingested corpus",
                             &Reach::ingest_sds_loaded};
constexpr Counter kRecipeParsed{"a recipe parsed", &Reach::recipes_parsed};
constexpr Counter kRecipeApplied{"a recipe applied", &Reach::recipes_applied};
constexpr Counter kReport{"an export report was read and bound",
                          &Reach::reports_read};

/// Every counter a seed is held to: the corpus as a whole moves them all.
constexpr Counter kEveryCounter[] = {
    kSmfLoaded,        kSdsLoaded,      kPlanAccepted, kExecuted,
    kTrained,          kEvaluated,      kMerged,       kRelaxed,
    kCompiled,         kTokenizer,      kEncoded,      kJsonl,
    kWindows,          kIngestSds,      kReport,       kRecipeParsed,
    kCompiledAccepted, kCorpusCompiled, kRecipeApplied};

struct SeedCase {
  const char* name;
  seeml::fuzz::Arm arm;
  std::vector<Counter> counters;  // each non-zero after the seed
  std::vector<OpCode> opcodes;    // each executed by the step the seed ran
};

/// A sealed plan: accepted, one step trained, merged.
std::vector<Counter> PlanRun(std::vector<Counter> more = {}) {
  std::vector<Counter> all{kPlanAccepted, kExecuted, kTrained, kMerged};
  all.insert(all.end(), more.begin(), more.end());
  return all;
}

/// A compile whose plan the loader takes and the engine runs.
std::vector<Counter> CompileRun(std::vector<Counter> more = {}) {
  std::vector<Counter> all{kSmfLoaded,    kCompiled, kCompiledAccepted,
                           kPlanAccepted, kExecuted, kTrained,
                           kEvaluated,    kMerged};
  all.insert(all.end(), more.begin(), more.end());
  return all;
}

/// The seed corpus, seed by seed: the arm, the counters, and the opcodes
/// that make the seed its fixture family's (the int8 seed runs an int8
/// GEMM, the tiled one the tiled attention, and so on).
const std::vector<SeedCase>& SeedCases() {
  using enum seeml::fuzz::Arm;
  static const std::vector<SeedCase> cases = {
      // [0] the model reader.
      {"smf-mlp", kArmSmf, {kSmfLoaded}, {}},
      {"smf-dec", kArmSmf, {kSmfLoaded}, {}},
      {"smf-segdec", kArmSmf, {kSmfLoaded}, {}},
      {"smf-textdec", kArmSmf, {kSmfLoaded}, {}},
      // [2] the corpus reader.
      {"sds-class", kArmSds, {kSdsLoaded}, {}},
      {"sds-dense", kArmSds, {kSdsLoaded}, {}},
      {"sds-tokens", kArmSds, {kSdsLoaded}, {}},
      {"sds-masked", kArmSds, {kSdsLoaded}, {}},
      {"sds-segments", kArmSds, {kSdsLoaded}, {}},
      // [1] one sealed plan per fixture family.
      {"plan-mlp",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kGemmNN, OpCode::kAddBias, OpCode::kReluFwd,
        OpCode::kSoftmaxXEntFwd, OpCode::kAdamWStep}},
      {"plan-mlp-mse-sgd",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kMseFwd, OpCode::kMseBwd, OpCode::kSgdStep}},
      {"plan-mlp-clip-global",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kSqNormAccum, OpCode::kClipGlobal}},
      {"plan-mlp-accum",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kAccumulate}},
      // #192: the 8-bit moment step (plan v27), its codes and scales.
      {"plan-mlp-moments-int8",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kAdamW8Step}},
      // A narrow-storage plan's eval program reads the source model file,
      // which a plan arm does not have: no eval.
      {"plan-mlp-int8", kArmPlan, PlanRun(), {OpCode::kGemmNNQ8}},
      {"plan-dec-bf16",
       kArmPlan,
       PlanRun(),
       {OpCode::kGemmNNBF16, OpCode::kAttnFwd}},
      {"plan-dec-cached",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kEmbedFwd, OpCode::kRmsNormFwd, OpCode::kRopeFwd,
        OpCode::kAttnFwd, OpCode::kAttnDP, OpCode::kAttnDQ, OpCode::kSiluFwd}},
      {"plan-dec-tiled",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kAttnFwdTiled, OpCode::kAttnDQTiled, OpCode::kAttnDKTiled,
        OpCode::kAttnDVTiled}},
      {"plan-dec-relaxed",
       kArmPlan,
       PlanRun({kEvaluated, kRelaxed}),
       {OpCode::kAttnFwdTiled}},
      {"plan-dec-masked",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kSoftmaxXEntMaskedFwd, OpCode::kSoftmaxXEntMaskedBwd}},
      {"plan-dec-segment",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kSegmentStarts, OpCode::kAttnFwdSeg}},
      {"plan-dec-segment-tiled",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kSegmentStarts, OpCode::kAttnFwdTiledSeg}},
      {"plan-dec-clip-global-accum",
       kArmPlan,
       PlanRun({kEvaluated}),
       {OpCode::kAccumulate, OpCode::kSqNormAccum, OpCode::kClipGlobal,
        OpCode::kAttnFwd}},
      // [4] the same bytes, the seal as it is.
      {"plan-raw-mlp", kArmPlanRaw, PlanRun({kEvaluated}), {OpCode::kGemmNN}},
      // [3] the compile with no corpus.
      {"compile-mlp", kArmCompile, CompileRun(), {OpCode::kGemmNN}},
      {"compile-dec", kArmCompile, CompileRun(), {OpCode::kAttnFwd}},
      {"compile-segdec", kArmCompile, CompileRun(), {OpCode::kEmbedFwd}},
      // [5] the tokenizer section and the text ingest.
      {"tokenizer-jsonl",
       kArmTokenizer,
       {kTokenizer, kEncoded, kJsonl, kWindows, kIngestSds},
       {}},
      {"tokenizer-txt",
       kArmTokenizer,
       {kTokenizer, kEncoded, kWindows, kIngestSds},
       {}},
      // [6] the recipe file.
      {"recipe", kArmRecipe, {kRecipeParsed, kRecipeApplied}, {}},
      // [7] the export report.
      {"report-segdec", kArmReport, {kSmfLoaded, kReport}, {}},
      {"report-textdec", kArmReport, {kSmfLoaded, kReport}, {}},
      // [8] the compile over a recipe and a corpus.
      {"compile-data-jsonl",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kEmbedFwd}},
      {"compile-data-chat",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kSoftmaxXEntMaskedFwd}},
      {"compile-data-txt",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kEmbedFwd}},
      {"compile-data-sds",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kSegmentStarts, OpCode::kAttnFwdSeg}},
      {"compile-data-masked",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kSoftmaxXEntMaskedFwd}},
      {"compile-data-class",
       kArmCompileData,
       CompileRun({kRecipeApplied, kCorpusCompiled}),
       {OpCode::kGemmNN}},
  };
  return cases;
}

/// The failures of `seed` against its case, as text; empty when it holds.
std::string Unreached(const SeedCase& seed, const std::vector<uint8_t>& input,
                      const Reach& r) {
  std::string missing;
  auto add = [&](const std::string& what) {
    missing += (missing.empty() ? "" : "; ") + what;
  };
  if (input.empty()) {
    add("the seed file is missing or empty");
    return missing;
  }
  if (input[0] % seeml::fuzz::kArmCount != seed.arm)
    add("selects arm " + std::to_string(input[0] % seeml::fuzz::kArmCount) +
        ", not " + std::to_string(seed.arm));
  for (const Counter& c : seed.counters)
    if (r.*c.field == 0) add(std::string("not reached: ") + c.what);
  for (const OpCode op : seed.opcodes)
    if (r.opcode_executed[static_cast<size_t>(op)] == 0)
      add("opcode " + std::to_string(static_cast<int>(op)) + " not executed");
  // Whatever the arm: nothing the compiler emitted was refused by the
  // loader, and nothing Ingest wrote was refused by the corpus reader.
  if (r.compiled_plans_refused != 0)
    add("the loader refused the plan the compiler emitted");
  if (r.ingest_sds_refused != 0)
    add("the corpus reader refused the corpus Ingest wrote");
  return missing;
}

TEST(FuzzReplay, EverySeedReachesWhatItsArmExists) {
  for (const SeedCase& seed : SeedCases()) {
    const std::vector<uint8_t> input = ReadSeed(seed.name);
    const std::string missing = Unreached(seed, input, Run(input));
    if (missing.empty()) continue;
    // Again, narrated: the refusal that stopped it goes to stderr.
    seeml::fuzz::SetNarration(true);
    (void)Run(input);
    seeml::fuzz::SetNarration(false);
    ADD_FAILURE("seed " + std::string(seed.name) + ": " + missing +
                kRegenerate);
  }
}

TEST(FuzzReplay, TheSeedDirectoryIsTheTable) {
  // A seed with no case asserts nothing, and a case with no seed fails
  // above: the directory and the table name the same seeds. (regressions/
  // is replayed by its own test.)
  std::vector<std::string> on_disk, in_table;
  for (const auto& entry : std::filesystem::directory_iterator(SeedDir()))
    if (entry.is_regular_file())
      on_disk.push_back(entry.path().filename().string());
  for (const SeedCase& seed : SeedCases()) in_table.emplace_back(seed.name);
  std::sort(on_disk.begin(), on_disk.end());
  std::sort(in_table.begin(), in_table.end());
  EXPECT_TRUE(on_disk == in_table);
  for (const std::string& name : on_disk)
    if (!std::binary_search(in_table.begin(), in_table.end(), name))
      ADD_FAILURE("test/fuzz/seeds/" + name +
                  " has no case in SeedCases(): say what it must reach");
}

TEST(FuzzReplay, EveryArmAndEveryCounterIsReachedByTheCorpus) {
  // The corpus as a whole, as a fuzzing run starts from it: no arm without
  // a seed, and every counter the harness keeps moved by some seed.
  Reach total;
  std::vector<uint64_t> per_arm(seeml::fuzz::kArmCount, 0);
  uint64_t opcodes = 0;
  for (const SeedCase& seed : SeedCases()) {
    const std::vector<uint8_t> input = ReadSeed(seed.name);
    if (input.empty()) continue;
    ++per_arm[input[0] % seeml::fuzz::kArmCount];
    const Reach r = Run(input);
    for (const Counter& c : kEveryCounter) total.*c.field += r.*c.field;
    for (size_t op = 0; op < r.opcode_executed.size(); ++op)
      total.opcode_executed[op] += r.opcode_executed[op];
  }
  for (size_t arm = 0; arm < per_arm.size(); ++arm)
    if (per_arm[arm] == 0)
      ADD_FAILURE("arm " + std::to_string(arm) + " has no seed");
  for (const Counter& c : kEveryCounter)
    if (total.*c.field == 0)
      ADD_FAILURE(std::string("no seed reaches: ") + c.what + kRegenerate);
  for (const uint64_t n : total.opcode_executed) opcodes += n != 0;
  // The families between them execute most of the instruction set (the
  // distillation losses, GELU and LayerNorm have no seed).
  EXPECT_GE(opcodes, uint64_t{40});
}

// --- The re-seal -------------------------------------------------------------

std::vector<uint8_t> WithArm(std::vector<uint8_t> input, seeml::fuzz::Arm arm) {
  input[0] = arm;
  return input;
}

TEST(FuzzReplay, AMutatedPlanMeetsTheContractOnTheResealArmAndTheSealOnTheRaw) {
  const std::vector<uint8_t> seed = ReadSeed("plan-mlp");
  ASSERT_GT(seed.size(), sizeof(PlanHeader) + 1);
  // The seal alone broken — the hash zeroed, every other byte the
  // compiler's: the re-seal arm restores it and the plan runs; the raw arm
  // refuses it at the seal.
  std::vector<uint8_t> unsealed = seed;
  std::memset(unsealed.data() + 1 + offsetof(PlanHeader, plan_hash), 0,
              sizeof(uint64_t));
  Reach r = Run(WithArm(unsealed, seeml::fuzz::kArmPlan));
  EXPECT_EQ(r.plans_accepted, uint64_t{1});
  EXPECT_EQ(r.train_steps, uint64_t{1});
  r = Run(WithArm(unsealed, seeml::fuzz::kArmPlanRaw));
  EXPECT_EQ(r.plans_accepted, uint64_t{0});
  EXPECT_EQ(r.plans_refused, uint64_t{1});
  // A mutation the contract admits (another step budget): accepted and
  // run after the re-seal, refused as corrupt without it.
  std::vector<uint8_t> edited = seed;
  uint64_t steps = 0;
  std::memcpy(&steps, edited.data() + 1 + offsetof(PlanHeader, default_steps),
              sizeof(steps));
  ++steps;
  std::memcpy(edited.data() + 1 + offsetof(PlanHeader, default_steps), &steps,
              sizeof(steps));
  r = Run(WithArm(edited, seeml::fuzz::kArmPlan));
  EXPECT_EQ(r.plans_accepted, uint64_t{1});
  EXPECT_EQ(r.train_steps, uint64_t{1});
  EXPECT_EQ(Run(WithArm(edited, seeml::fuzz::kArmPlanRaw)).plans_accepted,
            uint64_t{0});
  // A mutation the contract does not admit, re-sealed: it is the contract
  // that refuses — the seal was made whole.
  std::vector<uint8_t> hostile = seed;
  const uint64_t huge = ~uint64_t{0} / 2;
  std::memcpy(hostile.data() + 1 + offsetof(PlanHeader, train_instr_count),
              &huge, sizeof(huge));
  r = Run(WithArm(hostile, seeml::fuzz::kArmPlan));
  EXPECT_EQ(r.plans_accepted, uint64_t{0});
  EXPECT_EQ(r.plans_refused, uint64_t{1});
}

// --- Malformed inputs are refused --------------------------------------------

/// The counters that mean "the arm's consumer accepted its input".
uint64_t Accepted(const Reach& r, seeml::fuzz::Arm arm) {
  using enum seeml::fuzz::Arm;
  switch (arm) {
    case kArmSmf:
      return r.smf_loaded;
    case kArmSds:
      return r.sds_loaded;
    case kArmPlan:
    case kArmPlanRaw:
      return r.plans_accepted;
    case kArmCompile:
    case kArmCompileData:
      return r.compiles_completed;
    case kArmTokenizer:
      return r.tokenizers_loaded;
    case kArmRecipe:
      return r.recipes_parsed;
    case kArmReport:
      return r.reports_read;
    case kArmCount:
      break;
  }
  return 0;
}

TEST(FuzzReplay, ATruncatedSeedIsRefusedOnEveryArm) {
  for (const SeedCase& seed : SeedCases()) {
    const std::vector<uint8_t> input = ReadSeed(seed.name);
    if (input.size() < 8) continue;
    // Half of it, and all but its last byte of the first file's header.
    for (const size_t keep : {input.size() / 2, size_t{7}}) {
      const std::vector<uint8_t> cut(
          input.begin(), input.begin() + static_cast<std::ptrdiff_t>(keep));
      const Reach r = Run(cut);
      if (Accepted(r, seed.arm) != 0)
        ADD_FAILURE("seed " + std::string(seed.name) + " cut to " +
                    std::to_string(keep) + " bytes was accepted");
    }
  }
}

TEST(FuzzReplay, ABadMagicIsRefusedByEveryBinaryReader) {
  for (const char* name :
       {"smf-mlp", "smf-textdec", "sds-class", "sds-masked", "plan-mlp",
        "plan-dec-cached", "plan-raw-mlp", "compile-mlp", "tokenizer-txt",
        "report-segdec", "compile-data-sds"}) {
    std::vector<uint8_t> input = ReadSeed(name);
    ASSERT_GT(input.size(), size_t{8});
    input[1] ^= 0x20;  // the first byte of the first file: its magic
    const auto arm = static_cast<seeml::fuzz::Arm>(input[0]);
    if (Accepted(Run(input), arm) != 0)
      ADD_FAILURE(std::string(name) + " with a bad magic was accepted");
  }
}

/// `input` with the little-endian `value` written `at` bytes into its
/// first file.
template <typename T>
std::vector<uint8_t> Poke(std::vector<uint8_t> input, size_t at, T value) {
  std::memcpy(input.data() + 1 + at, &value, sizeof(value));
  return input;
}

TEST(FuzzReplay, AnOversizedCountIsRefusedNotAllocated) {
  using enum seeml::fuzz::Arm;
  // SMF: the preamble's tensor and op counts (magic, version, tensors, ops).
  for (const size_t at : {size_t{8}, size_t{12}}) {
    EXPECT_EQ(Run(Poke(ReadSeed("smf-dec"), at, 0xFFFFFFFFu)).smf_loaded,
              uint64_t{0});
    EXPECT_EQ(
        Run(Poke(ReadSeed("compile-dec"), at, 0xFFFFFFFFu)).compiles_started,
        uint64_t{0});
  }
  // SDS: the sample count and the input width (magic, version, n, d).
  for (const size_t at : {size_t{8}, size_t{16}})
    EXPECT_EQ(Run(Poke(ReadSeed("sds-class"), at, ~uint64_t{0} / 3)).sds_loaded,
              uint64_t{0});
  // The plan, re-sealed, so the validator is what meets each count.
  for (const size_t at :
       {offsetof(PlanHeader, arena_size),
        offsetof(PlanHeader, train_instr_count),
        offsetof(PlanHeader, rodata_size), offsetof(PlanHeader, emit_count),
        offsetof(PlanHeader, persist_init_size), offsetof(PlanHeader, batch)}) {
    const Reach r =
        Run(Poke(ReadSeed("plan-dec-cached"), at, ~uint64_t{0} / 3));
    EXPECT_EQ(r.train_steps, uint64_t{0});
    EXPECT_EQ(r.instructions_executed, uint64_t{0});
  }
  // The tokenizer section's vocabulary and merge counts (the head's sixth
  // and seventh words).
  for (const size_t at : {size_t{20}, size_t{24}})
    EXPECT_EQ(
        Run(Poke(ReadSeed("tokenizer-txt"), at, 0x7FFFFFFFu)).tokenizers_loaded,
        uint64_t{0});
}

std::vector<uint8_t> TextInput(seeml::fuzz::Arm arm, std::string_view text) {
  std::vector<uint8_t> input{arm};
  input.insert(input.end(), text.begin(), text.end());
  return input;
}

TEST(FuzzReplay, MalformedTextIsRefused) {
  using enum seeml::fuzz::Arm;
  // The recipe file: not an object, an unknown field, a value past its
  // bound, a nested object, a number no JSON parser takes.
  for (const char* text :
       {"", "[]", "{\"no_such_field\": 1}",
        "{\"lora_rank\": 1, \"lora_rank\": 2}", "{\"lora_rank\": {\"a\": 1}}",
        "{\"lr\": .5}", "{\"lora_rank\": 2} x"})
    EXPECT_EQ(Run(TextInput(kArmRecipe, text)).recipes_parsed, uint64_t{0});
  for (const char* text :
       {"{\"lora_rank\": 99999999999999999999999}", "{\"lora_rank\": 5000}",
        "{\"lr\": 1e999}", "{\"loss\": \"hinge\"}"}) {
    const Reach r = Run(TextInput(kArmRecipe, text));
    EXPECT_EQ(r.recipes_parsed, uint64_t{1});
    EXPECT_EQ(r.recipes_applied, uint64_t{0});
  }
  // The report beside its model: another model's hash, a truncated
  // object, a duplicate key — and no report at all for a model that
  // carries a tokenizer section.
  const std::vector<uint8_t> seed = ReadSeed("report-segdec");
  const std::string_view sep = seeml::fuzz::kPartSeparator;
  const auto at = std::search(
      seed.begin(), seed.end(), sep.begin(), sep.end(),
      [](uint8_t a, char b) { return a == static_cast<uint8_t>(b); });
  ASSERT_TRUE(at != seed.end());
  const std::vector<uint8_t> model(seed.begin(), at);  // arm byte + SMF
  const std::string report(at + static_cast<std::ptrdiff_t>(sep.size()),
                           seed.end());
  auto with_report = [&](const std::string& text) {
    std::vector<uint8_t> input = model;
    input.insert(input.end(), sep.begin(), sep.end());
    input.insert(input.end(), text.begin(), text.end());
    return input;
  };
  EXPECT_EQ(Run(with_report(report)).reports_read, uint64_t{1});
  std::string foreign = report;
  const size_t hash_at = foreign.find("\"smf_hash\": \"") + 13;
  foreign[hash_at] = foreign[hash_at] == '0' ? '1' : '0';
  EXPECT_EQ(Run(with_report(foreign)).reports_read, uint64_t{0});
  EXPECT_EQ(Run(with_report(report.substr(0, report.size() / 2))).reports_read,
            uint64_t{0});
  EXPECT_EQ(Run(with_report(report + report)).reports_read, uint64_t{0});
  EXPECT_EQ(Run(model).reports_read, uint64_t{0});
  // The text ingest: a JSONL line that is no object, an unterminated
  // string, a document that is not text; the section still loads.
  const std::vector<uint8_t> tok = ReadSeed("tokenizer-jsonl");
  const auto tok_at = std::search(
      tok.begin(), tok.end(), sep.begin(), sep.end(),
      [](uint8_t a, char b) { return a == static_cast<uint8_t>(b); });
  ASSERT_TRUE(tok_at != tok.end());
  for (const char* jsonl :
       {"not json\n", "{\"text\": \"unterminated\n", "{\"text\": 3}\n",
        "{\"prompt\": \"only a prompt\"}\n", "[1, 2]\n"}) {
    std::vector<uint8_t> input(
        tok.begin(), tok_at + static_cast<std::ptrdiff_t>(sep.size()));
    input.push_back(seeml::fuzz::kTextJsonl);
    input.push_back(4);
    input.insert(input.end(), jsonl, jsonl + std::strlen(jsonl));
    const Reach r = Run(input);
    EXPECT_EQ(r.tokenizers_loaded, uint64_t{1});
    EXPECT_EQ(r.jsonl_parsed, uint64_t{0});
    EXPECT_EQ(r.ingests, uint64_t{0});
  }
}

// --- The regression inputs ---------------------------------------------------

TEST(FuzzReplay, EveryRegressionInputRunsToTheEnd) {
  // test/fuzz/seeds/regressions/: inputs that once crashed, hung or
  // tripped a sanitizer, kept in the fuzzing corpus and replayed here.
  // Returning at all is the assertion (the sanitizer jobs make it sharp).
  const std::filesystem::path dir = SeedDir() + "/regressions";
  if (!std::filesystem::is_directory(dir)) return;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    std::ifstream f(entry.path(), std::ios::binary);
    const std::vector<uint8_t> input{std::istreambuf_iterator<char>(f),
                                     std::istreambuf_iterator<char>()};
    const Reach r = Run(input);
    EXPECT_EQ(r.compiled_plans_refused, uint64_t{0});
    EXPECT_EQ(r.ingest_sds_refused, uint64_t{0});
    // A slow unit must run nothing. slow-unit-grad-accum is plan-mlp-accum
    // with one bit of grad_accum_steps flipped, 2 -> 2^30 + 2 — a "single
    // step" of a billion micro-batches until the harness bounded the
    // accumulation — and since #223 the load contract refuses it (the count
    // is bounded at kMaxGradAccumSteps). Any other slow unit is a plan the
    // loader accepts and the harness's work bounds must turn away:
    // slow-unit-grad-accum-in-contract (2 -> 2^15 + 2) is inside the
    // contract's bound, and the harness's tighter cap holds it.
    const std::string name = entry.path().filename().string();
    if (name.starts_with("slow-unit-")) {
      if (name == "slow-unit-grad-accum") {
        EXPECT_EQ(r.plans_accepted, uint64_t{0});
        EXPECT_EQ(r.plans_refused, uint64_t{1});
      } else {
        EXPECT_EQ(r.plans_accepted, uint64_t{1});
      }
      EXPECT_EQ(r.train_steps + r.train_refused, uint64_t{0});
      EXPECT_EQ(r.evals + r.merges, uint64_t{0});
    }
    // timeout-nightly-grad-accum is the hang reproduced in the nightly's
    // own harness over its cached corpus (CR3 #207, 2026-11-24): a 325-byte
    // plan asking for 1,090,519,040 micro-batches a step. The load
    // contract's bound (#223) refuses it.
    if (name == "timeout-nightly-grad-accum") {
      EXPECT_EQ(r.plans_accepted, uint64_t{0});
      EXPECT_EQ(r.plans_refused, uint64_t{1});
      EXPECT_EQ(r.instructions_executed, uint64_t{0});
    }
  }
}

// --- A deterministic mutation pass -------------------------------------------

/// SplitMix64: the mutator's only state, so a pass is its seed.
struct Rng {
  uint64_t state;
  uint64_t Next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint64_t Below(uint64_t n) { return n == 0 ? 0 : Next() % n; }
};

/// One structured edit of `input`: a field-sized write of an interesting
/// value (a count, a dimension, an offset, an op kind, an attribute — the
/// formats are little-endian fields of 1, 2, 4 and 8 bytes), biased to the
/// head of the input where a container keeps its metadata; or a cut, a
/// repeated run, a flipped bit, or bytes of another seed spliced in.
std::vector<uint8_t> Mutate(std::vector<uint8_t> input, Rng& rng,
                            const std::vector<uint8_t>& other) {
  static constexpr uint64_t kInteresting[] = {
      0,          1,          2,          3,          4,          7,
      8,          16,         31,         32,         33,         63,
      64,         127,        128,        255,        256,        4095,
      4096,       65535,      65536,      0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
      1ull << 32, 1ull << 40, 1ull << 62, ~0ull >> 1, 1ull << 63, ~0ull};
  if (input.size() < 2) return input;
  const size_t body = input.size() - 1;
  // Three in four edits land in the first 1.5 KiB: the metadata.
  auto offset = [&](size_t width) {
    const size_t span = rng.Below(4) != 0 ? std::min<size_t>(body, 1536) : body;
    return 1 + rng.Below(span > width ? span - width : 1);
  };
  switch (rng.Below(10)) {
    case 0:
    case 1:
    case 2:
    case 3: {  // a field takes an interesting value
      const size_t width = size_t{1} << rng.Below(4);
      if (body < width) break;
      const uint64_t v = kInteresting[rng.Below(std::size(kInteresting))];
      std::memcpy(input.data() + offset(width), &v, width);
      break;
    }
    case 4: {  // a field moves by a little
      const size_t width = size_t{1} << rng.Below(4);
      if (body < width) break;
      const size_t at = offset(width);
      uint64_t v = 0;
      std::memcpy(&v, input.data() + at, width);
      v += rng.Below(2) ? 1 + rng.Below(4) : ~uint64_t{0} - rng.Below(4);
      std::memcpy(input.data() + at, &v, width);
      break;
    }
    case 5:  // a bit flips
      input[offset(1)] ^= static_cast<uint8_t>(1u << rng.Below(8));
      break;
    case 6:  // a byte takes any value
      input[offset(1)] = static_cast<uint8_t>(rng.Next());
      break;
    case 7:  // the input is cut
      input.resize(1 + rng.Below(body));
      break;
    case 8: {  // a run is repeated in place (a duplicated record or name)
      const size_t len = 1 + rng.Below(std::min<size_t>(body, 64));
      const size_t at = offset(len);
      if (at + len > input.size()) break;
      const std::vector<uint8_t> run(
          input.begin() + static_cast<std::ptrdiff_t>(at),
          input.begin() + static_cast<std::ptrdiff_t>(at + len));
      input.insert(input.begin() + static_cast<std::ptrdiff_t>(at), run.begin(),
                   run.end());
      break;
    }
    case 9: {  // bytes of another seed are spliced in
      if (other.size() < 2) break;
      const size_t len = 1 + rng.Below(std::min<size_t>(other.size() - 1, 64));
      const size_t from = 1 + rng.Below(other.size() - len);
      const size_t at = offset(len);
      if (at + len > input.size()) break;
      std::memcpy(input.data() + at, other.data() + from, len);
      break;
    }
  }
  return input;
}

TEST(FuzzReplay, MutatedSeedsAreRefusedOrRunNeverCrash) {
  // Not a fuzzing campaign: a fixed pass of structured mutations over
  // every seed, so each sanitizer job runs the harness on a few thousand
  // hostile-but-near-valid inputs. What must hold on every one: the
  // harness returns; a plan the compiler emitted is one the loader takes;
  // a corpus Ingest wrote is one the corpus reader takes.
  //
  //   SEEML_FUZZ_MUTATIONS=N    mutated inputs per seed (default 128)
  //   SEEML_FUZZ_MUTATION_SEED  the pass's seed (default 1)
  //   SEEML_FUZZ_KEEP=1         each input is written, before it runs, to
  //                             seeml_fuzz_keep.unit in the system's temp
  //                             directory, so a crash or a hang leaves its
  //                             reproducer on disk. A switch, not a path:
  //                             nothing the environment says names a file
  //                             this suite opens (cpp/path-injection).
  uint64_t per_seed = 128, pass_seed = 1;
  if (const char* n = std::getenv("SEEML_FUZZ_MUTATIONS"))
    per_seed = std::strtoull(n, nullptr, 10);
  if (const char* s = std::getenv("SEEML_FUZZ_MUTATION_SEED"))
    pass_seed = std::strtoull(s, nullptr, 10);
  const bool keep = std::getenv("SEEML_FUZZ_KEEP") != nullptr;
  const std::filesystem::path keep_path =
      std::filesystem::temp_directory_path() / "seeml_fuzz_keep.unit";
  std::vector<std::vector<uint8_t>> seeds;
  for (const SeedCase& seed : SeedCases()) seeds.push_back(ReadSeed(seed.name));
  Reach total;
  uint64_t inputs = 0;
  for (size_t s = 0; s < seeds.size(); ++s) {
    Rng rng{pass_seed * 0x100000001B3ull + s};
    for (uint64_t i = 0; i < per_seed; ++i) {
      // One to four edits: most inputs stay near a valid one.
      std::vector<uint8_t> input = seeds[s];
      const uint64_t edits = 1 + rng.Below(4);
      for (uint64_t e = 0; e < edits; ++e)
        input = Mutate(std::move(input), rng, seeds[rng.Below(seeds.size())]);
      if (keep) {
        std::ofstream f(keep_path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(input.data()),
                static_cast<std::streamsize>(input.size()));
      }
      const Reach r = Run(input);
      ++inputs;
      if (r.compiled_plans_refused != 0 || r.ingest_sds_refused != 0) {
        ADD_FAILURE(
            "seed " + std::string(SeedCases()[s].name) + ", pass " +
            std::to_string(pass_seed) + ", mutation " + std::to_string(i) +
            (r.compiled_plans_refused != 0
                 ? ": the loader refused a plan the compiler emitted"
                 : ": the corpus reader refused a corpus Ingest wrote"));
      }
      total.plans_accepted += r.plans_accepted;
      total.plans_refused += r.plans_refused;
      total.train_steps += r.train_steps;
      total.compiles_started += r.compiles_started;
      total.compiles_completed += r.compiles_completed;
      total.tokenizers_loaded += r.tokenizers_loaded;
      total.ingests += r.ingests;
      total.recipes_parsed += r.recipes_parsed;
      total.reports_read += r.reports_read;
    }
  }
  // What the pass reached, for whoever reads the log.
  std::printf(
      "             %llu mutated inputs (pass %llu): %llu plans accepted, "
      "%llu refused, %llu steps trained; %llu compiles started, %llu "
      "completed; %llu tokenizer sections, %llu ingests, %llu recipes, "
      "%llu reports\n",
      static_cast<unsigned long long>(inputs),
      static_cast<unsigned long long>(pass_seed),
      static_cast<unsigned long long>(total.plans_accepted),
      static_cast<unsigned long long>(total.plans_refused),
      static_cast<unsigned long long>(total.train_steps),
      static_cast<unsigned long long>(total.compiles_started),
      static_cast<unsigned long long>(total.compiles_completed),
      static_cast<unsigned long long>(total.tokenizers_loaded),
      static_cast<unsigned long long>(total.ingests),
      static_cast<unsigned long long>(total.recipes_parsed),
      static_cast<unsigned long long>(total.reports_read));
  // The pass is neither all refusals nor all accepts: mutations reach
  // past the seal and the loaders, and the contract refuses some.
  if (per_seed >= 128) {
    EXPECT_GT(total.plans_accepted, uint64_t{0});
    EXPECT_GT(total.plans_refused, uint64_t{0});
    EXPECT_GT(total.train_steps, uint64_t{0});
    EXPECT_GT(total.compiles_completed, uint64_t{0});
  }
  EXPECT_EQ(inputs, per_seed * seeds.size());
}

// --- The plan-aware edit (test/fuzz/plan_mutate.h) ---------------------------

TEST(FuzzReplay, APlanOperandEditRewritesOneInstructionAndNothingElse) {
  const std::vector<uint8_t> seed = ReadSeed("plan-mlp");
  ASSERT_GT(seed.size(), sizeof(PlanHeader) + 1);
  PlanHeader header;
  std::memcpy(&header, seed.data() + 1, sizeof(header));
  constexpr size_t kIns = sizeof(seeml::update::UpdateInstruction);
  Rng rng{7};
  uint64_t changed = 0;
  for (int i = 0; i < 256; ++i) {
    std::vector<uint8_t> edited = seed;
    const uint64_t r0 = rng.Next(), r1 = rng.Next(), r2 = rng.Next();
    ASSERT_TRUE(seeml::fuzz::MutatePlanOperand(edited, r0, r1, r2, rng.Next()));
    ASSERT_EQ(edited.size(), seed.size());
    size_t first = seed.size(), last = 0;
    for (size_t b = 0; b < seed.size(); ++b)
      if (edited[b] != seed[b]) {
        first = std::min(first, b);
        last = b;
      }
    if (first == seed.size()) continue;  // the value it already held
    ++changed;
    // Inside one instruction of one of the four streams the header names.
    bool inside = false;
    for (const auto& [offset, count] :
         {std::pair{header.train_instr_offset, header.train_instr_count},
          std::pair{header.merge_instr_offset, header.merge_instr_count},
          std::pair{header.eval_instr_offset, header.eval_instr_count},
          std::pair{header.step_instr_offset, header.step_instr_count}}) {
      const size_t begin = 1 + offset, end = begin + count * kIns;
      if (first >= begin && last < end)
        inside |= (first - begin) / kIns == (last - begin) / kIns;
    }
    EXPECT_TRUE(inside);
  }
  EXPECT_GT(changed, uint64_t{200});
  // Not a plan: left alone. A model, a header cut short, and a header
  // whose streams lie outside the input.
  for (std::vector<uint8_t> input :
       {ReadSeed("smf-dec"),
        std::vector<uint8_t>(seed.begin(), seed.begin() + 64),
        std::vector<uint8_t>(seed.begin(),
                             seed.begin() + 1 + sizeof(PlanHeader))}) {
    const std::vector<uint8_t> before = input;
    EXPECT_FALSE(seeml::fuzz::MutatePlanOperand(input, 1, 2, 3, 4));
    EXPECT_TRUE(input == before);
  }
}

TEST(FuzzReplay, PlanOperandEditsAreRefusedOrRunNeverCrash) {
  // The validator's operand rules under the edits a byte-level pass almost
  // never makes: every re-sealed plan seed, each instruction field pushed
  // to a boundary or past it. What must hold is that the harness returns —
  // under a sanitizer build, with no report: a rule that stops refusing
  // what it refused lets a kernel read or write outside the arena here.
  // Both verdicts must occur, or the edits are not reaching the rules.
  uint64_t per_seed = 256;
  if (const char* n = std::getenv("SEEML_FUZZ_MUTATIONS"))
    per_seed = 2 * std::strtoull(n, nullptr, 10);
  Rng rng{11};
  uint64_t accepted = 0, refused = 0, plans = 0;
  for (const SeedCase& seed : SeedCases()) {
    if (seed.arm != seeml::fuzz::kArmPlan) continue;
    const std::vector<uint8_t> clean = ReadSeed(seed.name);
    ++plans;
    for (uint64_t i = 0; i < per_seed; ++i) {
      std::vector<uint8_t> input = clean;
      for (uint64_t e = 0, edits = 1 + rng.Below(2); e < edits; ++e) {
        const uint64_t r0 = rng.Next(), r1 = rng.Next(), r2 = rng.Next();
        ASSERT_TRUE(
            seeml::fuzz::MutatePlanOperand(input, r0, r1, r2, rng.Next()));
      }
      const Reach r = Run(input);
      accepted += r.plans_accepted;
      refused += r.plans_refused;
    }
  }
  std::printf(
      "             %llu operand edits over %llu plan seeds: %llu accepted, "
      "%llu refused\n",
      static_cast<unsigned long long>(plans * per_seed),
      static_cast<unsigned long long>(plans),
      static_cast<unsigned long long>(accepted),
      static_cast<unsigned long long>(refused));
  EXPECT_GT(plans, uint64_t{5});
  EXPECT_GT(accepted, uint64_t{0});
  EXPECT_GT(refused, uint64_t{0});
}

}  // namespace
