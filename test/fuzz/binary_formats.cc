// =============================================================================
// libFuzzer harness over the attacker/corruption-facing parsers and the
// code that trusts what they accept:
//
//   [0] LoadSmf                  — the model container reader
//   [1] UpdateEngine load + run  — the .seeu plan loader + validator, the
//       plan RE-SEALED first so a mutated plan meets the contract and the
//       validator, not the seal; an accepted plan is executed
//   [2] Dataset::LoadFromFile    — the corpus reader
//   [3] LoadSmf + UpdateCompiler::Compile — the semantic layer past the
//       loader (sema, parser, passes, driver), which consumes
//       attacker-controlled loader-ACCEPTED models; a compiled plan is
//       loaded and executed
//   [4] UpdateEngine load + run, the plan as it is — the seal check itself
//   [5] ByteLevelBpe::Load + DocumentsFromJsonl + Ingest — the tokenizer
//       section and the text ingest (CR3 #208)
//   [6] ParseRecipeFile + Apply  — the recipe file
//   [7] LoadSmf + ReadExportReport — the exporter's report, held to its model
//   [8] [3] over a recipe and a corpus (.sds, .jsonl or .txt): the compile's
//       data path
//
// The first input byte selects the arm; the rest is the arm's input, the
// files of a multi-file arm joined by kPartSeparator (fuzz_reach.h states
// the framing; test/fuzz/make_seeds.py writes the seed corpus in it). The
// contract under test: arbitrary bytes may be REJECTED but must never
// crash, overflow, or over-read — the runtime trusts these validators
// completely (Execute() runs unchecked after Initialize() accepts a plan).
//
// Every arm's work is bounded before it is done (CR3 #207): a unit the
// harness lets through costs milliseconds, so a slow one is a finding.
//
// Build: cmake -DSEEML_FUZZ=ON (clang), then run seeml_fuzz_formats on a
// copy of test/fuzz/seeds. Every other build compiles this file into
// seeml_fuzz_replay_test, which replays those seeds through the same entry
// point and asserts what they reach (the counters of fuzz_reach.h).
// =============================================================================

#include "test/fuzz/fuzz_reach.h"
#include "test/fuzz/plan_mutate.h"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "source/plan/update_types.h"
#include "compiler/diagnostics/logger.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/ingressor/export_report.h"
#include "runtime/pipeline/dataset.h"
#include "runtime/dispatcher/update_engine.h"
#include "source/contract/split.h"
#include "source/identity/hash.h"
#include "source/recipe/recipe.h"
#include "source/recipe/recipe_file.h"
#include "source/statistics/plan_work.h"
#include "source/tokenizer/byte_level_bpe.h"
#include "source/tokenizer/text_ingest.h"
#include "compiler/frontend/ingressor/model_reader.h"

namespace seeml::fuzz {

Reach& TheReach() {
  static Reach reach;
  return reach;
}

namespace {
bool narrating = false;
}  // namespace

void SetNarration(bool on) { narrating = on; }

}  // namespace seeml::fuzz

namespace {

using seeml::fuzz::Reach;
using seeml::fuzz::TheReach;
using seeml::update_rt::Dataset;

using Bytes = std::span<const uint8_t>;

/// A refusal, narrated when asked (SetNarration); `r` passes through.
template <typename T>
const std::expected<T, std::string>& Seen(
    const char* stage, const std::expected<T, std::string>& r) {
  if (!r && seeml::fuzz::narrating)
    std::fprintf(stderr, "fuzz: %s: %s\n", stage, r.error().c_str());
  return r;
}

std::string TempPath(const char* tag) {
  return std::string("/tmp/seeml_fuzz_") + tag + "_" +
         std::to_string(::getpid());
}

void WriteBytes(const std::string& path, Bytes bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  if (!bytes.empty()) std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

/// The input's files: split at every kPartSeparator.
std::vector<Bytes> SplitParts(Bytes input) {
  const std::string_view sep = seeml::fuzz::kPartSeparator;
  std::vector<Bytes> parts;
  size_t from = 0;
  while (true) {
    const auto begin = input.begin() + static_cast<std::ptrdiff_t>(from);
    const auto hit = std::search(
        begin, input.end(), sep.begin(), sep.end(),
        [](uint8_t a, char b) { return a == static_cast<uint8_t>(b); });
    parts.push_back(input.subspan(from, static_cast<size_t>(hit - begin)));
    if (hit == input.end()) break;
    from += static_cast<size_t>(hit - begin) + sep.size();
  }
  return parts;
}

std::string_view AsText(Bytes b) {
  return {reinterpret_cast<const char*>(b.data()), b.size()};
}

// --- The plan arms -----------------------------------------------------------

// The bounds an accepted plan must meet before the harness executes it:
// what one unit may cost, read from the plan's own header and instructions
// (the loader has proved every section and every operand extent in
// bounds). Three costs, each saturating so a fuzzed dimension cannot wrap
// a cap away:
//   - the dense products — the GEMMs and the attention family, whose work
//     is not bounded by what they touch (a tiled attention over S
//     positions does S*S work in O(S) memory): plan_work.h's FLOPs;
//   - every other instruction, linear in the bytes it touches, which the
//     arena and the plan bound: instructions x (arena + plan) bytes;
//   - how often each program runs: the train program once per micro-batch
//     — the header's grad_accum_steps, which nothing bounded until a
//     one-bit mutation of an accumulating plan (2 -> 2^30 + 2) ran a
//     "single step" of a billion micro-batches (CR3 #207,
//     test/fuzz/seeds/regressions/) — and the eval program once per batch
//     of the corpus, per evaluation. The load contract now bounds the
//     count at kMaxGradAccumSteps = 2^16 (#223): a bound against days per
//     step, not against one fuzz unit's second, so the harness keeps its
//     own cap of 8 far below it (2^15 + 2 is contract-legal and still
//     32,770 train passes; slow-unit-grad-accum-in-contract holds the cap).
constexpr uint64_t kFuzzArenaCap = 64ull << 20;  // 64 MiB
constexpr uint64_t kFuzzInstructionCap = 4096;
constexpr uint64_t kFuzzAccumulationCap = 8;
constexpr uint64_t kFuzzFlopsCap = 100000000ull;  // 5e7 multiply-adds
constexpr uint64_t kFuzzTouchCap = 1ull << 30;    // instructions x bytes

/// One program of a loaded plan, decoded.
std::vector<seeml::update::UpdateInstruction> Program(Bytes plan,
                                                      uint64_t offset,
                                                      uint64_t count) {
  std::vector<seeml::update::UpdateInstruction> program(
      static_cast<size_t>(count));
  if (count)
    std::memcpy(program.data(), plan.data() + offset,
                static_cast<size_t>(count) * sizeof(program[0]));
  return program;
}

/// The four programs of a plan the loader accepted.
struct Programs {
  std::vector<seeml::update::UpdateInstruction> train, step, merge, eval;
};

/// Whether one training step (the train program `micro_batches` times, the
/// step program once), `evaluations` passes of the eval program and one
/// merge stay inside the harness's bounds.
bool WithinWorkBounds(const Programs& p, uint64_t micro_batches,
                      uint64_t evaluations, uint64_t bytes) {
  using seeml::update::AttentionFlops;
  using seeml::update::GemmFlops;
  using seeml::update::SatAdd;
  using seeml::update::SatMul;
  auto flops = [](const std::vector<seeml::update::UpdateInstruction>& prog) {
    return SatAdd(GemmFlops(prog), AttentionFlops(prog));
  };
  const uint64_t dense =
      SatAdd(SatAdd(SatMul(flops(p.train), micro_batches), flops(p.step)),
             SatAdd(SatMul(flops(p.eval), evaluations), flops(p.merge)));
  if (dense > kFuzzFlopsCap) return false;
  const uint64_t runs =
      SatAdd(SatAdd(SatMul(p.train.size(), micro_batches), p.step.size()),
             SatAdd(SatMul(p.eval.size(), evaluations), p.merge.size()));
  return SatMul(runs, bytes) <= kFuzzTouchCap;
}

/// A corpus synthesized from the header's geometry — zero features or
/// token ids, zero labels, every id below any vocabulary, every position a
/// target under a masked loss.
std::expected<Dataset, std::string> SynthesizeCorpus(
    const seeml::update::PlanHeader& h) {
  auto none = [] { return std::unexpected(std::string("no corpus")); };
  if (h.batch == 0) return none();
  if (h.input_kind == 1) {
    if (h.seq_len == 0 || h.batch % h.seq_len != 0) return none();
    const uint64_t records = h.batch / h.seq_len;
    std::vector<int32_t> tokens(records * (h.seq_len + 1), 0);
    std::vector<uint8_t> mask;
    if (h.data_flags & seeml::update::kDataFlagMaskedLoss)
      mask.assign(records * h.seq_len, 1);
    auto corpus = Dataset::FromTokens(std::move(tokens), records, h.seq_len,
                                      std::move(mask), h.tokenizer_hash);
    // A plan pinned to its corpus (v22) is a second seal on the data side:
    // the synthesized corpus claims the pinned identity, so the kernels of
    // a pinned plan run too.
    if (corpus && (h.data_flags & seeml::update::kDataFlagCorpusPinned))
      corpus->set_content_hash(h.corpus_hash);
    return corpus;
  }
  if (h.input_floats % h.batch != 0) return none();
  const uint64_t input_dim = h.input_floats / h.batch;
  std::vector<float> inputs(h.input_floats, 0.0f);
  std::vector<uint8_t> labels;
  uint64_t label_dim = 0;
  if (h.label_kind == 1) {
    if (h.label_bytes != h.batch * sizeof(int32_t)) return none();
    labels.assign(h.label_bytes, 0);
  } else if (h.label_kind == 2) {
    if (h.label_bytes % (h.batch * sizeof(float)) != 0) return none();
    label_dim = h.label_bytes / (h.batch * sizeof(float));
    labels.assign(h.label_bytes, 0);
  } else if (h.label_kind != 0) {
    return none();
  }
  auto corpus =
      Dataset::FromMemory(std::move(inputs), std::move(labels), h.batch,
                          input_dim, h.label_kind, label_dim);
  if (corpus && (h.data_flags & seeml::update::kDataFlagCorpusPinned))
    corpus->set_content_hash(h.corpus_hash);
  return corpus;
}

/// One training step of an accepted plan, then its eval and merge programs
/// — on `corpus` when the caller has the plan's own (split first, as the
/// runner splits it, when the plan compiled a held-out split), else on a
/// synthesized one. Nothing runs when the plan is outside the harness's
/// bounds or no corpus fits its header.
void ExecuteOnce(seeml::update_rt::UpdateEngine& engine, Bytes plan,
                 Dataset* corpus) {
  seeml::update::PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  if (h.arena_size > kFuzzArenaCap) return;
  using seeml::update::SatAdd;
  if (SatAdd(SatAdd(h.train_instr_count, h.step_instr_count),
             SatAdd(h.merge_instr_count, h.eval_instr_count)) >
      kFuzzInstructionCap)
    return;
  const uint64_t micro_batches = std::max<uint64_t>(h.grad_accum_steps, 1);
  if (micro_batches > kFuzzAccumulationCap) return;
  const Programs programs{
      Program(plan, h.train_instr_offset, h.train_instr_count),
      Program(plan, h.step_instr_offset, h.step_instr_count),
      Program(plan, h.merge_instr_offset, h.merge_instr_count),
      Program(plan, h.eval_instr_offset, h.eval_instr_count)};
  std::expected<Dataset, std::string> synthesized =
      std::unexpected(std::string("unused"));
  const bool given = corpus != nullptr;
  if (!given) {
    synthesized = SynthesizeCorpus(h);
    if (!Seen("corpus", synthesized)) return;
    corpus = &*synthesized;
  }
  // The eval program runs once per compiled batch of what it scores: the
  // corpus by Evaluate below and, with a held-out split, that split before
  // and after the step. Three passes over the whole corpus bound them all.
  const uint64_t rows = corpus->input_kind() == 1 && h.seq_len != 0
                            ? std::max<uint64_t>(h.batch / h.seq_len, 1)
                            : std::max<uint64_t>(h.batch, 1);
  const uint64_t evaluations =
      seeml::update::SatMul(3, corpus->num_samples() / rows + 1);
  if (!WithinWorkBounds(programs, micro_batches, evaluations,
                        SatAdd(h.arena_size, plan.size())))
    return;
  std::expected<Dataset, std::string> held_out =
      std::unexpected(std::string("unused"));
  seeml::update_rt::TrainOptions options;
  options.log_every = 0;
  // The first step of the schedule the plan was compiled for: a one-step
  // horizon ends inside any compiled warmup, which Train refuses — as it
  // refused every plan compiled under the default (cosine) schedule that
  // this harness loaded before CR3 #208.
  options.horizon_steps = h.default_steps;
  if (given && (h.data_flags & seeml::update::kDataFlagSplit)) {
    const std::vector<uint64_t> held = seeml::update::HeldOutIndices(
        corpus->num_samples(), h.split_seed,
        seeml::update::ValFracOf(h.val_frac_bits));
    if (!held.empty()) {
      held_out = corpus->SplitHeldOut(held, seeml::update::SplitHash(held));
      if (Seen("SplitHeldOut", held_out)) options.validation = &*held_out;
    }
  }
  Reach& reach = TheReach();
  if (Seen("Train", engine.Train(*corpus, 1, options))) {
    ++reach.train_steps;
    // What the step ran: the train program and, under gradient
    // accumulation, the step program.
    for (const auto* program : {&programs.train, &programs.step})
      for (const seeml::update::UpdateInstruction& ins : *program) {
        ++reach.instructions_executed;
        if (ins.opcode < reach.opcode_executed.size())
          ++reach.opcode_executed[ins.opcode];
        if (ins.flags & seeml::update::kFlagRelaxed) ++reach.relaxed_executed;
      }
  } else {
    ++reach.train_refused;
  }
  if (Seen("Evaluate", engine.Evaluate(*corpus))) ++reach.evals;
  if (Seen("RunMerge", engine.RunMerge())) ++reach.merges;
}

/// Loads `plan` and, when the loader accepts it, executes it. Bounded
/// before the load as well as before the step: the load allocates the
/// header's arena, and a header asking for more than the harness's cap is
/// not loaded — the engine refuses an arena beyond the host's memory on
/// its own (a unit test holds it to that), and anything between the cap
/// and that bound would only exhaust the fuzzing host.
bool LoadAndExecute(Bytes plan, Dataset* corpus,
                    const std::string* source_model) {
  if (plan.size() >= sizeof(seeml::update::PlanHeader)) {
    seeml::update::PlanHeader h;
    std::memcpy(&h, plan.data(), sizeof(h));
    if (h.arena_size > kFuzzArenaCap) return false;
  }
  // The measurement permit (F2 #206): a plan that carries relaxed
  // instructions — a mutated one, or the relaxed seed — is executed too.
  // The relaxed kernels are the ones to run under the sanitizer, and no
  // certificate can exist for a mutated plan; the refusal itself is the
  // loader tests'.
  seeml::update_rt::UpdateEngine engine;
  if (!Seen("load", engine.LoadFromMemory(plan.data(), plan.size(),
                                          seeml::update_rt::RelaxedPlanPermit::
                                              UncertifiedForMeasurement()))) {
    ++TheReach().plans_refused;
    return false;
  }
  ++TheReach().plans_accepted;
  // An accepted plan is EXECUTED (CR3 #144 F22): one training step on a
  // corpus synthesized from the header's geometry, so the kernels run over
  // the fuzzed instruction stream under the sanitizer, not only the
  // validators. Bounded first (WithinWorkBounds), so no input turns into a
  // minutes-long step that libFuzzer would report as a hang.
  // A narrow-storage plan's eval program reads the source model file
  // (v17); the compile arms have it.
  if (source_model)
    (void)Seen("BindSourceModel", engine.BindSourceModel(*source_model));
  ExecuteOnce(engine, plan, corpus);
  return true;
}

// --- The compile arms --------------------------------------------------------

// The bounds a loader-accepted model and its recipe must meet before the
// harness compiles them (CR3 #207: the nightly's lost unit was last seen
// here). The model's constant tensors are backed by the input's bytes, so
// they are bounded already; what is NOT backed by bytes is bounded here —
// the sequence length, a declared input width, the batch, the adapter
// rank, the accumulation — together with the counts every pass walks. The
// memory gate is always configured, so a footprint past it is refused by
// the compiler before the backend allocates an image of it.
constexpr size_t kFuzzModelNodes = 512;  // tensors, and ops
constexpr uint64_t kFuzzSeqCap = 32;
constexpr int64_t kFuzzDimCap = 4096;  // a dimension no payload backs
constexpr uint64_t kFuzzConstBytes = 4ull << 20;  // the tensors, summed
constexpr int64_t kFuzzBatchCap = 64;
constexpr int64_t kFuzzRankCap = 8;
constexpr uint32_t kFuzzAccumCap = 4;
constexpr uint64_t kFuzzStepsCap = 1ull << 20;
constexpr uint64_t kFuzzMemoryBudget = 64ull << 20;
constexpr size_t kFuzzCorpusBytes = 64u << 10;

bool ModelWithinBounds(const seeml::update::SmfModel& model) {
  if (model.tensors.size() > kFuzzModelNodes ||
      model.ops.size() > kFuzzModelNodes || model.seq_len > kFuzzSeqCap)
    return false;
  uint64_t const_bytes = 0;
  for (const seeml::update::SmfTensor& t : model.tensors) {
    // Tensors may share a payload; their sizes may not sum past the cap.
    if (t.byte_size > kFuzzConstBytes) return false;
    const_bytes += t.byte_size;
    if (const_bytes > kFuzzConstBytes) return false;
    if (t.is_const) continue;
    for (const int64_t d : t.dims)
      if (d > kFuzzDimCap) return false;
  }
  return true;
}

/// Holds a recipe's config to the harness's bounds.
void BoundConfig(seeml::update::UpdateConfig& c) {
  c.batch = std::clamp<int64_t>(c.batch, 1, kFuzzBatchCap);
  c.lora.rank = std::clamp<int64_t>(c.lora.rank, 1, kFuzzRankCap);
  c.grad_accum_steps =
      std::clamp<uint32_t>(c.grad_accum_steps, 1, kFuzzAccumCap);
  c.default_steps = std::clamp<uint64_t>(c.default_steps, 1, kFuzzStepsCap);
  c.memory_budget_bytes = kFuzzMemoryBudget;
  c.attention_cache_budget_bytes =
      std::min(c.attention_cache_budget_bytes, kFuzzMemoryBudget);
  c.dump_sir = false;
}

/// Compiles the model at `model_path` under `config` (bounded here) and
/// runs what comes out: a plan the compiler emits is one the loader takes.
/// `fit_batch` sets the batch to the smallest a token model's sequence
/// admits — two sequences — whatever the config says.
void CompileAndExecute(const std::string& model_path,
                       seeml::update::UpdateConfig config, bool fit_batch) {
  auto model = seeml::update::LoadSmf(model_path);
  if (!Seen("LoadSmf", model)) return;
  ++TheReach().smf_loaded;
  if (!ModelWithinBounds(*model)) return;
  if (fit_batch && model->seq_len != 0)
    config.batch = static_cast<int64_t>(2 * model->seq_len);
  BoundConfig(config);
  // The exporter's report is bound to the model's bytes — a second seal,
  // which a mutated model never meets (arm [7] reads reports). A model
  // that needs one gets it in code, so the driver's refusal is not where
  // every tokenizer-carrying model stops.
  if (seeml::update::NeedsExportReport(*model)) {
    config.export_report.present = true;
    config.export_report.report_hash = 1;
  }
  Reach& reach = TheReach();
  ++reach.compiles_started;
  auto compiled = seeml::update::UpdateCompiler(config).Compile(*model);
  if (!Seen("Compile", compiled)) return;
  ++reach.compiles_completed;
  if (!config.data_path.empty()) ++reach.corpora_compiled;
  // The plan's own corpus, when the compile took one: the canonical SDS it
  // wrote from text, or the SDS it was given.
  std::expected<Dataset, std::string> corpus =
      std::unexpected(std::string("none"));
  if (!compiled->corpus_sds.empty()) {
    const std::string path = TempPath("csds");
    WriteBytes(path, compiled->corpus_sds);
    corpus = Dataset::LoadFromFile(path);
    std::remove(path.c_str());
  } else if (compiled->corpus_from_sds) {
    corpus = Dataset::LoadFromFile(config.data_path);
  }
  if (LoadAndExecute(compiled->plan, corpus ? &*corpus : nullptr, &model_path))
    ++reach.compiled_plans_accepted;
  else if (compiled->arena_size <= kFuzzArenaCap)
    ++reach.compiled_plans_refused;
}

// --- The text arms -----------------------------------------------------------

constexpr size_t kFuzzTextBytes = 16u << 10;

/// [section | kind seq text]: the tokenizer section, then text through it.
void TokenizerArm(Bytes input) {
  const std::vector<Bytes> parts = SplitParts(input);
  namespace tk = seeml::tokenizer;
  auto tokenizer = tk::ByteLevelBpe::Load(parts[0]);
  if (!Seen("ByteLevelBpe::Load", tokenizer)) return;
  Reach& reach = TheReach();
  ++reach.tokenizers_loaded;
  if (parts.size() < 2 || parts[1].size() < 2) return;
  const uint8_t kind = parts[1][0] % 2;
  tk::IngestOptions options;
  options.seq_len = 1 + parts[1][1] % 32;
  options.one_document = kind == seeml::fuzz::kTextOneDocument;
  Bytes text_bytes = parts[1].subspan(2);
  if (text_bytes.size() > kFuzzTextBytes)
    text_bytes = text_bytes.first(kFuzzTextBytes);
  const std::string_view text = AsText(text_bytes);
  std::vector<int32_t> ids;
  if (Seen("Encode", tokenizer->Encode(text, ids))) ++reach.texts_encoded;
  std::vector<tk::Document> documents;
  if (options.one_document) {
    documents.push_back({std::string(text), {}, false});
  } else {
    auto parsed = tk::DocumentsFromJsonl(text);
    if (!Seen("DocumentsFromJsonl", parsed)) return;
    ++reach.jsonl_parsed;
    documents = std::move(*parsed);
  }
  auto ingested =
      tk::Ingest(*tokenizer, tk::SectionHash(parts[0]), documents, options);
  if (!Seen("Ingest", ingested)) return;
  ++reach.ingests;
  reach.ingest_windows += ingested->manifest.windows;
  (void)tk::ManifestJson(ingested->manifest);
  // The corpus Ingest writes is one the corpus reader takes.
  const std::string path = TempPath("ingest");
  WriteBytes(path, ingested->sds);
  if (Seen("the ingested SDS", Dataset::LoadFromFile(path)))
    ++reach.ingest_sds_loaded;
  else
    ++reach.ingest_sds_refused;
  std::remove(path.c_str());
}

/// The recipe file: parsed, applied, resolved.
std::expected<seeml::update::UpdateConfig, std::string> ReadRecipe(
    Bytes text, bool with_data) {
  namespace recipe = seeml::update::recipe;
  auto values = recipe::ParseRecipeFile(AsText(text));
  if (!Seen("ParseRecipeFile", values)) return std::unexpected(values.error());
  ++TheReach().recipes_parsed;
  seeml::update::UpdateConfig config;
  if (auto ok = recipe::Apply(*values, {.teacher = false, .data = with_data},
                              &config);
      !Seen("recipe::Apply", ok))
    return std::unexpected(ok.error());
  ++TheReach().recipes_applied;
  (void)recipe::ConfigViolation(config);
  (void)recipe::ResolvedText(config);
  (void)recipe::RecipeHash(config);
  return config;
}

/// [smf | report]: the report read beside its model and held to it.
void ReportArm(Bytes input) {
  const std::vector<Bytes> parts = SplitParts(input);
  const std::string path = TempPath("rsmf");
  const std::string report_path = path + seeml::update::kExportReportSuffix;
  WriteBytes(path, parts[0]);
  if (parts.size() >= 2) WriteBytes(report_path, parts[1]);
  if (auto model = seeml::update::LoadSmf(path)) {
    ++TheReach().smf_loaded;
    auto report =
        Seen("ReadExportReport", seeml::update::ReadExportReport(path, *model));
    if (report && report->present) ++TheReach().reports_read;
  }
  std::remove(path.c_str());
  std::remove(report_path.c_str());
}

/// [smf | recipe | kind corpus]: the compile over a recipe and a corpus.
void CompileDataArm(Bytes input) {
  const std::vector<Bytes> parts = SplitParts(input);
  if (parts.size() < 3 || parts[2].empty()) return;
  auto config = ReadRecipe(parts[1], /*with_data=*/true);
  if (!config) return;
  static constexpr const char* kExtensions[] = {".sds", ".jsonl", ".txt"};
  const std::string model_path = TempPath("dsmf");
  const std::string corpus_path =
      TempPath("corpus") + kExtensions[parts[2][0] % 3];
  Bytes corpus = parts[2].subspan(1);
  if (corpus.size() > kFuzzCorpusBytes) corpus = corpus.first(kFuzzCorpusBytes);
  WriteBytes(model_path, parts[0]);
  WriteBytes(corpus_path, corpus);
  config->data_path = corpus_path;
  CompileAndExecute(model_path, std::move(*config), /*fit_batch=*/false);
  std::remove(model_path.c_str());
  std::remove(corpus_path.c_str());
}

}  // namespace

#if defined(SEEML_LIBFUZZER)
// libFuzzer's own mutator, for the edits this one does not make.
extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

/// Three in four edits of a re-sealed plan unit go to an instruction's
/// operands (test/fuzz/plan_mutate.h says why); every other edit, and
/// every other arm's, is libFuzzer's. Only the libFuzzer build defines the
/// hook: the replay suite links this file without libFuzzer and calls
/// MutatePlanOperand itself.
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size,
                                          size_t max_size, unsigned int seed) {
  uint64_t state = seed;
  const auto next = [&state] {  // SplitMix64
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  if (size >= 1 && data[0] % seeml::fuzz::kArmCount == seeml::fuzz::kArmPlan &&
      next() % 4 != 0) {
    const uint64_t r0 = next(), r1 = next(), r2 = next(), r3 = next();
    if (seeml::fuzz::MutatePlanOperand({data, size}, r0, r1, r2, r3))
      return size;
  }
  return LLVMFuzzerMutate(data, size, max_size);
}
#endif  // SEEML_LIBFUZZER

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // The compiler narrates each pass at INFO; a fuzzing run would write
  // nothing else, and the log's tail is what a finding is read from.
  static const bool quiet = [] {
    seeml::diag::Logger::SetLevel(seeml::diag::LogLevel::kError);
    return true;
  }();
  (void)quiet;
  if (size < 1) return 0;
  Reach& reach = TheReach();
  ++reach.inputs;
  const uint8_t selector = data[0] % seeml::fuzz::kArmCount;
  const Bytes body(data + 1, size - 1);

  switch (selector) {
    case seeml::fuzz::kArmSmf: {
      const std::string path = TempPath("smf");
      WriteBytes(path, body);
      if (Seen("LoadSmf", seeml::update::LoadSmf(path))) ++reach.smf_loaded;
      std::remove(path.c_str());
      break;
    }
    case seeml::fuzz::kArmPlan: {
      // The integrity hash would reject virtually every mutated input at the
      // first gate and starve the deeper validators of coverage. The hash
      // exists to catch corruption, not adversaries — so the harness seals
      // the blob the same way the compiler does, letting the fuzzer attack
      // the section/instruction/bounds validation underneath.
      std::vector<uint8_t> plan(body.begin(), body.end());
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
      LoadAndExecute(plan, nullptr, nullptr);
      break;
    }
    case seeml::fuzz::kArmPlanRaw: {
      // The plan as the input holds it: the seal check is the first gate,
      // and a plan that passes it (a seed, or one the fuzzer left whole)
      // runs as arm [1]'s does. An engine borrows the bytes it loads.
      const std::vector<uint8_t> plan(body.begin(), body.end());
      LoadAndExecute(plan, nullptr, nullptr);
      break;
    }
    case seeml::fuzz::kArmSds: {
      const std::string path = TempPath("sds");
      WriteBytes(path, body);
      if (Seen("Dataset::LoadFromFile", Dataset::LoadFromFile(path)))
        ++reach.sds_loaded;
      std::remove(path.c_str());
      break;
    }
    case seeml::fuzz::kArmCompile: {
      // The contract under test: a loader-accepted model may be REFUSED by
      // sema/parser/driver but must never crash them. A tiny batch — two
      // rows, or two of a sequential model's sequences — keeps the
      // per-input compile cost bounded.
      const std::string path = TempPath("smfc");
      WriteBytes(path, body);
      seeml::update::UpdateConfig config;
      config.batch = 2;
      CompileAndExecute(path, config, /*fit_batch=*/true);
      std::remove(path.c_str());
      break;
    }
    case seeml::fuzz::kArmTokenizer:
      TokenizerArm(body);
      break;
    case seeml::fuzz::kArmRecipe:
      (void)ReadRecipe(body, /*with_data=*/true);
      break;
    case seeml::fuzz::kArmReport:
      ReportArm(body);
      break;
    case seeml::fuzz::kArmCompileData:
      CompileDataArm(body);
      break;
  }
  return 0;
}
