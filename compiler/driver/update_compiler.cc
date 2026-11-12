#include "compiler/driver/update_compiler.h"

#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "compiler/backend/allocation/arena_binder.h"
#include "compiler/backend/assembly/plan_assembly.h"
#include "compiler/backend/selection/instruction_lowering.h"
#include "compiler/diagnostics/generating/error.h"
#include "compiler/driver/contract.h"
#include "compiler/driver/statistics.h"
#include "compiler/frontend/accountant/resource_analyzer.h"
#include "compiler/frontend/computation/parser.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "compiler/frontend/tokenization/corpus_ingress.h"
#include "source/contract/split.h"
#include "compiler/frontend/topology/sema.h"
#include "compiler/frontend/representation/sir.h"
#include "compiler/analysis/update_passes.h"
#include "source/plan/metadata.h"
#include "source/recipe/recipe.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace generating = seeml::diag::generating;

// =============================================================================
// UpdateCompiler::Compile
//
// The driver is orchestration: one Compilation holds what the stages hand
// each other, and each stage is one step of the pipeline (S6 #152) —
// frontend (config, ingress, topology, data, the accountant's gate, the
// forward graph and the loss), analysis (the three pass phases, the merge
// program, storage selection), backend (binding, rematerialization,
// lowering, assembly), then the report.
// =============================================================================

namespace {

using Status = std::expected<void, std::string>;

// The diagnostics contract: every error that crosses the driver boundary
// must name a registered unit, or the failing process cannot be delimited.
std::expected<CompiledUpdate, std::string> Attributed(
    std::expected<CompiledUpdate, std::string> compiled) {
  if (!compiled && !WellFormedDiagnostic(compiled.error()))
    return generating::Error(
        generating::kDriver,
        "unattributed diagnostic escaped a subsystem: " + compiled.error());
  return compiled;
}

/// The f32 / i32 bytes of every result of `ops`: what stops being written
/// when those ops are fused away.
uint64_t ResultBytes(const std::vector<sir::Operation*>& ops) {
  uint64_t bytes = 0;
  for (const sir::Operation* op : ops)
    for (const auto& r : op->results())
      bytes += static_cast<uint64_t>(r->shape().volume()) * 4;
  return bytes;
}

struct Compilation {
  const UpdateConfig& config_;
  const SmfModel& source;
  const SmfModel* teacher;
  const bool consume;

  // Every pass (PassManager) and every driver phase between them, timed:
  // the compile report carries these, so a compile-side performance claim
  // is a measurement anyone can repeat (E5, #84).
  using PhaseClock = std::chrono::steady_clock;
  std::vector<PassTiming> timings;
  PhaseClock::time_point phase_start = PhaseClock::now();
  void phase(const char* name, size_t ops) {
    const auto now = PhaseClock::now();
    timings.push_back(
        {name, ops,
         std::chrono::duration<double, std::milli>(now - phase_start).count()});
    phase_start = now;
  }

  // Frontend.
  bool wants_teacher = false;
  CompileFacts facts;
  int64_t batch = 0;
  std::optional<sema::Topology> topology;
  std::optional<sema::Topology> teacher_topology;
  bool token_input = false;
  int64_t in_dim = 0;
  std::optional<tokenization::CorpusIngress> corpus;
  bool masked_loss = false;
  TrainingFootprint footprint;
  AttentionKind attention = AttentionKind::kAuto;
  sir::Block block;
  GraphBuild build;
  int64_t out_dim = 0;
  sir::Value* teacher_out = nullptr;
  sir::Value* labels = nullptr;
  uint32_t label_kind = 0;
  uint64_t label_bytes = 0;
  sir::Value* loss = nullptr;

  // Analysis.
  AttilingDecision attention_decision;
  std::vector<GraftedAdapter> adapters;
  std::vector<sir::Value*> trainables;
  std::vector<sir::Operation*> primal_ops;
  std::unordered_map<sir::Value*, sir::Value*> param_grads;
  uint32_t accum = 1;
  std::optional<MergeProgram> merge;
  std::unordered_map<const sir::Value*, float> quant_scales;
  std::unordered_set<const sir::Value*> bf16_weights;

  // Backend.
  std::unordered_set<const sir::Value*> pinned;
  std::optional<ArenaBinding> binding;
  std::unordered_set<const sir::Value*> merge_pinned;
  std::unordered_map<const sir::Value*, uint64_t> merge_bound;
  uint64_t plain_arena = 0;
  bool remat_on = false;
  Rematerialization remat;
  PlanPrograms programs;
  uint64_t relaxed_gemms = 0, relaxed_attention = 0;
  std::unordered_set<const sir::Value*> shipped;
  PlanHeader header;
  uint32_t recorded_accum = 1;
  uint64_t accumulator_bytes = 0;
  CompiledUpdate result;

  Compilation(const UpdateConfig& config, const SmfModel& source_model,
              const SmfModel* teacher_model, bool consume_models)
      : config_(config),
        source(source_model),
        teacher(teacher_model),
        consume(consume_models) {}

  // The config alone: every float finite and in range, the adapter-only
  // preconditions, the kernel tile, the schedule triple.
  Status ValidateConfig();
  // The weights cross into the compiler finite; the sweep records ranges.
  Status Ingress();
  // Topology to completion, then the cross-input rules.
  Status VerifyTopology();
  // --data: the corpus held to the model, and the compiled split.
  Status IngestData();
  // The masked objective: asked for, or implied by the corpus.
  Status DecideMaskedLoss();
  // The model as read, for the frontend screen.
  Status RecordFrontendFacts();
  // The accountant's early gate and the attention decision.
  Status ProveFootprint();
  // Forward SIR: student, and the frozen teacher sharing its input.
  Status BuildForwardGraph();
  // The loss, grafted onto the forward graph.
  Status GraftLoss();
  // Phase A, and the primal snapshot that becomes the eval program.
  Status StructuralPasses();
  // Phase B: autodiff pruned to the adapters, optimizer synthesis.
  Status CalculusPasses();
  // The merge program and the frozen-weight storage selection.
  Status MergeAndReview();
  // Phase C: the three fusers, then the DCE sweep.
  Status FusionPasses();
  // Segmented arena binding, the merge program bound above it.
  Status BindArenas();
  // Rematerialization, decided on the plain binding.
  Status Rematerialize();
  // The train stream, its relaxed instructions, the step program split.
  Status LowerTrain();
  // The eval program, which scores what ships, and the merge program.
  Status LowerEvalAndMerge();
  // The header words the compile decided; assembly lays the sections out.
  Status FillHeader();
  // The plan blob, sealed, and the exact memory gate.
  Status Assemble();
  // Debug hooks: the SIR dump, the timings, per-parameter and per-adapter refs.
  Status ReportDebug();
  // The compile's numbers, the screens, and the backend boundary check.
  Status ReportSummary();
  // The merge program bound into the training arena (again after a rebind).
  Status BindMerge();
  // The plan's metadata section: the manifest words and the ledger (v23).
  std::vector<uint8_t> BuildMetadata();
  ResolveFn ResolveTrain();

  std::expected<CompiledUpdate, std::string> Run();
};

Status Compilation::ValidateConfig() {
  wants_teacher = config_.loss == LossKind::kKLDistill ||
                             config_.loss == LossKind::kXEntPlusKL;
  // The recipe's bounds a config built in code must hold too (S4 #150,
  // review N18): an unbounded rank overflowed into an invalid plan.
  if (auto why = recipe::ConfigViolation(config_))
    return generating::Error(generating::kDriver, "config " + *why);
  // CR1 #142: every float the config carries is finite and within the
  // range the runtime's load contract and the validator apply — the library
  // path enforced nothing, so a NaN learning rate compiled into the plan
  // header and a NaN temperature into the KL words, and passed every
  // comparison downstream. (The recipe's table, source/recipe/recipe.h, is
  // narrower: its upper bounds on lr, weight decay, the clip, alpha and the
  // distillation weight are the command line's and are not held here.)
  {
    const OptimizerSpec& o = config_.optimizer;
    struct Rule {
      const char* name;
      float value;
      bool in_range;
      bool applies;
    };
    const Rule rules[] = {
        {"lr", o.lr, o.lr > 0.0f, true},
        {"beta1", o.beta1, o.beta1 >= 0.0f && o.beta1 < 1.0f, true},
        {"beta2", o.beta2, o.beta2 >= 0.0f && o.beta2 < 1.0f, true},
        {"eps", o.eps, o.eps > 0.0f, true},
        {"weight_decay", o.weight_decay, o.weight_decay >= 0.0f, true},
        {"clip_norm", o.clip_norm, o.clip_norm >= 0.0f, true},
        {"min_lr_factor", o.min_lr_factor,
         o.min_lr_factor >= 0.0f && o.min_lr_factor <= 1.0f, true},
        {"lora.alpha", config_.lora.alpha, config_.lora.alpha > 0.0f, true},
        {"distill_weight", config_.distill_weight,
         config_.distill_weight >= 0.0f, wants_teacher},
        {"temperature", config_.temperature,
         config_.temperature >= kKlTemperatureMin &&
             config_.temperature <= kKlTemperatureMax,
         wants_teacher},
    };
    for (const Rule& r : rules)
      if (r.applies && (!std::isfinite(r.value) || !r.in_range))
        return generating::Error(
            generating::kDriver,
            std::string("config ") + r.name +
                " is not a finite value in its range (got " +
                std::to_string(r.value) +
                (std::string(r.name) == "temperature"
                     ? std::string("; the range is ") +
                           kKlTemperatureRangeText
                     : std::string()) +
                ")");
  }
  if (wants_teacher && !teacher)
    return generating::Error(generating::kDriver,
                             "the selected loss requires a teacher model");
  // The adapter-only package (F6 #134, plan v21): the frozen weights stay
  // in the model file the device already holds, bound by hash at run time,
  // so the plan ships at adapter scale. Its preconditions are the config
  // and the file alone, so they are checked here, before any work: a model
  // loaded from a file (the offsets and the hash), the f32 base the file
  // holds, the exact kernels (the relaxed ones read rodata weights), an
  // evaluation that reads the file (the plan has no weights of its own to
  // score), and no teacher — its weights live in another file, and the one
  // file BindSourceModel maps is the student's, so a teacher's source ref
  // would read the student's bytes at the teacher's offsets.
  if (config_.adapter_only) {
    if (source.content_hash == 0)
      return generating::Error(
          generating::kDriver,
          "--adapter-only needs a model loaded from a file: the plan reads "
          "the frozen weights from it at run time, bound by its hash");
    if (config_.quantize_base || config_.bf16_base)
      return generating::Error(
          generating::kDriver,
          "--adapter-only keeps the frozen weights in the source model, which "
          "is f32: --quantize-base / --bf16-base would store narrow copies "
          "in the plan");
    if (config_.precision != Precision::kF32)
      return generating::Error(
          generating::kDriver,
          "--adapter-only runs the exact kernels: the relaxed GEMMs read "
          "rodata weights, and an adapter-only plan carries none");
    if (config_.eval_on_plan)
      return generating::Error(
          generating::kDriver,
          "--validation-scores plan: an adapter-only plan has no weights of "
          "its own to score; every program reads the source model");
    if (wants_teacher)
      return generating::Error(
          generating::kDriver,
          "--adapter-only binds every frozen weight to the source model file, "
          "and a teacher's weights live in another file the runtime does not "
          "bind: distillation keeps its rodata");
  }

  // The kernel policy is written into the header verbatim; the only
  // contract the kernels have is the K tile's 4-wide unroll alignment
  // (kernel_policy.h), and a plan that breaks it would be rejected by
  // every runtime's load-time contract — so it is refused here, where the
  // author can read it.
  if (config_.gemm_tile_k % 4 != 0)
    return generating::Error(
        generating::kDriver,
        "gemm_tile_k " + std::to_string(config_.gemm_tile_k) +
            " is not a multiple of the kernel's 4-wide unroll");

  // The schedule triple (E8, #91): optimization's rule
  // (analysis/optimization/schedule.h).
  if (auto ok = VerifySchedule(config_.optimizer, config_.default_steps); !ok)
    return std::unexpected(ok.error());
  return {};
}

Status Compilation::Ingress() {
  // --- Ingress: the weights cross into the compiler finite (CR1 #142 F12
  // / F34; the ingressor's rule, S1 #147), and the same pass records each
  // tensor's range for the frontend screen.
  {
    auto swept = SweepWeights(source, "student");
    if (!swept) return std::unexpected(swept.error());
    facts.frontend.ingressor.ranges = std::move(swept->ranges);
    if (teacher)
      if (auto t = SweepWeights(*teacher, "teacher"); !t)
        return std::unexpected(t.error());
  }
  phase("ingress", 0);
  return {};
}

Status Compilation::VerifyTopology() {
  batch = config_.batch;

  // --- Topology, to completion (S1 #147): the whole model verified and
  // every value shaped from the tensor table before the accountant walks
  // it or computation builds it; then what no single model can know — the
  // teacher against the student, the loss against the input kind, the
  // adapter filters against the weights, the reserved `t::` namespace.
  auto verified = sema::VerifyModel(source, batch);
  if (!verified) return std::unexpected(verified.error());
  topology = std::move(*verified);
  // The input's kind and width are topology's one reading of the tensor
  // table: a rank-1 dynamic ({-1}) input declares i32 token ids (SMF v4),
  // otherwise the width is the static, positive last dim (topology refuses
  // the rest — a non-positive width would flow into the SIR as a dynamic
  // dim and seal a plan the runtime rejects).
  token_input = topology->token_input;
  in_dim = topology->input_width;
  if (token_input) {
    if (source.seq_len == 0)
      return generating::Error(generating::kDriver,
                               "token models must declare a seq_len");
    if (source.seq_len > static_cast<uint64_t>(batch) ||
        batch % static_cast<int64_t>(source.seq_len) != 0)
      return generating::Error(
          generating::kDriver,
          "the compiled batch must be a whole number of sequences");
  }
  if (wants_teacher) {
    auto tt = sema::VerifyModel(*teacher, batch);
    if (!tt) return std::unexpected(tt.error());
    teacher_topology = std::move(*tt);
  }
  if (auto ok = sema::VerifyCrossInputs(
          source, *topology, wants_teacher ? teacher : nullptr,
          teacher_topology ? &*teacher_topology : nullptr,
          sema::CrossInputs{batch, config_.loss, config_.lora.target_filters});
      !ok)
    return std::unexpected(ok.error());
  phase("topology", source.ops.size());
  return {};
}

Status Compilation::IngestData() {
  // --- The data path (S2 #148): --data runs the vendored stages over the
  // corpus with the model's own tokenizer section, holds the result to the
  // model, and pins the plan to the canonical corpus by hash. Before the
  // accountant: the corpus rows are its.
  if (!config_.data_path.empty()) {
    auto c = tokenization::IngestCorpus(
        config_.data_path, source, *topology,
        tokenization::LabelsOf(
            config_.loss, static_cast<uint64_t>(topology->output.dims.at(1))),
        static_cast<uint64_t>(batch) * config_.grad_accum_steps,
        config_.default_steps);
    if (!c) return std::unexpected(c.error());
    corpus = std::move(*c);
    // The compiled split (S3 #149): the held-out records drawn here, by
    // the function the device recomputes; the plan carries the seed, the
    // fraction and the indices' hash.
    if (config_.val_frac != 0.0f) {
      if (!(config_.val_frac > 0.0f && config_.val_frac < 1.0f))
        return generating::Error(generating::kDriver,
                                 "val_frac must lie in (0, 1)");
      if (corpus->facts.records < 2)
        return generating::Error(
            generating::kDriver,
            "a held-out split needs at least two records");
      const std::vector<uint64_t> held = HeldOutIndices(
          corpus->facts.records, config_.split_seed, config_.val_frac);
      corpus->facts.val_records = held.size();
      corpus->facts.split_seed = config_.split_seed;
      corpus->facts.val_split_hash = SplitHash(held);
      corpus->facts.split = true;
      // The epoch arithmetic is over the records the plan trains on: the
      // held-out ones are scored, not stepped through. Both rows scale
      // with the record count (batches per epoch with it, epochs covered
      // against it).
      const double train_share =
          static_cast<double>(corpus->facts.records - held.size()) /
          static_cast<double>(corpus->facts.records);
      corpus->facts.batches_per_epoch *= train_share;
      corpus->facts.epochs_covered /= train_share;
    }
    facts.frontend.corpus = corpus->facts;
    phase("data", source.ops.size());
  } else if (config_.val_frac != 0.0f) {
    return generating::Error(generating::kDriver,
                             "a compiled validation split needs the corpus: "
                             "pass --data with --val-frac");
  }
  return {};
}

Status Compilation::DecideMaskedLoss() {
  // The masked loss (S3 #149, plan v22 kDataFlagMaskedLoss): asked for, or
  // implied by a corpus that carries a mask. The cross-entropy alone has a
  // masked form; the other objectives take every position.
  masked_loss =
      config_.masked_loss || (corpus && corpus->shape.masked);
  if (masked_loss && config_.loss != LossKind::kSoftmaxXEnt)
    return generating::Error(
        generating::kDriver,
        "a loss mask applies to the cross-entropy alone (--loss xent): the "
        "distillation and MSE objectives take every position");
  // A mask marks positions of a token record; a feature corpus carries
  // none (the loader refuses one), so a masked feature plan could be fed
  // by no dataset.
  if (masked_loss && !topology->token_input)
    return generating::Error(
        generating::kDriver,
        "a loss mask applies to token models alone: a feature corpus "
        "carries no per-position mask");
  // Asked for over a corpus that carries no mask: the plan would be pinned
  // to the one corpus the device must then refuse ("takes a loss mask the
  // dataset does not carry") — unsatisfiable by construction.
  if (config_.masked_loss && corpus && !corpus->shape.masked)
    return generating::Error(
        generating::kDriver,
        "--masked-loss with a corpus that carries no loss mask: the plan "
        "would refuse its own corpus; drop --masked-loss, or pass a prompt / "
        "response corpus (or a masked .sds)");
  return {};
}

Status Compilation::RecordFrontendFacts() {
  // The model as read, for the frontend screen (S0 #146). Captured before
  // any pass: the consuming compile releases the weight payloads as plan
  // assembly packs them, so the byte counts come from the tensors' sizes.
  {
    IngressorFacts& ing = facts.frontend.ingressor;
    ing.model_ops = source.ops.size();
    ing.model_tensors = source.tensors.size();
    ing.token_input = token_input;
    ing.input_width = token_input ? 0 : static_cast<uint64_t>(in_dim);
    ing.seq_len = source.seq_len;
    ing.content_hash = source.content_hash;
    TopologyFacts& top = facts.frontend.topology;
    top.vocabulary = topology->vocabulary;
    top.activations = topology->activations;
    top.activation_bytes = topology->activation_bytes;
    top.embedding_ops = topology->embedding_ops;
    top.attention_ops = topology->attention_ops;
    top.has_teacher = teacher != nullptr;
    if (teacher) top.teacher_ops = teacher->ops.size();
    // The weight census is the ingress sweep's: one range per constant
    // tensor, each with its payload bytes — the one statement of what a
    // weight weighs, for the whole table and for the adapted share.
    const auto adapted =
        sema::AdaptedWeights(source, config_.lora.target_filters);
    top.adapted_tensors = adapted.size();
    ing.weight_tensors = ing.ranges.size();
    for (const WeightRange& w : ing.ranges) {
      ing.weight_bytes += w.bytes;
      if (adapted.contains(w.name)) top.adapted_bytes += w.bytes;
    }
  }
  return {};
}

Status Compilation::ProveFootprint() {
  // --- 0. Fail fast: prove the training footprint fits local memory. -------
  // The teacher is a frozen forward — no backward caches its activations —
  // so it is estimated at its peak, not its sum; and --quantize-base stores
  // the student's frozen weights as int8 rodata at 1/4 size (under-counting the
  // unselected remainder keeps the bound a lower bound). Both matter:
  // an over-count here refuses compiles the exact final gate would pass.
  footprint =
      EstimateTrainingFootprint(source, batch, config_.lora.target_filters);
  // Only the student's weights take narrow storage: the teacher stays f32
  // (E12 — it is the distillation target), so it is added after the cut.
  if (config_.quantize_base)
    footprint.weight_bytes /= 4;
  else if (config_.bf16_base)
    footprint.weight_bytes /= 2;
  // An adapter-only plan (F6 #134) carries no weights at all: every frozen
  // tensor is read from the model file the device already holds, through
  // a clean, evictable mapping. The exact side of the bound — the arena
  // plus the plan — holds none of those bytes, so the bound counts none
  // either; the screen says so (S1 #147: the estimate is a lower bound on
  // what the plan and the arena hold, never above it).
  if (config_.adapter_only) footprint.weight_bytes = 0;
  // Rematerialization (step 8b) frees the recomputable activations in the
  // forward wherever it may run — forced, or `auto` against a configured
  // budget (with none, auto keeps the plan plain) — so there the bound is
  // taken without them: a refusal must hold for the plan that would
  // actually be built, and the final gate checks that plan exactly.
  if (config_.remat == RematKind::kOn ||
      (config_.remat == RematKind::kAuto && config_.memory_budget_bytes != 0))
    footprint.activation_bytes -=
        std::min(footprint.recomputable_bytes, footprint.activation_bytes);
  if (wants_teacher)
    footprint += EstimateFrozenForwardFootprint(*teacher, batch);
  // The merged deltas are full-size f32 whatever the base is stored as
  // (finding #8): under --quantize-base, 4x the weights they patch.
  footprint.delta_bytes =
      EstimateLoraDeltaBytes(source, config_.lora.target_filters);
  // The attention decision (E11, plan v15), made on the footprint by
  // DecideAttention (below) — the cache budget, and the memory budget when
  // one is configured; never the compiling host's memory. --attention
  // forces either.
  attention = DecideAttention(footprint, config_);
  if (attention == AttentionKind::kTiled) {
    if (footprint.probs_cache_bytes > 0)
      seeml::diag::Note(generating::kDriver,
                        "attention tiled: " +
                            std::to_string(footprint.probs_cache_bytes) +
                            " B of probability caches become " +
                            std::to_string(footprint.attention_stats_bytes) +
                            " B of stats rows");
    footprint.probs_cache_bytes = footprint.attention_stats_bytes;
  }
  {
    AccountantFacts& acc = facts.frontend.accountant;
    acc.footprint_weights = footprint.weight_bytes;
    acc.footprint_activations = footprint.activation_bytes;
    acc.footprint_deltas = footprint.delta_bytes;
    acc.footprint_caches = footprint.probs_cache_bytes;
    acc.footprint_total = footprint.total_bytes();
    acc.memory_budget = config_.memory_budget_bytes != 0
                            ? config_.memory_budget_bytes
                            : DetectLocalMemoryBytes();
    acc.weights_mapped = config_.adapter_only;
  }
  facts.remat_budget_configured = config_.memory_budget_bytes != 0;
  if (auto fits = CheckTrainableLocally(footprint, config_.memory_budget_bytes);
      !fits) {
    // The caches were kept — by force, or by `auto` with no memory budget
    // to weigh them against — and the footprint does not fit: say when the
    // tiled family would, so the refusal names its own remedy.
    std::string why = fits.error();
    if (attention == AttentionKind::kCached &&
        footprint.probs_cache_bytes > footprint.attention_stats_bytes) {
      TrainingFootprint tiled = footprint;
      tiled.probs_cache_bytes = footprint.attention_stats_bytes;
      if (CheckTrainableLocally(tiled, config_.memory_budget_bytes))
        why += config_.attention == AttentionKind::kAuto
                   ? " (--attention tiled would fit; --attention auto tiles "
                     "by itself only against a configured "
                     "--memory-budget-mib, never against this host's "
                     "memory)"
                   : " (--attention tiled would fit)";
    }
    return std::unexpected(std::move(why));
  }
  return {};
}

Status Compilation::BuildForwardGraph() {
  // --- 1. Forward SIR: student (+ frozen teacher sharing the input). -------
  build.input = token_input
                    ? block.addArgument(sir::DataType::I32, sir::Shape{batch})
                    : block.addArgument(sir::DataType::F32,
                                        sir::Shape{batch, in_dim});

  auto student_out = BuildForward(block, source, "", build.input, batch, build,
                                  &*topology);
  if (!student_out) return std::unexpected(student_out.error());
  build.output = *student_out;
  out_dim = build.output->shape().dims.at(1);
  if (out_dim <= 0)
    return generating::Error(generating::kDriver,
                             "model output width must be static and positive");

  if (wants_teacher) {
    // Topology's cross-input check proved the teacher's input and output
    // agree with the student's; the build is a translation.
    auto t_out = BuildForward(block, *teacher, std::string(sema::kTeacherPrefix),
                              build.input, batch, build, &*teacher_topology);
    if (!t_out) return std::unexpected(t_out.error());
    teacher_out = *t_out;
  }

  // Frontend boundary: the forward SIR and graph-build state must be fit
  // for analysis before anything is grafted onto them.
  if (auto ok = VerifyFrontendContract(block, build); !ok)
    return std::unexpected(ok.error());
  return {};
}

Status Compilation::GraftLoss() {
  // --- 2. Loss grafting. ----------------------------------------------------

  auto make_xent = [&]() {
    labels = block.addArgument(sir::DataType::I32, sir::Shape{batch});
    label_kind = 1;
    label_bytes = static_cast<uint64_t>(batch) * sizeof(int32_t);
    sir::Operation* op = block.appendOp(
        masked_loss ? "sc_high.softmax_xent_masked" : "sc_high.softmax_xent");
    op->addOperand(build.output);
    op->addOperand(labels);
    sir::Value* l = op->addResult("loss.xent", sir::DataType::F32, sir::Shape{});
    op->addResult("loss.xent_probs", sir::DataType::F32,
                  sir::Shape{batch, out_dim});
    return l;
  };
  auto make_kl = [&]() {
    sir::Operation* op = block.appendOp("sc_high.kl_distill");
    op->setAttribute("temperature", config_.temperature);
    // Hinton et al.: the soft-target loss is scaled by T^2 so its gradient
    // (T*(p_s - p_t)/N) stays commensurate with the hard-label term at any
    // temperature; without it, --temperature covertly re-weights xent+kl.
    // The objective is decided here, once; the runtime only multiplies.
    op->setAttribute("loss_scale", config_.temperature * config_.temperature);
    op->addOperand(build.output);
    op->addOperand(teacher_out);
    sir::Value* l = op->addResult("loss.kl", sir::DataType::F32, sir::Shape{});
    op->addResult("loss.kl_ps", sir::DataType::F32, sir::Shape{batch, out_dim});
    op->addResult("loss.kl_pt", sir::DataType::F32, sir::Shape{batch, out_dim});
    return l;
  };

  switch (config_.loss) {
    case LossKind::kSoftmaxXEnt:
      loss = make_xent();
      break;
    case LossKind::kMse: {
      labels =
          block.addArgument(sir::DataType::F32, sir::Shape{batch, out_dim});
      label_kind = 2;
      label_bytes =
          static_cast<uint64_t>(batch) * static_cast<uint64_t>(out_dim) *
          sizeof(float);
      sir::Operation* op = block.appendOp("sc_high.mse");
      op->addOperand(build.output);
      op->addOperand(labels);
      loss = op->addResult("loss.mse", sir::DataType::F32, sir::Shape{});
      break;
    }
    case LossKind::kKLDistill:
      loss = make_kl();
      break;
    case LossKind::kXEntPlusKL: {
      sir::Value* xent = make_xent();
      sir::Value* kl = make_kl();
      sir::Operation* sx = block.appendOp("sc_high.scale");
      sx->setAttribute("alpha", 1.0f - config_.distill_weight);
      sx->addOperand(xent);
      sir::Value* wx =
          sx->addResult("loss.xent_w", sir::DataType::F32, sir::Shape{});
      sir::Operation* sk = block.appendOp("sc_high.scale");
      sk->setAttribute("alpha", config_.distill_weight);
      sk->addOperand(kl);
      sir::Value* wk =
          sk->addResult("loss.kl_w", sir::DataType::F32, sir::Shape{});
      sir::Operation* total = block.appendOp("sc_high.add");
      total->addOperand(wx);
      total->addOperand(wk);
      loss = total->addResult("loss.total", sir::DataType::F32, sir::Shape{});
      break;
    }
  }

  // The loss now owns the only live reference to the network output. The
  // rewriting passes below redirect or destroy values without maintaining
  // build.output, so it is retired here — a later read would otherwise see
  // the pre-adapter (or freed) value and silently miss every rewrite.
  build.output = nullptr;

  facts.ops_in = block.numOps();
  phase("frontend", block.numOps());
  return {};
}

Status Compilation::StructuralPasses() {
  // --- 3. Structural passes (phase A): the attention decision, LoRA
  // grafting, the RoPE table — everything that must precede the primal
  // snapshot. The pass
  // manager re-verifies the block after each pass, so a corrupting rewrite
  // is reported against its author.
  {
    PassManager pm;
    // Attention memory (E11, plan v15): before the snapshot, so the eval
    // program runs the same family, and before autodiff, whose VJP rule
    // reads the decision off the op.
    pm.Add("attention-tiling",
           [&](sir::Block& b) -> std::expected<void, std::string> {
             auto decided =
                 AttentionTiling(config_.attention,
                                 attention == AttentionKind::kTiled,
                                 config_.attention_cache_budget_bytes)
                     .Run(b);
             if (!decided) return std::unexpected(decided.error());
             attention_decision = *decided;
             return {};
           });
    pm.Add("lora-graft",
           [&](sir::Block& b) -> std::expected<void, std::string> {
             auto grafted = LoraGrafter(config_.lora).Run(b);
             if (!grafted) return std::unexpected(grafted.error());
             adapters = std::move(*grafted);
             return {};
           });
    // The RoPE angle table (E3, plan v12): before the snapshot, so the
    // eval program builds and reads it too, and before autodiff, which
    // hands the same table to every backward rotation.
    if (config_.rope_table)
      pm.Add("rope-table",
             [this](sir::Block& b) -> std::expected<void, std::string> {
               auto tables = RopeTableHoister().Run(b);
               if (!tables) return std::unexpected(tables.error());
               facts.rope_tables = *tables;
               if (*tables != 0)
                 seeml::diag::Note(generating::kDriver,
                                   "hoisted the RoPE angles into " +
                                       std::to_string(*tables) +
                                       " table(s)");
               return {};
             });
    if (auto ok = pm.Run(block); !ok) return std::unexpected(ok.error());
    timings.insert(timings.end(), pm.timings().begin(), pm.timings().end());
    phase_start = PhaseClock::now();  // passes are timed by their manager
  }

  for (const GraftedAdapter& a : adapters) {
    trainables.push_back(a.A);
    trainables.push_back(a.B);
  }

  // Snapshot the primal program (adapted forward + loss, nothing else): it
  // becomes the evaluation program, lowered against the same arena binding.
  // Running it computes the current validation loss without touching any
  // parameter — the basis of the runtime's regression gate.
  primal_ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { primal_ops.push_back(op); });
  return {};
}

Status Compilation::CalculusPasses() {
  // --- 4/5. Calculus passes (phase B): reverse-mode autodiff pruned to the
  // adapters, then optimizer synthesis (one program execution == one full
  // step) — again under the per-pass verification gate, which subsumes the
  // former end-of-pipeline SSA validation.
  // Gradient accumulation (roadmap 2a): each micro-batch's gradient is a
  // mean over its own rows, so the seed dL/dL = 1/G makes G folded
  // gradients the mean over the effective batch of batch x G rows.
  accum = std::max<uint32_t>(1, config_.grad_accum_steps);
  {
    PassManager pm;
    pm.Add("autodiff", [&](sir::Block& b) -> std::expected<void, std::string> {
      auto grads = TrainableAutodiff(AccumulationSeed(accum))
                       .Run(b, loss, trainables);
      if (!grads) return std::unexpected(grads.error());
      param_grads = std::move(*grads);
      return {};
    });
    // The optimizer pass also owns the accumulators, so under accumulation
    // it runs even for the optimizer-less finite-difference builds (which
    // then get the folds and no step).
    if (config_.emit_optimizer || accum > 1)
      pm.Add("optimizer", [&](sir::Block& b) {
        return OptimizerSynthesizer(config_.optimizer.kind,
                                    config_.optimizer.clip_norm, accum,
                                    config_.emit_optimizer, config_.fuse_clip)
            .Run(b, param_grads);
      });
    if (auto ok = pm.Run(block); !ok) return std::unexpected(ok.error());
    timings.insert(timings.end(), pm.timings().begin(), pm.timings().end());
    phase_start = PhaseClock::now();  // passes are timed by their manager
  }

  facts.gradients = param_grads.size();
  // What the calculus added (S5 #151): the ops autodiff emitted, and the
  // forward values its backward reads — the activations a training step
  // must keep from forward to backward.
  for (const PassTiming& t : timings)
    if (t.name == "autodiff" && t.ops_after > primal_ops.size())
      facts.adjoint_ops = t.ops_after - primal_ops.size();
  {
    const std::unordered_set<const sir::Operation*> primal(primal_ops.begin(),
                                                           primal_ops.end());
    for (const sir::Operation* op : primal_ops) {
      if (op->isMemoryOp()) continue;
      for (const auto& r : op->results()) {
        bool read_by_backward = false;
        for (const sir::Operation* user : r->users())
          read_by_backward |= !primal.contains(user);
        if (read_by_backward)
          facts.saved_activation_bytes +=
              static_cast<uint64_t>(r->shape().volume()) * 4;
      }
    }
  }
  return {};
}

Status Compilation::MergeAndReview() {
  // --- 6. Merge program: Δ = (α/r)·A@B (commit adds Δ to the model file). ---
  MergeBuilder merger;
  auto merged = merger.Run(adapters);
  if (!merged) return std::unexpected(merged.error());
  merge = std::move(*merged);
  // Analysis boundary: grafts, gradients, and the merge program must be fit
  // for code generation (subsumes the former ad-hoc merge validation).
  if (auto ok = VerifyAnalysisContract(block, adapters, param_grads, *merge);
      !ok)
    return std::unexpected(ok.error());

  // --- 7. Frozen-weight quantization selection. ------------------------------
  // Before the fusion phase: the epilogue fuser must know which GEMMs will
  // lower to the q8 opcode, whose in[3] slot the dequant scale occupies —
  // a fused bias ref has nowhere to ride there.
  if (config_.quantize_base && config_.bf16_base)
    return generating::Error(generating::kDriver,
                             "quantize_base and bf16_base are mutually "
                             "exclusive (one storage precision per weight)");
  if (config_.quantize_base) {
    quant_scales = SelectQuantizedWeights(block, build);
    seeml::diag::Note(generating::kDriver,
                      "quantized " + std::to_string(quant_scales.size()) +
                          " frozen weight(s) to int8 rodata");
  }
  if (config_.bf16_base) {
    bf16_weights = SelectBf16Weights(block, build);
    seeml::diag::Note(generating::kDriver,
                      "stored " + std::to_string(bf16_weights.size()) +
                          " frozen weight(s) as bf16 rodata");
  }

  facts.int8_weights = quant_scales.size();
  facts.bf16_weights = bf16_weights.size();
  phase("merge-and-review", block.numOps());
  return {};
}

Status Compilation::FusionPasses() {
  // --- 7b. Optimization passes (phase C): epilogue fusion, then the DCE
  // sweep — after autodiff, so fusion legality is read off the use-lists
  // (an intermediate the backward program consumes carries that consumer
  // as an extra user and never matches), and after quantization selection.
  {
    PassManager pm;
    if (config_.fuse_epilogues)
      pm.Add("fuse-gemm-epilogue",
             [&](sir::Block& b) -> std::expected<void, std::string> {
               std::unordered_set<const sir::Value*> quantized;
               for (const auto& [w, s] : quant_scales) quantized.insert(w);
               std::unordered_set<const sir::Value*> protected_values;
               protected_values.insert(loss);
               for (const auto& [p, g] : param_grads)
                 protected_values.insert(g);
               auto fused =
                   GemmEpilogueFuser().Run(b, quantized, protected_values);
               if (!fused) return std::unexpected(fused.error());
               facts.epilogue_fusions = fused->fused_chains;
               facts.epilogue_bytes_removed = ResultBytes(fused->fused_away);
               // Fuse-then-rebind: the primal snapshot is the eval program
               // and the DCE root set; the orphaned ops leave it here, so
               // eval lowers the fused GEMMs and the sweep may free them.
               if (!fused->fused_away.empty()) {
                 std::unordered_set<const sir::Operation*> dead(
                     fused->fused_away.begin(), fused->fused_away.end());
                 std::erase_if(primal_ops, [&](const sir::Operation* op) {
                   return dead.contains(op);
                 });
               }
               return {};
             });
    // Before the chain fuser: an add that can absorb a GEMM stops being an
    // elementwise stage at all (its whole pass disappears), which beats
    // sharing one with its neighbours.
    if (config_.fuse_gemm_addend)
      pm.Add("fuse-gemm-addend",
             [&](sir::Block& b) -> std::expected<void, std::string> {
               std::unordered_set<const sir::Value*> narrow;
               for (const auto& [w, s] : quant_scales) narrow.insert(w);
               for (const sir::Value* w : bf16_weights) narrow.insert(w);
               std::unordered_set<const sir::Value*> protected_values;
               protected_values.insert(loss);
               for (const auto& [p, g] : param_grads)
                 protected_values.insert(g);
               auto fused = GemmAddendFuser().Run(b, narrow, protected_values);
               if (!fused) return std::unexpected(fused.error());
               facts.addend_fusions = fused->fused;
               facts.addend_bytes_removed = ResultBytes(fused->fused_away);
               // Fuse-then-rebind, as above: a folded forward GEMM leaves
               // the eval snapshot; the add that absorbed it stays.
               if (!fused->fused_away.empty()) {
                 std::unordered_set<const sir::Operation*> dead(
                     fused->fused_away.begin(), fused->fused_away.end());
                 std::erase_if(primal_ops, [&](const sir::Operation* op) {
                   return dead.contains(op);
                 });
               }
               return {};
             });
    if (config_.fuse_elementwise)
      pm.Add("fuse-elementwise",
             [&](sir::Block& b) -> std::expected<void, std::string> {
               std::unordered_set<const sir::Value*> protected_values;
               protected_values.insert(loss);
               for (const auto& [p, g] : param_grads)
                 protected_values.insert(g);
               auto fused = ElementwiseChainFuser().Run(b, protected_values);
               if (!fused) return std::unexpected(fused.error());
               facts.chain_fusions = fused->fused_chains;
               facts.chain_bytes_removed = ResultBytes(fused->fused_away);
               // Fuse-then-rebind, as above: the folded ops leave the eval
               // snapshot (their tail stays in it, now carrying the chain).
               if (!fused->fused_away.empty()) {
                 std::unordered_set<const sir::Operation*> dead(
                     fused->fused_away.begin(), fused->fused_away.end());
                 std::erase_if(primal_ops, [&](const sir::Operation* op) {
                   return dead.contains(op);
                 });
               }
               return {};
             });
    // The optimization phase's sweep: no op whose results nothing reads may
    // survive to lowering. Beyond the fusion orphans this proves an
    // invariant (the driver builds its programs minimally) — and it is the
    // seam where future rewriting passes get to leave dead ops behind.
    // Roots are every value read outside the stream: the loss slot, the
    // parameter gradients, and the whole primal snapshot, which lowers
    // again as the eval program.
    pm.Add("dce", [&](sir::Block& b) -> std::expected<void, std::string> {
      std::unordered_set<const sir::Value*> roots;
      roots.insert(loss);
      for (const auto& [p, g] : param_grads) roots.insert(g);
      for (const sir::Operation* op : primal_ops)
        for (const auto& r : op->results()) roots.insert(r.get());
      auto removed = DeadCodeElimination().Run(b, roots);
      if (!removed) return std::unexpected(removed.error());
      facts.dce_removed = *removed;
      if (*removed != 0)
        seeml::diag::Note(generating::kDriver,
                          "dce removed " + std::to_string(*removed) +
                              " dead op(s)");
      return {};
    });
    if (auto ok = pm.Run(block); !ok) return std::unexpected(ok.error());
    timings.insert(timings.end(), pm.timings().begin(), pm.timings().end());
    phase_start = PhaseClock::now();  // passes are timed by their manager
  }
  return {};
}

Status Compilation::BindArenas() {
  // --- 8. Segmented arena binding. -------------------------------------------
  // Pin the loss slot and every parameter gradient: they are read outside the
  // program (engine / optimizer-less debug builds) and must survive the step.
  pinned.insert(loss);
  for (const auto& [p, g] : param_grads) pinned.insert(g);

  // An adapter-only plan (F6 #134, v21; its preconditions were checked by
  // ValidateConfig) binds every frozen weight to a source ref and lays out
  // no rodata.
  auto bound = BindArena(block, build, pinned, quant_scales, bf16_weights,
                         config_.adapter_only);
  if (!bound) return std::unexpected(bound.error());
  binding = std::move(*bound);

  // Bind the merge program into the same arena: A/B mirrors resolve through
  // the alias map; delta outputs are pinned fresh transients ABOVE the
  // training high-water mark. They must not share offsets with train/eval
  // transients: the engine's public sequence Train -> RunMerge -> Evaluate
  // -> CommitToModel would otherwise let the eval forward pass overwrite
  // the materialized deltas before commit patches them into the model.
  for (const auto& [delta, adapter_index] : merge->outputs)
    merge_pinned.insert(delta);

  if (auto ok = BindMerge(); !ok) return std::unexpected(ok.error());

  phase("bind-arena", block.numOps());
  return {};
}

Status Compilation::Rematerialize() {
  // --- 8b. Rematerialization (roadmap 2b, F5 #133). --------------------------
  // Decided on the plain binding, which is the only way to know the arena:
  // `auto` recomputes when that arena plus the plan — bounded from above
  // here, before lowering, by one instruction per op over the rodata and
  // the persistent image — would fail the gate the sealed plan must pass
  // (CheckPlanFitsLocally, below); `on` recomputes regardless, so the
  // trade can be measured and tested at any size. The pass re-emits the
  // cheap forward ops before their backward readers (same opcodes, same
  // operands: the same bits), then the transient segment alone is scanned
  // again — storage keeps its refs — and the deltas re-pinned above the
  // new high-water mark. What the plain binding took is kept for the
  // report: bytes_saved is a measurement, not the pass's estimate.
  plain_arena = AlignUp(binding->arena_size);
  remat_on = config_.remat == RematKind::kOn;
  std::string over_budget;  // auto: the gate's verdict, with its numbers
  if (config_.remat == RematKind::kAuto) {
    auto instructions_of = [](const auto& ops) {
      uint64_t n = 0;
      for (const sir::Operation* op : ops)
        if (!op->isMemoryOp()) ++n;
      return n;
    };
    std::vector<const sir::Operation*> stream, merge_stream;
    block.walk([&](sir::Operation* op) { stream.push_back(op); });
    merge->block->walk(
        [&](sir::Operation* op) { merge_stream.push_back(op); });
    const uint64_t plan_bound =
        AlignUp(sizeof(PlanHeader)) +
        (instructions_of(stream) + instructions_of(primal_ops) +
         instructions_of(merge_stream)) *
            sizeof(UpdateInstruction) +
        2 * kSeeuRodataAlignment + AlignUp(binding->rodata_size) +
        AlignUp(binding->persistent_size) +
        merge->outputs.size() * sizeof(EmitEntry) + kSeeuRodataAlignment;
    // Against a CONFIGURED budget only: the plan's bytes, offsets and
    // hash must depend on its inputs, never on the compiling host, and
    // the gate's fallback is the host's memory. With no budget, auto is
    // off and says so once; --remat on forces it, --memory-budget-mib N
    // gives auto a budget the plan can be reproduced against.
    if (config_.memory_budget_bytes == 0) {
      seeml::diag::Note(generating::kDriver,
                        "rematerialization (auto): no memory budget is "
                        "configured, so the plan stays plain — the decision "
                        "would otherwise follow the compiling host's memory; "
                        "--remat on forces it, --memory-budget-mib N decides "
                        "it against a budget");
    } else if (auto fits = CheckPlanFitsLocally(plain_arena, plan_bound,
                                                config_.memory_budget_bytes);
               !fits) {
      remat_on = true;
      over_budget = std::move(fits.error());
    }
  }
  if (remat_on) {
    PassManager pm;
    pm.Add("rematerialize",
           [&](sir::Block& b) -> std::expected<void, std::string> {
             auto done = Rematerializer().Run(b, primal_ops, pinned);
             if (!done) return std::unexpected(done.error());
             remat = *done;
             return {};
           });
    if (auto ok = pm.Run(block); !ok) return std::unexpected(ok.error());
    timings.insert(timings.end(), pm.timings().begin(), pm.timings().end());
    phase_start = PhaseClock::now();  // passes are timed by their manager
    RebindTransients(block, pinned, *binding);
    if (auto ok = BindMerge(); !ok) return std::unexpected(ok.error());
    seeml::diag::Note(
        generating::kDriver,
        std::string("rematerialization ") +
            (config_.remat == RematKind::kAuto
                 ? "(auto: the plain binding fails the configured gate — " +
                       over_budget + ")"
                 : std::string("(forced)")) +
            ": " + std::to_string(remat.recomputed_values) +
            " forward value(s) of " + std::to_string(remat.recomputed_ops) +
            " op(s) recomputed in the backward, " +
            std::to_string(remat.declined) +
            " declined (no net saving); arena " +
            std::to_string(plain_arena) + " -> " +
            std::to_string(AlignUp(binding->arena_size)) + " B");
    phase("rebind-arena", block.numOps());
  }
  return {};
}

Status Compilation::LowerTrain() {
  // --- 9. Lowering: train, eval (primal prefix), and merge programs. ---------
  const ResolveFn resolve_train = ResolveTrain();
  std::vector<sir::Operation*> train_ops;
  train_ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { train_ops.push_back(op); });
  auto train_instrs = LowerOps(train_ops, resolve_train, quant_scales,
                               bf16_weights, binding->quant_column_scales);
  if (!train_instrs) return std::unexpected(train_instrs.error());

  // --- Relaxed arithmetic (plan v18, F2 / F4; v20, F3): which instructions
  // carry kFlagRelaxed is the precision assignment's decision
  // (analysis/statistics/precision.h); the eval program lowered separately
  // below stays exact.
  if (config_.precision == Precision::kCertifiedBf16) {
    const RelaxedCounts relaxed = AssignRelaxedPrecision(*train_instrs);
    relaxed_gemms = relaxed.gemms;
    relaxed_attention = relaxed.attention;
    seeml::diag::Note(generating::kDriver,
                      "relaxed arithmetic: " + std::to_string(relaxed_gemms) +
                          " frozen-weight GEMM(s) and " +
                          std::to_string(relaxed_attention) +
                          " tiled attention instruction(s) carry "
                          "kFlagRelaxed; the package needs a numerics "
                          "certificate");
  }

  // Under gradient accumulation the lowered stream is two programs: the
  // grad program (forward, backward, the folds) and the step program (clip,
  // step, zero), split at the first step-program instruction — the
  // synthesizer appended every fold before the first of those. With no
  // accumulation the whole stream is the one-batch training program.
  std::vector<UpdateInstruction> step_instrs;
  if (accum > 1 && config_.emit_optimizer) {
    auto is_step = [](const UpdateInstruction& ins) {
      const auto op = static_cast<OpCode>(ins.opcode);
      // (kClipNorm precedes the step only with fuse_clip off.)
      return op == OpCode::kClipNorm || op == OpCode::kSgdStep ||
             op == OpCode::kAdamWStep;
    };
    auto split = std::find_if(train_instrs->begin(), train_instrs->end(),
                              is_step);
    if (split == train_instrs->end())
      return generating::Error(generating::kDriver,
                               "gradient accumulation requested but the "
                               "stream carries no optimizer step");
    step_instrs.assign(split, train_instrs->end());
    train_instrs->erase(split, train_instrs->end());
    for (const UpdateInstruction& ins : *train_instrs)
      if (is_step(ins))
        return generating::Error(generating::kDriver,
                                 "optimizer instruction inside the grad "
                                 "program");
  }
  programs.train = std::move(*train_instrs);
  programs.step = std::move(step_instrs);
  return {};
}

Status Compilation::LowerEvalAndMerge() {
  // --- The eval program scores what ships (E12, #95, plan v17). ------------
  // Training runs against the narrow copies (that is the memory the int8 /
  // bf16 base buys); the commit patches the f32 weights of the source file.
  // So the eval program — every gate and best-state score — reads those
  // f32 weights straight from the file (source refs, bound at run time),
  // and its GEMMs lower as plain f32 ones: W_f32 + (α/r)·A·B, the committed
  // function up to the rounding of one addition per element. Possible only
  // for a model that came from a file (the offsets and the hash that binds
  // them); an in-memory compile keeps the in-plan proxy, and says so.
  const ResolveFn resolve_train = ResolveTrain();
  const bool from_file = source.content_hash != 0;
  auto narrow_student = [&](const sir::Value* v) {
    auto src = build.weight_sources.find(v);
    return src != build.weight_sources.end() &&
           !std::string_view(v->id()).starts_with("t::") &&
           src->second->data_offset != 0;
  };
  if (from_file && !config_.eval_on_plan) {
    for (const auto& [w, s] : quant_scales)
      if (narrow_student(w)) shipped.insert(w);
    for (const sir::Value* w : bf16_weights)
      if (narrow_student(w)) shipped.insert(w);
  }
  if (!from_file && (!quant_scales.empty() || !bf16_weights.empty()))
    seeml::diag::Note(generating::kDriver,
                      "the model was not loaded from a file, so the eval "
                      "program scores the narrow in-plan weights — a proxy "
                      "for the committed f32 model");
  if (from_file && config_.eval_on_plan &&
      (!quant_scales.empty() || !bf16_weights.empty()))
    seeml::diag::Note(generating::kDriver,
                      "--validation-scores plan: the eval program scores the "
                      "narrow in-plan weights — a proxy for the committed f32 "
                      "model, for a device that cannot hold the source model "
                      "beside the arena; the report says validation_scores "
                      "\"plan\"");
  auto resolve_eval =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (shipped.contains(v))
      return MakeSourceRef(build.weight_sources.at(v)->data_offset);
    return resolve_train(v);
  };
  std::unordered_map<const sir::Value*, float> eval_quant;
  for (const auto& [w, s] : quant_scales)
    if (!shipped.contains(w)) eval_quant.emplace(w, s);
  std::unordered_set<const sir::Value*> eval_bf16;
  for (const sir::Value* w : bf16_weights)
    if (!shipped.contains(w)) eval_bf16.insert(w);
  auto eval_instrs = LowerOps(primal_ops, resolve_eval, eval_quant, eval_bf16,
                              binding->quant_column_scales);
  if (!eval_instrs) return std::unexpected(eval_instrs.error());

  auto resolve_merge =
      [&](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = merge_bound.find(v); it != merge_bound.end())
      return it->second;
    return generating::Error(
        generating::kDriver,
        "unbound merge value '" + std::string(v->id()) + "'");
  };
  std::vector<sir::Operation*> merge_ops;
  merge_ops.reserve(merge->block->numOps());
  merge->block->walk([&](sir::Operation* op) { merge_ops.push_back(op); });
  auto merge_instrs = LowerOps(merge_ops, resolve_merge, quant_scales,
                               bf16_weights, binding->quant_column_scales);
  if (!merge_instrs) return std::unexpected(merge_instrs.error());

  programs.eval = std::move(*eval_instrs);
  programs.merge = std::move(*merge_instrs);
  facts.ops_out = block.numOps();
  phase("lower", block.numOps());
  return {};
}

Status Compilation::FillHeader() {
  header.arena_size = AlignUp(binding->arena_size);
  header.persistent_size = binding->persistent_size;
  header.input_ref = binding->refs.at(build.input);
  // Token plans stage one i32 id per batch row; feature plans stage
  // batch x in_dim f32 elements. Both are 4-byte elements, so the slot's
  // byte size is input_floats * 4 either way.
  header.input_floats = token_input ? static_cast<uint64_t>(batch)
                                    : static_cast<uint64_t>(batch) *
                                          static_cast<uint64_t>(in_dim);
  header.input_kind = token_input ? 1 : 0;
  header.seq_len = source.seq_len;
  header.label_ref = labels ? binding->refs.at(labels) : kNullRef;
  header.label_bytes = label_bytes;
  header.label_kind = label_kind;
  header.optimizer_kind = static_cast<uint32_t>(config_.optimizer.kind);
  header.loss_ref = binding->refs.at(loss);
  header.lr = config_.optimizer.lr;
  header.beta1 = config_.optimizer.beta1;
  header.beta2 = config_.optimizer.beta2;
  header.eps = config_.optimizer.eps;
  header.weight_decay = config_.optimizer.weight_decay;
  header.gemm_tile_k = config_.gemm_tile_k;  // v11; 0 = runtime default
  header.gemm_tile_n = config_.gemm_tile_n;
  header.batch = static_cast<uint64_t>(batch);
  header.default_steps = config_.default_steps;
  // An optimizer-less plan (the finite-difference hook) has nothing to
  // step: it keeps the 1/G seed and the folds, but records no
  // accumulation — the runtime's contract (a step program exists exactly
  // when the plan accumulates) is about plans that train.
  recorded_accum = config_.emit_optimizer ? accum : 1;
  header.grad_accum_steps = recorded_accum;
  accumulator_bytes = 0;
  for (const ParamInit& pi : binding->params)
    if (pi.value->id().ends_with(".grad_acc")) accumulator_bytes += pi.bytes;
  // The schedule words: the floor and warmup are the cosine schedule's
  // (analysis/optimization/schedule.h).
  const ScheduleWords schedule = ScheduleWordsOf(config_.optimizer);
  header.lr_schedule = schedule.lr_schedule;
  header.warmup_steps = schedule.warmup_steps;
  header.min_lr_factor = schedule.min_lr_factor;
  header.clip_norm = config_.optimizer.clip_norm;
  header.source_model_hash = source.content_hash;
  // The data contract (v22, S2 #148): a model carrying a tokenizer section
  // binds its plan to that tokenizer; the corpus the device feeds must name
  // the same one (or none). The split and mask words stay zero until a
  // compile takes data (S3 #149).
  header.tokenizer_hash = source.TokenizerHash();
  if (masked_loss) header.data_flags |= kDataFlagMaskedLoss;
  if (corpus) {
    header.corpus_hash = corpus->shape.corpus_hash;
    header.data_flags |= kDataFlagCorpusPinned;
    if (corpus->facts.split) {
      header.split_seed = corpus->facts.split_seed;
      header.val_frac_bits = ValFracBits(config_.val_frac);
      header.val_split_hash = corpus->facts.val_split_hash;
      header.data_flags |= kDataFlagSplit;
    }
    // The device's own feeder rule over the header the plan will carry
    // and the corpus it is pinned to (its split identity being the one
    // the device recomputes): a plan can never be pinned to a corpus it
    // would refuse — whatever the stage above missed, this does not.
    CorpusShape as_fed = corpus->shape;
    as_fed.split_hash = corpus->facts.split ? corpus->facts.val_split_hash : 0;
    if (auto why = CorpusGeometryViolation(header, as_fed))
      return generating::Error(
          generating::kDriver,
          "the corpus the plan is pinned to would be refused by the device: " +
              *why);
  }
  return {};
}

Status Compilation::Assemble() {
  // --- 10 - 12. The persistent image, the emit table, the layout, the blob
  // and its seal are the backend's assembly stage (backend/assembly/).
  const std::vector<uint8_t> persist_init = SeedPersistentImage(*binding);
  auto emit_table = BuildEmitTable(*merge, adapters, build, merge_bound);
  if (!emit_table) return std::unexpected(emit_table.error());
  if (auto ok = FillHeader(); !ok) return std::unexpected(ok.error());
  const std::vector<uint8_t> metadata = BuildMetadata();
  header.data_flags |= kDataFlagMetadata;
  const uint64_t plan_bytes = LayOutPlan(header, programs, *binding,
                                         persist_init.size(),
                                         emit_table->size(), metadata.size());
  phase("persist-init", block.numOps());
  result.plan = AssemblePlan(header, programs, *binding, persist_init,
                             *emit_table, metadata, plan_bytes, consume);
  phase("assemble", block.numOps());
  SealPlan(result.plan);

  // The step-0 gate proved a lower bound before any work was done; every
  // byte is bound now, so prove the exact resident footprint — the arena
  // the runtime will allocate plus the plan blob it keeps loaded — also
  // fits the budget. Optimizer state, gradients, and transients, which the
  // early estimate deliberately excludes, are all inside the arena here.
  if (auto fits = CheckPlanFitsLocally(header.arena_size, result.plan.size(),
                                       config_.memory_budget_bytes);
      !fits)
    return std::unexpected(fits.error());

  phase("seal", block.numOps());
  return {};
}

Status Compilation::ReportDebug() {
  // --- 13. Debug hooks + report. ----------------------------------------------
  // The SIR dump is debug output, produced only when asked for (E5, #84):
  // streaming the whole graph through an ostringstream and retaining the
  // string was the largest pure-CPU graph cost in the driver, paid by
  // every compile for a string almost none of them read.
  if (config_.dump_sir) {
    std::ostringstream dump;
    block.print(dump);
    dump << "  --- merge program ---\n";
    merge->block->print(dump);
    result.sir_dump = dump.str();
    phase("sir-dump", block.numOps());
  }
  result.pass_timings = std::move(timings);

  for (const auto& [p, g] : param_grads) {
    ParamDebugInfo info{.id = std::string(p->id()),
                        .param_ref = binding->refs.at(p),
                        .grad_ref = binding->refs.at(g),
                        .count = static_cast<uint64_t>(p->shape().volume())};
    // The accumulator the step program consumes under accumulation.
    for (const ParamInit& pi : binding->params)
      if (pi.value->id() == std::string(p->id()) + ".grad_acc")
        info.acc_ref = binding->refs.at(pi.value);
    result.params.push_back(info);
  }
  std::sort(result.params.begin(), result.params.end(),
            [](const ParamDebugInfo& a, const ParamDebugInfo& b) {
              return a.id < b.id;
            });

  for (size_t i = 0; i < adapters.size(); ++i) {
    const GraftedAdapter& a = adapters[i];
    const auto q = quant_scales.find(a.frozen_weight);
    result.adapters.push_back(
        {.weight_name = std::string(a.frozen_weight->id()),
         .weight_rodata_ref = binding->refs.at(a.frozen_weight),
         .a_ref = binding->refs.at(a.A),
         .b_ref = binding->refs.at(a.B),
         .delta_ref = merge_bound.at(merge->outputs[i].first),
         .k = a.frozen_weight->shape().dims.at(0),
         .m = a.frozen_weight->shape().dims.at(1),
         .r = a.A->shape().dims.at(1),
         .scale = a.scale,
         .quant_scale = q != quant_scales.end() ? q->second : 0.0f,
         .bf16 = bf16_weights.contains(a.frozen_weight)});
  }
  return {};
}

Status Compilation::ReportSummary() {
  result.arena_size = header.arena_size;
  result.persistent_size = header.persistent_size;
  result.io_size = binding->io_end - binding->persistent_size;
  result.transient_size = header.arena_size - binding->io_end;
  // The step-0 bound the early gate proved feasibility from, beside the
  // exact footprint the final gate just checked: estimate <= arena + plan
  // is the bound's contract, and their ratio is the accountant's honesty
  // at this shape (F5 #133).
  result.estimate_weight_bytes = footprint.weight_bytes;
  result.estimate_activation_bytes = footprint.activation_bytes;
  result.estimate_delta_bytes = footprint.delta_bytes;
  result.estimate_cache_bytes = footprint.probs_cache_bytes;
  result.estimate_bytes = footprint.total_bytes();
  seeml::diag::Note(generating::kDriver,
                    "memory: the step-0 bound is " +
                        std::to_string(result.estimate_bytes) +
                        " B against the exact arena + plan of " +
                        std::to_string(header.arena_size + result.plan.size()) +
                        " B");
  result.remat_policy = config_.remat;
  result.remat_applied = remat_on;
  result.remat_recomputed_values = remat.recomputed_values;
  result.remat_bytes_saved =
      plain_arena > header.arena_size ? plain_arena - header.arena_size : 0;
  result.train_instruction_count = header.train_instr_count;
  result.step_instruction_count = programs.step.size();
  result.grad_accum_steps = recorded_accum;
  result.merge_instruction_count = header.merge_instr_count;
  result.eval_instruction_count = header.eval_instr_count;
  result.rodata_size = header.rodata_size;
  // The eval program scores the f32 file either way: the narrow students'
  // source refs (E12), or every weight's (an adapter-only plan).
  result.scores_shipped = !shipped.empty() || config_.adapter_only;
  result.adapter_only = config_.adapter_only;
  if (corpus) {
    result.corpus_sds = std::move(corpus->sds);
    result.corpus_manifest = std::move(corpus->manifest_json);
    result.corpus_from_sds = corpus->from_sds;
  }
  if (config_.adapter_only)
    seeml::diag::Note(generating::kDriver,
                      "adapter-only: the plan carries no rodata — " +
                          std::to_string(build.weight_sources.size()) +
                          " frozen weight(s) are read from the source model "
                          "at run time, in every program");
  result.relaxed_gemms = relaxed_gemms;
  result.relaxed_attention = relaxed_attention;
  // What the plan carries, from the pass that decided on exact shapes.
  result.attention_tiled =
      attention_decision.tiled && attention_decision.attention_ops > 0;
  result.probs_cache_bytes = attention_decision.probs_cache_bytes;
  result.gemm_tile_k = header.gemm_tile_k;
  result.gemm_tile_n = header.gemm_tile_n;

  seeml::diag::Note(generating::kDriver, "plan compiled — " +
               std::to_string(header.train_instr_count) + " train + " +
               std::to_string(header.eval_instr_count) + " eval + " +
               std::to_string(header.merge_instr_count) + " merge" +
               (header.step_instr_count
                    ? " + " + std::to_string(header.step_instr_count) + " step"
                    : std::string()) +
               " instrs, arena " + std::to_string(header.arena_size) + " B (" +
               std::to_string(header.persistent_size) + " B persistent" +
               (recorded_accum > 1
                    ? ", " + std::to_string(accumulator_bytes) +
                          " B of it gradient accumulators for an effective "
                          "batch of " +
                          std::to_string(static_cast<uint64_t>(batch) *
                                         recorded_accum)
                    : std::string()) +
               "), rodata " + std::to_string(header.rodata_size) + " B");

  // The five largest values the arena holds, with their segment (S6 #152).
  {
    std::vector<TensorFact> tensors;
    for (const auto& [value, ref] : binding->refs) {
      if (IsRodataRef(ref) || IsSourceRef(ref)) continue;
      const uint64_t at = RefOffset(ref);
      tensors.push_back(
          {std::string(value->id()),
           static_cast<uint64_t>(value->shape().volume()) * 4,
           at < binding->persistent_size ? "persistent"
           : at < binding->io_end        ? "io"
                                         : "transient"});
    }
    std::sort(tensors.begin(), tensors.end(),
              [](const TensorFact& a, const TensorFact& b) {
                return a.bytes != b.bytes ? a.bytes > b.bytes : a.name < b.name;
              });
    if (tensors.size() > 5) tensors.resize(5);
    facts.largest_tensors = std::move(tensors);
  }
  // The screens (S0 #146): what the compile did, as numbers.
  facts.attention_ops = attention_decision.attention_ops;
  FillCompileScreens(facts, config_, result);

  // Backend boundary: the assembled plan must be internally consistent with
  // the adapters and gradients it claims to train.
  if (auto ok = VerifyGeneratedPlan(result, adapters.size()); !ok)
    return std::unexpected(ok.error());
  return {};
}

Status Compilation::BindMerge() {
  merge_bound.clear();
  for (const auto& [mirror, original] : merge->aliases) {
    auto it = binding->refs.find(original);
    if (it == binding->refs.end())
      return generating::Error(generating::kDriver,
                               "merge alias to unbound value");
    merge_bound[mirror] = it->second;
  }
  const uint64_t merge_high = LinearScanTransients(
      *merge->block, AlignUp(binding->arena_size), merge_bound,
      merge_pinned, merge_bound);
  binding->arena_size = std::max(binding->arena_size, merge_high);
  return {};
}

// One ledger line per pass that ran, in order: every analysis pass computes
// the bits of its input (each is proven so by the fused-vs-unfused, tiled-
// vs-cached and remat-vs-plain twin tests), and the precision assignment is
// the one decision that lets a backend reorder a reduction — under the
// numerics certificate the package must then carry.
std::vector<uint8_t> Compilation::BuildMetadata() {
  struct Reason {
    std::string_view pass, why;
  };
  static constexpr Reason kExactPasses[] = {
      {"attention-tiling", "same expressions; probabilities recomputed"},
      {"lora-graft", "adds the adapter path; B is zero at step 0"},
      {"rope-table", "angles hoisted; the kernels' own recurrence"},
      {"autodiff", "adds the backward; the forward is untouched"},
      {"optimizer", "adds the update; the forward is untouched"},
      {"fuse-gemm-epilogue", "bias and activation inside the GEMM's loop"},
      {"fuse-gemm-addend", "d + s over the complete dot product"},
      {"fuse-elementwise", "stages run per block, in program order"},
      {"dce", "removes ops nothing reads"},
      {"rematerialize", "same opcode over the same operands"},
  };
  std::vector<LedgerEntry> ledger;
  for (const PassTiming& t : timings)
    for (const Reason& r : kExactPasses)
      if (t.name == r.pass)
        ledger.push_back(MakeLedgerEntry(r.pass, kLedgerExact, 0, r.why));
  if (relaxed_gemms + relaxed_attention != 0)
    ledger.push_back(MakeLedgerEntry(
        "precision", kLedgerReordered,
        static_cast<uint32_t>(relaxed_gemms + relaxed_attention),
        "relaxed kernels; needs the numerics certificate"));
  // The RESOLVED recipe is what is hashed (S4 #150): the request with every
  // decision the compile took written in — which attention family the plan
  // runs, whether it rematerializes, whether the loss is masked, which
  // weights the eval program reads — and the knobs those decisions consumed
  // (the two budgets) dropped. Two requests that resolve to the same plan
  // share a hash; a plan's bytes depend on what it is, not on how it was
  // asked for.
  UpdateConfig resolved = config_;
  resolved.attention =
      attention_decision.tiled && attention_decision.attention_ops > 0
          ? AttentionKind::kTiled
          : AttentionKind::kCached;
  resolved.attention_cache_budget_bytes =
      UpdateConfig{}.attention_cache_budget_bytes;
  resolved.remat = remat_on ? RematKind::kOn : RematKind::kOff;
  resolved.memory_budget_bytes = 0;
  resolved.masked_loss = masked_loss;
  resolved.eval_on_plan = shipped.empty() && !config_.adapter_only;
  PlanMetadata meta;
  meta.recipe_hash = recipe::RecipeHash(resolved);
  const ExportReport& e = config_.export_report;
  if (e.present) {
    meta.export_flags = kExportReportPresent |
                        (e.tokenizer_parity ? kExportTokenizerParity : 0) |
                        (e.logits_parity ? kExportLogitsParity : 0);
    meta.export_report_hash = e.report_hash;
    meta.tokenizer_parity_ids = e.tokenizer_parity ? e.tokenizer_parity_ids : 0;
    meta.logits_parity_max_delta =
        e.logits_parity ? e.logits_parity_max_delta : 0.0f;
  }
  result.recipe_hash = meta.recipe_hash;
  result.ledger = ledger;
  return EncodePlanMetadata(meta, ledger);
}

ResolveFn Compilation::ResolveTrain() {
  return [this](const sir::Value* v) -> std::expected<uint64_t, std::string> {
    if (auto it = binding->refs.find(v); it != binding->refs.end())
      return it->second;
    return generating::Error(generating::kDriver,
                             "unbound value '" + std::string(v->id()) + "'");
  };
}

std::expected<CompiledUpdate, std::string> Compilation::Run() {
  using Stage = Status (Compilation::*)();
  constexpr Stage kStages[] = {
      &Compilation::ValidateConfig,
      &Compilation::Ingress,
      &Compilation::VerifyTopology,
      &Compilation::IngestData,
      &Compilation::DecideMaskedLoss,
      &Compilation::RecordFrontendFacts,
      &Compilation::ProveFootprint,
      &Compilation::BuildForwardGraph,
      &Compilation::GraftLoss,
      &Compilation::StructuralPasses,
      &Compilation::CalculusPasses,
      &Compilation::MergeAndReview,
      &Compilation::FusionPasses,
      &Compilation::BindArenas,
      &Compilation::Rematerialize,
      &Compilation::LowerTrain,
      &Compilation::LowerEvalAndMerge,
      &Compilation::Assemble,
      &Compilation::ReportDebug,
      &Compilation::ReportSummary,
  };
  for (const Stage stage : kStages)
    if (auto ok = (this->*stage)(); !ok) return std::unexpected(ok.error());
  return std::move(result);
}

}  // namespace

AttentionKind DecideAttention(const TrainingFootprint& footprint,
                              const UpdateConfig& config) {
  if (config.attention != AttentionKind::kAuto) return config.attention;
  const bool over_budget =
      footprint.probs_cache_bytes > config.attention_cache_budget_bytes;
  // Against a CONFIGURED memory budget only (see the header): with none,
  // the cache budget decides by itself, and the gate that follows keeps
  // its own, host-sized refusal. CheckTrainableLocally falls back to the
  // host's memory for a zero budget, which is exactly what a decision
  // that changes the plan's bytes may not consult.
  const bool cached_fits =
      config.memory_budget_bytes == 0 ||
      CheckTrainableLocally(footprint, config.memory_budget_bytes).has_value();
  return (over_budget || !cached_fits) ? AttentionKind::kTiled
                                       : AttentionKind::kCached;
}

std::expected<CompiledUpdate, std::string> UpdateCompiler::Compile(
    const SmfModel& source, const SmfModel* teacher) {
  return Attributed(CompileImpl(source, teacher, /*consume=*/false));
}

std::expected<CompiledUpdate, std::string> UpdateCompiler::Compile(
    SmfModel&& source, SmfModel* teacher) {
  // `source` binds to a caller's non-const model (as `teacher` points to
  // one), which is what makes releasing their payloads well-defined.
  return Attributed(CompileImpl(source, teacher, /*consume=*/true));
}

std::expected<CompiledUpdate, std::string> UpdateCompiler::CompileImpl(
    const SmfModel& source, const SmfModel* teacher, bool consume) {
  return Compilation(config_, source, teacher, consume).Run();
}

}  // namespace seeml::update
