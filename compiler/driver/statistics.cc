#include "compiler/driver/statistics.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

#include <span>

#include "compiler/backend/architecture/host_arch.h"
#include "source/plan/metadata.h"
#include "source/statistics/plan_work.h"

namespace seeml::update {

namespace {

bool IsFrontendPhase(std::string_view name) {
  return name == "ingress" || name == "topology" || name == "data" ||
         name == "frontend";
}

bool IsBackendPhase(std::string_view name) {
  return name == "bind-arena" || name == "rebind-arena" || name == "lower" ||
         name == "persist-init" || name == "assemble" || name == "seal" ||
         name == "sir-dump";
}

std::string Hex16(uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016" PRIx64, v);
  return buf;
}

}  // namespace

namespace {

/// What more than one screen reads, computed once from the plan.
struct Totals {
  double frontend_ms = 0.0, analysis_ms = 0.0, backend_ms = 0.0;
  uint64_t device_bytes = 0;  // arena + plan
  PlanHeader header{};
  std::span<const UpdateInstruction> train, eval, step;
  PlanWork work;
  uint64_t flops_forward = 0, flops_backward = 0, flops_optimizer = 0;
  bool peak = false;          // a finite positive --peak-gflops
  double predicted_ms = 0.0;  // ms = FLOP / (GFLOP/s * 1e9) * 1e3
};

/// One program of `plan`; empty when the header's section does not lie
/// inside the blob — the screens are filled before the backend boundary
/// check has proven the header (MeasurePlanWork's rule).
std::span<const UpdateInstruction> Program(const std::vector<uint8_t>& plan,
                                           uint64_t offset, uint64_t count) {
  uint64_t bytes = 0;
  if (!MulOk(count, sizeof(UpdateInstruction), &bytes) ||
      !RangeOk(offset, bytes, plan.size()))
    return {};
  return {reinterpret_cast<const UpdateInstruction*>(plan.data() + offset),
          static_cast<size_t>(count)};
}

Totals Measure(const UpdateConfig& config, const CompiledUpdate& r) {
  Totals t;
  for (const PassTiming& p : r.pass_timings) {
    if (IsFrontendPhase(p.name)) t.frontend_ms += p.ms;
    else if (IsBackendPhase(p.name)) t.backend_ms += p.ms;
    else t.analysis_ms += p.ms;
  }
  t.device_bytes = r.arena_size + r.plan.size();
  if (r.plan.size() >= sizeof(t.header)) {
    std::memcpy(&t.header, r.plan.data(), sizeof(t.header));
    t.train = Program(r.plan, t.header.train_instr_offset,
                      t.header.train_instr_count);
    t.eval = Program(r.plan, t.header.eval_instr_offset,
                     t.header.eval_instr_count);
    t.step = Program(r.plan, t.header.step_instr_offset,
                     t.header.step_instr_count);
  }
  t.work = MeasurePlanWork(r.plan);
  // Forward, backward, optimizer (S5 #151), with the unit seeml-plan-probe
  // reports per section: the eval program is the forward and the loss, the
  // train program is that plus the backward (and, without accumulation, the
  // update), the step program is the update under accumulation.
  auto flops = [](std::span<const UpdateInstruction> p) {
    return SatAdd(GemmFlops(p), AttentionFlops(p));
  };
  t.flops_forward = flops(t.eval);
  const uint64_t train =
      SatAdd(t.work.train_gemm_flops, t.work.train_attention_flops);
  t.flops_backward = train > t.flops_forward ? train - t.flops_forward : 0;
  t.flops_optimizer = flops(t.step);
  t.peak = std::isfinite(config.peak_gflops) && config.peak_gflops > 0.0;
  t.predicted_ms = t.peak ? static_cast<double>(t.work.FlopsPerStep()) /
                                (config.peak_gflops * 1e6)
                          : 0.0;
  return t;
}

void FillFrontend(const CompileFacts& f, const UpdateConfig& config,
                  const CompiledUpdate& r, const Totals& t, Screen& fe) {
  const IngressorFacts& ing = f.frontend.ingressor;
  const TopologyFacts& top = f.frontend.topology;
  const AccountantFacts& acc = f.frontend.accountant;
  fe.Set("model_ops", ing.model_ops);
  fe.Set("model_tensors", ing.model_tensors);
  fe.Set("weight_tensors", ing.weight_tensors);
  fe.Set("weight_bytes", ing.weight_bytes);
  fe.Set("parameters", ing.weight_bytes / 4);
  fe.Set("input", ing.token_input
                      ? "token ids, " + std::to_string(ing.seq_len) +
                            " per sequence"
                      : std::to_string(ing.input_width) + " features per row");
  if (top.vocabulary) fe.Set("vocabulary", top.vocabulary);
  if (top.has_teacher) fe.Set("teacher_ops", top.teacher_ops);
  fe.Set("adapted_tensors", top.adapted_tensors);
  fe.Set("adapted_bytes", top.adapted_bytes);
  fe.Set("activations", top.activations);
  fe.Set("activation_bytes", top.activation_bytes);
  fe.Set("weights_source", acc.weights_mapped
                               ? "mapped from the model file, not counted"
                               : "in the plan");
  if (f.frontend.corpus) {
    const CorpusFacts& c = *f.frontend.corpus;
    const bool text = c.kind != "sds";
    fe.Set("corpus", std::to_string(c.records) + " records of " +
                         std::to_string(c.record_width) +
                         (c.tokens ? " ids" : " features") + " (" + c.kind +
                         (c.masked ? ", masked)" : ")"));
    if (text) {
      fe.Set("corpus_documents", std::to_string(c.documents_in) + " in, " +
                                     std::to_string(c.documents_kept) +
                                     " kept");
      fe.Set("corpus_dropped", std::to_string(c.dropped_empty) + " empty, " +
                                   std::to_string(c.dropped_duplicate) +
                                   " duplicate, " +
                                   std::to_string(c.dropped_degenerate) +
                                   " degenerate");
      fe.Set("corpus_sanitized",
             std::to_string(c.invalid_utf8_bytes) + " invalid UTF-8 byte(s), " +
                 std::to_string(c.control_chars) + " control char(s), " +
                 std::to_string(c.bidi_controls) + " bidi control(s)" +
                 (c.kind == "txt" ? " (counted, kept: one document)" : ""));
      fe.Set("corpus_remainder", c.remainder_tokens);
      fe.Set("corpus_straddles", c.straddling_windows);
    }
    if (c.tokens) {
      fe.Set("corpus_tokens", c.tokens);
      fe.Set("corpus_distinct_ids", std::to_string(c.distinct_ids) + " of " +
                                        std::to_string(c.vocabulary));
      fe.Set("corpus_targets", std::to_string(c.target_tokens) + " of " +
                                   std::to_string(c.total_tokens));
    }
    fe.Set("batches_per_epoch", c.batches_per_epoch);
    fe.Set("epochs_covered", c.epochs_covered);
    if (c.label_entropy_nats) fe.Set("label_entropy", *c.label_entropy_nats);
    fe.Set("corpus_hash", Hex16(c.corpus_hash));
    if (c.split)
      fe.Set("validation", std::to_string(c.val_records) + " of " +
                               std::to_string(c.records) +
                               " records held out, seed " +
                               std::to_string(c.split_seed) + ", split " +
                               Hex16(c.val_split_hash));
  }
  // The identity manifest (S4 #150): every input of the fine-tune, hashed —
  // the accountant prints what the plan's header and metadata carry.
  fe.Set("model_hash", ing.content_hash ? Hex16(ing.content_hash)
                                        : std::string("none: built in memory"));
  if (t.header.tokenizer_hash)
    fe.Set("tokenizer_hash", Hex16(t.header.tokenizer_hash));
  fe.Set("recipe_hash", Hex16(r.recipe_hash));
  if (config.export_report.present) {
    const ExportReport& e = config.export_report;
    char delta[32];
    std::snprintf(delta, sizeof(delta), "%.3g",
                  static_cast<double>(e.logits_parity_max_delta));
    fe.Set("export_report",
           Hex16(e.report_hash) + "; tokenizer parity " +
               (e.tokenizer_parity
                    ? std::to_string(e.tokenizer_parity_ids) + " ids equal"
                    : std::string("not run")) +
               "; logits parity " +
               (e.logits_parity ? std::string("max delta ") + delta
                                : std::string("not run")));
  }
  fe.Set("footprint_weights", acc.footprint_weights);
  fe.Set("footprint_activations", acc.footprint_activations);
  fe.Set("footprint_deltas", acc.footprint_deltas);
  fe.Set("footprint_caches", acc.footprint_caches);
  fe.Set("footprint_total", acc.footprint_total);
  fe.Set("bound_ratio", t.device_bytes ? static_cast<double>(r.estimate_bytes) /
                                           static_cast<double>(t.device_bytes)
                                     : 0.0);
  fe.Set("memory_budget", acc.memory_budget);
  fe.Set("frontend_ms", t.frontend_ms);
  for (const WeightRange& w : ing.ranges)
    fe.AddRow("tensors", {w.name, w.bytes, static_cast<double>(w.min),
                          static_cast<double>(w.max)});

}

void FillAnalysis(const CompileFacts& f, const UpdateConfig& config,
                  const CompiledUpdate& r, const Totals& t, Screen& an) {
  const IngressorFacts& ing = f.frontend.ingressor;
  an.Set("passes", r.pass_timings.size());
  an.Set("ops_in", f.ops_in);
  an.Set("ops_out", f.ops_out);
  an.Set("adapters", r.adapters.size());
  an.Set("lora_rank", static_cast<uint64_t>(config.lora.rank));
  an.Set("lora_alpha", static_cast<double>(config.lora.alpha));
  {
    char scale[64];
    std::snprintf(scale, sizeof(scale), "%.6g = alpha / %s",
                  static_cast<double>(config.lora.Scale()),
                  config.lora.scaling == LoraScaling::kAlphaOverSqrtRank
                      ? "sqrt(r)"
                      : "r");
    an.Set("lora_scale", std::string(scale));
  }
  uint64_t trainable = 0;
  for (const AdapterDebugInfo& a : r.adapters)
    trainable += static_cast<uint64_t>(a.r) *
                 static_cast<uint64_t>(a.k + a.m);
  an.Set("trainable_parameters", trainable);
  const uint64_t parameters = ing.weight_bytes / 4;
  an.Set("trainable_share",
         parameters ? 100.0 * static_cast<double>(trainable) /
                          static_cast<double>(parameters)
                    : 0.0);
  an.Set("gradients", f.gradients);
  an.Set("rope_tables", f.rope_tables);
  an.Set("epilogue_fusions", f.epilogue_fusions);
  an.Set("addend_fusions", f.addend_fusions);
  an.Set("chain_fusions", f.chain_fusions);
  an.Set("dce_removed", f.dce_removed);
  an.Set("attention",
         f.attention_ops == 0
             ? std::string("none")
             : std::string(r.attention_tiled ? "tiled" : "cached") + ", " +
                   std::to_string(f.attention_ops) + " op(s)");
  std::string remat;
  if (r.remat_applied)
    remat = "on, " + std::to_string(r.remat_recomputed_values) +
            " value(s) recomputed, " + std::to_string(r.remat_bytes_saved) +
            " B saved";
  else if (config.remat == RematKind::kOff)
    remat = "off";
  else if (config.remat == RematKind::kAuto && !f.remat_budget_configured)
    remat = "off: auto, no memory budget";  // --remat on / --memory-budget-mib N
  else
    remat = "off: the plain plan fits the budget";
  an.Set("rematerialization", remat);
  an.Set("narrow_weights",
         f.int8_weights   ? std::to_string(f.int8_weights) + " int8"
         : f.bf16_weights ? std::to_string(f.bf16_weights) + " bf16"
         : r.adapter_only ? std::string("none: read from the model file")
                          : std::string("none: f32"));
  // What the calculus added and what the fusions removed (S5 #151).
  an.Set("adjoint_ops", f.adjoint_ops);
  an.Set("saved_activation_bytes", f.saved_activation_bytes);
  an.Set("flops_forward", t.flops_forward);
  an.Set("flops_backward", t.flops_backward);
  an.Set("flops_optimizer", t.flops_optimizer);
  an.Set("epilogue_bytes_removed", f.epilogue_bytes_removed);
  an.Set("addend_bytes_removed", f.addend_bytes_removed);
  an.Set("chain_bytes_removed", f.chain_bytes_removed);
  uint64_t reordered = 0;
  for (const LedgerEntry& e : r.ledger) reordered += e.instructions;
  an.Set("reordered_instructions", reordered);
  an.Set("analysis_ms", t.analysis_ms);
  for (const PassTiming& p : r.pass_timings)
    an.AddRow("passes", {p.name, static_cast<uint64_t>(p.ops_after), p.ms});
  for (const AdapterDebugInfo& a : r.adapters)
    an.AddRow("adapter_sites",
              {a.weight_name, static_cast<uint64_t>(a.k),
               static_cast<uint64_t>(a.m), static_cast<uint64_t>(a.r),
               std::string(a.quant_scale != 0.0f ? "int8"
                           : a.bf16              ? "bf16"
                           : r.adapter_only      ? "file"
                                                 : "f32")});
  // The reduction-order ledger, as the plan's metadata carries it.
  for (const LedgerEntry& e : r.ledger)
    an.AddRow("ledger",
              {std::string(e.pass),
               std::string(e.arithmetic == kLedgerExact ? "bit-identical"
                                                         : "reordered"),
               static_cast<uint64_t>(e.instructions), std::string(e.why)});
}

void FillBackend(const CompileFacts& f, const UpdateConfig& config,
                 const CompiledUpdate& r, const Totals& t, Screen& be) {
  be.Set("train_instructions", r.train_instruction_count);
  be.Set("eval_instructions", r.eval_instruction_count);
  be.Set("merge_instructions", r.merge_instruction_count);
  if (r.step_instruction_count)
    be.Set("step_instructions", r.step_instruction_count);
  be.Set("arena_bytes", r.arena_size);
  be.Set("persistent_bytes", r.persistent_size);
  be.Set("io_bytes", r.io_size);
  be.Set("transient_bytes", r.transient_size);
  be.Set("rodata_bytes", r.rodata_size);
  be.Set("plan_bytes", r.plan.size());
  be.Set("device_bytes", t.device_bytes);
  be.Set("gemm_flops", t.work.GemmFlopsPerStep());
  be.Set("attention_flops", t.work.AttentionFlopsPerStep());
  if (t.peak) {
    be.Set("peak_gflops", config.peak_gflops);
    be.Set("predicted_step_ms", t.predicted_ms);
  }
  // The GPU half of the target (S6 #152): the same work over the GPU's
  // stated peak, so a Metal run's measured step has a prediction beside it.
  if (std::isfinite(config.gpu_peak_gflops) && config.gpu_peak_gflops > 0.0) {
    be.Set("gpu_peak_gflops", config.gpu_peak_gflops);
    be.Set("predicted_step_gpu_ms",
           static_cast<double>(t.work.FlopsPerStep()) /
               (config.gpu_peak_gflops * 1e6));
  }
  {
    const HostArchInfo host = DetectHostArch();
    be.Set("target", HostKey(host) + "; GPU " +
                         (host.gpu_family.empty() ? std::string("none")
                                                  : host.gpu_family));
  }
  be.Set("kernel_policy",
         r.gemm_tile_k == 0 && r.gemm_tile_n == 0
             ? std::string("runtime default")
             : "K " + std::to_string(r.gemm_tile_k) + ", N " +
                   std::to_string(r.gemm_tile_n));
  be.Set("plan_hash", Hex16(t.header.plan_hash));
  be.Set("backend_ms", t.backend_ms);
  for (const TensorFact& tensor : f.largest_tensors)
    be.AddRow("largest_tensors", {tensor.name, tensor.bytes, tensor.segment});
  // The GEMM census: the train program's products by shape.
  const uint64_t train_gemm = t.work.train_gemm_flops;
  for (const GemmShape& g : GemmCensus(t.train))
    be.AddRow("gemm_census",
              {std::to_string(g.m) + " x " + std::to_string(g.n) + " x " +
                   std::to_string(g.k),
               g.count, g.flops,
               train_gemm ? 100.0 * static_cast<double>(g.flops) /
                                static_cast<double>(train_gemm)
                          : 0.0});
}

void FillRuntime(const Totals& t, Screen& rt) {
  rt.Set("device_bytes", t.device_bytes);
  rt.Set("flops_per_step", t.work.FlopsPerStep());
  if (t.peak) rt.Set("predicted_step_ms", t.predicted_ms);
}

}  // namespace

void FillCompileScreens(const CompileFacts& f, const UpdateConfig& config,
                        CompiledUpdate& r) {
  const Totals t = Measure(config, r);
  FillFrontend(f, config, r, t, r.screens.frontend);
  FillAnalysis(f, config, r, t, r.screens.analysis);
  FillBackend(f, config, r, t, r.screens.backend);
  FillRuntime(t, r.screens.runtime);
}

std::string RenderCompileScreens(const CompileScreens& screens, bool verbose) {
  return screens.frontend.Text(verbose) + screens.analysis.Text(verbose) +
         screens.backend.Text(verbose) + screens.runtime.Text(verbose);
}

}  // namespace seeml::update
