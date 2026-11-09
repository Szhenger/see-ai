#include "compiler/driver/statistics.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

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

void FillCompileScreens(const CompileFacts& f, const UpdateConfig& config,
                        CompiledUpdate& r) {
  CompileScreens& s = r.screens;
  double frontend_ms = 0.0, analysis_ms = 0.0, backend_ms = 0.0;
  for (const PassTiming& t : r.pass_timings) {
    if (IsFrontendPhase(t.name)) frontend_ms += t.ms;
    else if (IsBackendPhase(t.name)) backend_ms += t.ms;
    else analysis_ms += t.ms;
  }

  // --- frontend ---------------------------------------------------------------
  const IngressorFacts& ing = f.frontend.ingressor;
  const TopologyFacts& top = f.frontend.topology;
  const AccountantFacts& acc = f.frontend.accountant;
  Screen& fe = s.frontend;
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
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(c.corpus_hash));
    fe.Set("corpus_hash", std::string(hex));
    if (c.split) {
      char split_hex[32];
      std::snprintf(split_hex, sizeof(split_hex), "%016llx",
                    static_cast<unsigned long long>(c.val_split_hash));
      fe.Set("validation", std::to_string(c.val_records) + " of " +
                               std::to_string(c.records) +
                               " records held out, seed " +
                               std::to_string(c.split_seed) + ", split " +
                               split_hex);
    }
  }
  fe.Set("footprint_weights", acc.footprint_weights);
  fe.Set("footprint_activations", acc.footprint_activations);
  fe.Set("footprint_deltas", acc.footprint_deltas);
  fe.Set("footprint_caches", acc.footprint_caches);
  fe.Set("footprint_total", acc.footprint_total);
  const uint64_t device_bytes = r.arena_size + r.plan.size();
  fe.Set("bound_ratio", device_bytes ? static_cast<double>(r.estimate_bytes) /
                                           static_cast<double>(device_bytes)
                                     : 0.0);
  fe.Set("memory_budget", acc.memory_budget);
  fe.Set("frontend_ms", frontend_ms);
  for (const WeightRange& w : ing.ranges)
    fe.AddRow("tensors", {w.name, w.bytes, static_cast<double>(w.min),
                          static_cast<double>(w.max)});

  // --- analysis ---------------------------------------------------------------
  Screen& an = s.analysis;
  an.Set("passes", r.pass_timings.size());
  an.Set("ops_in", f.ops_in);
  an.Set("ops_out", f.ops_out);
  an.Set("adapters", r.adapters.size());
  an.Set("lora_rank", static_cast<uint64_t>(config.lora.rank));
  an.Set("lora_alpha", static_cast<double>(config.lora.alpha));
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
  an.Set("analysis_ms", analysis_ms);
  for (const PassTiming& t : r.pass_timings)
    an.AddRow("passes", {t.name, static_cast<uint64_t>(t.ops_after), t.ms});
  for (const AdapterDebugInfo& a : r.adapters)
    an.AddRow("adapter_sites",
              {a.weight_name, static_cast<uint64_t>(a.k),
               static_cast<uint64_t>(a.m), static_cast<uint64_t>(a.r),
               std::string(a.quant_scale != 0.0f ? "int8"
                           : a.bf16              ? "bf16"
                           : r.adapter_only      ? "file"
                                                 : "f32")});

  // --- backend ----------------------------------------------------------------
  Screen& be = s.backend;
  be.Set("train_instructions", r.train_instruction_count);
  be.Set("eval_instructions", r.eval_instruction_count);
  be.Set("merge_instructions", r.merge_instruction_count);
  if (r.step_instruction_count)
    be.Set("step_instructions", r.step_instruction_count);
  be.Set("arena_bytes", r.arena_size);
  be.Set("persistent_bytes", r.persistent_size);
  be.Set("transient_bytes", r.transient_size);
  be.Set("rodata_bytes", r.rodata_size);
  be.Set("plan_bytes", r.plan.size());
  be.Set("device_bytes", device_bytes);
  const PlanWork work = MeasurePlanWork(r.plan);
  be.Set("gemm_flops", work.GemmFlopsPerStep());
  be.Set("attention_flops", work.AttentionFlopsPerStep());
  // A finite positive peak only: an infinite one would predict a 0 ms step.
  const bool peak =
      std::isfinite(config.peak_gflops) && config.peak_gflops > 0.0;
  // ms = FLOP ÷ (GFLOP/s · 1e9) · 1e3.
  const double predicted_ms =
      peak ? static_cast<double>(work.FlopsPerStep()) /
                 (config.peak_gflops * 1e6)
           : 0.0;
  if (peak) {
    be.Set("peak_gflops", config.peak_gflops);
    be.Set("predicted_step_ms", predicted_ms);
  }
  be.Set("kernel_policy",
         r.gemm_tile_k == 0 && r.gemm_tile_n == 0
             ? std::string("runtime default")
             : "K " + std::to_string(r.gemm_tile_k) + ", N " +
                   std::to_string(r.gemm_tile_n));
  PlanHeader header{};
  if (r.plan.size() >= sizeof(header))
    std::memcpy(&header, r.plan.data(), sizeof(header));
  be.Set("plan_hash", Hex16(header.plan_hash));
  be.Set("backend_ms", backend_ms);

  // --- runtime, as planned -------------------------------------------------------
  Screen& rt = s.runtime;
  rt.Set("device_bytes", device_bytes);
  rt.Set("flops_per_step", work.FlopsPerStep());
  if (peak) rt.Set("predicted_step_ms", predicted_ms);
}

std::string RenderCompileScreens(const CompileScreens& screens, bool verbose) {
  return screens.frontend.Text(verbose) + screens.analysis.Text(verbose) +
         screens.backend.Text(verbose) + screens.runtime.Text(verbose);
}

}  // namespace seeml::update
