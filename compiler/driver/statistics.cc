#include "compiler/driver/statistics.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "source/statistics/plan_work.h"

namespace seeml::update {

namespace {

bool IsFrontendPhase(std::string_view name) {
  return name == "finite-sweep" || name == "frontend";
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
  Screen& fe = s.frontend;
  fe.Set("model_ops", f.model_ops);
  fe.Set("model_tensors", f.model_tensors);
  fe.Set("weight_tensors", f.weight_tensors);
  fe.Set("weight_bytes", f.weight_bytes);
  fe.Set("parameters", f.weight_bytes / 4);
  fe.Set("input", f.token_input
                      ? "token ids, " + std::to_string(f.seq_len) +
                            " per sequence"
                      : std::to_string(f.input_width) + " features per row");
  if (f.vocabulary) fe.Set("vocabulary", f.vocabulary);
  if (f.has_teacher) fe.Set("teacher_ops", f.teacher_ops);
  fe.Set("footprint_weights", f.footprint_weights);
  fe.Set("footprint_activations", f.footprint_activations);
  fe.Set("footprint_deltas", f.footprint_deltas);
  fe.Set("footprint_caches", f.footprint_caches);
  fe.Set("footprint_total", f.footprint_total);
  fe.Set("memory_budget", f.memory_budget);
  fe.Set("frontend_ms", frontend_ms);

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
  const uint64_t parameters = f.weight_bytes / 4;
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
  const uint64_t device_bytes = r.arena_size + r.plan.size();
  be.Set("device_bytes", device_bytes);
  be.Set("bound_ratio", device_bytes ? static_cast<double>(r.estimate_bytes) /
                                           static_cast<double>(device_bytes)
                                     : 0.0);
  const PlanWork work = MeasurePlanWork(r.plan);
  be.Set("gemm_flops", work.GemmFlopsPerStep());
  be.Set("attention_flops", work.AttentionFlopsPerStep());
  const bool peak = config.peak_gflops > 0.0;
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
