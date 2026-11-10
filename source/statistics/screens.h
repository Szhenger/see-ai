#ifndef SEEML_SOURCE_STATISTICS_SCREENS_H_
#define SEEML_SOURCE_STATISTICS_SCREENS_H_

#include "source/statistics/screen.h"

// =============================================================================
// The four screens (S0 #146; docs/next-project/seeai.md §2): one per
// subsystem — the compiler's frontend, analysis and backend, and the runtime.
// Each table below is the one statement of its screen: the console, the
// reports, the golden blocks the tests pin and the schema seeml-abi
// publishes all read it.
//
// The runtime screen is filled twice. The compiler fills its planned rows —
// what the device will hold, the work of a step, and, given the target's
// peak, how long a step should take — and the device's runner fills the
// rest from what it measured, printing each measured number beside its plan.
// =============================================================================

namespace seeml::update {

namespace screens_detail {
inline constexpr uint8_t kV = kStatVerbose;
inline constexpr uint8_t kX = kStatVolatile;
inline constexpr uint8_t kO = kStatOptional;
}  // namespace screens_detail

// --- frontend: the model the compiler read -------------------------------------

inline constexpr StatRow kFrontendRows[] = {
    {"model_ops", "operators", StatKind::kCount, "ops",
     "operators in the SMF model file"},
    {"model_tensors", "tensors", StatKind::kCount, "",
     "tensors the file declares, the input included"},
    {"weight_tensors", "frozen weights", StatKind::kCount, "",
     "the file's constant tensors"},
    {"weight_bytes", "weight payload", StatKind::kBytes, "B",
     "those tensors' bytes in the file"},
    {"parameters", "parameters", StatKind::kCount, "",
     "weight elements: the payload ÷ 4 B (f32)"},
    {"input", "input", StatKind::kText, "",
     "token ids per sequence, or features per row"},
    {"vocabulary", "vocabulary", StatKind::kCount, "tokens",
     "rows of the embedding table", screens_detail::kO},
    {"teacher_ops", "teacher operators", StatKind::kCount, "ops",
     "operators of the distillation teacher", screens_detail::kO},
    {"adapted_tensors", "adapted weights", StatKind::kCount, "",
     "frozen weights LoRA will adapt: read only as a MatMul's weight"},
    {"adapted_bytes", "adapted payload", StatKind::kBytes, "B",
     "those tensors' f32 bytes", screens_detail::kV},
    {"activations", "activations", StatKind::kCount, "",
     "forward values the graph computes at this batch", screens_detail::kV},
    {"activation_bytes", "activation payload", StatKind::kBytes, "B",
     "their f32 bytes, every value kept", screens_detail::kV},
    {"weights_source", "frozen weights live", StatKind::kText, "",
     "in the plan, or mapped from the model file (adapter-only)"},
    // The corpus (S2 #148, --data): present only when the compile took data.
    {"corpus", "corpus", StatKind::kText, "",
     "records and their width, and what --data named (text, jsonl, sds)",
     screens_detail::kO},
    {"corpus_documents", "documents", StatKind::kText, "",
     "documents read and kept (raw text)", screens_detail::kO},
    {"corpus_dropped", "dropped", StatKind::kText, "",
     "documents dropped: empty, exact duplicates, degenerate (raw text)",
     screens_detail::kO},
    {"corpus_sanitized", "sanitized", StatKind::kText, "",
     "bytes and characters the sanitizer removed or counted (raw text)",
     screens_detail::kO | screens_detail::kV},
    {"corpus_tokens", "corpus tokens", StatKind::kCount, "tokens",
     "ids the records hold", screens_detail::kO},
    {"corpus_remainder", "remainder", StatKind::kCount, "tokens",
     "ids the last partial window dropped (raw text)",
     screens_detail::kO | screens_detail::kV},
    {"corpus_straddles", "straddling windows", StatKind::kCount, "",
     "records that span a document boundary (no segment mask yet)",
     screens_detail::kO},
    {"corpus_distinct_ids", "distinct ids", StatKind::kText, "",
     "distinct token ids the corpus uses, of the vocabulary",
     screens_detail::kO},
    {"corpus_targets", "loss targets", StatKind::kText, "",
     "positions the loss counts, of all positions", screens_detail::kO},
    {"batches_per_epoch", "batches per epoch", StatKind::kReal, "",
     "training records (less a compiled split) ÷ sequences per optimizer step",
     screens_detail::kO, 2},
    {"epochs_covered", "epochs covered", StatKind::kReal, "",
     "the step budget × sequences per step ÷ training records",
     screens_detail::kO, 3},
    {"label_entropy", "label entropy", StatKind::kReal, "nats",
     "the labels' unigram entropy: the input-blind loss floor",
     screens_detail::kO, 4},
    {"corpus_hash", "corpus hash", StatKind::kText, "",
     "ContentHash64 of the canonical SDS the plan is pinned to",
     screens_detail::kO},
    {"validation", "validation", StatKind::kText, "",
     "the compiled held-out split: records, seed, the indices' hash",
     screens_detail::kO},
    {"footprint_weights", "step-0 weights", StatKind::kBytes, "B",
     "frozen weights at the base's storage", screens_detail::kV},
    {"footprint_activations", "step-0 activations", StatKind::kBytes, "B",
     "forward activations the backward reads", screens_detail::kV},
    {"footprint_deltas", "step-0 deltas", StatKind::kBytes, "B",
     "the merged f32 deltas, one per adapted weight", screens_detail::kV},
    {"footprint_caches", "step-0 caches", StatKind::kBytes, "B",
     "attention probability caches, or tiled stats rows", screens_detail::kV},
    {"footprint_total", "step-0 bound", StatKind::kBytes, "B",
     "the accountant's lower bound the early gate proved fits"},
    {"bound_ratio", "bound ÷ exact", StatKind::kReal, "",
     "the step-0 bound ÷ arena + plan: the accountant's honesty, ≤ 1", 0, 4},
    {"memory_budget", "memory budget", StatKind::kBytes, "B",
     "the configured budget, else this host's memory", screens_detail::kX},
    {"frontend_ms", "frontend time", StatKind::kReal, "ms",
     "ingress, topology, the accountant and the forward SIR, wall time",
     screens_detail::kV | screens_detail::kX, 3},
};
inline constexpr StatColumn kTensorColumns[] = {
    {"name", StatKind::kText, ""},
    {"bytes", StatKind::kBytes, "B"},
    // The fixtures' weights are drawn by the standard library's normal
    // distribution, whose algorithm differs between libc++ and libstdc++:
    // the values are the file's, not the toolchain's, so they stay out of
    // the golden blocks.
    {"min", StatKind::kReal, "", screens_detail::kX, 6},
    {"max", StatKind::kReal, "", screens_detail::kX, 6},
};
inline constexpr StatTable kFrontendTables[] = {
    {"tensors", "frozen weights",
     "each constant tensor: its payload and its value range, from the "
     "ingress sweep",
     kTensorColumns},
};
inline constexpr ScreenSpec kFrontendScreen = {
    "frontend", "the model the compiler read", kFrontendRows, kFrontendTables};

// --- analysis: what the passes did ------------------------------------------------

inline constexpr StatRow kAnalysisRows[] = {
    {"passes", "passes", StatKind::kCount, "",
     "passes and driver phases run, each timed"},
    {"ops_in", "ops in", StatKind::kCount, "ops",
     "the forward SIR the frontend built"},
    {"ops_out", "ops out", StatKind::kCount, "ops",
     "the training SIR after every pass, as lowered"},
    {"adapters", "adapters", StatKind::kCount, "",
     "LoRA adapters grafted, one per targeted matmul weight"},
    {"lora_rank", "rank", StatKind::kCount, "", "r, each adapter's inner width"},
    {"lora_alpha", "alpha", StatKind::kReal, "",
     "each adapter's delta is scaled by alpha ÷ r", screens_detail::kV, 1},
    {"trainable_parameters", "trainable", StatKind::kCount, "",
     "the sum of r·(k + m) over the adapters"},
    {"trainable_share", "trainable share", StatKind::kReal, "%",
     "trainable ÷ the model's parameters", 0, 3},
    {"gradients", "gradients", StatKind::kCount, "",
     "trainable tensors autodiff reached (A and B per adapter)",
     screens_detail::kV},
    {"rope_tables", "RoPE tables", StatKind::kCount, "",
     "angle tables hoisted, one per geometry", screens_detail::kV},
    {"epilogue_fusions", "epilogue fusions", StatKind::kCount, "",
     "bias and activation chains folded into a GEMM"},
    {"addend_fusions", "addend fusions", StatKind::kCount, "",
     "GEMMs folded into the add that reads them"},
    {"chain_fusions", "chain fusions", StatKind::kCount, "",
     "elementwise chains folded into one kernel"},
    {"dce_removed", "dead ops removed", StatKind::kCount, "ops",
     "ops nothing reads, swept before lowering"},
    {"attention", "attention", StatKind::kText, "",
     "cached probabilities, or the tiled family's stats rows"},
    {"rematerialization", "rematerialization", StatKind::kText, "",
     "forward values the backward recomputes instead of keeping"},
    {"narrow_weights", "narrow weights", StatKind::kText, "",
     "frozen weights stored int8 or bf16 in rodata"},
    {"analysis_ms", "analysis time", StatKind::kReal, "ms",
     "the passes' wall time", screens_detail::kV | screens_detail::kX, 3},
};
inline constexpr StatColumn kPassColumns[] = {
    {"name", StatKind::kText, ""},
    {"ops", StatKind::kCount, "ops"},
    {"ms", StatKind::kReal, "ms", screens_detail::kX, 3},
};
inline constexpr StatColumn kAdapterColumns[] = {
    {"weight", StatKind::kText, ""},
    {"k", StatKind::kCount, ""},
    {"m", StatKind::kCount, ""},
    {"rank", StatKind::kCount, ""},
    {"storage", StatKind::kText, ""},
};
inline constexpr StatTable kAnalysisTables[] = {
    {"passes", "passes", "each pass and phase: the ops after it, its time",
     kPassColumns},
    {"adapter_sites", "adapters",
     "each adapted weight [k, m], the rank, the base's storage",
     kAdapterColumns},
};
inline constexpr ScreenSpec kAnalysisScreen = {
    "analysis", "what the passes did", kAnalysisRows, kAnalysisTables};

// --- backend: the plan it emitted --------------------------------------------------

inline constexpr StatRow kBackendRows[] = {
    {"train_instructions", "train program", StatKind::kCount, "instructions",
     "one micro-batch: forward, backward, optimizer"},
    {"eval_instructions", "eval program", StatKind::kCount, "instructions",
     "the forward and the loss, for validation"},
    {"merge_instructions", "merge program", StatKind::kCount, "instructions",
     "each adapter's delta, (alpha ÷ r)·A·B"},
    {"step_instructions", "step program", StatKind::kCount, "instructions",
     "the optimizer step after G micro-batches", screens_detail::kO},
    {"arena_bytes", "arena", StatKind::kBytes, "B",
     "the one allocation the device makes"},
    {"persistent_bytes", "persistent", StatKind::kBytes, "B",
     "adapters and optimizer state, kept across steps", screens_detail::kV},
    {"transient_bytes", "transient", StatKind::kBytes, "B",
     "activations and gradients alive within a step", screens_detail::kV},
    {"rodata_bytes", "rodata", StatKind::kBytes, "B",
     "frozen weights packed into the plan"},
    {"plan_bytes", "plan file", StatKind::kBytes, "B", "the sealed .seeu"},
    {"device_bytes", "arena + plan", StatKind::kBytes, "B",
     "what the device holds: the arena and the mapped plan"},
    {"gemm_flops", "GEMM work", StatKind::kCount, "FLOP",
     "per step: 2·M·N·K over the train GEMMs, times G"},
    {"attention_flops", "attention work", StatKind::kCount, "FLOP",
     "per step: 4·B·H·S²·d forward + 8·B·H·S²·d backward"},
    {"peak_gflops", "target peak", StatKind::kReal, "GFLOP/s",
     "the target's peak, as passed with --peak-gflops", screens_detail::kO, 1},
    {"predicted_step_ms", "predicted step", StatKind::kReal, "ms",
     "(GEMM + attention work) ÷ the target peak", screens_detail::kO, 3},
    {"kernel_policy", "kernel policy", StatKind::kText, "",
     "the CPU GEMM tiles the header carries", screens_detail::kX},
    {"plan_hash", "plan hash", StatKind::kText, "",
     "the plan's self-hash, which checkpoints bind to", screens_detail::kX},
    {"backend_ms", "backend time", StatKind::kReal, "ms",
     "binding, lowering, assembly and sealing, wall time",
     screens_detail::kV | screens_detail::kX, 3},
};
inline constexpr ScreenSpec kBackendScreen = {
    "backend", "the plan it emitted", kBackendRows};

// --- runtime: planned by the compiler, measured by the device -------------------

inline constexpr StatRow kRuntimeRows[] = {
    {"device_bytes", "planned memory", StatKind::kBytes, "B",
     "arena + plan: the resource contract the device checks"},
    {"flops_per_step", "work per step", StatKind::kCount, "FLOP",
     "GEMM + attention work of one optimizer step"},
    {"predicted_step_ms", "predicted step", StatKind::kReal, "ms",
     "the work ÷ the target peak", screens_detail::kO, 3},
    {"backend", "backend", StatKind::kText, "",
     "the executor that ran the plan", screens_detail::kO | screens_detail::kX},
    {"steps", "steps", StatKind::kCount, "", "optimizer steps trained",
     screens_detail::kO},
    {"measured_step_ms", "measured step", StatKind::kReal, "ms",
     "training wall time ÷ steps, evaluations excluded",
     screens_detail::kO | screens_detail::kX, 3},
    {"achieved_gflops", "achieved", StatKind::kReal, "GFLOP/s",
     "the work per step ÷ the measured step",
     screens_detail::kO | screens_detail::kX, 1},
    {"utilization", "utilization", StatKind::kReal, "%",
     "achieved ÷ the target peak", screens_detail::kO | screens_detail::kX, 1},
    {"peak_rss_bytes", "peak RSS", StatKind::kBytes, "B",
     "the process's resident peak, as the OS reports it",
     screens_detail::kO | screens_detail::kX},
    {"rss_over_planned", "RSS ÷ planned", StatKind::kReal, "",
     "the resident peak ÷ the planned memory",
     screens_detail::kO | screens_detail::kX, 3},
    // The storage role (CR4 #145): what a checkpoint costs on disk.
    {"checkpoints", "checkpoints", StatKind::kCount, "",
     "the durable checkpoints the run saved", screens_detail::kO},
    {"checkpoint_bytes", "per checkpoint", StatKind::kBytes, "B",
     "bytes written per checkpoint, the mean", screens_detail::kO},
    {"checkpoint_fsyncs", "fsyncs per checkpoint", StatKind::kReal, "",
     "the file's and its directory's, the mean",
     screens_detail::kO | screens_detail::kX, 1},
    {"train_loss_first", "train loss, start", StatKind::kReal, "nats",
     "the mean loss over the run's first window of steps", screens_detail::kO, 4},
    {"train_loss_last", "train loss, end", StatKind::kReal, "nats",
     "the mean loss over the run's last window of steps", screens_detail::kO, 4},
    {"validation_loss_first", "validation loss, start", StatKind::kReal,
     "nats", "the held-out loss before training", screens_detail::kO, 4},
    {"validation_loss_last", "validation loss, end", StatKind::kReal, "nats",
     "the held-out loss of the state the gate weighed", screens_detail::kO, 4},
    {"label_entropy", "input-blind loss", StatKind::kReal, "nats",
     "the training labels' entropy: an input-blind predictor's loss",
     screens_detail::kO, 4},
    {"validation_accuracy", "validation accuracy", StatKind::kReal, "",
     "the held-out accuracy of the state the gate weighed",
     screens_detail::kO, 4},
    {"verdict", "verdict", StatKind::kText, "",
     "the gate's decision on the update", screens_detail::kO},
};
inline constexpr ScreenSpec kRuntimeScreen = {
    "runtime", "planned by the compiler, measured by the device",
    kRuntimeRows};

/// Every screen, in the order the console prints them.
inline constexpr const ScreenSpec* kScreens[] = {
    &kFrontendScreen, &kAnalysisScreen, &kBackendScreen, &kRuntimeScreen};

}  // namespace seeml::update

#endif  // SEEML_SOURCE_STATISTICS_SCREENS_H_
