#ifndef SEEML_SOURCE_PLAN_CONFIG_H_
#define SEEML_SOURCE_PLAN_CONFIG_H_

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// =============================================================================
// config/ discipline of the plan ABI: the compilation request. What the user
// hands the update compiler — loss selection, LoRA and optimizer
// specifications, batch and budget — before any byte of the plan exists.
// The runtime never reads these; their compiled consequences land in the
// PlanHeader (schema.h).
// =============================================================================

namespace seeml::update {

enum class LossKind : uint8_t {
  kSoftmaxXEnt = 0,  // supervised: cross-entropy over class labels
  kMse = 1,          // supervised: mean squared error over dense targets
  kKLDistill = 2,    // teacher-driven: KL(teacher || student) on logits
  kXEntPlusKL = 3,   // composite: (1-w)*xent + w*kl
};

enum class OptimizerKind : uint8_t { kSgd = 0, kAdamW = 1 };

// Learning-rate schedule applied by the runtime on top of OptimizerSpec::lr.
// Warmup ramps linearly from 0 over `warmup_steps`; cosine then decays to
// lr * min_lr_factor across the plan's default_steps horizon.
enum class LrSchedule : uint32_t { kConstant = 0, kCosineWithWarmup = 1 };
// Which attention family a plan compiles to (E11, #94): see
// UpdateConfig::attention.
enum class AttentionKind : uint8_t { kAuto = 0, kCached = 1, kTiled = 2 };
// Whether a plan recomputes cheap forward activations in the backward
// instead of keeping them across the forward (roadmap 2b, F5 #133): see
// UpdateConfig::remat.
enum class RematKind : uint8_t { kAuto = 0, kOff = 1, kOn = 2 };

// How the adapter's product is scaled (S5 #151). kAlphaOverRank is LoRA's
// own s = alpha / r; kAlphaOverSqrtRank is rank-stabilized LoRA's
// s = alpha / sqrt(r), whose update does not shrink as the rank grows.
enum class LoraScaling : uint8_t { kAlphaOverRank = 0, kAlphaOverSqrtRank = 1 };

struct LoRASpec {
  int64_t rank = 8;
  float alpha = 16.0f;
  LoraScaling scaling = LoraScaling::kAlphaOverRank;
  /// The scale s of the adapter product, C' = C + s * (x @ A @ B).
  float Scale() const {
    const float r = static_cast<float>(rank);
    return scaling == LoraScaling::kAlphaOverSqrtRank ? alpha / std::sqrt(r)
                                                      : alpha / r;
  }
  uint64_t seed = 42;
  // Substring filters matched against the frozen weight's tensor name.
  // Empty => every eligible student MatMul is adapted.
  std::vector<std::string> target_filters;
};

struct OptimizerSpec {
  OptimizerKind kind = OptimizerKind::kAdamW;
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  // Per-tensor L2 gradient clipping applied before each optimizer step;
  // 0 disables (no clip instructions are emitted).
  float clip_norm = 0.0f;
  LrSchedule lr_schedule = LrSchedule::kConstant;
  uint64_t warmup_steps = 0;
  // The cosine floor as a fraction of lr. 0.1 (E8, #91), not 0: a floor of
  // zero makes the schedule's last step — and, before the horizon followed
  // the run, every step past it — train at LR 0, moving nothing while
  // still updating AdamW's moments. A zero floor must be asked for twice:
  // min_lr_factor = 0 AND allow_zero_lr.
  float min_lr_factor = 0.1f;
  bool allow_zero_lr = false;
};

enum class Precision : uint8_t { kF32 = 0, kCertifiedBf16 = 1 };

/// What the exporter proved about the source model, read from the report it
/// wrote beside the SMF and verified against the file by the frontend (S4
/// #150). Recorded in the plan's metadata; not a recipe field.
struct ExportReport {
  bool present = false;
  uint64_t report_hash = 0;           // ContentHash64 of the report file
  bool tokenizer_parity = false;      // the parity harness was run
  uint64_t tokenizer_parity_ids = 0;  // ids it compared against the reference
  bool logits_parity = false;         // the logits-parity check was run
  float logits_parity_max_delta = 0.0f;
};

struct UpdateConfig {
  int64_t batch = 32;
  LossKind loss = LossKind::kSoftmaxXEnt;
  float distill_weight = 0.5f;  // weight on the KL term for kXEntPlusKL
  float temperature = 2.0f;     // distillation softmax temperature
  LoRASpec lora;
  OptimizerSpec optimizer;
  uint64_t default_steps = 1000;
  // Gradient accumulation (roadmap 2a): micro-steps per optimizer step.
  // The plan compiles at micro-batch `batch` and accumulates G gradients
  // before each optimizer step, so activation memory scales with `batch`
  // while the effective batch is batch * G. 1 = the classic one-batch step.
  uint32_t grad_accum_steps = 1;
  // Quantize frozen base/teacher weights that feed MatMuls to per-tensor
  // symmetric int8 in the plan's rodata (QLoRA-style). Adapters, gradients,
  // and all activations stay f32; the source .smf on disk is untouched.
  bool quantize_base = false;
  // Store frozen base/teacher weights that feed MatMuls as bfloat16 in the
  // plan's rodata (roadmap 2c): half of f32, f32's exponent range, 8 bits
  // of mantissa, widened exactly inside the GEMM — compute stays f32 and
  // the committed model is patched from the pristine f32 file. Mutually
  // exclusive with quantize_base (one storage precision per weight).
  bool bf16_base = false;
  // Arithmetic (plan v18, F2 #130 / F4 #132). kF32 is the reference: every
  // kernel exact, every backend's bits its own. kCertifiedBf16 marks the
  // frozen-weight GEMMs of the train and step programs kFlagRelaxed — the
  // backend may run them on a certified relaxed kernel (Metal tensor ops
  // over bf16-rounded activations and the exact stored weight; Accelerate
  // on the CPU) — and the package must carry a numerics certificate.
  Precision precision = Precision::kF32;
  // Fuse GEMM -> AddBias -> activation chains into flagged GEMM epilogues
  // wherever the SIR use-lists prove no other reader of the intermediates
  // (teacher subgraphs, unadapted layers). Bitwise-neutral by construction;
  // exposed so a fused and an unfused compilation of the same model can be
  // compared bit-for-bit.
  bool fuse_epilogues = true;
  // Attention memory (E11, #94). kAuto tiles every attention op when the
  // probability caches of all of them together would exceed the cache
  // budget, or when the step-0 footprint would not fit a CONFIGURED
  // memory_budget_bytes with them (F3 #131: never the compiling host's
  // memory, so the plan's bytes depend on its inputs alone) — the cached
  // family holds P [B*H*S, S] from forward to backward for every layer,
  // and dP and dS at that size — at about twice the attention arithmetic;
  // kCached / kTiled force one family. Identical bits either way (the
  // tiled kernels reproduce the cached kernels' expressions).
  AttentionKind attention = AttentionKind::kAuto;
  uint64_t attention_cache_budget_bytes = 256ull << 20;
  // Rematerialization (roadmap 2b, F5 #133). The forward activations the
  // backward reads stay resident across the whole forward; the transient
  // segment they make up is about 98 % of a decoder's arena. kOn re-emits
  // the cheap ones — the norm outputs, the SwiGLU hidden, every folded
  // elementwise chain, never a GEMM or attention output — right before the
  // backward instruction that reads them, with the same opcode over the
  // same operands, so the originals are freed in the forward and the bits
  // are identical to the non-remat plan's (the train program is longer by
  // the recomputes; the eval program is untouched). kAuto turns it on only
  // when the plain binding's arena plus the plan would exceed the budget
  // the memory gate applies (memory_budget_bytes, or the host's memory);
  // kOff never does. Reported in the compile report as `remat`.
  RematKind remat = RematKind::kAuto;
  // The adapter-only package (F6 #134, plan v21): every frozen weight is a
  // source ref — read at run time from the model file the plan is bound to
  // by hash, in every program — and the plan carries no rodata: it ships
  // the instructions, the adapters' initial image and the emit table, at
  // adapter scale rather than model scale. The device commits onto its own
  // copy of the model. An f32 base only (the narrow bases are copies the
  // plan would have to carry), exact arithmetic (the relaxed kernels read
  // rodata weights), and a model loaded from a file (the offsets and the
  // hash that binds them).
  bool adapter_only = false;
  // Where the eval program reads the frozen weights of a narrow-base plan
  // compiled from a file (E12, plan v17): the source model's f32 — the
  // function that ships, scored exactly — unless this asks for the plan's
  // own narrow copies, a proxy for a device that cannot hold the f32 file
  // beside the arena (the 1.5B rung on 16 GB, F5 #133; a phone). The
  // report says which (`validation_scores`); an adapter-only plan has no
  // weights of its own and refuses the request.
  bool eval_on_plan = false;
  // The bitwise-safe kernel batch (E3, plan v12), each exposed for the same
  // reason: so a compilation with and without it can be compared
  // bit-for-bit. rope_table hoists the RoPE angles into one device-built
  // table per geometry; fuse_clip folds the per-tensor clip into the
  // optimizer step, removing a full read+write pass over every gradient.
  // Debug: render the final SIR (training block + merge program) into
  // CompiledUpdate::sir_dump. Off by default — it is the largest pure-CPU
  // graph cost in the driver, and nothing but a debugger reads it.
  bool dump_sir = false;
  // Fold same-shape elementwise chains with single-reader intermediates
  // into kFusedMap instructions (E4, roadmap Phase 1b, plan v13). Bitwise-
  // neutral, and switchable for the same reason the epilogue pass is.
  bool fuse_elementwise = true;
  // Fold a single-reader f32 GEMM into the elementwise add that reads it
  // (E10, #93, plan v14: kFlagGemmAddend) — the two activation-sized sums
  // of every LoRA site. Bitwise-neutral, and switchable like its siblings.
  bool fuse_gemm_addend = true;
  bool rope_table = true;
  bool fuse_clip = true;
  // The CPU blocked-GEMM cache tiles written into the plan header (v11).
  // 0 = the runtime's compiled-in default for that dimension. A throughput
  // knob only — no result bit depends on it — decided on the build host
  // from a measured, host-keyed kernel-policy table (tool/autotune.py) or
  // an explicit --gemm-tiles; the K tile must be a multiple of 4 (the
  // kernel's unroll width), which the driver checks before assembly.
  uint32_t gemm_tile_k = 0;
  uint32_t gemm_tile_n = 0;
  // The target's peak arithmetic rate in GFLOP/s, as the user states it
  // (seeml-bench's --peak-gflops convention). Never written into the plan:
  // it only turns the backend screen's work per step into a predicted step
  // time (S0 #146). 0 = unstated, no prediction.
  double peak_gflops = 0.0;
  // Test hook: when false, the plan contains forward+backward only (no
  // parameter mutation), enabling finite-difference gradient verification.
  bool emit_optimizer = true;
  // Fail-fast memory gate (ingressor resource analyzer): compilation is
  // rejected up front when the statically proven lower bound of the training
  // footprint exceeds this many bytes. 0 = detect the host's physical memory.
  uint64_t memory_budget_bytes = 0;
  // The training data (S2 #148, plan v22): raw text (`.txt` — one document;
  // `.jsonl` — one {"text": ...} per line) the frontend tokenizes with the
  // model's own section into a canonical SDS, or a pre-tokenized `.sds` it
  // holds to the model. The plan is then pinned to that corpus by hash
  // (kDataFlagCorpusPinned) and the frontend screen carries the corpus
  // rows. Empty = no data at compile time: the plan takes any corpus of the
  // right geometry, as every plan did before v22.
  std::string data_path;
  // The compiled held-out split (S3 #149, plan v22 kDataFlagSplit): with
  // --data, this fraction of the corpus's records — drawn by
  // HeldOutIndices(records, split_seed, val_frac), source/contract/split.h
  // — is the validation set, named in the plan header and recomputed on the
  // device from the same corpus. 0 = no compiled split: the runner's
  // --val-frac decides, by position, as before.
  float val_frac = 0.0f;
  uint64_t split_seed = 0;
  // The masked cross-entropy (S3 #149, plan v22 kDataFlagMaskedLoss): the
  // labels carry -1 at the positions the loss ignores (a prompt/response
  // corpus's prompt), the mean runs over the target positions. Implied by
  // a --data corpus that carries a mask; set here for a plan compiled
  // without data that will be fed one. The cross-entropy alone.
  bool masked_loss = false;
  // The exporter's report on the source model (S4 #150). A model that
  // carries a tokenizer section and was loaded from a file — an imported
  // checkpoint — is refused without one: the exporter's output is verified,
  // not trusted. Exporter-native demo models have no reference to be in
  // parity with and need none.
  ExportReport export_report;
  // The target's GPU peak in GFLOP/s (S6 #152), as the user states it:
  // the backend screen's GPU-denominated predicted step. Never written
  // into the plan. 0 = unstated.
  double gpu_peak_gflops = 0.0;
};

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_CONFIG_H_
