#include "runtime/executor/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/metal_kernels.h"
#include "runtime/verifier/plan_validator.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// MetalBackend (roadmap Project 5, G1b-2/3/4): the Apple-GPU executor
// behind the ExecutorBackend seam.
//
// Residency (G1b-2): the engine's page-aligned arena is wrapped ONCE as a
// shared MTLBuffer (newBufferWithBytesNoCopy) — the GPU and the CPU read
// and write the same pages, so checkpoints, the loss slot and the merge
// deltas need no copies. The frozen weights are wrapped zero-copy too when
// the plan blob is page-aligned (the .incbin embedding) and page-padded
// (the compiler pads it); otherwise — a file-loaded plan in a heap vector
// — they are copied into one shared buffer at Bind, once, and device()
// says so.
//
// Batching (G1b-3): consecutive GPU-resident instructions are encoded into
// one command buffer through one serial compute encoder (Metal orders
// dispatches within it and tracks their memory hazards). A CPU-resident
// instruction — the loss kernels, the embedding gather — runs inline, but
// only after any pending GPU instruction it depends on has completed: the
// validator's operand extents (DescribeInstruction) decide, so dependency
// tracking is never looser than the bounds proof. The engine flushes at
// the end of every range, which makes each program (or timed phase) the
// batching unit. A command buffer that ends in any state but Completed
// aborts the update with an executor diagnostic — never a silent fallback
// to stale arena contents.
//
// Coverage (G1b-4): the GEMM family (simdgroup-MMA tiles with fused
// epilogues and the int8 variants), the transformer family, elementwise,
// normalization, RoPE, ClipNorm (the CPU's chunk geometry, partials
// combined in chunk order) and both optimizer steps run on the GPU; the
// three loss families and the embedding gather stay on the CPU (once per
// step, small). Determinism is per-backend: bitwise run-to-run on the same
// device, tolerance against the CPU (docs/roadmap.md).
// =============================================================================

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

// Mirrors `struct KArgs` in metal_kernels.h, field for field.
struct KArgs {
  uint64_t off[6] = {0, 0, 0, 0, 0, 0};
  uint32_t space = 0;
  uint32_t m = 0, n = 0, k = 0;
  uint32_t rows = 0, cols = 0;
  uint32_t B = 0, S = 0, H = 0, D = 0;
  uint32_t flags = 0;
  uint32_t step = 0;
  uint32_t pad0 = 0, pad1 = 0;
  float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // GEMM family: leading dimensions, batch geometry, per-operand batch
  // strides (elements; [2i] per outer batch, [2i+1] per inner) for A, B, C.
  uint32_t lda = 0, ldb = 0, ldc = 0;
  uint32_t batch = 1, batch_h = 1;
  uint32_t pad2 = 0, pad3 = 0, pad4 = 0;
  uint64_t bs[6] = {0, 0, 0, 0, 0, 0};
  // F7 #138 step 4: the low-rank epilogue's operands (PlanLowRank).
  uint64_t lr_off[3] = {0, 0, 0};
  uint32_t lr_k = 0;
  float lr_alpha = 1.0f;
};
static_assert(sizeof(KArgs) == 248, "KArgs must match the MSL layout");

enum Pipe : int {
  kPGemmNN, kPGemmNT, kPGemmTN, kPGemmAcc, kPGemmNNQ8, kPGemmNTQ8,
  kPGemmNNBF16, kPGemmNTBF16,
  // The skinny-shape variants, each in the same order (pipe + offset):
  // _small (K <= 16: one thread per output), _rows (N <= 16: one simdgroup
  // per row), _cols (M <= 16, wide N: one thread per column), _colsg
  // (M <= 16, narrower N: one simdgroup per column).
  kPGemmNNSmall, kPGemmNTSmall, kPGemmTNSmall, kPGemmAccSmall,
  kPGemmNNQ8Small, kPGemmNTQ8Small, kPGemmNNBF16Small, kPGemmNTBF16Small,
  kPGemmNNRows, kPGemmNTRows, kPGemmTNRows, kPGemmAccRows,
  kPGemmNNQ8Rows, kPGemmNTQ8Rows, kPGemmNNBF16Rows, kPGemmNTBF16Rows,
  kPGemmNNCols, kPGemmNTCols, kPGemmTNCols, kPGemmAccCols,
  kPGemmNNQ8Cols, kPGemmNTQ8Cols, kPGemmNNBF16Cols, kPGemmNTBF16Cols,
  kPGemmNNColsG, kPGemmNTColsG, kPGemmTNColsG, kPGemmAccColsG,
  kPGemmNNQ8ColsG, kPGemmNTQ8ColsG, kPGemmNNBF16ColsG, kPGemmNTBF16ColsG,
  // F7 step 3: the staged skinny forms — _rows_tg (N <= 16, A transposed:
  // the [32 k x 8 m] block of A through threadgroup memory) and _colsg_tg
  // (M <= 16, B untransposed: the [32 k x 8 n] block of B likewise).
  kPGemmNNRowsTg, kPGemmNTRowsTg, kPGemmTNRowsTg, kPGemmAccRowsTg,
  kPGemmNNQ8RowsTg, kPGemmNTQ8RowsTg, kPGemmNNBF16RowsTg, kPGemmNTBF16RowsTg,
  kPGemmNNColsGTg, kPGemmNTColsGTg, kPGemmTNColsGTg, kPGemmAccColsGTg,
  kPGemmNNQ8ColsGTg, kPGemmNTQ8ColsGTg, kPGemmNNBF16ColsGTg,
  kPGemmNTBF16ColsGTg,
  // F7 #138, the nt_rows floor: _rows_tgb (N <= 16, B transposed: the
  // [N x 32 k] block of B through threadgroup memory, read as float4s).
  kPGemmNNRowsTgB, kPGemmNTRowsTgB, kPGemmTNRowsTgB, kPGemmAccRowsTgB,
  kPGemmNNQ8RowsTgB, kPGemmNTQ8RowsTgB, kPGemmNNBF16RowsTgB,
  kPGemmNTBF16RowsTgB,
  kPGemmSplitKFin,
  // F7 #138 step 4: the tile kernel with the low-rank epilogue (NN forms).
  kPGemmNNLR, kPGemmNNQ8LR,
  kPAddEW, kPMulEW, kPAddBias, kPReluFwd, kPReluBwd, kPGeluFwd, kPGeluBwd,
  kPSiluFwd, kPSiluBwd, kPScale, kPFusedMap, kPFill, kPCopy, kPAccumulate, kPSgd, kPAdamW,
  kPReduceRows, kPLnFwd, kPLnBwd, kPRmsFwd, kPRmsBwd,
  kPClipPartials, kPClipFinish, kPClipApply, kPSgdClip, kPAdamWClip,
  kPSqNormFinish, kPClipGlobalScale,
  kPRopeFwd, kPRopeBwd, kPAttnSoftmax, kPSoftmaxRowsBwd,
  kPXentFwd, kPXentFinish, kPXentBwd,
  kPXentMaskedFwd, kPXentMaskedFinish, kPXentMaskedCount, kPXentMaskedBwd,
  kPRopeTable, kPEmbedFwd,
  kPAttnFwdTiled, kPAttnDQTiled, kPAttnDKTiled, kPAttnDVTiled,
  kPAttnFwdTiledBlk, kPAttnDQTiledBlk, kPAttnDKTiledBlk, kPAttnDVTiledBlk,
  kPAttnFwdTiledMma, kPAttnDQTiledMma, kPAttnDKTiledMma, kPAttnDVTiledMma,
  kPSelfTest,
  kPipeCount
};
const char* const kPipeNames[kPipeCount] = {
    "k_gemm_nn", "k_gemm_nt", "k_gemm_tn", "k_gemm_acc", "k_gemm_nn_q8",
    "k_gemm_nt_q8", "k_gemm_nn_bf16", "k_gemm_nt_bf16",
    "k_gemm_nn_small", "k_gemm_nt_small", "k_gemm_tn_small",
    "k_gemm_acc_small", "k_gemm_nn_q8_small", "k_gemm_nt_q8_small",
    "k_gemm_nn_bf16_small", "k_gemm_nt_bf16_small",
    "k_gemm_nn_rows", "k_gemm_nt_rows", "k_gemm_tn_rows", "k_gemm_acc_rows",
    "k_gemm_nn_q8_rows", "k_gemm_nt_q8_rows", "k_gemm_nn_bf16_rows",
    "k_gemm_nt_bf16_rows",
    "k_gemm_nn_cols", "k_gemm_nt_cols", "k_gemm_tn_cols", "k_gemm_acc_cols",
    "k_gemm_nn_q8_cols", "k_gemm_nt_q8_cols", "k_gemm_nn_bf16_cols",
    "k_gemm_nt_bf16_cols",
    "k_gemm_nn_colsg", "k_gemm_nt_colsg", "k_gemm_tn_colsg",
    "k_gemm_acc_colsg", "k_gemm_nn_q8_colsg", "k_gemm_nt_q8_colsg",
    "k_gemm_nn_bf16_colsg", "k_gemm_nt_bf16_colsg",
    "k_gemm_nn_rows_tg", "k_gemm_nt_rows_tg", "k_gemm_tn_rows_tg",
    "k_gemm_acc_rows_tg", "k_gemm_nn_q8_rows_tg", "k_gemm_nt_q8_rows_tg",
    "k_gemm_nn_bf16_rows_tg", "k_gemm_nt_bf16_rows_tg",
    "k_gemm_nn_colsg_tg", "k_gemm_nt_colsg_tg", "k_gemm_tn_colsg_tg",
    "k_gemm_acc_colsg_tg", "k_gemm_nn_q8_colsg_tg", "k_gemm_nt_q8_colsg_tg",
    "k_gemm_nn_bf16_colsg_tg", "k_gemm_nt_bf16_colsg_tg",
    "k_gemm_nn_rows_tgb", "k_gemm_nt_rows_tgb", "k_gemm_tn_rows_tgb",
    "k_gemm_acc_rows_tgb", "k_gemm_nn_q8_rows_tgb", "k_gemm_nt_q8_rows_tgb",
    "k_gemm_nn_bf16_rows_tgb", "k_gemm_nt_bf16_rows_tgb",
    "k_gemm_splitk_fin", "k_gemm_nn_lr", "k_gemm_nn_q8_lr", "k_add_ew", "k_mul_ew", "k_add_bias", "k_relu_fwd",
    "k_relu_bwd", "k_gelu_fwd", "k_gelu_bwd", "k_silu_fwd", "k_silu_bwd",
    "k_scale", "k_fused_map", "k_fill", "k_copy", "k_accumulate", "k_sgd", "k_adamw",
    "k_reduce_rows",
    "k_layernorm_fwd", "k_layernorm_bwd", "k_rmsnorm_fwd", "k_rmsnorm_bwd",
    "k_clip_partials", "k_clip_finish", "k_clip_apply", "k_sgd_clip",
    "k_adamw_clip", "k_sqnorm_finish", "k_clip_global_scale", "k_rope_fwd",
    "k_rope_bwd", "k_attn_softmax", "k_softmax_rows_bwd",
    "k_xent_fwd", "k_xent_finish", "k_xent_bwd",
    "k_xent_masked_fwd", "k_xent_masked_finish", "k_xent_masked_count",
    "k_xent_masked_bwd",
    "k_rope_table", "k_embed_fwd",
    "k_attn_fwd_tiled", "k_attn_dq_tiled", "k_attn_dk_tiled",
    "k_attn_dv_tiled",
    "k_attn_fwd_tiled_blk", "k_attn_dq_tiled_blk", "k_attn_dk_tiled_blk",
    "k_attn_dv_tiled_blk",
    "k_attn_fwd_tiled_mma", "k_attn_dq_tiled_mma", "k_attn_dk_tiled_mma",
    "k_attn_dv_tiled_mma", "k_selftest"};

// The relaxed GEMM family (plan v18): a second library, present only where
// Metal 4's tensor ops compile (macOS 26 on an Apple GPU). Two dispatches
// per GEMM — the activation rounded to bf16 into the backend's scratch
// (buffer 6), then the tensor-op tile with its epilogue; the exact kernels
// stand in everywhere the library is missing or a shape is skinny.
enum RxPipe : int {
  kRxToBf16, kRxNNQ8, kRxNTQ8, kRxNNBF16, kRxNTBF16, kRxNNF32, kRxNTF32,
  kRxPipeCount
};
const char* const kRxPipeNames[kRxPipeCount] = {
    "k_to_bf16", "k_gemm_rx_nn_q8", "k_gemm_rx_nt_q8", "k_gemm_rx_nn_bf16",
    "k_gemm_rx_nt_bf16", "k_gemm_rx_nn_f32", "k_gemm_rx_nt_f32"};
constexpr uint32_t kRxTileM = 64, kRxTileN = 32, kRxThreads = 128;

constexpr uint32_t kElementwiseGroup = 256;
constexpr uint32_t kGemmTile = 64;
constexpr uint32_t kGemmThreads = 128;  // 4 simdgroups of 32
constexpr int kGemmSmall = kPGemmNNSmall - kPGemmNN;
constexpr int kGemmRows = kPGemmNNRows - kPGemmNN;
constexpr int kGemmCols = kPGemmNNCols - kPGemmNN;
constexpr int kGemmColsG = kPGemmNNColsG - kPGemmNN;
constexpr int kGemmRowsTg = kPGemmNNRowsTg - kPGemmNN;
constexpr int kGemmColsGTg = kPGemmNNColsGTg - kPGemmNN;
constexpr int kGemmRowsTgB = kPGemmNNRowsTgB - kPGemmNN;
constexpr uint32_t kGemmStagedGroup = 256;  // 8 simdgroups: 8 rows or columns
// M <= 16: below this N a simdgroup per column (lanes striding K) beats a
// thread per column; above it the per-thread form's coalesced B reads win.
constexpr uint32_t kGemmColsPerThreadN = 4096;
// A GEMM with any dimension at or below this takes a skinny kernel rather
// than 64x64 tiles: an adapter's rank-8 factors, a K=8 product.
constexpr uint32_t kGemmSkinnyDim = 16;
// Split-K: a tiled GEMM with fewer output tiles than this splits K across
// threadgroup z (partials summed in a fixed order) so ten GPU cores stay
// fed; splits never exceed kGemmMaxSplits nor leave a split below
// kGemmMinSplitK.
constexpr uint32_t kGemmTargetTiles = 256;
constexpr uint32_t kGemmMaxSplits = 16;
constexpr uint32_t kGemmMinSplitK = 512;

uint32_t SplitsFor(uint32_t tiles, uint32_t k) {
  uint32_t splits = (kGemmTargetTiles + tiles - 1) / tiles;
  splits = std::max(splits, k / 4096);  // a very long K splits regardless
  splits = std::min(splits, kGemmMaxSplits);
  splits = std::min(splits, std::max(1u, k / kGemmMinSplitK));
  return std::max(splits, 1u);
}
constexpr size_t kScratchFloats = up::kMaxParallelChunks + 1;

std::string NsError(NSError* err, const char* what) {
  std::string s = what;
  if (err && err.localizedDescription)
    s += std::string(": ") + err.localizedDescription.UTF8String;
  return s;
}

float BitsToF32(uint64_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

struct Extent {
  uint64_t off, bytes;
};

// Whether every page of [ptr, ptr + len) is mapped writable. A no-copy
// MTLBuffer over read-only pages (an .incbin plan in __TEXT,__const) leaves
// them unreadable from the CPU afterwards — the embedding gather then dies
// with a protection fault — so zero-copy residency is offered only to
// writable memory; everything else is copied once at Bind.
bool PagesWritable(const void* ptr, uint64_t len) {
  mach_vm_address_t addr = reinterpret_cast<mach_vm_address_t>(ptr);
  const mach_vm_address_t end = addr + len;
  while (addr < end) {
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    mach_vm_address_t probe = addr;
    if (mach_vm_region(mach_task_self(), &probe, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &count,
                       &object) != KERN_SUCCESS)
      return false;
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (probe > addr) return false;  // a hole inside the range
    if (!(info.protection & VM_PROT_WRITE)) return false;
    addr = probe + size;
  }
  return true;
}
bool Overlaps(const Extent& a, const Extent& b) {
  return a.off < b.off + b.bytes && b.off < a.off + a.bytes;
}
// A diagnostic switch in the environment: set to anything but "" or "0".
bool EnvFlag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && *v != '0';
}

class MetalBackend final : public ExecutorBackend {
 public:
  static std::expected<std::unique_ptr<ExecutorBackend>, std::string> Create();

  ~MetalBackend() override {
    (void)Flush();
    if (profile_) PrintProfile();
    if (spans_) PrintSpans();
  }

  const char* name() const override { return "metal"; }
  std::string device() const override {
    std::string s = device_ ? device_.name.UTF8String : "no device";
    s += rodata_buf_ == nil ? "" : (rodata_zero_copy_ ? ", rodata zero-copy"
                                                        : ", rodata copied");
    s += relaxed_ ? ", relaxed GEMMs on tensor ops"
                  : ", relaxed GEMMs on the exact kernels (" +
                        relaxed_reason_ + ")";
    if (source_on_cpu_)
      s += ", source model on the CPU (" + std::to_string(source_bytes_) +
           " B; with the arena and the rodata it exceeds the GPU working "
           "set of " + std::to_string(working_set_limit_) + " B)";
    return s;
  }

  std::expected<void, std::string> Bind(uint8_t* arena, uint64_t arena_bytes,
                                        const uint8_t* rodata,
                                        uint64_t rodata_bytes,
                                        uint64_t rodata_mapped_bytes) override;
  std::expected<void, std::string> BindSource(const uint8_t* data,
                                              uint64_t bytes) override;
  std::expected<void, std::string> Execute(const up::UpdateInstruction& ins,
                                           const StepParams& params) override;
  std::expected<void, std::string> Flush() override;
  std::expected<void, std::string> SelfTest();
  void BeginRange(std::span<const up::UpdateInstruction> program, size_t begin,
                  size_t end) override {
    range_on_cpu_ = false;
    end = std::min(end, program.size());
    begin = std::min(begin, end);
    program_ = program;
    begin_ = begin;
    end_ = end;
    plan_ = nullptr;
    if (source_on_cpu_)
      for (const up::UpdateInstruction& ins :
           program.subspan(begin, end - begin))
        if (const InstructionExtents* ex = ExtentsOf(ins);
            ex && ReadsSource(*ex)) {
          range_on_cpu_ = true;
          return;
        }
    // F7 #138 step 4: the instruction pairs this program runs as one
    // dispatch — planned over the whole program, so a store dropped as
    // dead is dead in every range that follows (ExecuteRange's phases).
    if (fuse_lowrank_) plan_ = &PlanLowRank(program);
  }
  void EndRange() override {
    range_on_cpu_ = false;
    plan_ = nullptr;
    program_ = {};
    begin_ = end_ = 0;
  }
  static bool ReadsSource(const InstructionExtents& ex) {
    return std::any_of(ex.ranges, ex.ranges + ex.count,
                       [](const OperandExtent& r) { return r.source; });
  }
  // Its own tensor-op GEMMs and matrix-tile tiled attention dispatches
  // (v20) plus the relaxed GEMMs it hands the CPU
  // backend (a GEMM whose dimensions overflow 32 bits, DimsFit32), which
  // that backend runs on SGEMM in a SEEML_ACCELERATE build.
  uint64_t relaxed_executions() const override {
    return relaxed_executions_ + (cpu_ ? cpu_->relaxed_executions() : 0);
  }
  uint64_t fused_dispatches() const override { return lowrank_fused_; }

 private:
  const InstructionExtents* ExtentsOf(const up::UpdateInstruction& ins);
  // The validator's reason when ExtentsOf refuses an instruction, for the
  // dispatch-time message (CR1 #142: a refused threshold names itself).
  std::string revalidation_error_;
  bool HazardWithPending(const InstructionExtents& ex) const;
  void RecordPending(const InstructionExtents& ex);
  std::expected<void, std::string> EnsureEncoder();
  void Dispatch(Pipe pipe, const KArgs& a, uint32_t threads_x, uint32_t group,
                bool scratch = false);
  void DispatchGemm(Pipe pipe, KArgs a, bool at, bool bt, uint32_t b_elem);
  void EncodeAttentionGemm(Pipe pipe, bool at, bool bt, uint64_t a_ref,
                           uint64_t b_ref, uint64_t c_ref, uint32_t B,
                           uint32_t S, uint32_t H, uint32_t d,
                           uint32_t a_kind, uint32_t b_kind, uint32_t c_kind,
                           uint32_t m, uint32_t n, uint32_t k, float alpha);
  std::expected<void, std::string> Encode(const up::UpdateInstruction& ins,
                                          const StepParams& params);
  void PrintProfile() const;
  void PrintSpans() const;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pipes_[kPipeCount];
  id<MTLBuffer> scratch_ = nil;
  // The clip refusal flag (CR1 #142 F11): one float in its own buffer,
  // bound at index 7 for every encoder, raised by k_clip_finish on a
  // non-finite norm, honored by the kernels that would write parameters,
  // read and cleared by Flush.
  id<MTLBuffer> clip_flag_ = nil;
  id<MTLBuffer> splitk_ws_ = nil;  // split-K partials, grown on demand
  uint64_t splitk_ws_bytes_ = 0;
  // The relaxed family (v18): its pipelines when the Metal 4 library built,
  // the reason it did not otherwise, and the bf16 activation scratch.
  id<MTLComputePipelineState> rx_pipes_[kRxPipeCount];
  bool relaxed_ = false;
  std::string relaxed_reason_ = "not built";
  // GEMMs EncodeRelaxedGemm took (F42) and tiled attention instructions
  // dispatched on the matrix-tile kernels (v20).
  uint64_t relaxed_executions_ = 0;
  id<MTLBuffer> bf16_scratch_ = nil;
  uint64_t bf16_scratch_bytes_ = 0;
  bool EncodeRelaxedGemm(up::OpCode op, const up::UpdateInstruction& ins,
                         KArgs a);
  void NoteDispatchName(const char* name) {
    if (!profile_) return;
    if (!prof_pipes_.empty()) prof_pipes_ += '+';
    prof_pipes_ += name;
  }
  id<MTLBuffer> arena_buf_ = nil;
  id<MTLBuffer> rodata_buf_ = nil;
  bool rodata_zero_copy_ = false;
  // v17: the source model file the eval program's source refs read, wrapped
  // zero-copy when the mapping is page-aligned (the engine mmaps it), else
  // copied once. Bound at buffer index 5 on every encoder.
  id<MTLBuffer> source_buf_ = nil;
  // F5 #133, the 1.5B rung: beyond the device's working set the GPU
  // completes command buffers without running them (arena 7.7 GB + rodata
  // 2.6 GB + the 7.5 GB f32 source on a 16 GB M5: every loss read zero and
  // every buffer reported Completed). So Bind refuses an arena or an arena
  // plus rodata the GPU cannot hold, BindSource keeps the source model a
  // CPU mapping when it would not fit beside them — its readers, the eval
  // program's source-weight GEMMs, run on cpu_ — and a self-test at Bind
  // proves one dispatch executes. SEEML_METAL_WORKING_SET (bytes) overrides
  // the device's recommendedMaxWorkingSetSize, for tests.
  bool source_on_cpu_ = false;
  uint64_t source_bytes_ = 0;
  uint64_t working_set_limit_ = 0;
  // Set by BeginRange while source_on_cpu_: the range being executed reads
  // the source model, so every instruction of it runs on cpu_ — one
  // hand-off per range, not a drain around each source-weight GEMM (which
  // made the 1.5B rung's evaluation eight times the CPU backend's).
  bool range_on_cpu_ = false;
  // Whether this GPU reports command-buffer execution times (learned from
  // the self-test): where it does, a buffer that completes with none is
  // one the device declined, and Flush refuses it.
  bool gpu_timestamps_ = false;
  double last_gpu_start_ = 0, last_gpu_end_ = 0;
  uint8_t* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  const uint8_t* rodata_ = nullptr;
  uint64_t rodata_bytes_ = 0;
  std::unique_ptr<ExecutorBackend> cpu_;
  id<MTLCommandBuffer> cmd_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  std::vector<Extent> pending_reads_, pending_writes_;
  // Keyed on the instruction's 64 bytes, not its address: a caller that
  // reuses one UpdateInstruction object with different operands (the
  // hand-built test pattern) must never see stale extents.
  std::unordered_map<std::string, InstructionExtents> extents_;
  // SEEML_METAL_PROFILE=1: one command buffer per GPU instruction, its
  // GPU time (GPUStartTime..GPUEndTime) accumulated per kernel and shape,
  // CPU-resident instructions by wall clock; a table on stderr at
  // destruction. Diagnostic only — it serializes the stream.
  const bool profile_ = EnvFlag("SEEML_METAL_PROFILE");
  struct ProfRow { double seconds = 0; uint64_t count = 0; };
  std::map<std::string, ProfRow> prof_;
  std::string prof_pipes_;  // every kernel the current instruction dispatched
  double last_gpu_seconds_ = 0;
  // Forward and backward passes seen, counted at the loss opcodes (11 / 51
  // / 13 / 15 forward, 12 / 52 / 14 / 16 backward): the profile header prints them
  // so a reader sees how a run weights the two (F9 #140 — a run with
  // validation on is 7:1 forward).
  uint64_t fwd_passes_ = 0, bwd_passes_ = 0;
  // scratch_ holds kScratchFloats floats at creation (the clip partials);
  // the loss forward needs one float per row and grows it on demand.
  size_t scratch_floats_ = 0;
  // SEEML_METAL_SPANS=1 (F8 #139): per command buffer, the GPU's busy
  // span (GPUStartTime..GPUEndTime), the CPU wall from the first encode to
  // completion, and the dispatch count — the time outside kernels is the
  // difference, measured without serializing the stream the way the
  // profile does. A table on stderr at destruction. Diagnostic only.
  // SEEML_METAL_UNSTAGED_ROWS=1 keeps the transposed-B rows sites on the
  // unstaged kernel: the same-session control for the staged form's bits
  // and time (F7 #138), diagnostic only.
  const bool rows_tgb_ = !EnvFlag("SEEML_METAL_UNSTAGED_ROWS");
  // F7 #138 step 4: the low-rank epilogue fold (PlanLowRank) — a base
  // GEMM whose output the next rank-r GEMM takes as its addend runs as
  // one dispatch at the second instruction's place (k_gemm_*_lr), both
  // outputs written, the bits of the two dispatches.
  // SEEML_METAL_UNFUSED_LOWRANK=1 keeps them apart: the same-session
  // control for the fold's bits and time.
  const bool fuse_lowrank_ = !EnvFlag("SEEML_METAL_UNFUSED_LOWRANK");
  struct FusionPlan {
    std::vector<int32_t> role;  // kRoleNone, kRoleDeferred, or the base's index
    std::vector<int32_t> partner;  // a deferred base: its rank-r GEMM's index
    std::vector<uint8_t> drop_c;  // per fused j: the program never reads C again
    uint64_t pairs = 0;
  };
  static constexpr int32_t kRoleNone = -1, kRoleDeferred = -2;
  std::unordered_map<const void*, FusionPlan> fusion_plans_;
  const FusionPlan* plan_ = nullptr;
  // The program the current range belongs to and the range's bounds in it
  // (BeginRange): a pair is fused only when both instructions fall inside
  // the range the engine runs, since the engine reads the arena between
  // ranges.
  std::span<const up::UpdateInstruction> program_;
  size_t begin_ = 0, end_ = 0;
  // Fused dispatches, and those among them that dropped C's store (the
  // profile and the spans print both, so a run shows what the store-drop
  // rule decided on its program).
  uint64_t lowrank_fused_ = 0, lowrank_dropped_ = 0;
  const FusionPlan& PlanLowRank(std::span<const up::UpdateInstruction> program);
  std::expected<void, std::string> EncodeLowRank(
      const up::UpdateInstruction& base, const up::UpdateInstruction& lr,
      bool drop_c);
  const bool spans_ = EnvFlag("SEEML_METAL_SPANS");
  struct SpanRow { uint64_t dispatches; double gpu_seconds, wall_seconds; };
  std::vector<SpanRow> spans_rows_;
  std::chrono::steady_clock::time_point cb_t0_;
  uint64_t cb_dispatches_ = 0;
  void NoteDispatch(Pipe pipe) {
    if (!profile_) return;
    if (!prof_pipes_.empty()) prof_pipes_ += '+';
    prof_pipes_ += kPipeNames[pipe];
  }
};

// Dimensions AND the element indices built from them ride 32-bit in the
// kernels. Plain counts (GEMM M/N/K, elementwise counts, AddBias/ReduceRows
// rows x cols) and every index product (M·K, K·N, M·N; B·H·S·S for the
// probability matrix; B·S·H·d for activations) must fit; the transformer
// family's packed dim words are two 32-bit halves by construction, their
// products are not. A wider plan — one that could not exist on the devices
// this targets — takes the CPU path for that instruction.
bool DimsFit32(up::OpCode op, const up::UpdateInstruction& ins) {
  constexpr uint64_t kMax = 0xFFFFFFFFu;
  auto hi = [](uint64_t v) { return v >> 32; };
  auto lo = [](uint64_t v) { return v & kMax; };
  switch (op) {
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
    case up::OpCode::kGemmNNBF16:
    case up::OpCode::kGemmNTBF16:
      return ins.out[0] <= kMax && ins.out[1] <= kMax && ins.out[2] <= kMax &&
             ins.out[0] * ins.out[1] <= kMax && ins.out[0] * ins.out[2] <= kMax &&
             ins.out[2] * ins.out[1] <= kMax;
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd:
      return hi(ins.out[0]) * lo(ins.out[0]) * hi(ins.out[1]) * lo(ins.out[1]) <=
             kMax;
    case up::OpCode::kAttnFwd: {
      const uint64_t B = hi(ins.out[1]), S = lo(ins.out[1]);
      const uint64_t H = hi(ins.out[2]), d = lo(ins.out[2]);
      return B * H * S * S <= kMax && B * S * H * d <= kMax;
    }
    case up::OpCode::kAttnDP:
    case up::OpCode::kAttnDV:
    case up::OpCode::kAttnDQ:
    case up::OpCode::kAttnDK: {
      const uint64_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint64_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      return B * H * S * S <= kMax && B * S * H * d <= kMax;
    }
    case up::OpCode::kSoftmaxXEntFwd:
    case up::OpCode::kSoftmaxXEntBwd:
    case up::OpCode::kSoftmaxXEntMaskedFwd:
    case up::OpCode::kSoftmaxXEntMaskedBwd:
      return ins.out[0] <= kMax && ins.out[1] <= kMax &&
             ins.out[0] * ins.out[1] <= kMax;
    case up::OpCode::kSoftmaxRowsBwd:
    case up::OpCode::kLayerNormFwd:
    case up::OpCode::kRmsNormFwd:
      return hi(ins.out[0]) * lo(ins.out[0]) <= kMax;
    case up::OpCode::kLayerNormBwd:
      return hi(ins.out[2]) * lo(ins.out[2]) <= kMax;
    case up::OpCode::kRmsNormBwd:
      return hi(ins.out[1]) * lo(ins.out[1]) <= kMax;
    case up::OpCode::kAddBias:
    case up::OpCode::kReduceRows:
      return ins.out[0] <= kMax && ins.out[1] <= kMax &&
             ins.out[0] * ins.out[1] <= kMax;
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluFwd:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kSiluBwd:
    case up::OpCode::kScale:
    case up::OpCode::kFill:
    case up::OpCode::kCopy:
    case up::OpCode::kAccumulate:
    case up::OpCode::kFusedMap:
    case up::OpCode::kSgdStep:
    case up::OpCode::kAdamWStep:
    case up::OpCode::kClipNorm:
    case up::OpCode::kSqNormAccum:
    case up::OpCode::kClipGlobal:
      return ins.out[0] <= kMax;
    case up::OpCode::kRopeTable:
      return hi(ins.out[0]) * lo(ins.out[0]) <= kMax;
    // The tiled family: one simdgroup per row, no S x S matrix anywhere.
    case up::OpCode::kAttnFwdTiled: {
      const uint64_t B = hi(ins.out[1]), S = lo(ins.out[1]);
      const uint64_t H = hi(ins.out[2]), d = lo(ins.out[2]);
      return B * S * H * d <= kMax && B * H * S * 32 <= kMax;
    }
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled: {
      const auto geo = up::UnpackAttnGeometry(ins.out[2]);
      return geo.B * geo.S * geo.H * geo.d <= kMax &&
             geo.B * geo.H * geo.S * 32 <= kMax;
    }
    case up::OpCode::kEmbedFwd:
      return ins.out[0] * lo(ins.out[1]) <= kMax && hi(ins.out[1]) <= kMax;
    default:
      return true;
  }
}

bool IsGpuOpcode(up::OpCode op) {
  switch (op) {
    case up::OpCode::kNop:
    // The softmax cross-entropy pair (11 / 12) runs on the GPU since F8
    // (#139); the MSE and KL families stay here (MSE leaves the ISA under
    // S6; KL is the distillation path, not on the SmolLM step).
    case up::OpCode::kMseFwd:
    case up::OpCode::kMseBwd:
    case up::OpCode::kKLDistillFwd:
    case up::OpCode::kKLDistillBwd:
    // The embedding gather (41) and the RoPE angle table (45) run on the
    // GPU since F8 (#139) step 1's follow-up: the table was the last
    // CPU-resident instruction of a SmolLM-shaped train program, written
    // into a transient the GPU had just used, so the backend flushed and
    // waited there every step; the gather, instruction 0, went with it.
      return false;
    // Everything else, the tiled attention family (v15) included since F3
    // (#131; the CPU form stays the reference), runs on the GPU when its
    // dimensions fit (DimsFit32).
    default:
      return true;
  }
}

// The compiled kernel set, one per process. Both libraries compile from
// constant source for the one system device, and a pipeline state is an
// immutable, thread-safe object, so every backend the process creates
// shares them: the compile is paid once, not per backend. On a hosted
// macOS runner (a VM with no shader cache, 2–3 s a compile) the Metal
// suite creates a backend per case — 456 in the GEMM shape test alone —
// and ran 21 to 27 minutes against the job's 30, then crossed it three
// times on 2026-10-22; on an M5 the same suite is 7 s. A failed compile is
// not cached: every Create reports it, as before.
struct KernelSet {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipes[kPipeCount];
  id<MTLComputePipelineState> rx_pipes[kRxPipeCount];
  bool relaxed = false;
  std::string relaxed_reason = "not built";
};

std::expected<std::unique_ptr<KernelSet>, std::string> CompileKernels(
    id<MTLDevice> device) {
  auto set = std::make_unique<KernelSet>();
  set->device = device;
  MTLCompileOptions* options = [MTLCompileOptions new];
  // Precise math, both halves of it: `mathMode` governs algebraic
  // fast-math (reassociation, no NaN/inf), `mathFloatingPointFunctions`
  // governs which transcendental variants the unqualified tanh/exp/sin/
  // cos/sqrt calls resolve to — and the fast tanh returns NaN above
  // ~44, which through the GELU means NaN for any pre-activation past
  // ~10.25. The kernel library mirrors the CPU expressions and is
  // compared against them at tolerance; both knobs must be precise.
  // (The pre-15 `fastMathEnabled = NO` selects both at once.)
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else
#endif
  {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  NSError* err = nil;
  const std::string source = std::string(kMetalKernelHead) +
                             kMetalKernelPrelude + kMetalKernelSource;
  NSString* src = [NSString stringWithUTF8String:source.c_str()];
  id<MTLLibrary> lib = [device newLibraryWithSource:src
                                            options:options
                                              error:&err];
  if (!lib)
    return std::unexpected(NsError(err, "kernel library compilation failed"));
  // The relaxed family: Metal 4 (macOS 26) and a GPU whose compiler
  // accepts the tensor ops. Absent, relaxed instructions run exact.
  for (int i = 0; i < kRxPipeCount; ++i) set->rx_pipes[i] = nil;
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.0, *)) {
    MTLCompileOptions* rx_options = [MTLCompileOptions new];
    rx_options.languageVersion = MTLLanguageVersion4_0;
    rx_options.mathMode = MTLMathModeSafe;
    rx_options.mathFloatingPointFunctions =
        MTLMathFloatingPointFunctionsPrecise;
    const std::string rx_source = std::string(kMetalRelaxedKernelHead) +
                                  kMetalKernelPrelude +
                                  kMetalRelaxedKernelSource;
    NSError* rx_err = nil;
    id<MTLLibrary> rx_lib = [device
        newLibraryWithSource:[NSString stringWithUTF8String:rx_source.c_str()]
                     options:rx_options
                       error:&rx_err];
    if (!rx_lib) {
      set->relaxed_reason =
          NsError(rx_err, "Metal 4 tensor-op library did not compile");
    } else {
      set->relaxed = true;
      for (int i = 0; i < kRxPipeCount && set->relaxed; ++i) {
        id<MTLFunction> fn = [rx_lib
            newFunctionWithName:[NSString
                                    stringWithUTF8String:kRxPipeNames[i]]];
        set->rx_pipes[i] =
            fn ? [device newComputePipelineStateWithFunction:fn error:&rx_err]
               : nil;
        if (!set->rx_pipes[i]) {
          set->relaxed = false;
          set->relaxed_reason =
              std::string("pipeline '") + kRxPipeNames[i] + "' failed";
        }
      }
      if (set->relaxed &&
          set->rx_pipes[kRxNNQ8].threadExecutionWidth != 32) {
        set->relaxed = false;
        set->relaxed_reason = "tensor-op tiles need 32-wide simdgroups";
      }
    }
  } else
#endif
  {
    set->relaxed_reason = "Metal 4 needs macOS 26";
  }
  for (int i = 0; i < kPipeCount; ++i) {
    id<MTLFunction> fn = [lib
        newFunctionWithName:[NSString stringWithUTF8String:kPipeNames[i]]];
    if (!fn)
      return std::unexpected(std::string("kernel '") + kPipeNames[i] +
                             "' missing from the library");
    set->pipes[i] = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!set->pipes[i])
      return std::unexpected(NsError(err, "pipeline creation failed"));
  }
  // The GEMM tiles assume 32-wide simdgroups (four per 128-thread group)
  // and threadgroups of at least 256 for the elementwise kernels.
  if (set->pipes[kPGemmNN].threadExecutionWidth != 32)
    return std::unexpected(
        "unsupported SIMD width " +
        std::to_string(set->pipes[kPGemmNN].threadExecutionWidth) +
        " (the GEMM tiles need 32)");
  for (int i = 0; i < kPipeCount; ++i)
    if (set->pipes[i].maxTotalThreadsPerThreadgroup < kElementwiseGroup)
      return std::unexpected(
          "device threadgroup limit below " +
          std::to_string(kElementwiseGroup) + " for kernel " + kPipeNames[i]);
  return set;
}

// The process's kernel set for `device`, compiled on first use; a different
// device (none has been seen — the system default is one object) compiles
// its own and replaces it.
std::expected<const KernelSet*, std::string> SharedKernels(
    id<MTLDevice> device) {
  static std::mutex mu;
  static std::unique_ptr<KernelSet> shared;
  std::lock_guard<std::mutex> lock(mu);
  if (!shared || shared->device != device) {
    auto compiled = CompileKernels(device);
    if (!compiled) return std::unexpected(compiled.error());
    shared = std::move(*compiled);
  }
  return shared.get();
}

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
MetalBackend::Create() {
  @autoreleasepool {
    auto be = std::unique_ptr<MetalBackend>(new MetalBackend());
    be->device_ = MTLCreateSystemDefaultDevice();
    if (!be->device_) return std::unexpected("no Metal device available");
    be->queue_ = [be->device_ newCommandQueue];
    if (!be->queue_) return std::unexpected("cannot create a command queue");
    auto kernels = SharedKernels(be->device_);
    if (!kernels) return std::unexpected(kernels.error());
    for (int i = 0; i < kPipeCount; ++i) be->pipes_[i] = (*kernels)->pipes[i];
    for (int i = 0; i < kRxPipeCount; ++i)
      be->rx_pipes_[i] = (*kernels)->rx_pipes[i];
    be->relaxed_ = (*kernels)->relaxed;
    be->relaxed_reason_ = (*kernels)->relaxed_reason;
    be->scratch_ =
        [be->device_ newBufferWithLength:kScratchFloats * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    if (!be->scratch_) return std::unexpected("scratch allocation failed");
    be->scratch_floats_ = kScratchFloats;
    be->clip_flag_ =
        [be->device_ newBufferWithLength:16
                                 options:MTLResourceStorageModeShared];
    if (!be->clip_flag_)
      return std::unexpected("clip flag allocation failed");
    static_cast<float*>(be->clip_flag_.contents)[0] = 0.0f;  // not zeroed
    be->cpu_ = CreateCpuBackend();
    // Zero — a device that reports no working set, or an override that
    // does not parse — means unknown: the Bind and BindSource checks stand
    // down rather than refuse every arena and divert every source.
    be->working_set_limit_ = be->device_.recommendedMaxWorkingSetSize;
    if (const char* ws = std::getenv("SEEML_METAL_WORKING_SET");
        ws && *ws != '\0')
      if (const uint64_t v = std::strtoull(ws, nullptr, 10); v != 0)
        be->working_set_limit_ = v;
    return std::unique_ptr<ExecutorBackend>(std::move(be));
  }
}

std::expected<void, std::string> MetalBackend::Bind(
    uint8_t* arena, uint64_t arena_bytes, const uint8_t* rodata,
    uint64_t rodata_bytes, uint64_t rodata_mapped_bytes) {
  @autoreleasepool {
    // Nothing may still be in flight against the previous binding.
    if (auto r = Flush(); !r) return r;
    fusion_plans_.clear();  // the ranges of the previous plan
    plan_ = nullptr;
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    if (page == 0 || reinterpret_cast<uintptr_t>(arena) % page != 0 ||
        arena_bytes % page != 0)
      return std::unexpected("arena is not page-aligned (" +
                             std::to_string(page) + "-byte pages)");
    // The arena is one buffer, and the GPU holds it with the rodata for
    // every step: both limits are refusals with the numbers, never a run
    // that computes nothing (see source_on_cpu_).
    const uint64_t largest = device_.maxBufferLength;
    if (arena_bytes > largest)
      return std::unexpected(
          "the arena (" + std::to_string(arena_bytes) +
          " bytes) exceeds this GPU's largest buffer (" +
          std::to_string(largest) +
          " bytes): run on the CPU, or compile a smaller plan (a smaller "
          "micro-batch under --grad-accum, --remat on, --attention tiled)");
    id<MTLBuffer> arena_buf =
        [device_ newBufferWithBytesNoCopy:arena
                                   length:arena_bytes
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
    if (!arena_buf) return std::unexpected("cannot wrap the arena zero-copy");
    id<MTLBuffer> rodata_buf = nil;
    bool zero_copy = false;
    if (rodata_bytes > 0) {
      const uint64_t wrapped = (rodata_bytes + page - 1) & ~(page - 1);
      if (reinterpret_cast<uintptr_t>(rodata) % page == 0 &&
          wrapped <= rodata_mapped_bytes && PagesWritable(rodata, wrapped)) {
        rodata_buf = [device_
            newBufferWithBytesNoCopy:const_cast<uint8_t*>(rodata)
                              length:wrapped
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        zero_copy = rodata_buf != nil;
      }
      if (!rodata_buf) {
        // Documented cost of a heap-resident plan: one rodata-sized copy,
        // made once per Bind (device() reports it).
        rodata_buf = [device_ newBufferWithBytes:rodata
                                          length:rodata_bytes
                                         options:MTLResourceStorageModeShared];
        if (!rodata_buf)
          return std::unexpected("cannot allocate the rodata buffer");
      }
    }
    const uint64_t resident =
        arena_bytes + (rodata_buf ? static_cast<uint64_t>(rodata_buf.length) : 0);
    if (working_set_limit_ != 0 && resident > working_set_limit_)
      return std::unexpected(
          "the arena (" + std::to_string(arena_bytes) + " bytes) plus the "
          "rodata (" + std::to_string(resident - arena_bytes) +
          " bytes) exceed this GPU's working set (" +
          std::to_string(working_set_limit_) +
          " bytes): run on the CPU, or compile a smaller plan (a smaller "
          "micro-batch under --grad-accum, --remat on, --attention tiled)");
    if (auto r = cpu_->Bind(arena, arena_bytes, rodata, rodata_bytes,
                            rodata_mapped_bytes);
        !r)
      return r;
    arena_buf_ = arena_buf;
    rodata_buf_ = rodata_buf;
    rodata_zero_copy_ = zero_copy;
    arena_ = arena;
    arena_bytes_ = arena_bytes;
    rodata_ = rodata;
    rodata_bytes_ = rodata_bytes;
    extents_.clear();
    pending_reads_.clear();
    pending_writes_.clear();
    return SelfTest();
  }
}

std::expected<void, std::string> MetalBackend::SelfTest() {
  // One dispatch that writes the scratch buffer, read back: the device runs
  // what it completes. It also learns whether this GPU reports execution
  // times, which lets Flush refuse a later buffer that completes with none.
  @autoreleasepool {
    auto* s = static_cast<float*>(scratch_.contents);
    s[0] = 0.0f;
    gpu_timestamps_ = false;
    if (auto r = EnsureEncoder(); !r) return r;
    KArgs a;
    a.f[0] = 1.0f;
    Dispatch(kPSelfTest, a, 1, 1);
    if (auto r = Flush(); !r) return r;
    if (s[0] != 1.0f)
      return std::unexpected(
          "the GPU completed a command buffer without executing it (the "
          "self-test kernel wrote nothing)");
    gpu_timestamps_ = last_gpu_start_ > 0 && last_gpu_end_ > last_gpu_start_;
    return {};
  }
}

std::expected<void, std::string> MetalBackend::BindSource(const uint8_t* data,
                                                          uint64_t bytes) {
  @autoreleasepool {
    if (auto r = Flush(); !r) return r;  // nothing in flight reads the old one
    if (auto r = cpu_->BindSource(data, bytes); !r) return r;
    source_on_cpu_ = false;
    source_bytes_ = bytes;
    if (!data || bytes == 0) {
      source_buf_ = nil;
      return {};
    }
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    const uint64_t wrapped = (bytes + page - 1) & ~(page - 1);
    // The arena and the rodata are what every step touches; the source
    // model is read by the eval program alone. Where the three together
    // exceed the GPU's working set, the source stays a CPU mapping and the
    // instructions that read it run on cpu_ (Execute) — a slower validation
    // pass, and a correct one.
    const uint64_t resident =
        (arena_buf_ ? static_cast<uint64_t>(arena_buf_.length) : 0) +
        (rodata_buf_ ? static_cast<uint64_t>(rodata_buf_.length) : 0) + wrapped;
    if (working_set_limit_ != 0 && resident > working_set_limit_) {
      source_buf_ = nil;
      source_on_cpu_ = true;
      return {};
    }
    id<MTLBuffer> buf = nil;
    if (reinterpret_cast<uintptr_t>(data) % page == 0)
      buf = [device_ newBufferWithBytesNoCopy:const_cast<uint8_t*>(data)
                                       length:wrapped
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
    if (!buf)
      buf = [device_ newBufferWithBytes:data
                                 length:bytes
                                options:MTLResourceStorageModeShared];
    if (!buf) return std::unexpected("cannot wrap the source model for the GPU");
    source_buf_ = buf;
    return {};
  }
}

const InstructionExtents* MetalBackend::ExtentsOf(
    const up::UpdateInstruction& ins) {
  std::string key(reinterpret_cast<const char*>(&ins), sizeof(ins));
  auto it = extents_.find(key);
  if (it != extents_.end()) return &it->second;
  // The engine validated every program (the eval one with source refs);
  // describing an instruction here only recovers the extents it proved.
  auto ex = DescribeInstruction(ins, arena_bytes_, rodata_bytes_,
                                up::kSeeuVersion, /*allow_source=*/true);
  if (!ex) {
    revalidation_error_ = ex.error();
    return nullptr;
  }
  return &extents_.emplace(std::move(key), *ex).first->second;
}

bool MetalBackend::HazardWithPending(const InstructionExtents& ex) const {
  for (size_t i = 0; i < ex.count; ++i) {
    const OperandExtent& e = ex.ranges[i];
    if (e.rodata || e.source) continue;  // never written by anyone
    const Extent mine{e.off, e.bytes};
    for (const Extent& w : pending_writes_)
      if (Overlaps(mine, w)) return true;
    if (e.write)
      for (const Extent& r : pending_reads_)
        if (Overlaps(mine, r)) return true;
  }
  return false;
}

void MetalBackend::RecordPending(const InstructionExtents& ex) {
  for (size_t i = 0; i < ex.count; ++i) {
    const OperandExtent& e = ex.ranges[i];
    if (e.rodata || e.source) continue;
    (e.write ? pending_writes_ : pending_reads_).push_back({e.off, e.bytes});
  }
}

std::expected<void, std::string> MetalBackend::EnsureEncoder() {
  if (enc_) return {};
  cmd_ = [queue_ commandBuffer];
  enc_ = cmd_ ? [cmd_ computeCommandEncoder] : nil;  // serial dispatch type
  if (spans_) cb_t0_ = std::chrono::steady_clock::now();
  cb_dispatches_ = 0;
  if (!cmd_ || !enc_) {
    cmd_ = nil;
    enc_ = nil;
    return std::unexpected("cannot create a GPU command encoder");
  }
  [enc_ setBuffer:arena_buf_ offset:0 atIndex:0];
  [enc_ setBuffer:(rodata_buf_ ? rodata_buf_ : arena_buf_) offset:0 atIndex:1];
  [enc_ setBuffer:(source_buf_ ? source_buf_ : arena_buf_) offset:0 atIndex:5];
  [enc_ setBuffer:scratch_ offset:0 atIndex:3];
  [enc_ setBuffer:clip_flag_ offset:0 atIndex:7];
  // Every tiled GEMM kernel declares the split-K workspace at index 4 and
  // reads it only when it splits; the binding must still exist (an unbound
  // argument is a contract violation the debug layer aborts on), so the
  // scratch buffer stands in until a workspace is allocated.
  [enc_ setBuffer:(splitk_ws_ ? splitk_ws_ : scratch_) offset:0 atIndex:4];
  return {};
}

void MetalBackend::Dispatch(Pipe pipe, const KArgs& a, uint32_t threads_x,
                            uint32_t group, bool) {
  NoteDispatch(pipe);
  ++cb_dispatches_;
  [enc_ setComputePipelineState:pipes_[pipe]];
  [enc_ setBytes:&a length:sizeof(a) atIndex:2];
  // 64-bit: DimsFit32 admits counts up to 2^32 - 1, and the round-up must
  // not wrap to a near-empty grid.
  const uint64_t groups = (uint64_t{threads_x} + group - 1) / group;
  [enc_ dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

// `at`/`bt`: operand stored transposed; `b_elem`: B's element size. A
// caller that leaves lda/ldb/ldc zero gets the dense strides. Sets the
// vector-load flag bits (16: A, 32: B) when the operand's byte offset and
// leading dimension keep every 4-element vector 4*elem-aligned, picks a
// skinny kernel for skinny shapes, and splits K for an unbatched GEMM
// with too few tiles.
void MetalBackend::DispatchGemm(Pipe pipe, KArgs a, bool at, bool bt,
                                uint32_t b_elem) {
  if (a.lda == 0) a.lda = at ? a.m : a.k;
  if (a.ldb == 0) a.ldb = bt ? a.k : a.n;
  if (a.ldc == 0) a.ldc = a.n;
  if (a.off[0] % 16 == 0 && a.lda % 4 == 0 && a.bs[0] % 4 == 0 &&
      a.bs[1] % 4 == 0)
    a.flags |= 16u;
  if (a.off[1] % (4 * b_elem) == 0 && a.ldb % 4 == 0 && a.bs[2] % 4 == 0 &&
      a.bs[3] % 4 == 0)
    a.flags |= 32u;
  if (a.k <= kGemmSkinnyDim) {
    // Four outputs per thread (F7 step 3); flag 4096 = C and the addend
    // move as 16-byte vectors.
    if (a.off[2] % 16 == 0 && a.ldc % 4 == 0 && a.n % 4 == 0 &&
        a.bs[4] % 4 == 0 && a.bs[5] % 4 == 0 &&
        (!(a.flags & 2048u) || a.off[3] % 16 == 0))
      a.flags |= 4096u;
    Dispatch(static_cast<Pipe>(pipe + kGemmSmall), a,
             a.batch * a.m * ((a.n + 3) / 4), kElementwiseGroup);
    return;
  }
  if (a.n <= kGemmSkinnyDim) {
    if (at)  // A[k, m] with the lanes striding k: stage it (F7 step 3)
      Dispatch(static_cast<Pipe>(pipe + kGemmRowsTg), a,
               a.batch * ((a.m + 7) / 8) * kGemmStagedGroup, kGemmStagedGroup);
    else if (bt && rows_tgb_)  // B[n, k]: N rows per k — stage the block
      Dispatch(static_cast<Pipe>(pipe + kGemmRowsTgB), a,
               a.batch * ((a.m + 7) / 8) * kGemmStagedGroup, kGemmStagedGroup);
    else
      Dispatch(static_cast<Pipe>(pipe + kGemmRows), a, a.batch * a.m * 32u,
               kElementwiseGroup);
    return;
  }
  if (a.m <= kGemmSkinnyDim) {
    if (a.n >= kGemmColsPerThreadN)
      Dispatch(static_cast<Pipe>(pipe + kGemmCols), a, a.batch * a.n,
               kElementwiseGroup);
    else if (!bt)  // B[k, n] with the lanes striding k: stage it
      Dispatch(static_cast<Pipe>(pipe + kGemmColsGTg), a,
               a.batch * ((a.n + 7) / 8) * kGemmStagedGroup, kGemmStagedGroup);
    else
      Dispatch(static_cast<Pipe>(pipe + kGemmColsG), a, a.batch * a.n * 32u,
               kElementwiseGroup);
    return;
  }
  const uint32_t gx = (a.n + kGemmTile - 1) / kGemmTile;
  const uint32_t gy = (a.m + kGemmTile - 1) / kGemmTile;
  // The low-rank kernels (F7 #138 step 4) ignore p.pad0 and write both
  // outputs whole, so they never split: a split would leave the finish
  // kernel summing a workspace nobody wrote. PlanLowRank declines a base
  // the split-K policy would take; this keeps the dispatch right if the
  // two ever disagree.
  const bool lowrank = pipe == kPGemmNNLR || pipe == kPGemmNNQ8LR;
  const uint32_t splits =
      a.batch > 1 || lowrank ? 1 : SplitsFor(gx * gy, a.k);
  a.pad0 = splits;
  if (splits > 1) {
    const uint64_t need = uint64_t{splits} * a.m * a.n * sizeof(float);
    if (need > splitk_ws_bytes_) {
      // Grow geometrically; an encoder that still references the old
      // buffer keeps it alive until its command buffer completes.
      const uint64_t bytes = std::max(need, std::max<uint64_t>(
                                                splitk_ws_bytes_ * 2, 1 << 20));
      splitk_ws_ = [device_ newBufferWithLength:bytes
                                        options:MTLResourceStorageModePrivate];
      splitk_ws_bytes_ = splitk_ws_ ? bytes : 0;
      if (!splitk_ws_) {
        // No workspace: run unsplit (slower, never wrong).
        a.pad0 = 1;
      }
    }
    if (a.pad0 > 1) [enc_ setBuffer:splitk_ws_ offset:0 atIndex:4];
  }
  NoteDispatch(pipe);
  ++cb_dispatches_;
  [enc_ setComputePipelineState:pipes_[pipe]];
  [enc_ setBytes:&a length:sizeof(a) atIndex:2];
  [enc_ dispatchThreadgroups:MTLSizeMake(gx, gy, a.pad0 > 1 ? a.pad0 : a.batch)
       threadsPerThreadgroup:MTLSizeMake(kGemmThreads, 1, 1)];
  if (a.pad0 > 1) {
    // The finish kernel: flags bit 6 selects the accumulate form.
    if (pipe == kPGemmAcc) a.flags |= 64u;
    Dispatch(kPGemmSplitKFin, a, a.m * a.n, kElementwiseGroup);
  }
}

// Whether an opcode's write extents are stores alone. The validator
// describes a read-modify-write operand — an accumulator, a parameter an
// optimizer step updates, a norm clipped in place, dQ's delta column — as
// one write extent (plan_validator.cc, "dst (read+write, one operand)"), so
// a covering write by one of these still reads what it covers: for the
// store-drop rule below it is a read.
bool OverwritesWhatItWrites(up::OpCode op) {
  switch (op) {
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kAccumulate:
    case up::OpCode::kClipNorm:
    case up::OpCode::kSqNormAccum:
    case up::OpCode::kClipGlobal:
    case up::OpCode::kSgdStep:
    case up::OpCode::kAdamWStep:
    case up::OpCode::kAttnDQTiled:
      return false;
    default:
      return true;
  }
}

// F7 #138 step 4 — the low-rank epilogue, fused at dispatch. At a LoRA site
// the plan carries two instructions: the base product into C (kGemmNN /
// kGemmNNQ8 on the tile kernel), then C' = alpha·T@B + C — the rank-r
// product with C as its addend (kGemmNN, K <= 16, kFlagGemmAddend). Where
// nothing between them touches C or writes a base operand, the pair runs
// as ONE dispatch at the second instruction's place: the tile kernel with
// the low-rank epilogue (k_gemm_*_lr) writes C as before and C' as
// k_gemm_nn_small would have — serial k from zero, alpha on the finished
// sum, the addend first — so the arena holds the bits of the two dispatches
// with one pass over the activation fewer. Decided once per program from
// the extents the validator proved (cached by the program's identity,
// dropped at Bind), the whole program and not the range the engine runs,
// so what it finds dead is dead in every range that follows; the plan and
// the other backends are untouched, and SEEML_METAL_UNFUSED_LOWRANK=1 is
// the same-session control.
const MetalBackend::FusionPlan& MetalBackend::PlanLowRank(
    std::span<const up::UpdateInstruction> program) {
  FusionPlan& plan = fusion_plans_[program.data()];
  if (plan.role.size() == program.size()) return plan;
  plan.role.assign(program.size(), kRoleNone);
  plan.partner.assign(program.size(), kRoleNone);
  plan.drop_c.assign(program.size(), 0);
  plan.pairs = 0;
  auto arena_ref = [](uint64_t r) {
    return r != up::kNullRef && !up::IsRodataRef(r) && !up::IsSourceRef(r);
  };
  // Two extents of one address space that share a byte.
  auto overlap = [](const OperandExtent& a, const OperandExtent& b) {
    return a.rodata == b.rodata && a.source == b.source &&
           a.off < b.off + b.bytes && b.off < a.off + a.bytes;
  };
  auto covers = [](const OperandExtent& a, const OperandExtent& b) {
    return a.off <= b.off && a.off + a.bytes >= b.off + b.bytes;
  };
  for (size_t j = 0; j < program.size(); ++j) {
    const up::UpdateInstruction& lr = program[j];
    if (static_cast<up::OpCode>(lr.opcode) != up::OpCode::kGemmNN ||
        lr.flags != up::kFlagGemmAddend || lr.out[2] == 0 ||
        lr.out[2] > kGemmSkinnyDim || !arena_ref(lr.in[0]) ||
        !arena_ref(lr.in[1]) || !arena_ref(lr.in[2]) || !arena_ref(lr.in[3]) ||
        !DimsFit32(up::OpCode::kGemmNN, lr))
      continue;
    const InstructionExtents* lex = ExtentsOf(lr);
    if (!lex) continue;
    OperandExtent c;
    c.off = up::RefOffset(lr.in[3]);
    c.bytes = lr.out[0] * lr.out[1] * sizeof(float);
    // The base: scanning back, the first instruction that touches C must
    // be the GEMM that wrote exactly it.
    size_t i = program.size();
    for (size_t k = j; k-- > 0;) {
      const InstructionExtents* ex = ExtentsOf(program[k]);
      if (!ex) break;
      bool touches = false;
      for (size_t t = 0; t < ex->count && !touches; ++t)
        touches = overlap(ex->ranges[t], c);
      if (!touches) continue;
      i = k;
      break;
    }
    if (i == program.size() || plan.role[i] != kRoleNone) continue;
    const up::UpdateInstruction& base = program[i];
    const auto bop = static_cast<up::OpCode>(base.opcode);
    if ((bop != up::OpCode::kGemmNN && bop != up::OpCode::kGemmNNQ8) ||
        base.in[2] != lr.in[3] || base.out[0] != lr.out[0] ||
        base.out[1] != lr.out[1] || !DimsFit32(bop, base))
      continue;
    // The base must take the unsplit tile kernel, the one with the fused
    // epilogue: no skinny dimension, no split-K, not the tensor-op path.
    const auto bm = static_cast<uint32_t>(base.out[0]);
    const auto bn = static_cast<uint32_t>(base.out[1]);
    const auto bk = static_cast<uint32_t>(base.out[2]);
    if (bm <= kGemmSkinnyDim || bn <= kGemmSkinnyDim || bk <= kGemmSkinnyDim)
      continue;
    if (SplitsFor(((bn + kGemmTile - 1) / kGemmTile) *
                      ((bm + kGemmTile - 1) / kGemmTile),
                  bk) != 1)
      continue;
    if ((base.flags & up::kFlagRelaxed) && relaxed_) {
      // EncodeRelaxedGemm takes every non-skinny relaxed GEMM whose alpha
      // (the q8 per-tensor scale included) is exactly 1.
      const float f0 = bop == up::OpCode::kGemmNN
                           ? up::GemmAlphaOf(base.imm)
                           : (base.flags & up::kFlagQ8ColScale)
                                 ? 1.0f
                                 : BitsToF32(base.in[3]);
      if (f0 == 1.0f) continue;
    }
    const InstructionExtents* bex = ExtentsOf(base);
    if (!bex) continue;
    bool legal = true;
    // Across the two (the validator proves each alone): in the fused
    // kernel every threadgroup reads the base's operands while others
    // already write C and C', and stages T and B while others write C — so
    // C' may alias no base operand, and nothing of the rank-r instruction
    // may alias C but its addend, which is C exactly.
    for (size_t t = 0; t < lex->count && legal; ++t) {
      const OperandExtent& r = lex->ranges[t];
      if (r.write) {
        for (size_t u = 0; u < bex->count; ++u)
          if (overlap(r, bex->ranges[u])) legal = false;
      } else if (overlap(r, c) && !(r.off == c.off && r.bytes == c.bytes)) {
        legal = false;
      }
    }
    // Between the two: nothing writes a base operand (nothing touches C —
    // the base is the nearest instruction that does).
    for (size_t k = i + 1; k < j && legal; ++k) {
      const InstructionExtents* ex = ExtentsOf(program[k]);
      if (!ex) {
        legal = false;
        break;
      }
      for (size_t t = 0; t < ex->count && legal; ++t) {
        const OperandExtent& r = ex->ranges[t];
        if (!r.write) continue;
        for (size_t u = 0; u < bex->count; ++u)
          if (!bex->ranges[u].write && overlap(r, bex->ranges[u]))
            legal = false;
      }
    }
    if (!legal) continue;
    plan.role[i] = kRoleDeferred;
    plan.partner[i] = static_cast<int32_t>(j);
    plan.role[j] = static_cast<int32_t>(i);
    ++plan.pairs;
    // The base value's store is dropped when the program never reads C
    // again: the first instruction after j to touch its extent writes all
    // of it, reads none of it and overwrites what it writes
    // (OverwritesWhatItWrites), or there is none. A read, a write of part
    // of it, or a read-modify-write keeps the store. (The compiler's
    // addend fuser proves one reader for the rank-r product it folds, not
    // for the addend; this scan is the proof, and the engine's own reads
    // after a program — the loss slot, the probabilities, the deltas, the
    // persistent segment — are of slots no base product occupies.)
    bool dead = true;
    for (size_t k = j + 1; k < program.size(); ++k) {
      const InstructionExtents* ex = ExtentsOf(program[k]);
      if (!ex) {
        dead = false;
        break;
      }
      bool touched = false, read = false, covered = false;
      for (size_t t = 0; t < ex->count; ++t) {
        const OperandExtent& r = ex->ranges[t];
        if (!overlap(r, c)) continue;
        touched = true;
        if (!r.write)
          read = true;
        else if (covers(r, c))
          covered = true;
      }
      if (!touched) continue;
      dead = covered && !read &&
             OverwritesWhatItWrites(
                 static_cast<up::OpCode>(program[k].opcode));
      break;
    }
    plan.drop_c[j] = dead ? 1 : 0;
  }
  return plan;
}

// The fused dispatch: the base GEMM's arguments exactly as Encode builds
// them (the same tile, flags and geometry, so the base part computes the
// same bits), plus the rank-r instruction's operands for the epilogue.
std::expected<void, std::string> MetalBackend::EncodeLowRank(
    const up::UpdateInstruction& base, const up::UpdateInstruction& lr,
    bool drop_c) {
  KArgs a;
  auto ref = [&](int slot, uint64_t r) {
    a.off[slot] = up::RefOffset(r);
    if (up::IsRodataRef(r)) a.space |= 1u << slot;
    if (up::IsSourceRef(r)) a.space |= 1u << (8 + slot);
  };
  auto u32 = [](uint64_t v) { return static_cast<uint32_t>(v); };
  ref(0, base.in[0]);
  ref(1, base.in[1]);
  ref(2, base.in[2]);
  a.m = u32(base.out[0]);
  a.n = u32(base.out[1]);
  a.k = u32(base.out[2]);
  // DispatchGemm routes a skinny shape to pipe + offset, which from the
  // low-rank pipes is some other kernel: PlanLowRank never plans one, and
  // this refuses rather than dispatch it.
  if (a.m <= kGemmSkinnyDim || a.n <= kGemmSkinnyDim || a.k <= kGemmSkinnyDim)
    return std::unexpected("the low-rank fold planned a skinny base GEMM");
  Pipe pipe;
  uint32_t b_elem;
  if (static_cast<up::OpCode>(base.opcode) == up::OpCode::kGemmNN) {
    if (base.flags & up::kFlagEpilogueBias) ref(3, base.in[3]);
    a.f[0] = up::GemmAlphaOf(base.imm);
    a.flags = static_cast<uint32_t>(up::EpilogueActOf(base.flags)) |
              ((base.flags & up::kFlagEpilogueBias) ? 8u : 0u);
    if (base.flags & up::kFlagGemmAddend) {
      ref(3, base.in[3]);
      a.flags |= 2048u;
    }
    pipe = kPGemmNNLR;
    b_elem = 4;
  } else {
    a.flags = static_cast<uint32_t>(up::EpilogueActOf(base.flags));
    if (base.flags & up::kFlagQ8ColScale) {
      ref(3, base.in[3]);
      a.f[0] = 1.0f;
      a.flags |= 128u;
    } else {
      a.f[0] = BitsToF32(base.in[3]);
    }
    pipe = kPGemmNNQ8LR;
    b_elem = 1;
  }
  a.lr_off[0] = up::RefOffset(lr.in[0]);
  a.lr_off[1] = up::RefOffset(lr.in[1]);
  a.lr_off[2] = up::RefOffset(lr.in[2]);
  a.lr_k = u32(lr.out[2]);
  a.lr_alpha = up::GemmAlphaOf(lr.imm);
  if (drop_c) a.flags |= 8192u;  // the base value is not stored
  // C' moves as 16-byte vectors where its offset and N allow.
  if (a.lr_off[2] % 16 == 0 && a.n % 4 == 0) a.flags |= 16384u;
  DispatchGemm(pipe, a, false, false, b_elem);
  return {};
}

// The relaxed family (plan v18): true when the instruction carries
// kFlagRelaxed, the Metal 4 library built, and the shape is one the tile
// serves — then two dispatches (k_to_bf16, the tensor-op GEMM with its
// epilogue) replace the exact kernels and the caller returns. Skinny
// shapes (the exact dispatcher's per-thread forms) and batched GEMMs stay
// exact: the permission is the compiler's, the choice the backend's.
bool MetalBackend::EncodeRelaxedGemm(up::OpCode op,
                                     const up::UpdateInstruction& ins,
                                     KArgs a) {
  if (!(ins.flags & up::kFlagRelaxed) || !relaxed_) return false;
  if (a.batch > 1 || a.m <= kGemmSkinnyDim || a.n <= kGemmSkinnyDim ||
      a.k <= kGemmSkinnyDim)
    return false;
  if (a.f[0] != 1.0f) return false;  // v19 alpha: the validator refuses it here
  const bool nt = op == up::OpCode::kGemmNT || op == up::OpCode::kGemmNTQ8 ||
                  op == up::OpCode::kGemmNTBF16;
  RxPipe pipe;
  switch (op) {
    case up::OpCode::kGemmNN:     pipe = kRxNNF32; break;
    case up::OpCode::kGemmNT:     pipe = kRxNTF32; break;
    case up::OpCode::kGemmNNQ8:   pipe = kRxNNQ8; break;
    case up::OpCode::kGemmNTQ8:   pipe = kRxNTQ8; break;
    case up::OpCode::kGemmNNBF16: pipe = kRxNNBF16; break;
    case up::OpCode::kGemmNTBF16: pipe = kRxNTBF16; break;
    default: return false;
  }
  if (a.lda == 0) a.lda = a.k;
  if (a.ldb == 0) a.ldb = nt ? a.k : a.n;
  if (a.ldc == 0) a.ldc = a.n;
  // The exact path's flag words: 8 bias, 128 NN column scales, 256 NT
  // reduction scales (folded here into the conversion, 512), and the
  // addend the exact path adds afterwards (here 1024, in the epilogue).
  uint32_t conv_flags = 0;
  if (a.flags & 256u) {
    conv_flags = 512u;
    a.flags &= ~256u;
  }
  if (ins.flags & up::kFlagGemmAddend) {
    a.off[3] = up::RefOffset(ins.in[3]);
    if (up::IsRodataRef(ins.in[3])) a.space |= 1u << 3;
    a.flags |= 1024u;
  }
  const uint64_t need = uint64_t{a.m} * a.k * 2;
  if (need > bf16_scratch_bytes_) {
    const uint64_t bytes = std::max(need, std::max<uint64_t>(
                                              bf16_scratch_bytes_ * 2, 1 << 20));
    bf16_scratch_ = [device_ newBufferWithLength:bytes
                                         options:MTLResourceStorageModePrivate];
    bf16_scratch_bytes_ = bf16_scratch_ ? bytes : 0;
    if (!bf16_scratch_) return false;  // no scratch: the exact kernels
  }
  [enc_ setBuffer:bf16_scratch_ offset:0 atIndex:6];
  KArgs conv = a;
  conv.flags = conv_flags;
  NoteDispatchName(kRxPipeNames[kRxToBf16]);
  [enc_ setComputePipelineState:rx_pipes_[kRxToBf16]];
  [enc_ setBytes:&conv length:sizeof(conv) atIndex:2];
  const uint64_t elems = uint64_t{a.m} * a.k;
  [enc_ dispatchThreadgroups:MTLSizeMake((elems + kElementwiseGroup - 1) /
                                             kElementwiseGroup,
                                         1, 1)
       threadsPerThreadgroup:MTLSizeMake(kElementwiseGroup, 1, 1)];
  NoteDispatchName(kRxPipeNames[pipe]);
  [enc_ setComputePipelineState:rx_pipes_[pipe]];
  [enc_ setBytes:&a length:sizeof(a) atIndex:2];
  [enc_ dispatchThreadgroups:MTLSizeMake((a.n + kRxTileN - 1) / kRxTileN,
                                         (a.m + kRxTileM - 1) / kRxTileM, 1)
       threadsPerThreadgroup:MTLSizeMake(kRxThreads, 1, 1)];
  ++relaxed_executions_;
  return true;
}

// The tiled attention kernels read a head's rows as float4 (flag 1) when
// every operand's byte offset is 16-byte aligned and the head width and
// the row stride keep every row so.
static uint32_t TiledRowsVectorizable(const KArgs& a) {
  if (a.D % 4 != 0 || (a.H * a.D) % 4 != 0) return 0;
  for (int i = 0; i < 6; ++i)
    if (a.off[i] % 16 != 0) return 0;
  return 1u;
}

std::expected<void, std::string> MetalBackend::Encode(
    const up::UpdateInstruction& ins, const StepParams& params) {
  KArgs a;
  auto ref = [&](int slot, uint64_t r) {
    a.off[slot] = up::RefOffset(r);
    if (up::IsRodataRef(r)) a.space |= 1u << slot;
    if (up::IsSourceRef(r)) a.space |= 1u << (8 + slot);  // v17
  };
  auto u32 = [](uint64_t v) { return static_cast<uint32_t>(v); };
  auto hi = [](uint64_t v) { return static_cast<uint32_t>(v >> 32); };
  auto lo = [](uint64_t v) { return static_cast<uint32_t>(v & 0xFFFFFFFFu); };
  // The clip factor min(1, max/||g||) for the gradient in ref slot
  // `g_slot` of `args`, left in scratch[256] — kClipNorm's first two
  // dispatches, shared with the fused-clip steps (v12), whose `word` is
  // zero when they carry no clip. Runs on a copy: the clip kernels read g
  // through slot 0 and their own f[0]. The partial sums follow the CPU's
  // chunk geometry (a pure function of n), so they combine in the order
  // the reference combines them.
  auto EncodeClipScale = [&](const KArgs& args, uint64_t word,
                             int g_slot = 1) {
    if (word == 0) return false;
    KArgs c;
    c.off[0] = args.off[g_slot];
    c.space = (args.space >> g_slot) & 1u;
    c.n = args.n;
    c.f[0] = BitsToF32(word);
    c.k = static_cast<uint32_t>(
        up::ParallelChunkGrain(c.n, kernels::kGrainCheap));
    c.m = static_cast<uint32_t>(
        up::ParallelChunkCount(c.n, kernels::kGrainCheap));
    Dispatch(kPClipPartials, c, c.m, kElementwiseGroup, true);
    Dispatch(kPClipFinish, c, 1, 1, true);
    return true;
  };
  // The GEMM addend (v14): since F7 (#138) step 3 the exact kernels sum D
  // into their write-back (flag 2048, slot 3) — `d + s` over the finished
  // dot product, the bits of the GEMM + kAddEW pair this used to dispatch
  // on this backend, without the second dispatch and its pass over C. The
  // relaxed path sets its own flag (1024) for the same sum.
  auto FoldAddend = [&]() {
    if (!(ins.flags & up::kFlagGemmAddend)) return;
    ref(3, ins.in[3]);
    a.flags |= 2048u;
  };
  const auto op = static_cast<up::OpCode>(ins.opcode);
  switch (op) {
    case up::OpCode::kGemmNN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      if (ins.flags & up::kFlagEpilogueBias) ref(3, ins.in[3]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = up::GemmAlphaOf(ins.imm);  // v19: the write-back's alpha
      // flags: bit 3 = bias present, low bits = EpilogueAct (1..3)
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags)) |
                ((ins.flags & up::kFlagEpilogueBias) ? 8u : 0u);
      if (EncodeRelaxedGemm(op, ins, a)) return {};
      FoldAddend();
      DispatchGemm(kPGemmNN, a, false, false, 4);
      return {};
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = up::GemmAlphaOf(ins.imm);
      if (op == up::OpCode::kGemmNT && EncodeRelaxedGemm(op, ins, a))
        return {};
      FoldAddend();
      if (op == up::OpCode::kGemmNT)
        DispatchGemm(kPGemmNT, a, false, true, 4);
      else
        DispatchGemm(kPGemmTN, a, true, false, 4);
      return {};
    case up::OpCode::kGemmAccNN:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = BitsToF32(ins.in[3]);
      DispatchGemm(kPGemmAcc, a, false, false, 4);
      return {};
    case up::OpCode::kGemmNNBF16:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      if (ins.flags & up::kFlagEpilogueBias) ref(3, ins.in[3]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags)) |
                ((ins.flags & up::kFlagEpilogueBias) ? 8u : 0u);
      if (EncodeRelaxedGemm(op, ins, a)) return {};
      DispatchGemm(kPGemmNNBF16, a, false, false, 2);
      return {};
    case up::OpCode::kGemmNTBF16:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.f[0] = 1.0f;
      if (EncodeRelaxedGemm(op, ins, a)) return {};
      DispatchGemm(kPGemmNTBF16, a, false, true, 2);
      return {};
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.m = u32(ins.out[0]); a.n = u32(ins.out[1]); a.k = u32(ins.out[2]);
      a.flags = static_cast<uint32_t>(up::EpilogueActOf(ins.flags));
      if (ins.flags & up::kFlagQ8ColScale) {
        // v17: per-column scales in slot 3 — the NN form scales output
        // columns in the epilogue (128), the NT form scales B as it widens.
        ref(3, ins.in[3]);
        a.f[0] = 1.0f;
        a.flags |= op == up::OpCode::kGemmNNQ8 ? 128u : 256u;
      } else {
        a.f[0] = BitsToF32(ins.in[3]);  // the per-tensor dequant scale
      }
      if (EncodeRelaxedGemm(op, ins, a)) return {};
      if (op == up::OpCode::kGemmNNQ8)
        DispatchGemm(kPGemmNNQ8, a, false, false, 1);
      else
        DispatchGemm(kPGemmNTQ8, a, false, true, 1);
      return {};
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kAddEW ? kPAddEW : kPMulEW, a, a.n,
               kElementwiseGroup);
      return {};
    case up::OpCode::kAddBias:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      Dispatch(kPAddBias, a, a.rows * a.cols, kElementwiseGroup);
      return {};
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kReluFwd   ? kPReluFwd
               : op == up::OpCode::kGeluFwd ? kPGeluFwd
                                            : kPSiluFwd,
               a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(op == up::OpCode::kReluBwd   ? kPReluBwd
               : op == up::OpCode::kGeluBwd ? kPGeluBwd
                                            : kPSiluBwd,
               a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kScale:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.f[0] = BitsToF32(ins.in[2]);
      a.n = u32(ins.out[0]);
      Dispatch(kPScale, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kFill:
      ref(0, ins.in[0]);
      a.f[0] = BitsToF32(ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPFill, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kCopy:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPCopy, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kAccumulate:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      Dispatch(kPAccumulate, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kFusedMap:
      ref(0, ins.in[0]); ref(3, ins.in[3]);
      if (ins.in[1] != up::kNullRef) ref(1, ins.in[1]);
      if (ins.in[2] != up::kNullRef) ref(2, ins.in[2]);
      a.n = u32(ins.out[0]);
      a.flags = u32(ins.out[1]);
      a.f[0] = BitsToF32(ins.out[2]);
      a.f[1] = BitsToF32(ins.out[2] >> 32);
      Dispatch(kPFusedMap, a, a.n, kElementwiseGroup);
      return {};
    case up::OpCode::kSgdStep: {
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      const bool clip = EncodeClipScale(a, ins.out[1]);
      a.f[0] = params.lr; a.f[1] = params.weight_decay;
      Dispatch(clip ? kPSgdClip : kPSgd, a, a.n, kElementwiseGroup, clip);
      return {};
    }
    case up::OpCode::kAdamWStep: {
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.n = u32(ins.out[0]);
      const bool clip = EncodeClipScale(a, ins.out[1]);
      // The per-step factors exactly as the CPU kernel hoists them.
      const float step = static_cast<float>(params.step);
      a.f[0] = params.lr; a.f[1] = params.beta1; a.f[2] = params.beta2;
      a.f[3] = params.eps; a.f[4] = params.weight_decay;
      a.f[5] = 1.0f / (1.0f - std::pow(params.beta1, step));
      a.f[6] = 1.0f / (1.0f - std::pow(params.beta2, step));
      Dispatch(clip ? kPAdamWClip : kPAdamW, a, a.n, kElementwiseGroup, clip);
      return {};
    }
    case up::OpCode::kReduceRows:
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      Dispatch(kPReduceRows, a, a.cols, kElementwiseGroup);
      return {};
    case up::OpCode::kLayerNormFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[1]); ref(5, ins.out[2]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      a.f[0] = up::NormEpsOf(ins.imm);  // v16; 1e-5 by default
      Dispatch(kPLnFwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kLayerNormBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]); ref(5, ins.out[1]);
      a.rows = hi(ins.out[2]); a.cols = lo(ins.out[2]);
      Dispatch(kPLnBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kRmsNormFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      a.f[0] = up::NormEpsOf(ins.imm);
      Dispatch(kPRmsFwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kRmsNormBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]);
      a.rows = hi(ins.out[1]); a.cols = lo(ins.out[1]);
      Dispatch(kPRmsBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    case up::OpCode::kClipNorm: {
      // CR1 #142 F13 / F18: a threshold that is not a finite positive float
      // is refused before anything is encoded — a zero word skipped the
      // norm dispatches and k_clip_apply multiplied by stale scratch.
      const float cap = BitsToF32(ins.in[1]);
      if (!std::isfinite(cap) || !(cap > 0.0f))
        return std::unexpected(kClipThresholdRefusal);
      ref(0, ins.in[0]);
      a.n = u32(ins.out[0]);
      EncodeClipScale(a, ins.in[1], /*slot=*/0);
      Dispatch(kPClipApply, a, a.n, kElementwiseGroup, true);
      return {};
    }
    case up::OpCode::kSqNormAccum: {  // v24, #190: the CPU's chunk geometry
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      a.k = static_cast<uint32_t>(
          up::ParallelChunkGrain(a.n, kernels::kGrainCheap));
      a.m = static_cast<uint32_t>(
          up::ParallelChunkCount(a.n, kernels::kGrainCheap));
      Dispatch(kPClipPartials, a, a.m, kElementwiseGroup, true);
      Dispatch(kPSqNormFinish, a, 1, 1, true);
      return {};
    }
    case up::OpCode::kClipGlobal: {  // v24, #190: kClipNorm's refusals
      const float cap = BitsToF32(ins.in[2]);
      if (!std::isfinite(cap) || !(cap > 0.0f))
        return std::unexpected(kClipThresholdRefusal);
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.n = u32(ins.out[0]);
      a.f[0] = cap;
      Dispatch(kPClipGlobalScale, a, 1, 1, true);
      Dispatch(kPClipApply, a, a.n, kElementwiseGroup, true);
      return {};
    }
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd: {
      ref(0, ins.in[0]); ref(1, ins.in[1]);
      a.B = hi(ins.out[0]); a.S = lo(ins.out[0]);
      a.H = hi(ins.out[1]); a.D = lo(ins.out[1]);
      const float base = BitsToF32(ins.out[2]);
      a.f[0] = std::pow(base, -2.0f / static_cast<float>(a.D));
      // v12: slot 2 names the angle table (built by k_rope_table with this
      // same recurrence step, so reading it changes no bit); flag 1 says so.
      if (ins.in[2] != up::kNullRef) {
        ref(2, ins.in[2]);
        a.flags = 1u;
      }
      Dispatch(op == up::OpCode::kRopeFwd ? kPRopeFwd : kPRopeBwd, a,
               a.B * a.S * a.H, kElementwiseGroup);
      return {};
    }
    case up::OpCode::kAttnFwdTiled: {
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]);
      a.B = hi(ins.out[1]); a.S = lo(ins.out[1]);
      a.H = hi(ins.out[2]); a.D = lo(ins.out[2]);
      a.f[0] = 1.0f / std::sqrt(static_cast<float>(a.D));
      a.flags = TiledRowsVectorizable(a);
      // The relaxed form (F3 design item 2, plan v20) when the plan permits
      // it, the products on simdgroup_matrix tiles, 32 rows a threadgroup;
      // else the blocked form (F3 step 2) for head widths the 64-float row
      // registers cover; wider heads take the per-row form either way.
      if (a.D <= 64u && (ins.flags & up::kFlagRelaxed)) {
        Dispatch(kPAttnFwdTiledMma, a,
                 a.B * a.H * ((a.S + 31u) / 32u) * 128u, 128u);
        ++relaxed_executions_;
      } else if (a.D <= 64u) {
        Dispatch(kPAttnFwdTiledBlk, a,
                 a.B * a.H * ((a.S + 127u) / 128u) * 128u, 128u);
      } else {
        Dispatch(kPAttnFwdTiled, a, a.B * a.H * a.S * 32u, kElementwiseGroup);
      }
      return {};
    }
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled: {
      const auto geo = up::UnpackAttnGeometry(ins.out[2]);
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      ref(4, ins.out[0]); ref(5, ins.out[1]);
      a.B = u32(geo.B); a.S = u32(geo.S); a.H = u32(geo.H); a.D = u32(geo.d);
      a.f[0] = 1.0f / std::sqrt(static_cast<float>(a.D));
      a.flags = TiledRowsVectorizable(a);
      if (a.D <= 64u && (ins.flags & up::kFlagRelaxed)) {
        Dispatch(op == up::OpCode::kAttnDQTiled   ? kPAttnDQTiledMma
                 : op == up::OpCode::kAttnDKTiled ? kPAttnDKTiledMma
                                                  : kPAttnDVTiledMma,
                 a, a.B * a.H * ((a.S + 31u) / 32u) * 128u, 128u);
        ++relaxed_executions_;
      } else if (a.D <= 64u) {
        Dispatch(op == up::OpCode::kAttnDQTiled   ? kPAttnDQTiledBlk
                 : op == up::OpCode::kAttnDKTiled ? kPAttnDKTiledBlk
                                                  : kPAttnDVTiledBlk,
                 a, a.B * a.H * ((a.S + 127u) / 128u) * 128u, 128u);
      } else {
        Dispatch(op == up::OpCode::kAttnDQTiled   ? kPAttnDQTiled
                 : op == up::OpCode::kAttnDKTiled ? kPAttnDKTiled
                                                  : kPAttnDVTiled,
                 a, a.B * a.H * a.S * 32u, kElementwiseGroup);
      }
      return {};
    }
    case up::OpCode::kRopeTable: {
      ref(0, ins.in[0]);
      a.S = hi(ins.out[0]); a.D = lo(ins.out[0]);
      const float base = BitsToF32(ins.out[1]);
      a.f[0] = std::pow(base, -2.0f / static_cast<float>(a.D));
      Dispatch(kPRopeTable, a, a.S, kElementwiseGroup);
      return {};
    }
    case up::OpCode::kEmbedFwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.rows = u32(ins.out[0]); a.cols = lo(ins.out[1]); a.k = hi(ins.out[1]);
      Dispatch(kPEmbedFwd, a, a.rows * a.cols, kElementwiseGroup);
      return {};
    // The attention family (docs/runtime.md): each primitive is one or
    // more batched GEMMs over the [B*S, H*d] activations plus a row
    // softmax kernel — the same tile machinery as the projections.
    // Operand kinds: 0 = activation [B*S, H*d] (per-batch S*D, per-head
    // d), 1 = probability matrix [B*H*S, S] (per-batch H*S*S, per-head
    // S*S).
    case up::OpCode::kAttnFwd: {
      const uint32_t B = hi(ins.out[1]), S = lo(ins.out[1]);
      const uint32_t H = hi(ins.out[2]), d = lo(ins.out[2]);
      const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
      // scores = (Q K^T) / sqrt(d) into the P cache, then a causal softmax
      // in place, then O = P V.
      EncodeAttentionGemm(kPGemmNT, false, true, ins.in[0], ins.in[1],
                          ins.out[0], B, S, H, d, 0, 0, 1, S, S, d,
                          inv_sqrt_d);
      ref(0, ins.out[0]);
      a.rows = B * H * S; a.cols = S;
      Dispatch(kPAttnSoftmax, a, a.rows * 32u, kElementwiseGroup);
      EncodeAttentionGemm(kPGemmNN, false, false, ins.out[0], ins.in[2],
                          ins.in[3], B, S, H, d, 1, 0, 0, S, d, S, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDP: {  // dP = dO V^T
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmNT, false, true, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 0, 0, 1, S, S, d, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDV: {  // dV = P^T dO
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmTN, true, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S, 1.0f);
      return {};
    }
    case up::OpCode::kAttnDQ: {  // dQ = (dS K) / sqrt(d)
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmNN, false, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S,
                          1.0f / std::sqrt(static_cast<float>(d)));
      return {};
    }
    case up::OpCode::kAttnDK: {  // dK = (dS^T Q) / sqrt(d)
      const uint32_t B = hi(ins.out[0]), S = lo(ins.out[0]);
      const uint32_t H = hi(ins.out[1]), d = lo(ins.out[1]);
      EncodeAttentionGemm(kPGemmTN, true, false, ins.in[0], ins.in[1],
                          ins.in[2], B, S, H, d, 1, 0, 0, S, d, S,
                          1.0f / std::sqrt(static_cast<float>(d)));
      return {};
    }
    case up::OpCode::kSoftmaxRowsBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]);
      a.rows = hi(ins.out[0]); a.cols = lo(ins.out[0]);
      Dispatch(kPSoftmaxRowsBwd, a, a.rows * 32u, kElementwiseGroup);
      return {};
    // Softmax cross-entropy (F8): the row kernel leaves -log p[label] per
    // row in scratch, the finish kernel folds them into the loss slot.
    case up::OpCode::kSoftmaxXEntFwd: {
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      if (a.rows > scratch_floats_) {
        // The scratch is bound per encoder: end this one, grow, rebind.
        if (auto r = Flush(); !r) return r;
        scratch_ = [device_ newBufferWithLength:size_t{a.rows} * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        if (!scratch_) return std::unexpected("scratch allocation failed");
        scratch_floats_ = a.rows;
        if (auto r = EnsureEncoder(); !r) return r;
      }
      Dispatch(kPXentFwd, a, a.rows * 32u, kElementwiseGroup, true);
      Dispatch(kPXentFinish, a, 1, 1, true);
      return {};
    }
    case up::OpCode::kSoftmaxXEntBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      Dispatch(kPXentBwd, a, a.rows * a.cols, kElementwiseGroup);
      return {};
    // The masked pair (S3 #149): the row kernel leaves -log p[label] or 0
    // in scratch, the finish kernel counts the target rows and divides;
    // the backward reads the count the count kernel leaves in scratch[0].
    case up::OpCode::kSoftmaxXEntMaskedFwd: {
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      if (a.rows > scratch_floats_) {
        if (auto r = Flush(); !r) return r;
        scratch_ = [device_ newBufferWithLength:size_t{a.rows} * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        if (!scratch_) return std::unexpected("scratch allocation failed");
        scratch_floats_ = a.rows;
        if (auto r = EnsureEncoder(); !r) return r;
      }
      Dispatch(kPXentMaskedFwd, a, a.rows * 32u, kElementwiseGroup, true);
      Dispatch(kPXentMaskedFinish, a, 1, 1, true);
      return {};
    }
    case up::OpCode::kSoftmaxXEntMaskedBwd:
      ref(0, ins.in[0]); ref(1, ins.in[1]); ref(2, ins.in[2]); ref(3, ins.in[3]);
      a.rows = u32(ins.out[0]); a.cols = u32(ins.out[1]);
      // scratch_ holds kScratchFloats floats from creation and only grows,
      // so scratch[0] for the count is always there.
      Dispatch(kPXentMaskedCount, a, 1, 1, true);
      Dispatch(kPXentMaskedBwd, a, a.rows * a.cols, kElementwiseGroup, true);
      return {};
    default:
      return std::unexpected("opcode " + std::to_string(ins.opcode) +
                             " has no GPU kernel");
  }
}

void MetalBackend::EncodeAttentionGemm(
    Pipe pipe, bool at, bool bt, uint64_t a_ref, uint64_t b_ref,
    uint64_t c_ref, uint32_t B, uint32_t S, uint32_t H, uint32_t d,
    uint32_t a_kind, uint32_t b_kind, uint32_t c_kind, uint32_t m,
    uint32_t n, uint32_t k, float alpha) {
  KArgs a;
  const uint64_t D = uint64_t{H} * d;
  auto set = [&](int slot, uint64_t r, uint32_t kind, uint32_t& ld) {
    a.off[slot] = up::RefOffset(r);
    if (up::IsRodataRef(r)) a.space |= 1u << slot;
    if (kind == 0) {
      a.bs[2 * slot] = uint64_t{S} * D;
      a.bs[2 * slot + 1] = d;
      ld = static_cast<uint32_t>(D);
    } else {
      a.bs[2 * slot] = uint64_t{H} * S * S;
      a.bs[2 * slot + 1] = uint64_t{S} * S;
      ld = S;
    }
  };
  set(0, a_ref, a_kind, a.lda);
  set(1, b_ref, b_kind, a.ldb);
  set(2, c_ref, c_kind, a.ldc);
  a.m = m; a.n = n; a.k = k;
  a.batch = B * H; a.batch_h = H;
  a.f[0] = alpha;
  DispatchGemm(pipe, a, at, bt, 4);
}

std::expected<void, std::string> MetalBackend::Execute(
    const up::UpdateInstruction& ins, const StepParams& params) {
  @autoreleasepool {
    if (!arena_buf_) return std::unexpected("no plan bound");
    const auto op = static_cast<up::OpCode>(ins.opcode);
    if (profile_) {
      if (op == up::OpCode::kSoftmaxXEntFwd ||
          op == up::OpCode::kSoftmaxXEntMaskedFwd ||
          op == up::OpCode::kMseFwd || op == up::OpCode::kKLDistillFwd)
        ++fwd_passes_;
      else if (op == up::OpCode::kSoftmaxXEntBwd ||
               op == up::OpCode::kSoftmaxXEntMaskedBwd ||
               op == up::OpCode::kMseBwd || op == up::OpCode::kKLDistillBwd)
        ++bwd_passes_;
    }
    const InstructionExtents* ex = ExtentsOf(ins);
    if (!ex && op != up::OpCode::kNop)
      return std::unexpected("instruction " + std::to_string(ins.opcode) +
                             " failed re-validation at dispatch: " +
                             revalidation_error_);
    // A reader of the source model runs where the model is: on cpu_ when
    // BindSource kept it off the GPU (source_on_cpu_) — and so does every
    // instruction of a range that holds one (range_on_cpu_, BeginRange).
    const bool reads_source =
        source_on_cpu_ && (range_on_cpu_ || (ex && ReadsSource(*ex)));
    // F7 #138 step 4: a base GEMM deferred into the rank-r GEMM that
    // reads it (PlanLowRank) dispatches nothing here and runs there, fused.
    int32_t role = kRoleNone;
    size_t idx = 0;
    if (plan_ && !program_.empty() && &ins >= program_.data() &&
        &ins < program_.data() + program_.size()) {
      idx = static_cast<size_t>(&ins - program_.data());
      role = plan_->role[idx];
      // A pair the range the engine runs does not hold whole runs apart —
      // both halves decide it from the same bounds, so the base dispatches
      // at its own place in the earlier range and the rank-r GEMM encodes
      // alone in the later one.
      if (role == kRoleDeferred &&
          static_cast<size_t>(plan_->partner[idx]) >= end_)
        role = kRoleNone;
      else if (role >= 0 && static_cast<size_t>(role) < begin_)
        role = kRoleNone;
    }
    if (role == kRoleDeferred) return {};
    if (!IsGpuOpcode(op) || !DimsFit32(op, ins) || reads_source) {
      if (role >= 0)
        return std::unexpected("a fused low-rank pair fell to the CPU path");
      // CPU-resident: wait for any pending GPU work it depends on.
      if (ex && HazardWithPending(*ex))
        if (auto r = Flush(); !r) return r;
      if (!profile_) return cpu_->Execute(ins, params);
      const auto t0 = std::chrono::steady_clock::now();
      auto r = cpu_->Execute(ins, params);
      ProfRow& row = prof_["cpu opcode " + std::to_string(ins.opcode)];
      row.seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
      row.count++;
      return r;
    }
    if (auto r = EnsureEncoder(); !r) return r;
    prof_pipes_.clear();
    if (role >= 0) {
      const up::UpdateInstruction& base = program_[static_cast<size_t>(role)];
      const bool drop_c = plan_->drop_c[idx] != 0;
      if (auto r = EncodeLowRank(base, ins, drop_c); !r) return r;
      if (const InstructionExtents* bex = ExtentsOf(base)) RecordPending(*bex);
      ++lowrank_fused_;
      if (drop_c) ++lowrank_dropped_;
    } else if (auto r = Encode(ins, params); !r) {
      return r;
    }
    RecordPending(*ex);
    if (profile_) {
      if (auto r = Flush(); !r) return r;
      auto dim = [](uint64_t v) {
        return v >> 32 ? std::to_string(v >> 32) + "/" +
                             std::to_string(v & 0xFFFFFFFFu)
                       : std::to_string(v);
      };
      const up::UpdateInstruction& keyed =
          role >= 0 ? program_[static_cast<size_t>(role)] : ins;
      ProfRow& row = prof_[prof_pipes_ + " " + dim(keyed.out[0]) + "x" +
                           dim(keyed.out[1]) + "x" + dim(keyed.out[2])];
      row.seconds += last_gpu_seconds_;
      row.count++;
    }
    return {};
  }
}

std::expected<void, std::string> MetalBackend::Flush() {
  @autoreleasepool {
    if (!enc_) return {};
    [enc_ endEncoding];
    [cmd_ commit];
    [cmd_ waitUntilCompleted];
    const MTLCommandBufferStatus status = cmd_.status;
    NSError* err = cmd_.error;
    last_gpu_start_ = cmd_.GPUStartTime;
    last_gpu_end_ = cmd_.GPUEndTime;
    const uint64_t dispatches = cb_dispatches_;
    if (profile_) last_gpu_seconds_ = cmd_.GPUEndTime - cmd_.GPUStartTime;
    if (spans_)
      spans_rows_.push_back(
          {cb_dispatches_, cmd_.GPUEndTime - cmd_.GPUStartTime,
           std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         cb_t0_)
               .count()});
    enc_ = nil;
    cmd_ = nil;
    pending_reads_.clear();
    pending_writes_.clear();
    // Anything short of Completed means the arena holds a mixture of
    // finished and unfinished work: refuse to continue on it.
    if (status != MTLCommandBufferStatusCompleted)
      return std::unexpected(NsError(err, "GPU command buffer did not complete"));
    // Completed with no execution time, on a GPU that reports one for every
    // buffer that ran: the device declined the work — the silent form of a
    // working set it cannot hold (F5 #133). Refuse rather than read zeros.
    if (gpu_timestamps_ && dispatches > 0 &&
        !(last_gpu_start_ > 0 && last_gpu_end_ > last_gpu_start_))
      return std::unexpected(
          "GPU command buffer completed without executing its " +
          std::to_string(dispatches) + " dispatch(es): the device declined "
          "the work (resident: arena " + std::to_string(arena_bytes_) +
          " B, rodata " + std::to_string(rodata_bytes_) + " B, source " +
          std::to_string(source_buf_ ? source_bytes_ : 0) +
          " B; this GPU's working set is " +
          std::to_string(working_set_limit_) + " B)");
    // A clip in this buffer saw a non-finite norm (CR1 #142 F11): the
    // kernels after it wrote nothing, and the verdict is the CPU's, word
    // for word. The flag is the buffer's; clear it for the next.
    if (clip_flag_) {
      float* flag = static_cast<float*>(clip_flag_.contents);
      if (*flag != 0.0f) {
        *flag = 0.0f;
        return std::unexpected(kNonFiniteNormRefusal);
      }
    }
    return {};
  }
}

void MetalBackend::PrintProfile() const {
  std::vector<std::pair<std::string, ProfRow>> rows(prof_.begin(), prof_.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.second.seconds > b.second.seconds;
  });
  double total = 0;
  for (const auto& r : rows) total += r.second.seconds;
  std::fprintf(stderr,
               "metal profile: %.3f s over %zu keys; %llu forward pass(es), "
               "%llu backward pass(es) (counted at the loss opcodes)\n",
               total, rows.size(), (unsigned long long)fwd_passes_,
               (unsigned long long)bwd_passes_);
  if (lowrank_fused_)
    std::fprintf(stderr,
                 "  low-rank epilogue: %llu fused dispatch(es), %llu with C's "
                 "store dropped (k_gemm_*_lr; SEEML_METAL_UNFUSED_LOWRANK=1 "
                 "keeps the pairs apart)\n",
                 (unsigned long long)lowrank_fused_,
                 (unsigned long long)lowrank_dropped_);
  for (const auto& r : rows)
    std::fprintf(stderr, "  %8.3f s  %5.1f%%  %7llu x %8.1f us  %s\n",
                 r.second.seconds,
                 total > 0 ? 100.0 * r.second.seconds / total : 0.0,
                 (unsigned long long)r.second.count,
                 1e6 * r.second.seconds / (double)r.second.count,
                 r.first.c_str());
}

void MetalBackend::PrintSpans() const {
  double gpu = 0, wall = 0;
  uint64_t dispatches = 0;
  for (const SpanRow& s : spans_rows_) {
    gpu += s.gpu_seconds;
    wall += s.wall_seconds;
    dispatches += s.dispatches;
  }
  std::fprintf(stderr,
               "metal spans: %zu command buffer(s), %llu dispatch(es); GPU "
               "busy %.3f s, CPU wall %.3f s, outside the GPU %.3f s "
               "(%.1f%% of the wall; %.2f us per dispatch)\n",
               spans_rows_.size(), (unsigned long long)dispatches, gpu, wall,
               wall - gpu, wall > 0 ? 100.0 * (wall - gpu) / wall : 0.0,
               dispatches ? 1e6 * (wall - gpu) / (double)dispatches : 0.0);
  if (lowrank_fused_)
    std::fprintf(stderr,
                 "  low-rank epilogue: %llu fused dispatch(es), %llu with C's "
                 "store dropped\n",
                 (unsigned long long)lowrank_fused_,
                 (unsigned long long)lowrank_dropped_);
  // Per buffer size: the train, eval and merge programs are told apart by
  // their dispatch counts (one buffer per ExecuteRange).
  std::map<uint64_t, std::array<double, 3>> by_size;  // n, gpu, wall
  for (const SpanRow& s : spans_rows_) {
    auto& v = by_size[s.dispatches];
    v[0] += 1; v[1] += s.gpu_seconds; v[2] += s.wall_seconds;
  }
  for (const auto& [n, v] : by_size)
    std::fprintf(stderr,
                 "  %7llu dispatches x %4.0f: GPU %8.3f ms, wall %8.3f ms, "
                 "outside %7.3f ms (%.1f%%) per buffer\n",
                 (unsigned long long)n, v[0], 1e3 * v[1] / v[0],
                 1e3 * v[2] / v[0], 1e3 * (v[2] - v[1]) / v[0],
                 v[2] > 0 ? 100.0 * (v[2] - v[1]) / v[2] : 0.0);
}

}  // namespace

bool MetalBackendAvailable() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend() {
  return MetalBackend::Create();
}

}  // namespace seeml::update_rt
