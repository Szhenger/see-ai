#include "runtime/verifier/plan_validator.h"

#include <bit>
#include <cmath>
#include <string>

#include "runtime/diagnostics/validating/error.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Proves a kFusedMap micro-program (plan v13): 1..4 known stages,
/// zero-terminated with nothing after the terminator; binary stages name
/// operand slot 1 or 2 (recorded in `uses_slot`), scale stages immediate 0
/// or 1, unary stages carry no argument; every immediate a scale names is
/// finite and every one none names is zero.
bool FusedProgramOk(uint64_t stages_word, uint64_t imm_word,
                    bool uses_slot[3]) {
  if (up::HighWord(stages_word) != 0) return false;
  constexpr uint8_t kKnownBits =
      static_cast<uint8_t>(up::kFusedStageKindMask |
                           (up::kFusedStageArgMask << up::kFusedStageArgShift) |
                           up::kFusedStageRunIsRight);
  bool uses_imm[2] = {false, false};
  size_t stages = 0;
  for (; stages < up::kFusedMapMaxStages; ++stages) {
    const auto stage = static_cast<uint8_t>(stages_word >> (8 * stages));
    const up::FusedStage kind = up::FusedStageKind(stage);
    if (kind == up::FusedStage::kEnd) {
      if (stage != 0) return false;
      break;
    }
    if ((stage & ~kKnownBits) != 0) return false;
    const uint8_t arg = up::FusedStageArg(stage);
    if (kind == up::FusedStage::kAdd || kind == up::FusedStage::kMul) {
      if (arg != 1 && arg != 2) return false;
      uses_slot[arg] = true;
    } else if (kind == up::FusedStage::kScale) {
      if (arg > 1 || (stage & up::kFusedStageRunIsRight)) return false;
      uses_imm[arg] = true;
    } else if (kind == up::FusedStage::kRelu || kind == up::FusedStage::kGelu ||
               kind == up::FusedStage::kSilu) {
      if ((stage & ~up::kFusedStageKindMask) != 0) return false;
    } else {
      return false;
    }
  }
  if (stages == 0 || (stages_word >> (8 * stages)) != 0) return false;
  for (int i = 0; i < 2; ++i) {
    const uint32_t bits = static_cast<uint32_t>(imm_word >> (32 * i));
    if (uses_imm[i] ? !std::isfinite(std::bit_cast<float>(bits)) : bits != 0)
      return false;
  }
  return true;
}

/// The load-time proof of one instruction, held as the state its checks
/// share: the instruction, the plan it sits in, and the operand extents the
/// bounds checks record for the overlap proof. Run() is the whole proof —
/// the flags and imm discipline, the version gates, then the opcode's own
/// operand checks, one function per opcode family (S8 #154).
struct InstructionProof {
  InstructionProof(const up::UpdateInstruction& instruction, uint64_t arena,
                   uint64_t rodata, uint32_t version, bool source_ok,
                   InstructionExtents* out)
      : ins(instruction),
        arena_size(arena),
        rodata_size(rodata),
        plan_version(version),
        allow_source(source_ok),
        extents(out),
        d0(ins.out[0]),
        d1(ins.out[1]),
        d2(ins.out[2]) {}

  const up::UpdateInstruction& ins;
  const uint64_t arena_size;
  const uint64_t rodata_size;
  const uint32_t plan_version;
  const bool allow_source;
  InstructionExtents* const extents;
  const uint64_t d0, d1, d2;  // the instruction's three out[] words

  [[nodiscard]] std::expected<void, std::string> Run();

  // The preamble every opcode passes, in this order.
  [[nodiscard]] std::expected<void, std::string> FlagsOk() const;
  [[nodiscard]] std::expected<void, std::string> ImmOk() const;
  [[nodiscard]] std::expected<void, std::string> VersionOk() const;

  // The operand proofs, one per opcode family. Each handles exactly the
  // opcodes Run() routes to it and refuses any other.
  [[nodiscard]] std::expected<void, std::string> Gemm();
  [[nodiscard]] std::expected<void, std::string> Elementwise();
  [[nodiscard]] std::expected<void, std::string> Normalization();
  [[nodiscard]] std::expected<void, std::string> Loss();
  [[nodiscard]] std::expected<void, std::string> OptimizerStep();
  [[nodiscard]] std::expected<void, std::string> RopeAndEmbedding();
  [[nodiscard]] std::expected<void, std::string> AttentionCached();
  [[nodiscard]] std::expected<void, std::string> AttentionTiled();

  // elem_bytes: f32/i32 operands are 4 bytes; quantized weights are 1.
  // A ref is admitted only if its extent is nonzero (every kernel
  // dereferences element 0 unconditionally, so a zero-extent operand is not
  // "harmlessly empty" — it is an unchecked pointer), aligned to its
  // element size (the kernels cast the raw offset to a typed pointer), and
  // in bounds for its address space.
  bool RefOkW(uint64_t ref, uint64_t elems, bool write, uint64_t elem_bytes) {
    if (ref == up::kNullRef) return false;
    if (write && up::IsRodataRef(ref)) return false;
    // A source ref (v17): read-only f32, in the eval program only.
    const bool source = up::IsSourceRef(ref);
    if (source &&
        (write || !allow_source || plan_version < up::kSeeuShippedEvalVersion))
      return false;
    if (elems == 0) return false;
    uint64_t bytes = 0;
    if (!MulOk(elems, elem_bytes, &bytes)) return false;
    // Execute() reinterpret_casts the ref to its element type and
    // dereferences directly, so alignment is part of "safe to dispatch
    // blindly": a misaligned offset is UB, and a bus error on the
    // strict-alignment targets this runtime ships to.
    if (up::RefOffset(ref) % elem_bytes != 0) return false;
    const uint64_t space = source                 ? kSourceSpaceLimit
                           : up::IsRodataRef(ref) ? rodata_size
                                                  : arena_size;
    if (!RangeOk(up::RefOffset(ref), bytes, space)) return false;
    extents->ranges[extents->count++] = {up::RefOffset(ref), bytes, write,
                                         up::IsRodataRef(ref), source};
    return true;
  }
  bool RefOk(uint64_t ref, uint64_t elems, bool write) {
    return RefOkW(ref, elems, write, sizeof(float));
  }
  std::unexpected<std::string> Fail() const {
    return diag::validating::Error(
        "instruction operand out of bounds "
        "(opcode " +
        std::to_string(ins.opcode) + ")");
  }
  // Every kernel is compiled with SEEML_RESTRICT pointers: a written range
  // overlapping any *other* operand of the same instruction is undefined
  // behavior, not a wrong answer. Bounds alone don't rule that out, so each
  // validated operand's byte range is recorded and proven disjoint from
  // every written range before the instruction is accepted. A single ref
  // that is read and written through one pointer (SGD's param, GemmAcc's C)
  // is one operand, not an alias.
  // Accept only if no written range overlaps another operand's range in the
  // same address space. Called in place of a bare success return by every
  // case that records operands.
  [[nodiscard]] std::expected<void, std::string> Disjoint() const {
    const OperandExtent* ranges = extents->ranges;
    const size_t num_ranges = extents->count;
    for (size_t i = 0; i < num_ranges; ++i)
      for (size_t j = i + 1; j < num_ranges; ++j) {
        const OperandExtent& a = ranges[i];
        const OperandExtent& b = ranges[j];
        if (!(a.write || b.write) || a.rodata != b.rodata ||
            a.source != b.source)
          continue;
        if (a.bytes == 0 || b.bytes == 0) continue;
        if (a.off < b.off + b.bytes && b.off < a.off + a.bytes)
          return diag::validating::Error(
              "instruction operands alias a written range (opcode " +
              std::to_string(ins.opcode) + ")");
      }
    return {};
  }
  /// The refusal of an opcode this validator does not know: a load error,
  /// never a skip — Execute() would silently pass over it.
  std::unexpected<std::string> UnknownOpcode() const {
    return diag::validating::Error("unknown opcode " +
                                   std::to_string(ins.opcode));
  }
  // The KL temperature word: the low half is T at every version, and both
  // softmaxes divide by it — a non-positive or non-finite T inverts or
  // dissolves them (CR1 #142 F17 / F10), so it is a finite float within
  // the bounds the compiler and the CLI apply. The high half is the loss
  // scale (v8). Pre-v8 plans never wrote it: a nonzero high word there is
  // corruption, exactly like flags before v5. From v8 on, zero means
  // "absent" (scale 1.0) and any written scale must be a finite positive
  // float — Execute() multiplies by it blindly, and a zero or NaN scale
  // would train on nothing, silently.
  [[nodiscard]] std::expected<void, std::string> KlWordOk(uint64_t word) const {
    const float t = std::bit_cast<float>(static_cast<uint32_t>(word));
    if (!std::isfinite(t) || t < up::kKlTemperatureMin ||
        t > up::kKlTemperatureMax)
      return diag::validating::Error(
          std::string("kl_distill temperature must be a finite float in ") +
          up::kKlTemperatureRangeText);
    const uint64_t hi = up::HighWord(word);
    if (plan_version < up::kSeeuKlScaleVersion) {
      if (hi != 0)
        return diag::validating::Error(
            "kl_distill carries a loss scale in a pre-v" +
            std::to_string(up::kSeeuKlScaleVersion) + " plan");
      return {};
    }
    if (hi == 0) return {};
    const float scale = std::bit_cast<float>(static_cast<uint32_t>(hi));
    if (!std::isfinite(scale) || scale <= 0.0f)
      return diag::validating::Error(
          "kl_distill loss scale must be a finite positive float");
    return {};
  }
  // A clip threshold as f32 bits in a word's low half: finite and positive,
  // the high half zero — Execute() clips by it blindly. The standalone
  // kClipNorm carries one in in[1] at every version, zero included in the
  // refusal (CR1 #142 F13 / F18: a cap of 0 scaled g by 0 on the CPU and
  // by stale scratch on the GPU, and -1 was gradient ascent); the fused
  // form on the optimizer steps (v12, out[1]) is absent below v12 — a
  // nonzero word there is corruption — and from v12 on either zero or such
  // a float.
  [[nodiscard]] std::expected<void, std::string> ClipThresholdOk(
      uint64_t word, const char* what) const {
    const float clip = std::bit_cast<float>(static_cast<uint32_t>(word));
    if (up::HighWord(word) != 0 || !std::isfinite(clip) || clip <= 0.0f)
      return diag::validating::Error(std::string(what) +
                                     " must be a finite positive float");
    return {};
  }
  [[nodiscard]] std::expected<void, std::string> ClipWordOk(
      uint64_t word) const {
    if (word == 0) return {};
    if (plan_version < up::kSeeuKernelBatchVersion)
      return diag::validating::Error(
          "optimizer step carries a clip threshold in a pre-v" +
          std::to_string(up::kSeeuKernelBatchVersion) + " plan");
    return ClipThresholdOk(word, "optimizer step clip threshold");
  }
  // Shared geometry for the transformer family: activations are
  // [B*S, H*d] f32, the probability matrix [B*H*S, S]. Derived with the
  // same overflow-safe chain the kernels' loop bounds imply.
  bool AttnGeometryOk(uint64_t bs_word, uint64_t hd_word, uint64_t* td,
                      uint64_t* pn) const {
    const uint64_t B = up::HighWord(bs_word), S = up::LowWord(bs_word);
    const uint64_t H = up::HighWord(hd_word), d = up::LowWord(hd_word);
    uint64_t t = 0, dm = 0, bhs = 0;
    if (B == 0 || S == 0 || H == 0 || d == 0) return false;
    if (!MulOk(B, S, &t) || !MulOk(H, d, &dm) || !MulOk(t, dm, td))
      return false;
    if (!MulOk(t, H, &bhs) || !MulOk(bhs, S, pn)) return false;
    return true;
  }
};

std::expected<void, std::string> InstructionProof::FlagsOk() const {
  // Flags discipline before any operand math. Pre-v5 plans predate the
  // flags vocabulary: a nonzero word there is corruption, not a feature.
  // From v5 on, every set bit must be a defined epilogue bit AND defined
  // for this opcode — Execute() applies flags blindly, so an unknown or
  // misplaced bit must die here, loudly, not skip silently.
  if (plan_version < up::kSeeuFlagsVersion) {
    if (ins.flags != 0)
      return diag::validating::Error("instruction carries flags in a pre-v" +
                                     std::to_string(up::kSeeuFlagsVersion) +
                                     " plan (opcode " +
                                     std::to_string(ins.opcode) + ")");
  } else {
    const auto opcode = static_cast<up::OpCode>(ins.opcode);
    const bool addend_ok =
        plan_version >= up::kSeeuGemmAddendVersion &&
        (opcode == up::OpCode::kGemmNN || opcode == up::OpCode::kGemmNT ||
         opcode == up::OpCode::kGemmTN);
    const bool colscale_ok =
        plan_version >= up::kSeeuShippedEvalVersion &&
        (opcode == up::OpCode::kGemmNNQ8 || opcode == up::OpCode::kGemmNTQ8);
    // kFlagRelaxed (v18): the six frozen-weight GEMM opcodes. Which
    // PROGRAM may carry it is the executor contract's rule (the eval
    // program, which scores what ships in exact arithmetic, may not); a
    // backend re-validating a single instruction at dispatch sees the bit
    // as the opcode-level permission it is.
    // v20 (F3 #131): the four tiled attention opcodes as well.
    const bool relaxed_ok =
        (plan_version >= up::kSeeuRelaxedVersion &&
         (opcode == up::OpCode::kGemmNN || opcode == up::OpCode::kGemmNT ||
          opcode == up::OpCode::kGemmNNQ8 || opcode == up::OpCode::kGemmNTQ8 ||
          opcode == up::OpCode::kGemmNNBF16 ||
          opcode == up::OpCode::kGemmNTBF16)) ||
        (plan_version >= up::kSeeuRelaxedAttentionVersion &&
         (opcode == up::OpCode::kAttnFwdTiled ||
          opcode == up::OpCode::kAttnDQTiled ||
          opcode == up::OpCode::kAttnDKTiled ||
          opcode == up::OpCode::kAttnDVTiled)) ||
        // v26: the masked tiled forward, relaxed as the plain one is — the
        // backward opcodes it shares already are.
        (plan_version >= up::kSeeuTiledSegmentMaskVersion &&
         opcode == up::OpCode::kAttnFwdTiledSeg);
    const uint16_t allowed = static_cast<uint16_t>(
        ((opcode == up::OpCode::kGemmNN || opcode == up::OpCode::kGemmNNBF16)
             ? up::kEpilogueFlagsMask
         : opcode == up::OpCode::kGemmNNQ8 ? up::kFlagEpilogueActMask
                                           : uint16_t{0}) |
        (addend_ok ? up::kFlagGemmAddend : uint16_t{0}) |
        (colscale_ok ? up::kFlagQ8ColScale : uint16_t{0}) |
        (relaxed_ok ? up::kFlagRelaxed : uint16_t{0}));
    if (ins.flags & static_cast<uint16_t>(~allowed))
      return diag::validating::Error("unknown or misplaced instruction flags " +
                                     std::to_string(ins.flags) + " (opcode " +
                                     std::to_string(ins.opcode) + ")");
    // One ref slot, one tenant: the addend and the epilogue both want in[3].
    if ((ins.flags & up::kFlagGemmAddend) &&
        (ins.flags & up::kEpilogueFlagsMask))
      return diag::validating::Error(
          "GEMM addend combined with a bias / activation epilogue (opcode " +
          std::to_string(ins.opcode) + ")");
  }
  return {};
}

std::expected<void, std::string> InstructionProof::ImmOk() const {
  // The imm word: defined on the normalization forwards (v16) as a finite
  // positive epsilon's f32 bits (0 = 1e-5), and on the three f32 GEMMs
  // (v19) as a finite nonzero alpha's (0 = 1.0). Before v16 it was a pad
  // word every compiler wrote as zero; a nonzero value anywhere it is not
  // defined would be ignored by Execute(), so it dies here.
  if (ins.imm != 0) {
    const auto opcode = static_cast<up::OpCode>(ins.opcode);
    const bool norm_fwd = opcode == up::OpCode::kLayerNormFwd ||
                          opcode == up::OpCode::kRmsNormFwd;
    const bool gemm_f32 = opcode == up::OpCode::kGemmNN ||
                          opcode == up::OpCode::kGemmNT ||
                          opcode == up::OpCode::kGemmTN;
    const bool defined =
        (norm_fwd && plan_version >= up::kSeeuNormEpsVersion) ||
        (gemm_f32 && plan_version >= up::kSeeuGemmAlphaVersion);
    if (!defined)
      return diag::validating::Error(
          "instruction carries an imm word its opcode does not define "
          "(opcode " +
          std::to_string(ins.opcode) + ", plan v" +
          std::to_string(plan_version) + ")");
    const float value = std::bit_cast<float>(ins.imm);
    if (norm_fwd && (!std::isfinite(value) || !(value > 0.0f)))
      return diag::validating::Error(
          "normalization epsilon must be a finite positive float");
    if (gemm_f32 && (!std::isfinite(value) || value == 0.0f))
      return diag::validating::Error(
          "GEMM alpha must be a finite nonzero float");
    // The relaxed family is the frozen-weight products; an adapter
    // product never carries the bit, and no relaxed kernel takes an alpha.
    if (gemm_f32 && (ins.flags & up::kFlagRelaxed))
      return diag::validating::Error(
          "GEMM alpha on a relaxed instruction (opcode " +
          std::to_string(ins.opcode) + ")");
  }
  return {};
}

std::expected<void, std::string> InstructionProof::VersionOk() const {
  // Transformer opcodes exist only from v6: no earlier compiler emits them,
  // so their appearance in an older plan is corruption, not a feature.
  if (ins.opcode >= static_cast<uint16_t>(up::OpCode::kRmsNormFwd) &&
      ins.opcode <= static_cast<uint16_t>(up::OpCode::kAttnDK) &&
      plan_version < up::kSeeuTransformerVersion)
    return diag::validating::Error(
        "transformer opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuTransformerVersion) + " plan");
  if ((ins.opcode == static_cast<uint16_t>(up::OpCode::kGemmNNBF16) ||
       ins.opcode == static_cast<uint16_t>(up::OpCode::kGemmNTBF16)) &&
      plan_version < up::kSeeuBf16Version)
    return diag::validating::Error("bf16 GEMM opcode in a pre-v" +
                                   std::to_string(up::kSeeuBf16Version) +
                                   " plan");
  if (ins.opcode >= static_cast<uint16_t>(up::OpCode::kAttnFwdTiled) &&
      ins.opcode <= static_cast<uint16_t>(up::OpCode::kAttnDVTiled) &&
      plan_version < up::kSeeuTiledAttentionVersion)
    return diag::validating::Error(
        "tiled attention opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuTiledAttentionVersion) + " plan");
  if ((ins.opcode == static_cast<uint16_t>(up::OpCode::kSoftmaxXEntMaskedFwd) ||
       ins.opcode ==
           static_cast<uint16_t>(up::OpCode::kSoftmaxXEntMaskedBwd)) &&
      plan_version < up::kSeeuDataContractVersion)
    return diag::validating::Error(
        "masked cross-entropy opcode " + std::to_string(ins.opcode) +
        " in a pre-v" + std::to_string(up::kSeeuDataContractVersion) + " plan");
  if ((ins.opcode == static_cast<uint16_t>(up::OpCode::kSegmentStarts) ||
       ins.opcode == static_cast<uint16_t>(up::OpCode::kAttnFwdSeg)) &&
      plan_version < up::kSeeuSegmentMaskVersion)
    return diag::validating::Error(
        "segment mask opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuSegmentMaskVersion) + " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kAttnFwdTiledSeg) &&
      plan_version < up::kSeeuTiledSegmentMaskVersion)
    return diag::validating::Error(
        "tiled segment mask opcode " + std::to_string(ins.opcode) +
        " in a pre-v" + std::to_string(up::kSeeuTiledSegmentMaskVersion) +
        " plan");
  if ((ins.opcode == static_cast<uint16_t>(up::OpCode::kSqNormAccum) ||
       ins.opcode == static_cast<uint16_t>(up::OpCode::kClipGlobal)) &&
      plan_version < up::kSeeuGlobalClipVersion)
    return diag::validating::Error(
        "global clip opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuGlobalClipVersion) + " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kAccumulate) &&
      plan_version < up::kSeeuGradAccumVersion)
    return diag::validating::Error("accumulate opcode in a pre-v" +
                                   std::to_string(up::kSeeuGradAccumVersion) +
                                   " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kFusedMap) &&
      plan_version < up::kSeeuFusedMapVersion)
    return diag::validating::Error("fused_map opcode in a pre-v" +
                                   std::to_string(up::kSeeuFusedMapVersion) +
                                   " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kRopeTable) &&
      plan_version < up::kSeeuKernelBatchVersion)
    return diag::validating::Error("rope_table opcode in a pre-v" +
                                   std::to_string(up::kSeeuKernelBatchVersion) +
                                   " plan");
  if (ins.opcode == static_cast<uint16_t>(up::OpCode::kEmbedFwd) &&
      plan_version < up::kSeeuTokenVersion)
    return diag::validating::Error(
        "token opcode " + std::to_string(ins.opcode) + " in a pre-v" +
        std::to_string(up::kSeeuTokenVersion) + " plan");
  return {};
}

std::expected<void, std::string> InstructionProof::Gemm() {
  uint64_t mk = 0, kn = 0, mn = 0;
  // Every layout variant reads M*K (A) and K*N (B), writes M*N (C).
  const auto oc = static_cast<up::OpCode>(ins.opcode);
  const bool q8 = oc == up::OpCode::kGemmNNQ8 || oc == up::OpCode::kGemmNTQ8;
  const bool bf16 =
      oc == up::OpCode::kGemmNNBF16 || oc == up::OpCode::kGemmNTBF16;
  if (!MulOk(d0, d2, &mk) || !MulOk(d2, d1, &kn) || !MulOk(d0, d1, &mn))
    return Fail();
  // Narrow-storage B must live in rodata: only the compiler's own
  // packing produces it — 1 byte per element for int8, 2 for bf16.
  if ((q8 || bf16) && !up::IsRodataRef(ins.in[1])) return Fail();
  const uint64_t b_elem = q8 ? 1 : bf16 ? 2 : sizeof(float);
  if (!RefOk(ins.in[0], mk, false) || !RefOkW(ins.in[1], kn, false, b_elem) ||
      !RefOk(ins.in[2], mn, true))
    return Fail();
  // Fused-bias epilogue (flags proved valid for this opcode above):
  // in[3] is a read of N floats and joins the overlap discipline
  // against the written C range.
  if ((ins.flags & up::kFlagEpilogueBias) && !RefOk(ins.in[3], d1, false))
    return Fail();
  // The per-tensor dequant scale (the q8 GEMMs' in[3] without the
  // column-scale flag): every element of the widened weight is
  // multiplied by it, so it is a finite positive f32 in the low word,
  // the high word zero (CR1 #142 F33 — a zero or NaN scale passed, and
  // dequantized the whole weight to nothing or to NaN).
  if (q8 && !(ins.flags & up::kFlagQ8ColScale)) {
    const float scale = std::bit_cast<float>(static_cast<uint32_t>(ins.in[3]));
    if (up::HighWord(ins.in[3]) != 0 || !std::isfinite(scale) || scale <= 0.0f)
      return diag::validating::Error(
          "quantized GEMM dequant scale must be a finite positive float");
  }
  // Per-column int8 scales (v17): a rodata vector, one float per
  // output column of W — N in the forward (NN), the reduction extent K
  // in the dX GEMM (NT), where W is read as [N, K].
  if (ins.flags & up::kFlagQ8ColScale) {
    const bool nt = oc == up::OpCode::kGemmNTQ8;
    if (!up::IsRodataRef(ins.in[3]) || !RefOk(ins.in[3], nt ? d2 : d1, false))
      return Fail();
  }
  // The addend (v14, f32 GEMMs only — proved above): in[3] is a read of
  // M*N floats, disjoint from the written C like every other operand.
  if ((ins.flags & up::kFlagGemmAddend) && !RefOk(ins.in[3], mn, false))
    return Fail();
  return Disjoint();
}

std::expected<void, std::string> InstructionProof::Elementwise() {
  uint64_t mn = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      if (!RefOk(ins.in[0], d0, false) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], d0, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kAddBias:
      if (!MulOk(d0, d1, &mn)) return Fail();
      if (!RefOk(ins.in[0], mn, false) || !RefOk(ins.in[1], d1, false) ||
          !RefOk(ins.in[2], mn, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      if (!RefOk(ins.in[0], d0, false) || !RefOk(ins.in[1], d0, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kReduceRows:
      if (!MulOk(d0, d1, &mn)) return Fail();
      if (!RefOk(ins.in[0], mn, false) || !RefOk(ins.in[1], d1, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kFill:
      if (!RefOk(ins.in[0], d0, true)) return Fail();
      return Disjoint();
    case up::OpCode::kFusedMap: {
      // The micro-program is proven like everything else (FusedProgramOk);
      // then every operand slot a stage names must be present and count
      // floats long, and every slot none names absent.
      bool uses_slot[3] = {false, false, false};
      if (!FusedProgramOk(d1, d2, uses_slot)) return Fail();
      if (!RefOk(ins.in[0], d0, false)) return Fail();
      for (int slot = 1; slot <= 2; ++slot) {
        if (uses_slot[slot] ? !RefOk(ins.in[slot], d0, false)
                            : ins.in[slot] != up::kNullRef)
          return Fail();
      }
      if (!RefOk(ins.in[3], d0, true)) return Fail();
      return Disjoint();
    }
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::Normalization() {
  uint64_t nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kLayerNormFwd: {
      const uint64_t rows = up::HighWord(d0), cols = up::LowWord(d0);
      if (!MulOk(rows, cols, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], cols, false) ||
          !RefOk(ins.in[2], cols, false) || !RefOk(ins.in[3], nc, true) ||
          !RefOk(ins.out[1], rows, true) || !RefOk(ins.out[2], rows, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kLayerNormBwd: {
      const uint64_t rows = up::HighWord(d2), cols = up::LowWord(d2);
      if (!MulOk(rows, cols, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], nc, false) ||
          !RefOk(ins.in[2], cols, false) || !RefOk(ins.in[3], nc, true) ||
          !RefOk(ins.out[0], rows, false) || !RefOk(ins.out[1], rows, false))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kRmsNormFwd: {
      const uint64_t rows = up::HighWord(d0), cols = up::LowWord(d0);
      if (!MulOk(rows, cols, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], cols, false) ||
          !RefOk(ins.in[2], nc, true) || !RefOk(ins.in[3], rows, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kRmsNormBwd: {
      const uint64_t rows = up::HighWord(d1), cols = up::LowWord(d1);
      if (!MulOk(rows, cols, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], nc, false) ||
          !RefOk(ins.in[2], cols, false) || !RefOk(ins.in[3], nc, true) ||
          !RefOk(ins.out[0], rows, false))
        return Fail();
      return Disjoint();
    }
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::Loss() {
  uint64_t nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kSoftmaxXEntFwd:
    case up::OpCode::kSoftmaxXEntMaskedFwd:
      if (!MulOk(d0, d1, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], 1, true) || !RefOk(ins.in[3], nc, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kSoftmaxXEntBwd:
    case up::OpCode::kSoftmaxXEntMaskedBwd:
      if (!MulOk(d0, d1, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], 1, false) || !RefOk(ins.in[3], nc, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kMseFwd:
      if (!RefOk(ins.in[0], d0, false) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], 1, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kMseBwd:
      if (!RefOk(ins.in[0], d0, false) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], 1, false) || !RefOk(ins.in[3], d0, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kKLDistillFwd:
      if (!MulOk(up::HighWord(d1), up::LowWord(d1), &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], nc, false) ||
          !RefOk(ins.in[2], 1, true) || !RefOk(ins.in[3], nc, true) ||
          !RefOk(ins.out[0], nc, true))
        return Fail();
      if (auto r = KlWordOk(d2); !r) return r;
      return Disjoint();
    case up::OpCode::kKLDistillBwd:
      if (!MulOk(up::HighWord(d0), up::LowWord(d0), &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], nc, false) ||
          !RefOk(ins.in[2], 1, false) || !RefOk(ins.in[3], nc, true))
        return Fail();
      if (auto r = KlWordOk(d1); !r) return r;
      return Disjoint();
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::OptimizerStep() {
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kClipNorm:
      if (auto r = ClipThresholdOk(ins.in[1], "clip_norm threshold"); !r)
        return r;
      if (!RefOk(ins.in[0], d0, true)) return Fail();
      return Disjoint();
    case up::OpCode::kSqNormAccum:  // v24: reads g, adds into its slot
      if (!RefOk(ins.in[0], d0, false) || !RefOk(ins.in[1], 1, true))
        return Fail();
      return Disjoint();
    case up::OpCode::kClipGlobal:  // v24: reads the slot, scales g in place
      if (auto r = ClipThresholdOk(ins.in[2], "global clip threshold"); !r)
        return r;
      if (!RefOk(ins.in[0], d0, true) || !RefOk(ins.in[1], 1, false))
        return Fail();
      return Disjoint();
    case up::OpCode::kSgdStep:
      if (auto r = ClipWordOk(d1); !r) return r;
      if (!RefOk(ins.in[0], d0, true) || !RefOk(ins.in[1], d0, false))
        return Fail();
      return Disjoint();
    case up::OpCode::kAccumulate:  // dst (read+write, one operand), src
      if (!RefOk(ins.in[0], d0, true) || !RefOk(ins.in[1], d0, false))
        return Fail();
      return Disjoint();
    case up::OpCode::kAdamWStep:
      if (auto r = ClipWordOk(d1); !r) return r;
      if (!RefOk(ins.in[0], d0, true) || !RefOk(ins.in[1], d0, false) ||
          !RefOk(ins.in[2], d0, true) || !RefOk(ins.in[3], d0, true))
        return Fail();
      return Disjoint();
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::RopeAndEmbedding() {
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd: {
      uint64_t td = 0, pn = 0;
      if (!AttnGeometryOk(d0, d1, &td, &pn)) return Fail();
      // The rotation pairs (2c, 2c+1) require an even head width.
      if (up::LowWord(d1) % 2 != 0) return Fail();
      // The rotary base rides out[2] as f32 bits (SMF v5 attr1, or the
      // lowering default). A non-finite or <= 1 base would make every
      // angle degenerate or NaN — reject it here, not as a NaN loss later.
      const float base =
          std::bit_cast<float>(static_cast<uint32_t>(ins.out[2]));
      if (up::HighWord(ins.out[2]) != 0 || !std::isfinite(base) || base <= 1.0f)
        return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, true))
        return Fail();
      // The optional angle table (v12): [S, d/2, 2] = S*d floats, the
      // extent kRopeTable writes for this geometry. Below v12 the slot is
      // kNullRef by construction; anything else there is corruption.
      if (ins.in[2] != up::kNullRef) {
        if (plan_version < up::kSeeuKernelBatchVersion)
          return diag::validating::Error(
              "rope carries an angle table in a pre-v" +
              std::to_string(up::kSeeuKernelBatchVersion) + " plan");
        uint64_t table = 0;
        if (!MulOk(up::LowWord(d0), up::LowWord(d1), &table) ||
            !RefOk(ins.in[2], table, false))
          return Fail();
      }
      return Disjoint();
    }
    case up::OpCode::kRopeTable: {
      // out[0] = S<<32|d, out[1] = base bits; the table is S*d floats.
      const uint64_t S = up::HighWord(d0), d = up::LowWord(d0);
      const float base = std::bit_cast<float>(static_cast<uint32_t>(d1));
      uint64_t table = 0;
      if (S == 0 || d == 0 || d % 2 != 0 || up::HighWord(d1) != 0 ||
          !std::isfinite(base) || base <= 1.0f || !MulOk(S, d, &table))
        return Fail();
      if (!RefOk(ins.in[0], table, true)) return Fail();
      return Disjoint();
    }
    case up::OpCode::kEmbedFwd: {
      // out[t, :] = table[tokens[t], :] with T = d0 rows and a [V, D]
      // table packed as d1 = V<<32|D. The table must be rodata — only the
      // compiler's own packing produces it — or, from v21, the source
      // model's f32 table where source refs are admitted (an adapter-only
      // plan); never the arena. The gather's row bound (tokens[t] < V) is
      // the FEEDER contract's runtime obligation — the extents proven here
      // are the buffers, not the indices.
      const uint64_t vocab = up::HighWord(d1), dim = up::LowWord(d1);
      uint64_t vd = 0, td = 0;
      if (!MulOk(vocab, dim, &vd) || !MulOk(d0, dim, &td)) return Fail();
      const bool source_table = up::IsSourceRef(ins.in[1]) && allow_source &&
                                plan_version >= up::kSeeuSourceWeightsVersion;
      if (!up::IsRodataRef(ins.in[1]) && !source_table) return Fail();
      if (!RefOk(ins.in[0], d0, false) ||  // i32 tokens: 4-byte elements
          !RefOk(ins.in[1], vd, false) || !RefOk(ins.in[2], td, true))
        return Fail();
      return Disjoint();
    }
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::AttentionCached() {
  uint64_t nc = 0;
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kAttnFwd: {
      uint64_t td = 0, pn = 0;
      if (!AttnGeometryOk(d1, d2, &td, &pn)) return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, false) || !RefOk(ins.in[3], td, true) ||
          !RefOk(ins.out[0], pn, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kSegmentStarts: {  // v25: i32 tokens in, i32 starts out
      const uint64_t B = up::HighWord(d0), S = up::LowWord(d0);
      uint64_t t = 0;
      if (B == 0 || S == 0 || !MulOk(B, S, &t)) return Fail();
      if (d1 > 0x7FFFFFFFu)
        return diag::validating::Error("segment EOS id is not a token id");
      if (!RefOk(ins.in[0], t, false) || !RefOk(ins.in[1], t, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnFwdSeg: {  // v25: kAttnFwd, packed, plus starts
      const up::AttnGeometry g = up::UnpackAttnGeometry(d1);
      uint64_t td = 0, pn = 0, t = 0;
      if (!AttnGeometryOk(up::PackWords(g.B, g.S), up::PackWords(g.H, g.d), &td,
                          &pn) ||
          !MulOk(g.B, g.S, &t))
        return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, false) || !RefOk(ins.in[3], td, true) ||
          !RefOk(ins.out[0], pn, true) || !RefOk(d2, t, false))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnDP: {
      uint64_t td = 0, pn = 0;
      if (!AttnGeometryOk(d0, d1, &td, &pn)) return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], pn, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnDV: {
      uint64_t td = 0, pn = 0;
      if (!AttnGeometryOk(d0, d1, &td, &pn)) return Fail();
      if (!RefOk(ins.in[0], pn, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kSoftmaxRowsBwd: {
      const uint64_t rows = up::HighWord(d0), cols = up::LowWord(d0);
      if (!MulOk(rows, cols, &nc)) return Fail();
      if (!RefOk(ins.in[0], nc, false) || !RefOk(ins.in[1], nc, false) ||
          !RefOk(ins.in[2], nc, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnDQ:
    case up::OpCode::kAttnDK: {
      uint64_t td = 0, pn = 0;
      if (!AttnGeometryOk(d0, d1, &td, &pn)) return Fail();
      if (!RefOk(ins.in[0], pn, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, true))
        return Fail();
      return Disjoint();
    }
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::AttentionTiled() {
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kAttnFwdTiled: {
      // q, k, v read and o written at [B*S, H*d]; stats written at
      // [B*H*S, kAttnStatsWidth] — the family's geometry words.
      uint64_t td = 0, pn = 0, st = 0;
      if (!AttnGeometryOk(d1, d2, &td, &pn)) return Fail();
      const uint64_t bhs =
          up::HighWord(d1) * up::LowWord(d1) * up::HighWord(d2);
      if (!MulOk(bhs, up::kAttnStatsWidth, &st)) return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, false) || !RefOk(ins.in[3], td, true) ||
          !RefOk(ins.out[0], st, true))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnFwdTiledSeg: {
      // v26: kAttnFwdTiled with the geometry packed into one word and the
      // i32 starts [B*S] read through the slot that frees — kAttnFwdSeg's
      // layout, the stats row where the cache was.
      const up::AttnGeometry g = up::UnpackAttnGeometry(d1);
      uint64_t td = 0, pn = 0, st = 0, t = 0;
      if (!AttnGeometryOk(up::PackWords(g.B, g.S), up::PackWords(g.H, g.d), &td,
                          &pn) ||
          !MulOk(g.B * g.S * g.H, up::kAttnStatsWidth, &st) ||
          !MulOk(g.B, g.S, &t))
        return Fail();
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, false) || !RefOk(ins.in[3], td, true) ||
          !RefOk(ins.out[0], st, true) || !RefOk(d2, t, false))
        return Fail();
      return Disjoint();
    }
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled: {
      // q, k, v, dO read; the result written; stats read — and, for dQ,
      // written too (the delta column), through the one pointer it is.
      // Geometry is the packed 16-bit word: every field must be nonzero
      // and the products must be sound.
      const auto g = up::UnpackAttnGeometry(d2);
      const uint64_t bs = up::PackWords(g.B, g.S), hd = up::PackWords(g.H, g.d);
      uint64_t td = 0, pn = 0, st = 0;
      if (!AttnGeometryOk(bs, hd, &td, &pn)) return Fail();
      if (!MulOk(g.B * g.S * g.H, up::kAttnStatsWidth, &st)) return Fail();
      const bool dq =
          ins.opcode == static_cast<uint16_t>(up::OpCode::kAttnDQTiled);
      if (!RefOk(ins.in[0], td, false) || !RefOk(ins.in[1], td, false) ||
          !RefOk(ins.in[2], td, false) || !RefOk(ins.in[3], td, false) ||
          !RefOk(ins.out[0], st, dq) || !RefOk(ins.out[1], td, true))
        return Fail();
      return Disjoint();
    }
    default:
      break;
  }
  return UnknownOpcode();
}

std::expected<void, std::string> InstructionProof::Run() {
  if (auto r = FlagsOk(); !r) return r;
  if (auto r = ImmOk(); !r) return r;
  extents->count = 0;
  if (auto r = VersionOk(); !r) return r;
  // Every opcode is named here, so a new one is a compile error until it
  // is given a proof; an opcode word outside the enum falls through.
  switch (static_cast<up::OpCode>(ins.opcode)) {
    case up::OpCode::kNop:
      return Disjoint();
    case up::OpCode::kGemmNN:
    case up::OpCode::kGemmNT:
    case up::OpCode::kGemmTN:
    case up::OpCode::kGemmAccNN:
    case up::OpCode::kGemmNNQ8:
    case up::OpCode::kGemmNTQ8:
    case up::OpCode::kGemmNNBF16:
    case up::OpCode::kGemmNTBF16:
      return Gemm();
    case up::OpCode::kAddEW:
    case up::OpCode::kMulEW:
    case up::OpCode::kReluBwd:
    case up::OpCode::kGeluBwd:
    case up::OpCode::kSiluBwd:
      return Elementwise();
    case up::OpCode::kAddBias:
      return Elementwise();
    case up::OpCode::kReluFwd:
    case up::OpCode::kGeluFwd:
    case up::OpCode::kSiluFwd:
    case up::OpCode::kScale:
    case up::OpCode::kCopy:
      return Elementwise();
    case up::OpCode::kLayerNormFwd:
      return Normalization();
    case up::OpCode::kLayerNormBwd:
      return Normalization();
    case up::OpCode::kClipNorm:
      return OptimizerStep();
    case up::OpCode::kSqNormAccum:
      return OptimizerStep();
    case up::OpCode::kClipGlobal:
      return OptimizerStep();
    case up::OpCode::kReduceRows:
      return Elementwise();
    case up::OpCode::kSoftmaxXEntFwd:
    case up::OpCode::kSoftmaxXEntMaskedFwd:
      return Loss();
    case up::OpCode::kSoftmaxXEntBwd:
    case up::OpCode::kSoftmaxXEntMaskedBwd:
      return Loss();
    case up::OpCode::kMseFwd:
      return Loss();
    case up::OpCode::kMseBwd:
      return Loss();
    case up::OpCode::kKLDistillFwd:
      return Loss();
    case up::OpCode::kKLDistillBwd:
      return Loss();
    case up::OpCode::kSgdStep:
      return OptimizerStep();
    case up::OpCode::kAccumulate:
      return OptimizerStep();
    case up::OpCode::kAdamWStep:
      return OptimizerStep();
    case up::OpCode::kFill:
      return Elementwise();
    case up::OpCode::kRmsNormFwd:
      return Normalization();
    case up::OpCode::kRmsNormBwd:
      return Normalization();
    case up::OpCode::kRopeFwd:
    case up::OpCode::kRopeBwd:
      return RopeAndEmbedding();
    case up::OpCode::kFusedMap:
      return Elementwise();
    case up::OpCode::kRopeTable:
      return RopeAndEmbedding();
    case up::OpCode::kAttnFwd:
      return AttentionCached();
    case up::OpCode::kSegmentStarts:
      return AttentionCached();
    case up::OpCode::kAttnFwdSeg:
      return AttentionCached();
    case up::OpCode::kAttnDP:
      return AttentionCached();
    case up::OpCode::kAttnDV:
      return AttentionCached();
    case up::OpCode::kSoftmaxRowsBwd:
      return AttentionCached();
    case up::OpCode::kAttnDQ:
    case up::OpCode::kAttnDK:
      return AttentionCached();
    case up::OpCode::kAttnFwdTiled:
      return AttentionTiled();
    case up::OpCode::kAttnFwdTiledSeg:
      return AttentionTiled();
    case up::OpCode::kAttnDQTiled:
    case up::OpCode::kAttnDKTiled:
    case up::OpCode::kAttnDVTiled:
      return AttentionTiled();
    case up::OpCode::kEmbedFwd:
      return RopeAndEmbedding();
  }
  // Every known opcode returned through its proof above; anything else
  // must be a load error — Execute() would silently skip it.
  return UnknownOpcode();
}

[[nodiscard]] std::expected<void, std::string> ValidateInstructionImpl(
    const up::UpdateInstruction& ins, uint64_t arena_size, uint64_t rodata_size,
    uint32_t plan_version, bool allow_source, InstructionExtents* extents) {
  return InstructionProof(ins, arena_size, rodata_size, plan_version,
                          allow_source, extents)
      .Run();
}

}  // namespace

std::expected<void, std::string> ValidateInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size, uint64_t rodata_size,
    uint32_t plan_version, bool allow_source) {
  InstructionExtents extents;
  return ValidateInstructionImpl(ins, arena_size, rodata_size, plan_version,
                                 allow_source, &extents);
}

std::expected<InstructionExtents, std::string> DescribeInstruction(
    const up::UpdateInstruction& ins, uint64_t arena_size, uint64_t rodata_size,
    uint32_t plan_version, bool allow_source) {
  InstructionExtents extents;
  if (auto r = ValidateInstructionImpl(ins, arena_size, rodata_size,
                                       plan_version, allow_source, &extents);
      !r)
    return std::unexpected(r.error());
  return extents;
}

}  // namespace seeml::update_rt
