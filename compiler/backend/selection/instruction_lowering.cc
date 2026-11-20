#include "compiler/backend/selection/instruction_lowering.h"

#include <unordered_set>
#include "source/language/model_format.h"

#include <bit>

#include "compiler/diagnostics/generating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace generating = seeml::diag::generating;

namespace {

uint64_t F32Bits(float f) { return std::bit_cast<uint32_t>(f); }

/// A normalization forward's imm word (plan v16): the op's epsilon bits, or
/// 0 for the default 1e-5 — so a model without one lowers byte-identically.
uint32_t NormEpsImm(const sir::Operation& op) {
  const float eps = op.getAttrAs<float>("eps").value_or(kDefaultNormEps);
  return std::bit_cast<uint32_t>(eps) ==
                 std::bit_cast<uint32_t>(kDefaultNormEps)
             ? 0u
             : std::bit_cast<uint32_t>(eps);
}
// Two immediates in one word: (hi bits << 32) | lo bits. Used for the KL
// temperature word, whose high half is the v8 loss scale (instruction.h).
uint64_t F32BitsPair(float hi, float lo) {
  return PackWords(F32Bits(hi), F32Bits(lo));
}

/// One op's lowering: the op, the block-wide tables its arms read, and the
/// instruction being filled. Run() is the old if / else-if chain over the
/// mnemonic, cut into contiguous runs and tried in the chain's order — the
/// order is part of the meaning (S8 #154). A run returns true when the op
/// was its to lower, whether it filled `ins` or set `error`.
struct OpLowering {
  OpLowering(
      const sir::Operation* operation, const ResolveFn& resolve_fn,
      const std::unordered_map<const sir::Value*, float>& scales,
      const std::unordered_set<const sir::Value*>& bf16,
      const std::unordered_map<const sir::Value*, uint64_t>& column_scales,
      std::string& error_out)
      : op(operation),
        m(operation->mnemonic()),
        resolve(resolve_fn),
        quant_scales(scales),
        bf16_weights(bf16),
        quant_column_scales(column_scales),
        error(error_out) {}

  const sir::Operation* const op;
  const std::string_view m;
  const ResolveFn& resolve;
  const std::unordered_map<const sir::Value*, float>& quant_scales;
  const std::unordered_set<const sir::Value*>& bf16_weights;
  const std::unordered_map<const sir::Value*, uint64_t>& quant_column_scales;
  std::string& error;  // the block's first error; empty while there is none
  UpdateInstruction ins;

  uint64_t Ref(const sir::Value* v) {
    auto r = resolve(v);
    if (!r) {
      if (error.empty()) error = r.error();
      return kNullRef;
    }
    return r.value();
  }
  static uint64_t Vol(const sir::Value* v) {
    return static_cast<uint64_t>(v->shape().volume());
  }
  // Packed dim words carry two dims in 32-bit halves. Shapes are int64:
  // a half that does not fit must be a refusal, never a silent truncation —
  // the validator cannot tell a truncated word from an honest small one.
  uint64_t Pack32(int64_t hi, int64_t lo) {
    if (hi < 0 || lo < 0 || hi > 0xFFFFFFFFll || lo > 0xFFFFFFFFll) {
      if (error.empty())
        error = "dimension pair (" + std::to_string(hi) + ", " +
                std::to_string(lo) + ") exceeds the 32-bit ISA dim fields";
      return 0;
    }
    return PackWords(static_cast<uint64_t>(hi), static_cast<uint64_t>(lo));
  }
  void Set(OpCode oc) { ins.opcode = static_cast<uint16_t>(oc); }

  /// An elementwise chain folded into its tail.
  bool FusedMap();
  /// The GEMMs: matmul in its three layouts and three storages, the
  /// accumulating form, and an add that GemmAddendFuser made a GEMM — which
  /// is why this run precedes the plain add.
  bool Gemm();
  /// The add, the bias, the activations and their gradients, the product.
  bool Elementwise();
  /// LayerNorm and RmsNorm, forward and backward, and the embedding.
  bool NormAndEmbedding();
  /// The sequence family, over one shared geometry: RoPE, then attention —
  /// the masked forward (four operands) before the plain one, the tiled
  /// backward before the cached.
  bool RopeAndAttention();
  /// The row softmax gradient, the clips and their norm, the segment starts,
  /// the scale and the row reduction.
  bool ClipScaleAndReduce();
  /// The losses and their gradients.
  bool Loss();
  /// The RoPE table, the optimizer steps, and fill / zero / accumulate / copy.
  bool TableStepAndFill();

  void Run() {
    if (FusedMap() || Gemm() || Elementwise() || NormAndEmbedding() ||
        RopeAndAttention() || ClipScaleAndReduce() || Loss() ||
        TableStepAndFill())
      return;
    error = "cannot lower '" + std::string(m) + "'";
  }
};

bool OpLowering::FusedMap() {
  if (const auto stages = op->getAttrAs<int64_t>("fused_stages")) {
    // An elementwise chain folded into its tail (ElementwiseChainFuser,
    // plan v13): operands are [x, other tensors...], whatever the tail's
    // own mnemonic was.
    if (op->numOperands() < 1 || op->numOperands() > 3) {
      error = "fused elementwise chain with " +
              std::to_string(op->numOperands()) + " operands";
      return true;
    }
    Set(OpCode::kFusedMap);
    for (size_t i = 0; i < op->numOperands(); ++i)
      ins.in[i] = Ref(op->operand(i));
    ins.in[3] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
    ins.out[1] = static_cast<uint64_t>(*stages);
    ins.out[2] =
        F32Bits(op->getAttrAs<float>("fused_imm0").value_or(0.0f)) |
        (F32Bits(op->getAttrAs<float>("fused_imm1").value_or(0.0f)) << 32);
  } else {
    return false;
  }
  return true;
}

bool OpLowering::Gemm() {
  if (m == "sc_high.matmul" || m == "sc_low.matmul_nt" ||
      m == "sc_low.matmul_tn") {
    const sir::Value* a = op->operand(0);
    const sir::Value* c = op->result(0);
    // GEMMs whose B operand is a quantized frozen weight take the q8
    // opcode and carry the dequant scale in in[3].
    const auto q = m != "sc_low.matmul_tn" ? quant_scales.find(op->operand(1))
                                           : quant_scales.end();
    const bool q8 = q != quant_scales.end();
    // bf16 frozen weights (v10) take the widening opcodes; the quantization review
    // never selects a weight for both storages.
    const bool bf16 =
        !q8 && m != "sc_low.matmul_tn" && bf16_weights.contains(op->operand(1));
    Set(m == "sc_high.matmul"     ? (q8     ? OpCode::kGemmNNQ8
                                     : bf16 ? OpCode::kGemmNNBF16
                                            : OpCode::kGemmNN)
        : m == "sc_low.matmul_nt" ? (q8     ? OpCode::kGemmNTQ8
                                     : bf16 ? OpCode::kGemmNTBF16
                                            : OpCode::kGemmNT)
                                  : OpCode::kGemmTN);
    ins.in[0] = Ref(a);
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(c);
    if (q8) ins.in[3] = F32Bits(q->second);
    ins.out[0] = static_cast<uint64_t>(c->shape().dims.at(0));  // M
    ins.out[1] = static_cast<uint64_t>(c->shape().dims.at(1));  // N
    ins.out[2] = static_cast<uint64_t>(                         // K
        m == "sc_low.matmul_tn" ? a->shape().dims.at(0)
                                : a->shape().dims.at(1));
    // The GEMM alpha (plan v19): the f32 forms carry it in imm. Only the
    // LoRA grafter and autodiff set it, on adapter products whose B
    // operand lives in the arena, so a narrow-weight form here is a
    // compiler bug, not a plan.
    if (const float alpha = op->getAttrAs<float>("alpha").value_or(1.0f);
        alpha != 1.0f) {
      if (q8 || bf16) {
        error = "GEMM alpha on a narrow-weight GEMM ('" + std::string(c->id()) +
                "')";
        return true;
      }
      ins.imm = static_cast<uint32_t>(F32Bits(alpha));
    }
    // Fused epilogue (plan v5): a third operand is the fused bias, whose
    // ref rides the free in[3]; "epilogue_act" selects the activation.
    // Both are produced only by GemmEpilogueFuser, and only on
    // sc_high.matmul — the fuser also guarantees a bias never lands on a
    // q8 GEMM (in[3] is its scale), so a clash here is a compiler bug.
    const bool fused_bias = m == "sc_high.matmul" && op->numOperands() == 3;
    if (fused_bias && q8) {
      error = "fused bias on a quantized GEMM ('" + std::string(c->id()) + "')";
      return true;
    }
    if (fused_bias) ins.in[3] = Ref(op->operand(2));
    EpilogueAct act = EpilogueAct::kNone;
    if (auto act_name = op->getAttrAs<std::string>("epilogue_act")) {
      act = *act_name == "relu"   ? EpilogueAct::kRelu
            : *act_name == "gelu" ? EpilogueAct::kGelu
            : *act_name == "silu" ? EpilogueAct::kSilu
                                  : EpilogueAct::kNone;
      if (act == EpilogueAct::kNone) {
        error = "unknown epilogue activation '" + *act_name + "'";
        return true;
      }
    }
    ins.flags = MakeEpilogueFlags(fused_bias, act);
    // v17: the int8 weight's own column scales, by rodata ref.
    if (q8)
      if (auto cs = quant_column_scales.find(op->operand(1));
          cs != quant_column_scales.end()) {
        ins.in[3] = cs->second;
        ins.flags |= kFlagQ8ColScale;
      }
  } else if (m == "sc_low.gemm_acc") {
    const sir::Value* a = op->operand(0);
    const sir::Value* b = op->operand(1);
    Set(OpCode::kGemmAccNN);
    ins.in[0] = Ref(a);
    ins.in[1] = Ref(b);
    ins.in[2] = Ref(op->operand(2));
    ins.in[3] = F32Bits(op->getAttrAs<float>("alpha").value_or(1.0f));
    ins.out[0] = static_cast<uint64_t>(a->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(b->shape().dims.at(1));
    ins.out[2] = static_cast<uint64_t>(a->shape().dims.at(1));
  } else if (m == "sc_high.add" && op->hasAttribute("gemm_addend")) {
    // GemmAddendFuser's tail (v14): operands [addend, A, B]; the add IS
    // the GEMM now, summing as the dot products come out of the core.
    const std::string form =
        op->getAttrAs<std::string>("gemm_addend").value_or("");
    if (op->numOperands() != 3 ||
        (form != "nn" && form != "nt" && form != "tn")) {
      error =
          "malformed GEMM addend ('" + std::string(op->result(0)->id()) + "')";
      return true;
    }
    const sir::Value* a = op->operand(1);
    const sir::Value* c = op->result(0);
    Set(form == "nn"   ? OpCode::kGemmNN
        : form == "nt" ? OpCode::kGemmNT
                       : OpCode::kGemmTN);
    ins.in[0] = Ref(a);
    ins.in[1] = Ref(op->operand(2));
    ins.in[2] = Ref(c);
    ins.in[3] = Ref(op->operand(0));
    ins.out[0] = static_cast<uint64_t>(c->shape().dims.at(0));  // M
    ins.out[1] = static_cast<uint64_t>(c->shape().dims.at(1));  // N
    ins.out[2] = static_cast<uint64_t>(                         // K
        form == "tn" ? a->shape().dims.at(0) : a->shape().dims.at(1));
    ins.flags = kFlagGemmAddend;
    if (const float alpha = op->getAttrAs<float>("alpha").value_or(1.0f);
        alpha != 1.0f)
      ins.imm = static_cast<uint32_t>(F32Bits(alpha));
  } else {
    return false;
  }
  return true;
}

bool OpLowering::Elementwise() {
  if (m == "sc_high.add") {
    Set(OpCode::kAddEW);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_high.add_bias") {
    Set(OpCode::kAddBias);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->result(0));
    ins.out[0] = static_cast<uint64_t>(op->result(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->result(0)->shape().dims.at(1));
  } else if (m == "sc_high.relu" || m == "sc_high.gelu" ||
             m == "sc_high.silu") {
    Set(m == "sc_high.relu"   ? OpCode::kReluFwd
        : m == "sc_high.gelu" ? OpCode::kGeluFwd
                              : OpCode::kSiluFwd);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_low.relu_grad" || m == "sc_low.gelu_grad" ||
             m == "sc_low.silu_grad") {
    Set(m == "sc_low.relu_grad"   ? OpCode::kReluBwd
        : m == "sc_low.gelu_grad" ? OpCode::kGeluBwd
                                  : OpCode::kSiluBwd);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_high.mul" || m == "sc_low.mul") {
    Set(OpCode::kMulEW);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else {
    return false;
  }
  return true;
}

bool OpLowering::NormAndEmbedding() {
  if (m == "sc_high.layer_norm") {
    const sir::Value* y = op->result(0);
    Set(OpCode::kLayerNormFwd);
    ins.imm = NormEpsImm(*op);
    ins.in[0] = Ref(op->operand(0));  // x
    ins.in[1] = Ref(op->operand(1));  // gamma
    ins.in[2] = Ref(op->operand(2));  // beta
    ins.in[3] = Ref(y);
    ins.out[0] = Pack32(y->shape().dims.at(0), y->shape().dims.at(1));
    ins.out[1] = Ref(op->result(1));  // mean cache
    ins.out[2] = Ref(op->result(2));  // rstd cache
  } else if (m == "sc_low.layer_norm_grad") {
    const sir::Value* dx = op->result(0);
    Set(OpCode::kLayerNormBwd);
    ins.in[0] = Ref(op->operand(0));  // dy
    ins.in[1] = Ref(op->operand(1));  // x
    ins.in[2] = Ref(op->operand(2));  // gamma
    ins.in[3] = Ref(dx);
    ins.out[0] = Ref(op->operand(3));  // mean cache
    ins.out[1] = Ref(op->operand(4));  // rstd cache
    ins.out[2] = Pack32(dx->shape().dims.at(0), dx->shape().dims.at(1));
  } else if (m == "sc_high.embedding") {
    const sir::Value* out = op->result(0);
    const sir::Value* table = op->operand(1);
    Set(OpCode::kEmbedFwd);
    ins.in[0] = Ref(op->operand(0));  // i32 tokens
    ins.in[1] = Ref(table);           // rodata [V, D]
    ins.in[2] = Ref(out);
    ins.out[0] = static_cast<uint64_t>(out->shape().dims.at(0));
    ins.out[1] = Pack32(table->shape().dims.at(0), table->shape().dims.at(1));
  } else if (m == "sc_high.rms_norm") {
    const sir::Value* y = op->result(0);
    Set(OpCode::kRmsNormFwd);
    ins.imm = NormEpsImm(*op);
    ins.in[0] = Ref(op->operand(0));  // x
    ins.in[1] = Ref(op->operand(1));  // gamma
    ins.in[2] = Ref(y);
    ins.in[3] = Ref(op->result(1));  // rstd cache
    ins.out[0] = Pack32(y->shape().dims.at(0), y->shape().dims.at(1));
  } else if (m == "sc_low.rms_norm_grad") {
    const sir::Value* dx = op->result(0);
    Set(OpCode::kRmsNormBwd);
    ins.in[0] = Ref(op->operand(0));  // dy
    ins.in[1] = Ref(op->operand(1));  // x
    ins.in[2] = Ref(op->operand(2));  // gamma
    ins.in[3] = Ref(dx);
    ins.out[0] = Ref(op->operand(3));  // rstd cache
    ins.out[1] = Pack32(dx->shape().dims.at(0), dx->shape().dims.at(1));
  } else {
    return false;
  }
  return true;
}

bool OpLowering::RopeAndAttention() {
  if (m == "sc_high.rope" || m == "sc_low.rope_grad" ||
      m == "sc_high.attention" || m == "sc_low.attn_dp" ||
      m == "sc_low.attn_dv" || m == "sc_low.attn_dq" || m == "sc_low.attn_dk" ||
      m == "sc_low.attn_dq_tiled" || m == "sc_low.attn_dk_tiled" ||
      m == "sc_low.attn_dv_tiled") {
    // Shared sequence geometry, packed as B<<32|S and H<<32|d. Derived
    // from a designated [T, H*d] activation of the op plus the heads/seq
    // attributes the frontend validated (seq | T, heads | D).
    const sir::Value* act = m == "sc_high.attention" ? op->operand(0)
                            : m == "sc_low.attn_dp"  ? op->operand(0)
                                                     : op->result(0);
    const int64_t heads = op->getAttrAs<int64_t>("heads").value_or(0);
    const int64_t seq = op->getAttrAs<int64_t>("seq").value_or(0);
    const int64_t rows = act->shape().dims.at(0);
    const int64_t width = act->shape().dims.at(1);
    if (heads <= 0 || seq <= 0 || rows % seq != 0 || width % heads != 0) {
      error = "malformed sequence geometry on '" + std::string(m) + "'";
      return true;
    }
    const uint64_t bs = Pack32(rows / seq, seq);
    const uint64_t hd = Pack32(heads, width / heads);
    if (m == "sc_high.rope" || m == "sc_low.rope_grad") {
      Set(m == "sc_high.rope" ? OpCode::kRopeFwd : OpCode::kRopeBwd);
      ins.in[0] = Ref(op->operand(0));
      ins.in[1] = Ref(op->result(0));
      // v12: the hoisted angle table, when RopeTableHoister ran.
      if (op->numOperands() > 1) ins.in[2] = Ref(op->operand(1));
      ins.out[0] = bs;
      ins.out[1] = hd;
      // The parser always sets "base" (SMF v5 attr1, or the format
      // default); the fallback only covers SIR built by hand in tests.
      ins.out[2] =
          F32Bits(op->getAttrAs<float>("base").value_or(kSmfDefaultRopeBase));
    } else if (m == "sc_high.attention" && op->numOperands() == 4) {
      // The segment mask (v25, #191): the starts are the fourth operand,
      // the geometry packed into one word to free a slot for them.
      // Tiled (v26): the same layout with the stats row where the cache
      // was — the backward reads the row's start from there.
      Set(op->hasAttribute("tiled") ? OpCode::kAttnFwdTiledSeg
                                    : OpCode::kAttnFwdSeg);
      ins.in[0] = Ref(op->operand(0));
      ins.in[1] = Ref(op->operand(1));
      ins.in[2] = Ref(op->operand(2));
      ins.in[3] = Ref(op->result(0));
      ins.out[0] = Ref(op->result(1));
      ins.out[1] = PackAttnGeometry(
          static_cast<uint64_t>(rows / seq), static_cast<uint64_t>(seq),
          static_cast<uint64_t>(heads), static_cast<uint64_t>(width / heads));
      ins.out[2] = Ref(op->operand(3));
    } else if (m == "sc_high.attention") {
      // Tiled (v15): the same slots, the stats row where the cache was.
      Set(op->hasAttribute("tiled") ? OpCode::kAttnFwdTiled : OpCode::kAttnFwd);
      ins.in[0] = Ref(op->operand(0));  // q
      ins.in[1] = Ref(op->operand(1));  // k
      ins.in[2] = Ref(op->operand(2));  // v
      ins.in[3] = Ref(op->result(0));   // o
      ins.out[0] = Ref(op->result(1));  // probs cache | stats row
      ins.out[1] = bs;
      ins.out[2] = hd;
    } else if (m.ends_with("_tiled")) {
      // The tiled backward: q, k, v, dO in; stats and the result in the
      // out slots with the geometry packed into one word.
      Set(m == "sc_low.attn_dq_tiled"   ? OpCode::kAttnDQTiled
          : m == "sc_low.attn_dk_tiled" ? OpCode::kAttnDKTiled
                                        : OpCode::kAttnDVTiled);
      for (size_t s = 0; s < 4; ++s) ins.in[s] = Ref(op->operand(s));
      ins.out[0] = Ref(op->operand(4));  // stats
      ins.out[1] = Ref(op->result(0));
      ins.out[2] = PackAttnGeometry(
          static_cast<uint64_t>(rows / seq), static_cast<uint64_t>(seq),
          static_cast<uint64_t>(heads), static_cast<uint64_t>(width / heads));
    } else {
      Set(m == "sc_low.attn_dp"   ? OpCode::kAttnDP
          : m == "sc_low.attn_dv" ? OpCode::kAttnDV
          : m == "sc_low.attn_dq" ? OpCode::kAttnDQ
                                  : OpCode::kAttnDK);
      ins.in[0] = Ref(op->operand(0));
      ins.in[1] = Ref(op->operand(1));
      ins.in[2] = Ref(op->result(0));
      ins.out[0] = bs;
      ins.out[1] = hd;
    }
  } else {
    return false;
  }
  return true;
}

bool OpLowering::ClipScaleAndReduce() {
  if (m == "sc_low.softmax_rows_grad") {
    const sir::Value* ds = op->result(0);
    Set(OpCode::kSoftmaxRowsBwd);
    ins.in[0] = Ref(op->operand(0));  // probs
    ins.in[1] = Ref(op->operand(1));  // dp
    ins.in[2] = Ref(ds);
    ins.out[0] = Pack32(ds->shape().dims.at(0), ds->shape().dims.at(1));
  } else if (m == "sc_low.clip_norm") {
    Set(OpCode::kClipNorm);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = F32Bits(op->getAttrAs<float>("max_norm").value_or(0.0f));
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_low.segment_starts") {  // v25, #191
    Set(OpCode::kSegmentStarts);
    const sir::Value* tokens = op->operand(0);
    const int64_t seq = op->getAttrAs<int64_t>("seq").value_or(0);
    const int64_t t = tokens->shape().dims.at(0);
    if (seq <= 0 || t % seq != 0) {
      error = "malformed sequence geometry on 'sc_low.segment_starts'";
      return true;
    }
    ins.in[0] = Ref(tokens);
    ins.in[1] = Ref(op->result(0));
    ins.out[0] = Pack32(t / seq, seq);
    ins.out[1] =
        static_cast<uint64_t>(op->getAttrAs<int64_t>("eos").value_or(0));
  } else if (m == "sc_low.sqnorm_accum") {  // v24, #190
    Set(OpCode::kSqNormAccum);
    ins.in[0] = Ref(op->operand(0));  // g
    ins.in[1] = Ref(op->operand(1));  // the one-float sum
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_low.clip_global") {  // v24, #190
    Set(OpCode::kClipGlobal);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = F32Bits(op->getAttrAs<float>("max_norm").value_or(0.0f));
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_high.scale" || m == "sc_low.scale") {
    Set(OpCode::kScale);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->result(0));
    ins.in[2] = F32Bits(op->getAttrAs<float>("alpha").value_or(1.0f));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_low.reduce_rows") {
    Set(OpCode::kReduceRows);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->result(0));
    ins.out[0] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(1));
  } else {
    return false;
  }
  return true;
}

bool OpLowering::Loss() {
  if (m == "sc_high.softmax_xent") {
    Set(OpCode::kSoftmaxXEntFwd);
    ins.in[0] = Ref(op->operand(0));  // logits
    ins.in[1] = Ref(op->operand(1));  // labels (i32)
    ins.in[2] = Ref(op->result(0));   // loss
    ins.in[3] = Ref(op->result(1));   // probs cache
    ins.out[0] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(1));
  } else if (m == "sc_high.softmax_xent_masked") {
    Set(OpCode::kSoftmaxXEntMaskedFwd);
    ins.in[0] = Ref(op->operand(0));  // logits
    ins.in[1] = Ref(op->operand(1));  // labels (i32; -1 = ignored)
    ins.in[2] = Ref(op->result(0));   // loss
    ins.in[3] = Ref(op->result(1));   // probs cache
    ins.out[0] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->operand(0)->shape().dims.at(1));
  } else if (m == "sc_low.softmax_xent_masked_grad") {
    Set(OpCode::kSoftmaxXEntMaskedBwd);
    ins.in[0] = Ref(op->operand(0));  // probs
    ins.in[1] = Ref(op->operand(1));  // labels
    ins.in[2] = Ref(op->operand(2));  // seed
    ins.in[3] = Ref(op->result(0));   // dlogits
    ins.out[0] = static_cast<uint64_t>(op->result(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->result(0)->shape().dims.at(1));
  } else if (m == "sc_low.softmax_xent_grad") {
    Set(OpCode::kSoftmaxXEntBwd);
    ins.in[0] = Ref(op->operand(0));  // probs
    ins.in[1] = Ref(op->operand(1));  // labels
    ins.in[2] = Ref(op->operand(2));  // seed
    ins.in[3] = Ref(op->result(0));   // dlogits
    ins.out[0] = static_cast<uint64_t>(op->result(0)->shape().dims.at(0));
    ins.out[1] = static_cast<uint64_t>(op->result(0)->shape().dims.at(1));
  } else if (m == "sc_high.mse") {
    Set(OpCode::kMseFwd);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->result(0));
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_low.mse_grad") {
    Set(OpCode::kMseBwd);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->operand(2));
    ins.in[3] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_high.kl_distill") {
    const sir::Value* s = op->operand(0);
    Set(OpCode::kKLDistillFwd);
    ins.in[0] = Ref(s);
    ins.in[1] = Ref(op->operand(1));  // teacher logits
    ins.in[2] = Ref(op->result(0));   // loss
    ins.in[3] = Ref(op->result(1));   // p_s
    ins.out[0] = Ref(op->result(2));  // p_t
    ins.out[1] = Pack32(s->shape().dims.at(0), s->shape().dims.at(1));
    ins.out[2] =
        F32BitsPair(op->getAttrAs<float>("loss_scale").value_or(1.0f),
                    op->getAttrAs<float>("temperature").value_or(1.0f));
  } else if (m == "sc_low.kl_grad") {
    const sir::Value* d = op->result(0);
    Set(OpCode::kKLDistillBwd);
    ins.in[0] = Ref(op->operand(0));  // p_s
    ins.in[1] = Ref(op->operand(1));  // p_t
    ins.in[2] = Ref(op->operand(2));  // seed
    ins.in[3] = Ref(d);               // dlogits
    ins.out[0] = Pack32(d->shape().dims.at(0), d->shape().dims.at(1));
    ins.out[1] =
        F32BitsPair(op->getAttrAs<float>("loss_scale").value_or(1.0f),
                    op->getAttrAs<float>("temperature").value_or(1.0f));
  } else {
    return false;
  }
  return true;
}

bool OpLowering::TableStepAndFill() {
  if (m == "sc_low.rope_table") {
    Set(OpCode::kRopeTable);
    ins.in[0] = Ref(op->result(0));
    ins.out[0] = Pack32(op->getAttrAs<int64_t>("seq").value_or(0),
                        op->getAttrAs<int64_t>("width").value_or(0));
    ins.out[1] =
        F32Bits(op->getAttrAs<float>("base").value_or(kSmfDefaultRopeBase));
  } else if (m == "sc_low.sgd_step") {
    Set(OpCode::kSgdStep);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.out[0] = Vol(op->operand(0));
    // v12: the fused per-tensor clip threshold (0 = none).
    if (auto clip = op->getAttrAs<float>("clip_norm"))
      ins.out[1] = F32Bits(*clip);
  } else if (m == "sc_low.adamw_step") {
    Set(OpCode::kAdamWStep);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.in[2] = Ref(op->operand(2));
    ins.in[3] = Ref(op->operand(3));
    ins.out[0] = Vol(op->operand(0));
    if (auto clip = op->getAttrAs<float>("clip_norm"))
      ins.out[1] = F32Bits(*clip);
  } else if (m == "sc_low.fill") {
    Set(OpCode::kFill);
    ins.in[0] = Ref(op->result(0));
    ins.in[1] = F32Bits(op->getAttrAs<float>("value").value_or(0.0f));
    ins.out[0] = Vol(op->result(0));
  } else if (m == "sc_low.zero") {  // in-place: kFill over an operand
    Set(OpCode::kFill);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = F32Bits(0.0f);
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_low.accumulate") {  // dst += src (v9)
    Set(OpCode::kAccumulate);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->operand(1));
    ins.out[0] = Vol(op->operand(0));
  } else if (m == "sc_low.copy") {
    Set(OpCode::kCopy);
    ins.in[0] = Ref(op->operand(0));
    ins.in[1] = Ref(op->result(0));
    ins.out[0] = Vol(op->result(0));
  } else {
    return false;
  }
  return true;
}

}  // namespace

std::expected<std::vector<UpdateInstruction>, std::string> LowerOps(
    const std::vector<sir::Operation*>& ops, const ResolveFn& resolve,
    const std::unordered_map<const sir::Value*, float>& quant_scales,
    const std::unordered_set<const sir::Value*>& bf16_weights,
    const std::unordered_map<const sir::Value*, uint64_t>&
        quant_column_scales) {
  std::vector<UpdateInstruction> instrs;
  instrs.reserve(ops.size());  // ~1 instruction per non-storage op
  std::string error;

  for (sir::Operation* op : ops) {
    if (!error.empty()) break;
    if (op->mnemonic().starts_with("sc_mem.")) {
      continue;  // storage declaration, no code
    }
    OpLowering lowering(op, resolve, quant_scales, bf16_weights,
                        quant_column_scales, error);
    lowering.Run();
    // An error ends the block's lowering; the instruction it was found
    // on is not part of any result.
    if (!error.empty()) break;
    instrs.push_back(lowering.ins);
  }

  if (!error.empty())
    return generating::Error(generating::kInstructionLowering, error);
  return instrs;
}

}  // namespace seeml::update
