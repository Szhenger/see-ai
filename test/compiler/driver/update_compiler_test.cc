// =============================================================================
// UpdateCompiler tests: plan assembly (header contract, section layout,
// segmented arena binding, debug hooks), per-loss configuration, optimizer
// selection, and the compile-time error surface.
// =============================================================================

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/accountant/resource_analyzer.h"
#include "compiler/frontend/egressor/model_writer.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "runtime/pipeline/dataset.h"
#include "source/contract/split.h"
#include "source/identity/hash.h"
#include "source/plan/metadata.h"
#include "test/support/scoped_temp_dir.h"
#include "source/plan/update_types.h"
#include "source/parallel/parallel_for.h"
#include "source/language/model_format.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::BaseConfig;
using seeml::testing::MakeMlp;
using seeml::testing::MakeTiedMlp;

constexpr int64_t kInDim = 6;
constexpr int64_t kHidden = 10;
constexpr int64_t kOutDim = 3;
constexpr int64_t kBatch = 4;

PlanHeader HeaderOf(const CompiledUpdate& compiled) {
  PlanHeader h;
  std::memcpy(&h, compiled.plan.data(), sizeof(h));
  return h;
}

std::vector<UpdateInstruction> TrainProgramOf(const CompiledUpdate& compiled) {
  const PlanHeader h = HeaderOf(compiled);
  std::vector<UpdateInstruction> instrs(h.train_instr_count);
  std::memcpy(instrs.data(), compiled.plan.data() + h.train_instr_offset,
              h.train_instr_count * sizeof(UpdateInstruction));
  return instrs;
}

std::vector<UpdateInstruction> EvalProgramOf(const CompiledUpdate& compiled) {
  const PlanHeader h = HeaderOf(compiled);
  std::vector<UpdateInstruction> instrs(h.eval_instr_count);
  std::memcpy(instrs.data(), compiled.plan.data() + h.eval_instr_offset,
              h.eval_instr_count * sizeof(UpdateInstruction));
  return instrs;
}

size_t CountOpcode(const std::vector<UpdateInstruction>& instrs, OpCode oc) {
  size_t n = 0;
  for (const UpdateInstruction& ins : instrs)
    if (ins.opcode == static_cast<uint16_t>(oc)) ++n;
  return n;
}

TEST(UpdateCompiler, RopeBaseReachesForwardAndBackwardInstructions) {
  // SMF v5 attr1 → parser "base" attribute → autodiff copy → lowering
  // out[2] on both kRopeFwd and kRopeBwd. A Llama-3-class θ=500k must not
  // silently lower to 10000 anywhere in the train program.
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  const float theta = 500000.0f;
  for (SmfOp& op : model.ops)
    if (op.kind == SmfOpKind::kRope) op.attr1 = std::bit_cast<uint32_t>(theta);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(8)).Compile(model));
  const auto instrs = TrainProgramOf(compiled);
  size_t fwd = 0, bwd = 0;
  for (const UpdateInstruction& ins : instrs) {
    const auto op = static_cast<OpCode>(ins.opcode);
    if (op != OpCode::kRopeFwd && op != OpCode::kRopeBwd) continue;
    (op == OpCode::kRopeFwd ? fwd : bwd)++;
    EXPECT_EQ(static_cast<uint32_t>(ins.out[2]),
              std::bit_cast<uint32_t>(theta));
  }
  EXPECT_EQ(fwd, 2u);
  EXPECT_GE(bwd, 1u);
}

TEST(UpdateCompiler, PlanHeaderContract) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));

  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.magic, kSeeuMagic);
  EXPECT_EQ(h.version, kSeeuVersion);
  EXPECT_EQ(h.batch, static_cast<uint64_t>(kBatch));
  EXPECT_EQ(h.label_kind, 1u);  // softmax cross-entropy: class indices
  EXPECT_EQ(h.input_floats, static_cast<uint64_t>(kBatch * kInDim));
  EXPECT_EQ(h.label_bytes, kBatch * sizeof(int32_t));
  EXPECT_EQ(h.optimizer_kind, 1u);  // AdamW default

  // The memory contract is internally consistent.
  EXPECT_GT(h.arena_size, 0u);
  EXPECT_EQ(h.arena_size % 64, 0u);
  EXPECT_LE(h.persistent_size, h.arena_size);
  EXPECT_EQ(h.persist_init_size, h.persistent_size);
  EXPECT_EQ(h.train_instr_count, compiled.train_instruction_count);
  EXPECT_EQ(h.merge_instr_count, compiled.merge_instruction_count);
  EXPECT_GT(h.rodata_size, 0u);
  EXPECT_EQ(h.emit_count, compiled.adapters.size());

  // Every section lies inside the plan blob.
  EXPECT_LE(h.train_instr_offset +
                h.train_instr_count * sizeof(UpdateInstruction),
            compiled.plan.size());
  EXPECT_LE(h.rodata_offset + h.rodata_size, compiled.plan.size());
  EXPECT_LE(h.emit_table_offset + h.emit_count * sizeof(EmitEntry),
            compiled.plan.size());

  // I/O slots live in the mutable arena.
  EXPECT_FALSE(IsRodataRef(h.input_ref));
  EXPECT_FALSE(IsRodataRef(h.label_ref));
  EXPECT_FALSE(IsRodataRef(h.loss_ref));
}

TEST(UpdateCompiler, DebugHooksDescribeAdaptersAndParams) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 2);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));

  ASSERT_EQ(compiled.adapters.size(), 2u);
  ASSERT_EQ(compiled.params.size(), 4u);  // {A, B} x 2 layers

  for (const AdapterDebugInfo& a : compiled.adapters) {
    EXPECT_EQ(a.r, 4);
    EXPECT_NEAR(a.scale, 2.0f, 1e-6);
    EXPECT_TRUE(IsRodataRef(a.weight_rodata_ref));
    EXPECT_FALSE(IsRodataRef(a.a_ref));
    EXPECT_FALSE(IsRodataRef(a.b_ref));
    // Adapters live in the checkpointed persistent segment.
    EXPECT_LT(RefOffset(a.a_ref), compiled.persistent_size);
    EXPECT_LT(RefOffset(a.b_ref), compiled.persistent_size);
  }
  EXPECT_EQ(compiled.adapters[0].weight_name, "w1");
  EXPECT_EQ(compiled.adapters[1].weight_name, "w2");

  // Params are sorted by id and carry live gradient refs.
  for (size_t i = 1; i < compiled.params.size(); ++i)
    EXPECT_LT(compiled.params[i - 1].id, compiled.params[i].id);
  for (const ParamDebugInfo& p : compiled.params) {
    EXPECT_NE(p.param_ref, kNullRef);
    EXPECT_NE(p.grad_ref, kNullRef);
    EXPECT_GT(p.count, 0u);
  }

  // The SIR dump is a human-readable rendering of both programs.
  EXPECT_TRUE(compiled.sir_dump.empty());  // debug output is opt-in
  UpdateConfig dumping = BaseConfig(kBatch);
  dumping.dump_sir = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate dumped,
                       UpdateCompiler(dumping).Compile(model));
  EXPECT_TRUE(dumped.plan == compiled.plan);  // the dump changes no byte
  compiled = std::move(dumped);
  EXPECT_STR_CONTAINS(compiled.sir_dump, "sc_high.matmul");
  EXPECT_STR_CONTAINS(compiled.sir_dump, "merge program");
}

TEST(UpdateCompiler, CompilationIsDeterministic) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 3);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate b,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  EXPECT_TRUE(a.plan == b.plan);
}

TEST(UpdateCompiler, TiedWeightMaterializesOnce) {
  SmfModel model = MakeTiedMlp(4, 4);
  UpdateConfig config = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  // The tied tensor resolves to a single SIR value, and both consuming
  // MatMuls share ONE adapter pair: per-site pairs would train fine but
  // commit W + Δ_1 + Δ_2 to the single file range, polluting every site
  // with every other site's delta. One adapter -> one delta -> one emit
  // entry, and the committed weight is exactly the W + Δ every site
  // computed during training.
  const PlanHeader h = HeaderOf(compiled);
  ASSERT_EQ(compiled.adapters.size(), 1u);
  ASSERT_EQ(h.emit_count, 1u);
}

TEST(UpdateCompiler, MseLossUsesDenseLabels) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 5);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kMse;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));

  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 2u);
  EXPECT_EQ(h.label_bytes,
            static_cast<uint64_t>(kBatch * kOutDim) * sizeof(float));

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kMseFwd), 1u);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kSoftmaxXEntFwd), 0u);
}

TEST(UpdateCompiler, DistillationRequiresAndUsesTeacher) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 6);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 7);

  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student),
                        "requires a teacher");

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 0u);  // the teacher provides the signal in-graph
  EXPECT_EQ(h.label_ref, kNullRef);

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kKLDistillFwd), 1u);
  // Teacher weights ride along frozen: no adapters on them.
  EXPECT_EQ(compiled.adapters.size(), 2u);

  // The objective is decided here: the plan carries T in the low word and
  // the Hinton T^2 scale in the high word of the temperature operand, on
  // both the forward and its VJP (#13).
  const auto bits = [](float f) {
    return uint64_t{std::bit_cast<uint32_t>(f)};
  };
  const float T = config.temperature;
  const uint64_t want = (bits(T * T) << 32) | bits(T);
  size_t fwd = 0, bwd = 0;
  for (const auto& ins : instrs) {
    if (ins.opcode == static_cast<uint16_t>(OpCode::kKLDistillFwd)) {
      EXPECT_EQ(ins.out[2], want);
      ++fwd;
    } else if (ins.opcode == static_cast<uint16_t>(OpCode::kKLDistillBwd)) {
      EXPECT_EQ(ins.out[1], want);
      ++bwd;
    }
  }
  EXPECT_EQ(fwd, 1u);
  EXPECT_EQ(bwd, 1u);
  EXPECT_TRUE(HeaderOf(compiled).version >= seeml::update::kSeeuKlScaleVersion);
}

TEST(UpdateCompiler, GradientAccumulationSplitsTheStream) {
  // G = 4: the train section becomes the grad program (forward, backward,
  // one fold per trainable into a persistent accumulator) and a step
  // section carries clip / step / zero; the seed is 1/G. G = 1 (the
  // default) is the classic one-batch program with no step section.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 51);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.clip_norm = 1.0f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(config).Compile(model));
  config.grad_accum_steps = 4;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate accum,
                       UpdateCompiler(config).Compile(model));
  const PlanHeader hp = HeaderOf(plain), ha = HeaderOf(accum);
  EXPECT_EQ(hp.grad_accum_steps, 1u);
  EXPECT_EQ(hp.step_instr_count, 0u);
  EXPECT_EQ(ha.grad_accum_steps, 4u);
  EXPECT_GT(ha.step_instr_count, 0u);
  EXPECT_TRUE(ha.version >= kSeeuGradAccumVersion);
  const size_t trainables = 2 * accum.adapters.size();

  const auto grad = TrainProgramOf(accum);
  EXPECT_EQ(CountOpcode(grad, OpCode::kAccumulate), trainables);
  EXPECT_EQ(CountOpcode(grad, OpCode::kAdamWStep), 0u);
  EXPECT_EQ(CountOpcode(grad, OpCode::kClipNorm), 0u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAccumulate), 0u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAdamWStep), trainables);

  std::vector<UpdateInstruction> step(ha.step_instr_count);
  std::memcpy(step.data(), accum.plan.data() + ha.step_instr_offset,
              step.size() * sizeof(UpdateInstruction));
  // The clip rides the step (plan v12): no standalone pass over the
  // accumulator, its threshold in every step's out[1].
  EXPECT_EQ(CountOpcode(step, OpCode::kClipNorm), 0u);
  EXPECT_EQ(CountOpcode(step, OpCode::kAdamWStep), trainables);
  for (const auto& ins : step)
    if (ins.opcode == static_cast<uint16_t>(OpCode::kAdamWStep))
      EXPECT_EQ(std::bit_cast<float>(static_cast<uint32_t>(ins.out[1])), 1.0f);
  EXPECT_EQ(CountOpcode(step, OpCode::kFill), trainables);  // the zeroes
  EXPECT_EQ(CountOpcode(step, OpCode::kAccumulate), 0u);
  // The seed carries 1/G: exactly one fill of 0.25 in the grad program.
  size_t quarter_seeds = 0;
  for (const auto& ins : grad)
    if (ins.opcode == static_cast<uint16_t>(OpCode::kFill) &&
        std::bit_cast<float>(static_cast<uint32_t>(ins.in[1])) == 0.25f)
      ++quarter_seeds;
  EXPECT_EQ(quarter_seeds, 1u);
  // One extra grad-sized set in the persistent segment, activations
  // unchanged (the arena beyond it is the same micro-batch workspace).
  uint64_t grad_bytes = 0;  // each accumulator is a 64-byte-aligned slot
  for (const auto& p : accum.params) {
    grad_bytes += (p.count * sizeof(float) + 63) & ~uint64_t{63};
    EXPECT_NE(p.acc_ref, kNullRef);
  }
  for (const auto& p : plain.params) EXPECT_EQ(p.acc_ref, kNullRef);
  EXPECT_EQ(accum.persistent_size, plain.persistent_size + grad_bytes);
}

TEST(UpdateCompiler, GradientAccumulationSplitsAnSgdStream) {
  // SGD has no moments: per trainable the step program is step + zero with
  // the clip folded into the step (plan v12) — or, with fuse_clip off, the
  // v11 shape clip + step + zero — and the grad program's folds precede it.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 52);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.kind = OptimizerKind::kSgd;
  config.optimizer.clip_norm = 0.5f;
  config.grad_accum_steps = 2;
  for (const bool fuse : {true, false}) {
    config.fuse_clip = fuse;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate c,
                         UpdateCompiler(config).Compile(model));
    const PlanHeader h = HeaderOf(c);
    const size_t trainables = 2 * c.adapters.size();
    std::vector<UpdateInstruction> step(h.step_instr_count);
    std::memcpy(step.data(), c.plan.data() + h.step_instr_offset,
                step.size() * sizeof(UpdateInstruction));
    EXPECT_EQ(step.size(), (fuse ? 2 : 3) * trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kClipNorm), fuse ? 0u : trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kSgdStep), trainables);
    EXPECT_EQ(CountOpcode(step, OpCode::kFill), trainables);
    EXPECT_EQ(static_cast<OpCode>(step[0].opcode),
              fuse ? OpCode::kSgdStep : OpCode::kClipNorm);
    for (const auto& ins : step)
      if (ins.opcode == static_cast<uint16_t>(OpCode::kSgdStep))
        EXPECT_EQ(ins.out[1],
                  fuse ? uint64_t{std::bit_cast<uint32_t>(0.5f)} : 0u);
    const auto grad = TrainProgramOf(c);
    EXPECT_EQ(CountOpcode(grad, OpCode::kAccumulate), trainables);
    EXPECT_EQ(CountOpcode(grad, OpCode::kSgdStep), 0u);
    EXPECT_EQ(CountOpcode(grad, OpCode::kClipNorm), 0u);
  }
}

TEST(UpdateCompiler, CertifiedBf16MarksOnlyTheFrozenGemmsOfTraining) {
  // --precision certified-bf16 (plan v18): kFlagRelaxed on every train /
  // step GEMM whose B is rodata (NN forward, NT dX), on nothing else, and
  // on no eval instruction; the f32 default emits no bit at all, so the
  // two plans differ only in those flags.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 53);
  for (int storage = 0; storage < 3; ++storage) {
    UpdateConfig exact = BaseConfig(kBatch);
    exact.quantize_base = storage == 1;
    exact.bf16_base = storage == 2;
    UpdateConfig relaxed = exact;
    relaxed.precision = Precision::kCertifiedBf16;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate a, UpdateCompiler(exact).Compile(model));
    ASSERT_OK_AND_ASSIGN(CompiledUpdate b, UpdateCompiler(relaxed).Compile(model));
    EXPECT_EQ(a.relaxed_gemms, 0u);
    const auto ta = TrainProgramOf(a), tb = TrainProgramOf(b);
    ASSERT_EQ(ta.size(), tb.size());
    uint64_t flagged = 0;
    for (size_t i = 0; i < ta.size(); ++i) {
      const auto op = static_cast<OpCode>(tb[i].opcode);
      const bool frozen =
          (op == OpCode::kGemmNN || op == OpCode::kGemmNT ||
           op == OpCode::kGemmNNQ8 || op == OpCode::kGemmNTQ8 ||
           op == OpCode::kGemmNNBF16 || op == OpCode::kGemmNTBF16) &&
          IsRodataRef(tb[i].in[1]);
      EXPECT_EQ((tb[i].flags & kFlagRelaxed) != 0, frozen);
      EXPECT_EQ(tb[i].flags & ~kFlagRelaxed, ta[i].flags);
      EXPECT_EQ(ta[i].flags & kFlagRelaxed, 0u);
      flagged += frozen;
    }
    // Every adapted weight: one forward, and one dX for all but the first.
    EXPECT_EQ(flagged, 2 * b.adapters.size() - 1);
    EXPECT_EQ(b.relaxed_gemms, flagged);
    for (const auto& ins : EvalProgramOf(b))
      EXPECT_EQ(ins.flags & kFlagRelaxed, 0u);
    EXPECT_EQ(HeaderOf(b).version, kSeeuVersion);
    EXPECT_GE(kSeeuVersion, kSeeuRelaxedVersion);
  }
}

TEST(UpdateCompiler, CertifiedPrecisionAlsoMarksTheTiledAttentionOfTraining) {
  // Plan v20 (F3 #131): under --precision certified-bf16 the tiled
  // attention instructions of the train program carry kFlagRelaxed beside
  // the frozen-weight GEMMs — all four per attention op (forward, dQ, dK,
  // dV) — and the eval program's do not; the cached family has no relaxed
  // form, so a cached plan marks GEMMs only.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 8, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig tiled = BaseConfig(batch);
  tiled.attention = AttentionKind::kTiled;
  tiled.precision = Precision::kCertifiedBf16;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate t, UpdateCompiler(tiled).Compile(model));
  auto is_tiled = [](uint16_t opcode) {
    const auto op = static_cast<OpCode>(opcode);
    return op == OpCode::kAttnFwdTiled || op == OpCode::kAttnDQTiled ||
           op == OpCode::kAttnDKTiled || op == OpCode::kAttnDVTiled;
  };
  uint64_t tiled_train = 0, flagged_attention = 0;
  for (const auto& ins : TrainProgramOf(t)) {
    if (!is_tiled(ins.opcode)) continue;
    ++tiled_train;
    flagged_attention += (ins.flags & kFlagRelaxed) != 0;
  }
  EXPECT_GT(tiled_train, 0u);
  EXPECT_EQ(tiled_train % 4, 0u);
  EXPECT_EQ(flagged_attention, tiled_train);
  EXPECT_EQ(t.relaxed_attention, tiled_train);
  EXPECT_GT(t.relaxed_gemms, 0u);
  uint64_t tiled_eval = 0;
  for (const auto& ins : EvalProgramOf(t)) {
    tiled_eval += is_tiled(ins.opcode);
    EXPECT_EQ(ins.flags & kFlagRelaxed, 0u);
  }
  EXPECT_GT(tiled_eval, 0u);  // the eval forward is tiled too, and exact
  EXPECT_EQ(HeaderOf(t).version, kSeeuVersion);
  EXPECT_GE(kSeeuVersion, kSeeuRelaxedAttentionVersion);
  UpdateConfig cached = tiled;
  cached.attention = AttentionKind::kCached;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate c, UpdateCompiler(cached).Compile(model));
  EXPECT_EQ(c.relaxed_attention, 0u);
  EXPECT_EQ(c.relaxed_gemms, t.relaxed_gemms);
  // f32 precision marks nothing, tiled or not.
  UpdateConfig exact = tiled;
  exact.precision = Precision::kF32;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate e, UpdateCompiler(exact).Compile(model));
  EXPECT_EQ(e.relaxed_attention, 0u);
  for (const auto& ins : TrainProgramOf(e))
    EXPECT_EQ(ins.flags & kFlagRelaxed, 0u);
}

TEST(UpdateCompiler, Bf16BaseHalvesRodataAndLowersTheWideningGemms) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 53);
  UpdateConfig f32 = BaseConfig(kBatch);
  UpdateConfig bf16 = BaseConfig(kBatch);
  bf16.bf16_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a, UpdateCompiler(f32).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate b, UpdateCompiler(bf16).Compile(model));
  // Declared at the current version, which is at least the one that
  // introduced the widening opcodes (the validator gates them on it).
  EXPECT_EQ(HeaderOf(b).version, kSeeuVersion);
  EXPECT_GE(kSeeuVersion, kSeeuBf16Version);
  // Both matmul weights are adapted: a forward NN per weight, and a
  // backward NT (dX through the frozen weight) for every weight but the
  // first — nothing trainable sits before layer 1's input, so autodiff
  // prunes that adjoint.
  const auto instrs = TrainProgramOf(b);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNNBF16), b.adapters.size());
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNTBF16), b.adapters.size() - 1);
  // The adapter GEMMs (X@A, t@B and their backward) stay f32: A and B are
  // trainable arena parameters, not frozen rodata.
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNN),
            CountOpcode(TrainProgramOf(a), OpCode::kGemmNN) - b.adapters.size());
  EXPECT_EQ(CountOpcode(instrs, OpCode::kGemmNNQ8), 0u);
  for (const auto& ad : b.adapters) {
    EXPECT_TRUE(ad.bf16);
    EXPECT_EQ(ad.quant_scale, 0.0f);
  }
  for (const auto& ad : a.adapters) EXPECT_FALSE(ad.bf16);
  // Weights halve; the biases stay f32 (they are AddBias operands, not
  // matmul weights), so rodata shrinks by exactly the weight bytes / 2.
  uint64_t weight_bytes = 0;
  for (const auto& ad : b.adapters)
    weight_bytes += static_cast<uint64_t>(ad.k * ad.m) * sizeof(float);
  EXPECT_LT(b.rodata_size, a.rodata_size);
  EXPECT_GE(a.rodata_size - b.rodata_size, weight_bytes / 2 - 64 * b.adapters.size());
  // One storage precision per weight.
  UpdateConfig both = BaseConfig(kBatch);
  both.bf16_base = true;
  both.quantize_base = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(both).Compile(model),
                        "mutually exclusive");
}

TEST(UpdateCompiler, RejectsTeacherShapeMismatch) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 8);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;

  SmfModel narrow_input = MakeMlp(kInDim + 2, kHidden, kOutDim, 9);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &narrow_input),
                        "teacher input dimensionality");

  SmfModel wrong_output = MakeMlp(kInDim, kHidden, kOutDim + 1, 10);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &wrong_output),
                        "teacher output dimensionality");
}

TEST(UpdateCompiler, CompositeLossCombinesBothTerms) {
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 11);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 12);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kXEntPlusKL;
  config.distill_weight = 0.3f;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(student, &teacher));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_EQ(h.label_kind, 1u);  // the cross-entropy term still needs labels

  const auto instrs = TrainProgramOf(compiled);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kSoftmaxXEntFwd), 1u);
  EXPECT_EQ(CountOpcode(instrs, OpCode::kKLDistillFwd), 1u);
}

TEST(UpdateCompiler, LoraSitesCostNoActivationSizedPassOfTheirOwn) {
  // E10 (#93). Per adapted matmul the program used to spend two
  // activation-sized elementwise passes forward (scale, add) and two
  // backward (the scale's VJP over dC, the fan-out add into dX). Now the
  // scale lives on the rank-r activation and both adds ride their GEMM.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 31);
  UpdateConfig config = BaseConfig(kBatch);  // rank 4, alpha 8
  UpdateConfig unfolded = config;
  unfolded.fuse_gemm_addend = false;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate folded, UpdateCompiler(config).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(unfolded).Compile(model));
  ASSERT_EQ(folded.adapters.size(), 2u);

  // No scale instruction is left in either program (F7 #138, plan v19):
  // alpha/r = 2 is the alpha of the three rank-r products at every site —
  // forward u = t@B (NN), backward dt = dC@B^T (NT) and dB = t^T@dC (TN).
  for (const CompiledUpdate* c : {&folded, &plain}) {
    EXPECT_EQ(CountOpcode(TrainProgramOf(*c), OpCode::kScale), 0u);
    size_t nn_alpha = 0, nt_alpha = 0, tn_alpha = 0;
    for (const UpdateInstruction& ins : TrainProgramOf(*c)) {
      if (ins.imm == 0) continue;
      const auto op = static_cast<OpCode>(ins.opcode);
      if (op != OpCode::kGemmNN && op != OpCode::kGemmNT &&
          op != OpCode::kGemmTN)
        continue;  // the norms' epsilon word
      EXPECT_EQ(GemmAlphaOf(ins.imm), 2.0f);
      if (op == OpCode::kGemmNN) ++nn_alpha;
      if (op == OpCode::kGemmNT) ++nt_alpha;
      if (op == OpCode::kGemmTN) ++tn_alpha;
    }
    EXPECT_EQ(nn_alpha, folded.adapters.size());
    EXPECT_EQ(nt_alpha, folded.adapters.size());
    EXPECT_EQ(tn_alpha, folded.adapters.size());
  }

  // Folded: one forward addend per site (C' = C + ts@B), and one backward
  // wherever dX is needed at all — the first layer's input takes no
  // gradient — each naming a real addend; no standalone add is left.
  size_t nn = 0, nt = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(folded)) {
    if (!(ins.flags & kFlagGemmAddend)) continue;
    EXPECT_EQ(ins.flags, kFlagGemmAddend);
    EXPECT_NE(ins.in[3], kNullRef);
    if (ins.opcode == static_cast<uint16_t>(OpCode::kGemmNN)) ++nn;
    if (ins.opcode == static_cast<uint16_t>(OpCode::kGemmNT)) ++nt;
  }
  EXPECT_EQ(nn, 2u);
  EXPECT_EQ(nt, 1u);
  EXPECT_EQ(CountOpcode(TrainProgramOf(folded), OpCode::kAddEW), 0u);

  // Unfolded: the same sums as instructions of their own, and no flag.
  for (const UpdateInstruction& ins : TrainProgramOf(plain))
    EXPECT_EQ(ins.flags & kFlagGemmAddend, 0);
  EXPECT_EQ(CountOpcode(TrainProgramOf(plain), OpCode::kAddEW), 3u);
  EXPECT_EQ(HeaderOf(plain).train_instr_count,
            HeaderOf(folded).train_instr_count + 3);
  EXPECT_EQ(HeaderOf(plain).eval_instr_count,
            HeaderOf(folded).eval_instr_count + 2);
}

TEST(UpdateCompiler, PerColumnInt8ScalesKeepTheBulkOfAnOutlierWeight) {
  // E12 (#95): LLM projections carry |w| outliers 10-50x the bulk. Under a
  // per-tensor max-abs scale one outlier column sets the step for every
  // column, collapsing the bulk to a few int8 levels; per-column scales
  // (v17) give each column its own. Plant one column at 50x the bulk —
  // with raw mt19937_64 bits, never a distribution — compile, and read the
  // plan's own levels and scales back: the bulk's dequantization error
  // falls by well over 10x, while the outlier column's own error cannot
  // (its scale is the per-tensor one either way) and does not change.
  const int64_t in_dim = 64, hidden = 48, out_dim = 3;
  SmfModel model = MakeMlp(in_dim, hidden, out_dim, 57);
  SmfTensor* w = nullptr;
  for (SmfTensor& t : model.tensors)
    if (t.is_const && t.dims.size() == 2 && t.dims[0] == in_dim) w = &t;
  ASSERT_NE(w, nullptr);
  const size_t K = static_cast<size_t>(w->dims[0]);
  const size_t Mc = static_cast<size_t>(w->dims[1]);
  std::mt19937_64 bits(4242);
  auto unit = [&] {  // [-1, 1), exact in f32, portable
    return static_cast<float>(static_cast<double>(bits() >> 40) /
                                  static_cast<double>(1ull << 23) -
                              1.0);
  };
  std::vector<float> wf(K * Mc);
  const size_t outlier = 7;
  for (size_t k = 0; k < K; ++k)
    for (size_t m = 0; m < Mc; ++m)
      wf[k * Mc + m] = (m == outlier ? 50.0f : 1.0f) * 0.02f * unit();
  w->data = seeml::testing::AsPayload(wf);
  w->byte_size = wf.size() * sizeof(float);

  UpdateConfig config = BaseConfig(kBatch);
  config.quantize_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  uint64_t levels_ref = kNullRef;
  for (const auto& a : compiled.adapters)
    if (a.weight_name == w->name) levels_ref = a.weight_rodata_ref;
  ASSERT_NE(levels_ref, kNullRef);
  uint64_t scales_ref = kNullRef;
  for (const UpdateInstruction& ins : TrainProgramOf(compiled))
    if (ins.opcode == static_cast<uint16_t>(OpCode::kGemmNNQ8) &&
        ins.in[1] == levels_ref) {
      EXPECT_TRUE((ins.flags & kFlagQ8ColScale) != 0);
      scales_ref = ins.in[3];
    }
  ASSERT_TRUE(IsRodataRef(scales_ref));
  const PlanHeader h = HeaderOf(compiled);
  const uint8_t* rodata = compiled.plan.data() + h.rodata_offset;
  const auto* q = reinterpret_cast<const int8_t*>(rodata + RefOffset(levels_ref));
  const auto* cs = reinterpret_cast<const float*>(rodata + RefOffset(scales_ref));

  float tensor_max = 0.0f;
  for (float v : wf) tensor_max = std::max(tensor_max, std::fabs(v));
  const float per_tensor = tensor_max / 127.0f;
  double bulk_sq_col = 0.0, bulk_sq_ten = 0.0, bulk_abs_col = 0.0,
         bulk_abs_ten = 0.0, out_max_col = 0.0, out_max_ten = 0.0;
  for (size_t k = 0; k < K; ++k)
    for (size_t m = 0; m < Mc; ++m) {
      const float v = wf[k * Mc + m];
      const double e_col = std::fabs(q[k * Mc + m] * cs[m] - v);
      const float lvl = std::clamp(std::round(v / per_tensor), -127.0f, 127.0f);
      const double e_ten = std::fabs(lvl * per_tensor - v);
      if (m == outlier) {
        out_max_col = std::max(out_max_col, e_col);
        out_max_ten = std::max(out_max_ten, e_ten);
      } else {
        bulk_sq_col += e_col * e_col;
        bulk_sq_ten += e_ten * e_ten;
        bulk_abs_col += e_col;
        bulk_abs_ten += e_ten;
      }
    }
  const double rms_ratio = std::sqrt(bulk_sq_ten / bulk_sq_col);
  const double mean_ratio = bulk_abs_ten / bulk_abs_col;
  std::fprintf(stderr,
               "per-column int8: bulk RMS error down %.1fx, mean-abs %.1fx; "
               "outlier column max-abs %.3g vs %.3g\n",
               rms_ratio, mean_ratio, out_max_col, out_max_ten);
  EXPECT_GT(rms_ratio, 10.0);
  EXPECT_GT(mean_ratio, 10.0);
  EXPECT_EQ(cs[outlier], per_tensor);  // the outlier's scale is the tensor's
  EXPECT_NEAR(out_max_col, out_max_ten, 1e-9);
}

TEST(UpdateCompiler, NarrowWeightGemmsKeepTheirOwnInstruction) {
  // The addend exists for the f32 GEMMs only. At a LoRA site over an int8
  // or bf16 frozen weight the rank-r GEMM is the one that folds, so the
  // fold loses nothing — and the validator would refuse the other choice.
  for (const bool bf16 : {false, true}) {
    SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 32);
    UpdateConfig config = BaseConfig(kBatch);
    config.quantize_base = !bf16;
    config.bf16_base = bf16;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                         UpdateCompiler(config).Compile(model));
    size_t folded = 0;
    for (const UpdateInstruction& ins : TrainProgramOf(compiled)) {
      if (!(ins.flags & kFlagGemmAddend)) continue;
      ++folded;
      EXPECT_TRUE(ins.opcode == static_cast<uint16_t>(OpCode::kGemmNN) ||
                  ins.opcode == static_cast<uint16_t>(OpCode::kGemmNT));
      EXPECT_FALSE(IsRodataRef(ins.in[1]));  // the adapter, never the base
    }
    EXPECT_EQ(folded, 3u);
  }
}

TEST(UpdateCompiler, EpilogueFusionShrinksDistillPrograms) {
  // Under distillation the frozen teacher's GEMM -> AddBias -> activation
  // chains fuse into flagged GEMM epilogues; the orphaned ops are DCE'd, so
  // both the train and eval programs shrink and the arena loses their
  // transient slots. The same compile with fusion off is the reference.
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 31);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 32);
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;
  UpdateConfig unfused_config = config;
  unfused_config.fuse_epilogues = false;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate fused,
                       UpdateCompiler(config).Compile(student, &teacher));
  ASSERT_OK_AND_ASSIGN(
      CompiledUpdate unfused,
      UpdateCompiler(unfused_config).Compile(student, &teacher));

  const PlanHeader fh = HeaderOf(fused);
  const PlanHeader uh = HeaderOf(unfused);
  EXPECT_LT(fh.train_instr_count, uh.train_instr_count);
  EXPECT_LT(fh.eval_instr_count, uh.eval_instr_count);
  // No arena-size assertion: fusion removes transient values, but first-fit
  // offsets are not monotone in the interval set (the fused GEMM's result
  // inherits the chain output's longer liveness), so the high-water mark
  // may move either way by a packing accident.

  // The fused stream carries epilogue flags on forward GEMMs; the unfused
  // stream carries none anywhere (flags == 0 was the pre-v5 invariant).
  // (The GEMM addend, v14, is a different pass's flag and rides both.)
  size_t flagged = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(fused)) {
    EXPECT_EQ(ins.flags & static_cast<uint16_t>(~kKnownFlagsMask), 0);
    if ((ins.flags & kEpilogueFlagsMask) == 0) continue;
    ++flagged;
    EXPECT_EQ(ins.opcode, static_cast<uint16_t>(OpCode::kGemmNN));
  }
  EXPECT_GT(flagged, 0u);
  for (const UpdateInstruction& ins : TrainProgramOf(unfused))
    EXPECT_EQ(ins.flags & kEpilogueFlagsMask, 0);

  // The teacher's hidden layer fused bias+relu, its logits layer bias-only:
  // no teacher AddBias survives, and the student's (LoRA-interposed) ones
  // remain — 2 in the fused stream vs 4 unfused.
  EXPECT_EQ(CountOpcode(TrainProgramOf(fused), OpCode::kAddBias),
            CountOpcode(TrainProgramOf(unfused), OpCode::kAddBias) - 2);
}

TEST(UpdateCompiler, OptimizerSelectionShapesTheProgram) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 13);

  UpdateConfig adamw = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate with_adamw,
                       UpdateCompiler(adamw).Compile(model));
  EXPECT_EQ(CountOpcode(TrainProgramOf(with_adamw), OpCode::kAdamWStep), 4u);

  UpdateConfig sgd = BaseConfig(kBatch);
  sgd.optimizer.kind = OptimizerKind::kSgd;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate with_sgd,
                       UpdateCompiler(sgd).Compile(model));
  const auto sgd_instrs = TrainProgramOf(with_sgd);
  EXPECT_EQ(CountOpcode(sgd_instrs, OpCode::kSgdStep), 4u);
  EXPECT_EQ(CountOpcode(sgd_instrs, OpCode::kAdamWStep), 0u);
  // No AdamW moments: SGD's persistent segment holds only the adapters.
  EXPECT_LT(with_sgd.persistent_size, with_adamw.persistent_size);
  EXPECT_EQ(HeaderOf(with_sgd).optimizer_kind, 0u);

  UpdateConfig frozen = BaseConfig(kBatch);
  frozen.emit_optimizer = false;  // gradient-verification builds
  ASSERT_OK_AND_ASSIGN(CompiledUpdate no_opt,
                       UpdateCompiler(frozen).Compile(model));
  const auto no_opt_instrs = TrainProgramOf(no_opt);
  EXPECT_EQ(CountOpcode(no_opt_instrs, OpCode::kAdamWStep), 0u);
  EXPECT_EQ(CountOpcode(no_opt_instrs, OpCode::kSgdStep), 0u);
}

TEST(UpdateCompiler, HyperparametersReachThePlanHeader) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 14);
  UpdateConfig config = BaseConfig(kBatch);
  config.optimizer.lr = 0.025f;
  config.optimizer.weight_decay = 0.005f;
  config.default_steps = 123;

  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  const PlanHeader h = HeaderOf(compiled);
  EXPECT_NEAR(h.lr, 0.025f, 1e-9);
  EXPECT_NEAR(h.weight_decay, 0.005f, 1e-9);
  EXPECT_EQ(h.default_steps, 123u);
}

TEST(UpdateCompiler, RegatesTheBudgetOnTheExactCompiledFootprint) {
  // The step-0 estimate is a lower bound blind to gradients, optimizer
  // state, and transients; the driver must re-prove the budget against the
  // bytes the runtime actually keeps resident — arena + plan blob.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  UpdateConfig config = BaseConfig(kBatch);
  // The gate alone: under `auto` a budget one byte short would turn
  // rematerialization on and shrink the arena into it (see the remat test).
  config.remat = RematKind::kOff;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(model));
  const uint64_t need = compiled.arena_size + compiled.plan.size();

  config.memory_budget_bytes = need;
  EXPECT_OK(UpdateCompiler(config).Compile(model));

  // One byte short of the real footprint: the early lower bound admits it,
  // the final gate must not.
  config.memory_budget_bytes = need - 1;
  const auto r = UpdateCompiler(config).Compile(model);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "cannot run locally");
}

TEST(UpdateCompiler, RejectsModelWithoutInputMetadata) {
  // Topology's reading of the input (S1 #147): the driver no longer reads
  // the tensor table ahead of it, so this is topology's diagnostic.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 15);
  model.input_name = "not_a_tensor";
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(kBatch)).Compile(model),
                        "is not a tensor of the model with a shape");
}

TEST(UpdateCompiler, PropagatesGrafterFailure) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 16);
  UpdateConfig config = BaseConfig(kBatch);
  config.lora.target_filters = {"no_such_weight"};
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model), "no eligible");
}

TEST(UpdateCompiler, ATargetThatNamesOnlyATiedHeadIsAnError) {
  // P7 (#96): the tied embedding / LM head is ineligible (the gather reads
  // the same tensor), so --targets emb cannot be honoured: a compile error
  // that says why, not a silently unmet filter. A filter naming nothing,
  // beside one that does match, is an error too.
  SmfModel tied = seeml::testing::MakeTiedTokenDecoder(8, 2, 4, 12, 3);
  UpdateConfig config = BaseConfig(8);
  config.lora.target_filters = {"emb"};
  auto r = UpdateCompiler(config).Compile(tied);
  ASSERT_FALSE(r.has_value());
  EXPECT_STR_CONTAINS(r.error(), "--targets 'emb'");
  EXPECT_STR_CONTAINS(r.error(), "tied embedding / LM head");
  config.lora.target_filters = {"wq", "no_such_weight"};
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(tied),
                        "--targets 'no_such_weight' matches no eligible");
  // Without a filter the tied model compiles, adapting everything else.
  config.lora.target_filters = {};
  ASSERT_OK_AND_ASSIGN(CompiledUpdate compiled,
                       UpdateCompiler(config).Compile(tied));
  for (const auto& a : compiled.adapters) EXPECT_NE(a.weight_name, "emb");
  EXPECT_FALSE(compiled.adapters.empty());
}

TEST(UpdateCompiler, PlanBytesAreThreadCountInvariant) {
  // Compilation parallelizes its byte-heavy passes (int8 quantization,
  // seeded randn init of the persistent image); the emitted plan must be
  // bit-identical at any pool width — same blob, same integrity hash.
  UpdateConfig config = BaseConfig(kBatch);
  config.quantize_base = true;  // exercise the parallel max-abs scan + pack

  seeml::update::SetParallelThreadCount(1);
  SmfModel model_a = MakeMlp(kInDim, kHidden, kOutDim, 17);
  auto serial = UpdateCompiler(config).Compile(model_a);
  seeml::update::SetParallelThreadCount(8);
  SmfModel model_b = MakeMlp(kInDim, kHidden, kOutDim, 17);
  auto wide = UpdateCompiler(config).Compile(model_b);
  seeml::update::SetParallelThreadCount(0);

  ASSERT_OK(serial);
  ASSERT_OK(wide);
  EXPECT_TRUE(serial->plan == wide->plan);
}

// --- Single weight residency (E2, #81) ---------------------------------------

TEST(UpdateCompiler, TheConsumingCompileIsTheSamePlanAndReleasesWhatItPacked) {
  // Every storage form, a teacher, and the student/teacher split: the
  // consuming overload must assemble the identical blob — it changes WHEN
  // payloads die, nothing else — and leave no packed payload resident.
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 71);
  SmfModel teacher = MakeMlp(kInDim, 2 * kHidden, kOutDim, 72);
  for (int storage = 0; storage < 3; ++storage) {
    UpdateConfig config = BaseConfig(kBatch);
    config.loss = LossKind::kXEntPlusKL;
    config.quantize_base = storage == 1;
    config.bf16_base = storage == 2;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate kept,
                         UpdateCompiler(config).Compile(student, &teacher));
    SmfModel s = student, t = teacher;
    const uint64_t hash = s.content_hash;
    ASSERT_OK_AND_ASSIGN(CompiledUpdate consumed,
                         UpdateCompiler(config).Compile(std::move(s), &t));
    EXPECT_TRUE(kept.plan == consumed.plan);
    EXPECT_EQ(s.content_hash, hash);  // identity survives the release
    for (const SmfModel* m : {&s, &t}) {
      size_t released = 0;
      for (const SmfTensor& tensor : m->tensors) {
        if (!tensor.is_const) continue;
        EXPECT_GT(tensor.byte_size, 0u);  // metadata is kept
        if (tensor.data.empty()) ++released;
      }
      EXPECT_GT(released, 0u);
    }
    // The caller's originals were never touched.
    for (const SmfTensor& tensor : student.tensors)
      if (tensor.is_const) EXPECT_EQ(tensor.data.size(), tensor.byte_size);
  }
}

// --- The schedule triple is validated at compile time (E8, #91) ---------------

TEST(UpdateCompiler, RefusesSchedulesThatWouldMisbehaveSilently) {
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 91);
  auto compile = [&](auto&& edit) {
    UpdateConfig config = BaseConfig(kBatch);
    config.default_steps = 1000;
    edit(config);
    return UpdateCompiler(config).Compile(model);
  };
  auto cosine = [](UpdateConfig& c) {
    c.optimizer.lr_schedule = LrSchedule::kCosineWithWarmup;
  };
  EXPECT_ERROR_CONTAINS(compile([](UpdateConfig& c) { c.default_steps = 0; }),
                        "step budget must be positive");
  EXPECT_ERROR_CONTAINS(compile([&](UpdateConfig& c) {
                          cosine(c);
                          c.optimizer.warmup_steps = 1000;  // == the budget
                        }),
                        "must be shorter than the step budget");
  EXPECT_ERROR_CONTAINS(
      compile([](UpdateConfig& c) { c.optimizer.warmup_steps = 10; }),
      "cosine schedule only");
  EXPECT_ERROR_CONTAINS(compile([&](UpdateConfig& c) {
                          cosine(c);
                          c.optimizer.min_lr_factor = 0.0f;
                        }),
                        "learning rate of zero");
  ASSERT_OK_AND_ASSIGN(CompiledUpdate zero, compile([&](UpdateConfig& c) {
                         cosine(c);
                         c.optimizer.min_lr_factor = 0.0f;
                         c.optimizer.allow_zero_lr = true;  // asked for twice
                       }));
  EXPECT_EQ(HeaderOf(zero).min_lr_factor, 0.0f);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate normal, compile([&](UpdateConfig& c) {
                         cosine(c);
                         c.optimizer.warmup_steps = 999;
                       }));
  EXPECT_EQ(HeaderOf(normal).min_lr_factor, 0.1f);  // the default floor
  // A constant-schedule plan records no floor: byte-identical to the plans
  // compiled before the default moved.
  ASSERT_OK_AND_ASSIGN(CompiledUpdate constant, compile([](UpdateConfig&) {}));
  EXPECT_EQ(HeaderOf(constant).min_lr_factor, 0.0f);
  EXPECT_EQ(HeaderOf(constant).warmup_steps, 0u);
}

TEST(UpdateCompiler, RematerializationShrinksTheArenaAndLeavesEvalAlone) {
  // Roadmap 2b (F5, #133). The forward activations the backward reads stay
  // resident across the whole forward; --remat on re-emits the cheap ones
  // (the norm outputs, silu(gate), the SwiGLU product) right before their
  // backward readers, so the binder frees the originals in the forward.
  // The train program grows by exactly those recomputes, the eval program
  // — the primal snapshot — is the same instruction sequence, and the
  // persistent segment is untouched.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig plain = BaseConfig(batch);
  plain.remat = RematKind::kOff;
  UpdateConfig remat = plain;
  remat.remat = RematKind::kOn;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate off, UpdateCompiler(plain).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate on, UpdateCompiler(remat).Compile(model));

  EXPECT_FALSE(off.remat_applied);
  EXPECT_EQ(off.remat_recomputed_values, 0u);
  EXPECT_EQ(off.remat_bytes_saved, 0u);
  EXPECT_TRUE(on.remat_applied);
  EXPECT_GT(on.remat_recomputed_values, 0u);
  EXPECT_LT(on.arena_size, off.arena_size);
  EXPECT_EQ(on.remat_bytes_saved, off.arena_size - on.arena_size);
  EXPECT_EQ(on.persistent_size, off.persistent_size);
  EXPECT_EQ(on.io_size, off.io_size);
  EXPECT_LT(on.transient_size, off.transient_size);
  EXPECT_EQ(on.persistent_size + on.io_size + on.transient_size,
            on.arena_size);
  // Every recompute is one more train instruction, none of them a GEMM.
  const auto train_off = TrainProgramOf(off);
  const auto train_on = TrainProgramOf(on);
  EXPECT_GT(train_on.size(), train_off.size());
  for (OpCode gemm : {OpCode::kGemmNN, OpCode::kGemmNT, OpCode::kGemmTN,
                      OpCode::kGemmNNQ8, OpCode::kGemmNTQ8})
    EXPECT_EQ(CountOpcode(train_on, gemm), CountOpcode(train_off, gemm));
  // All three norms (the first one's input is the embedding gather, itself
  // recomputed for free from storage), silu(gate) and the SwiGLU product;
  // not the RoPE rotations (pre-rotation Q and K have no backward reader,
  // so recomputing them would only swap which tensor stays live).
  EXPECT_EQ(CountOpcode(train_on, OpCode::kRmsNormFwd),
            2 * CountOpcode(train_off, OpCode::kRmsNormFwd));
  EXPECT_EQ(CountOpcode(train_on, OpCode::kEmbedFwd),
            2 * CountOpcode(train_off, OpCode::kEmbedFwd));
  EXPECT_EQ(CountOpcode(train_on, OpCode::kRopeFwd),
            CountOpcode(train_off, OpCode::kRopeFwd));
  EXPECT_EQ(on.remat_recomputed_values, 8u);  // 3 x (y, rstd), silu, product
  EXPECT_EQ(on.merge_instruction_count, off.merge_instruction_count);
  // The eval program: the same opcodes, flags and immediates in the same
  // order (its transient refs move with the rescan, so not the same bytes).
  const auto eval_off = EvalProgramOf(off);
  const auto eval_on = EvalProgramOf(on);
  ASSERT_EQ(eval_on.size(), eval_off.size());
  for (size_t i = 0; i < eval_on.size(); ++i) {
    EXPECT_EQ(eval_on[i].opcode, eval_off[i].opcode);
    EXPECT_EQ(eval_on[i].flags, eval_off[i].flags);
    EXPECT_EQ(eval_on[i].imm, eval_off[i].imm);
  }

  // `auto` is the memory gate's decision: with the budget one byte short
  // of the plain footprint it recomputes, and the plan is the forced one
  // byte for byte; with room to spare it does not, and the plan is the
  // plain one.
  UpdateConfig automatic = plain;
  automatic.remat = RematKind::kAuto;
  automatic.memory_budget_bytes = off.arena_size + off.plan.size() - 1;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate squeezed,
                       UpdateCompiler(automatic).Compile(model));
  EXPECT_TRUE(squeezed.remat_applied);
  EXPECT_TRUE(squeezed.plan == on.plan);
  automatic.memory_budget_bytes = 2 * (off.arena_size + off.plan.size());
  ASSERT_OK_AND_ASSIGN(CompiledUpdate roomy,
                       UpdateCompiler(automatic).Compile(model));
  EXPECT_FALSE(roomy.remat_applied);
  EXPECT_TRUE(roomy.plan == off.plan);
  // With no budget configured, `auto` never consults the host's memory:
  // the plan is the plain one, byte for byte, on every machine.
  automatic.memory_budget_bytes = 0;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate unconfigured,
                       UpdateCompiler(automatic).Compile(model));
  EXPECT_FALSE(unconfigured.remat_applied);
  EXPECT_TRUE(unconfigured.plan == off.plan);
}

TEST(UpdateCompiler, AnAdapterOnlyPlanCarriesNoRodataAndReadsTheSourceModel) {
  // F6 (#134), plan v21: --adapter-only binds every frozen weight to its
  // f32 payload in the model file — a source ref, in the train program too
  // — and lays out no rodata, so the plan is the instructions, the
  // adapters' initial image and the emit table. Only for a model loaded
  // from a file, the f32 base and exact arithmetic; the driver names each
  // refusal.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig config = BaseConfig(batch);
  config.adapter_only = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model),
                        "needs a model loaded from a file");
  seeml::testing::ScopedTempDir dir;
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(path));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain,
                       UpdateCompiler(BaseConfig(batch)).Compile(saved));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate adapter,
                       UpdateCompiler(config).Compile(saved));
  const PlanHeader h = HeaderOf(adapter);
  EXPECT_EQ(h.version, kSeeuVersion);
  EXPECT_EQ(h.rodata_size, 0u);
  EXPECT_EQ(adapter.rodata_size, 0u);
  EXPECT_GT(HeaderOf(plain).rodata_size, 0u);
  // No larger (both toy plans pad to a 16 KiB multiple); the bytes an
  // adapter-only package saves are the SmolLM measurement's claim.
  EXPECT_LE(adapter.plan.size(), plain.plan.size());
  EXPECT_TRUE(adapter.scores_shipped);
  // Every weight read in the train program is a source ref and nothing in
  // it is a rodata ref (an unused slot is kNullRef, whose top bit is not a
  // rodata ref); the validator, which admitted the plan, refuses a write
  // through a source ref in any program.
  size_t source_reads = 0;
  for (const UpdateInstruction& ins : TrainProgramOf(adapter))
    for (uint64_t ref : ins.in) {
      if (ref == kNullRef) continue;
      EXPECT_FALSE(IsRodataRef(ref));
      if (IsSourceRef(ref)) ++source_reads;
    }
  EXPECT_GT(source_reads, 0u);
  // The persistent segment is the plain plan's: the adapters are the same
  // values, initialized the same way.
  EXPECT_EQ(adapter.persistent_size, plain.persistent_size);
  // The narrow bases and the relaxed kernels are refused by name.
  config.quantize_base = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(saved), "narrow copies");
  config.quantize_base = false;
  config.bf16_base = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(saved), "narrow copies");
  config.bf16_base = false;
  config.precision = Precision::kCertifiedBf16;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(saved),
                        "exact kernels");
  // A teacher's weights live in another file than the one the runtime
  // binds: refused by name, never bound to the student's bytes at the
  // teacher's offsets.
  config.precision = Precision::kF32;
  config.loss = LossKind::kKLDistill;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(saved, &saved),
                        "teacher");
}

TEST(UpdateCompiler, TheEvalProgramScoresThePlanByRequest) {
  // E12 scores the shipped f32 file for a narrow-base plan compiled from a
  // file; --validation-scores plan keeps the eval program on the plan's own
  // narrow copies — a proxy, for a device that cannot hold the file beside
  // the arena (F5 #133's 1.5B rung on 16 GB) — and the report says so. An
  // adapter-only plan has no weights of its own and refuses it.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  seeml::testing::ScopedTempDir dir;
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel saved, LoadSmf(path));
  UpdateConfig config = BaseConfig(batch);
  config.quantize_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate shipped,
                       UpdateCompiler(config).Compile(saved));
  EXPECT_TRUE(shipped.scores_shipped);
  config.eval_on_plan = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate proxy, UpdateCompiler(config).Compile(saved));
  EXPECT_FALSE(proxy.scores_shipped);
  // The eval program of the proxy plan reads no source ref; the shipped
  // plan's does.
  auto source_reads = [](const CompiledUpdate& c) {
    const PlanHeader h = HeaderOf(c);
    size_t n = 0;
    for (uint64_t i = 0; i < h.eval_instr_count; ++i) {
      UpdateInstruction ins;
      std::memcpy(&ins, c.plan.data() + h.eval_instr_offset + i * sizeof(ins),
                  sizeof(ins));
      for (uint64_t ref : ins.in)
        if (ref != kNullRef && IsSourceRef(ref)) ++n;
    }
    return n;
  };
  EXPECT_GT(source_reads(shipped), 0u);
  EXPECT_EQ(source_reads(proxy), 0u);
  // The train programs are the same bytes: the request touches eval alone.
  const std::vector<UpdateInstruction> train_proxy = TrainProgramOf(proxy);
  const std::vector<UpdateInstruction> train_shipped = TrainProgramOf(shipped);
  ASSERT_EQ(train_proxy.size(), train_shipped.size());
  EXPECT_EQ(std::memcmp(train_proxy.data(), train_shipped.data(),
                        train_proxy.size() * sizeof(UpdateInstruction)),
            0);
  config.quantize_base = false;
  config.adapter_only = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(saved),
                        "no weights of its own");
}

TEST(UpdateCompiler, AttentionAutoDecidesAgainstAConfiguredBudgetOnly) {
  // F3 (#131) design item 3, by the rule --remat auto already follows: a
  // decision that changes the plan's bytes may weigh the footprint against
  // a CONFIGURED memory budget, never against the compiling host's memory,
  // so identical inputs compile identical plans on every machine. With no
  // budget the cache budget decides by itself.
  TrainingFootprint fp;
  fp.weight_bytes = 1000;
  fp.activation_bytes = 2000;
  fp.delta_bytes = 500;
  fp.probs_cache_bytes = 4000;
  fp.attention_stats_bytes = 100;
  UpdateConfig config = BaseConfig(kBatch);
  ASSERT_EQ(config.attention, AttentionKind::kAuto);
  config.memory_budget_bytes = 0;  // no budget: the caches fit their own
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kCached);
  config.memory_budget_bytes = fp.total_bytes();  // exactly fits with them
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kCached);
  config.memory_budget_bytes = fp.total_bytes() - 1;  // not with them
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kTiled);
  config.memory_budget_bytes = 0;
  config.attention_cache_budget_bytes = fp.probs_cache_bytes - 1;
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kTiled);
  config.attention_cache_budget_bytes = fp.probs_cache_bytes;
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kCached);
  // A forced family is obeyed whatever the budgets say.
  config.memory_budget_bytes = 1;
  config.attention_cache_budget_bytes = 0;
  config.attention = AttentionKind::kCached;
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kCached);
  config.memory_budget_bytes = 0;
  config.attention_cache_budget_bytes = 1ull << 40;
  config.attention = AttentionKind::kTiled;
  EXPECT_EQ(DecideAttention(fp, config), AttentionKind::kTiled);

  // Through the driver: with no budget and with one nothing could exceed,
  // the same plan byte for byte — the cached family, by the cache budget.
  // (S = 8: at S = 4 a probability row is exactly a stats row's width, so
  // the two families' footprints coincide and no budget tells them apart.)
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 8, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig plain = BaseConfig(batch);
  plain.remat = RematKind::kOff;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate unbudgeted,
                       UpdateCompiler(plain).Compile(model));
  UpdateConfig roomy = plain;
  roomy.memory_budget_bytes = 1ull << 50;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate budgeted,
                       UpdateCompiler(roomy).Compile(model));
  EXPECT_FALSE(unbudgeted.attention_tiled);
  EXPECT_GT(unbudgeted.probs_cache_bytes, 0u);
  EXPECT_TRUE(unbudgeted.plan == budgeted.plan);

  // The step-0 gate's refusal names its remedy: the caches kept by force
  // under a budget the tiled family would have fit.
  TrainingFootprint est = EstimateTrainingFootprint(model, batch);
  est.delta_bytes = EstimateLoraDeltaBytes(model, plain.lora.target_filters);
  ASSERT_GT(est.probs_cache_bytes, est.attention_stats_bytes);
  UpdateConfig forced = plain;
  forced.attention = AttentionKind::kCached;
  forced.memory_budget_bytes = est.total_bytes() - 1;
  const auto refused = UpdateCompiler(forced).Compile(model);
  ASSERT_FALSE(refused.has_value());
  EXPECT_STR_CONTAINS(refused.error(), "too big to train locally");
  EXPECT_STR_CONTAINS(refused.error(), "--attention tiled would fit");
  // `auto` under the same budget tiles instead of refusing at this gate;
  // what it may still meet is the final gate on the exact arena, whose
  // refusal reads differently.
  UpdateConfig automatic = plain;
  automatic.memory_budget_bytes = est.total_bytes() - 1;
  const auto tiled = UpdateCompiler(automatic).Compile(model);
  if (tiled.has_value())
    EXPECT_TRUE(tiled->attention_tiled);
  else
    EXPECT_STR_CONTAINS(tiled.error(), "cannot run locally");
}

TEST(UpdateCompiler, TopologyRefusesWhatTheReviewFoundMissing) {
  // S1 #147: the cross-input checks that had no home while shape checks ran
  // inside the build loop — each a compile-time refusal naming the rule.
  // F06: the teacher's `t::` namespace is reserved.
  {
    SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 21);
    for (SmfTensor& t : model.tensors)
      if (t.name == "w1") t.name = "t::w1";
    for (SmfOp& op : model.ops)
      for (std::string& in : op.inputs)
        if (in == "w1") in = "t::w1";
    EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(kBatch)).Compile(model),
                          "reserved namespace 't::'");
  }
  // F68: the output's rows are the batch — a MatMul over a constant LHS
  // has its own row count, and the loss would read labels past the slot.
  {
    SmfModel model;
    model.input_name = "x";
    model.output_name = "logits";
    model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
    model.tensors.push_back({.name = "c",
                             .dims = {5, 4},
                             .is_const = true,
                             .data = seeml::testing::AsPayload(
                                 std::vector<float>(20, 0.5f))});
    model.tensors.back().byte_size = 80;
    model.tensors.push_back({.name = "w",
                             .dims = {4, 3},
                             .is_const = true,
                             .data = seeml::testing::AsPayload(
                                 std::vector<float>(12, 0.25f))});
    model.tensors.back().byte_size = 48;
    model.ops.push_back({SmfOpKind::kMatMul, "mm", {"c", "w"}, "logits"});
    EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(kBatch)).Compile(model),
                          "has 5 rows, but the compiled batch is 4");
  }
  // N19: a token model's labels are next-token classes; MSE has no
  // dataset that could satisfy its contract.
  {
    SmfModel model =
        seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 16, 41);
    UpdateConfig config = BaseConfig(16);
    config.loss = LossKind::kMse;
    EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model),
                          "--loss mse needs dense f32 labels");
  }
  // F02: a sequential teacher must agree on seq_len — the causal mask and
  // the rotary positions follow each model's own declaration.
  {
    SmfModel student = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
    SmfModel teacher = seeml::testing::MakeTinyDecoder(8, 2, 2, 12, 3, 10);
    UpdateConfig config = BaseConfig(4);
    config.loss = LossKind::kKLDistill;
    EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &teacher),
                          "teacher seq_len 2 disagrees with the student's 4");
  }
}

TEST(UpdateCompiler, TheTokenizerSectionMustMatchTheEmbeddingAndTheHead) {
  // S1 / S2: an SMF v7 tokenizer section's vocabulary is the id space the
  // corpus carries — the embedding table's rows and the head's width.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  auto section_with_vocab = [](uint32_t v) {
    // The SMTK head alone (kSmtkHeadBytes): magic, version 1, kind 1,
    // pre-tokenizer 1, flags 0, vocab, 0 merges, 0 special, -1 x 4, no
    // template. Topology reads the head; the encoder is not instantiated.
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t x) {
      for (int i = 0; i < 4; ++i) b.push_back((x >> (8 * i)) & 0xFF);
    };
    u32(0x4B544D53); u32(1); u32(1); u32(1); u32(0); u32(v); u32(0); u32(0);
    u32(0xFFFFFFFFu); u32(0xFFFFFFFFu); u32(0xFFFFFFFFu); u32(0xFFFFFFFFu);
    u32(0);
    return b;
  };
  model.tokenizer = section_with_vocab(16);
  auto bound = UpdateCompiler(BaseConfig(batch)).Compile(model);
  ASSERT_OK(bound);
  // The plan is bound to the section: its header carries the hash the
  // corpus must name (v22), and a model without a section binds to none.
  EXPECT_EQ(HeaderOf(*bound).tokenizer_hash, model.TokenizerHash());
  EXPECT_NE(HeaderOf(*bound).tokenizer_hash, 0u);
  model.tokenizer.clear();
  auto unbound = UpdateCompiler(BaseConfig(batch)).Compile(model);
  ASSERT_OK(unbound);
  EXPECT_EQ(HeaderOf(*unbound).tokenizer_hash, 0u);
  model.tokenizer = section_with_vocab(17);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(batch)).Compile(model),
                        "vocabulary (17 tokens) is not the embedding table's "
                        "rows (16)");
  SmfModel mlp = MakeMlp(kInDim, kHidden, kOutDim, 3);
  mlp.tokenizer = section_with_vocab(static_cast<uint32_t>(kOutDim));
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(kBatch)).Compile(mlp),
                        "takes feature rows, not token ids");
  // A head shorter than the fixed 13 words is refused as truncated (the
  // constant used to say 12, admitting a head without its template length).
  model.tokenizer = section_with_vocab(16);
  model.tokenizer.resize(50);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(batch)).Compile(model),
                        "truncated head");
  model.tokenizer = section_with_vocab(16);
  model.tokenizer[4] = 2;  // the version word
  EXPECT_ERROR_CONTAINS(UpdateCompiler(BaseConfig(batch)).Compile(model),
                        "unsupported version 2");
  // A teacher reads the student's token ids: one carrying another section
  // (same vocabulary, other bytes) is refused; the student's own, or none,
  // is admitted.
  model.tokenizer = section_with_vocab(16);
  SmfModel teacher =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 42);
  UpdateConfig kl = BaseConfig(batch);
  kl.loss = LossKind::kKLDistill;
  teacher.tokenizer = section_with_vocab(16);
  teacher.tokenizer[32] = 0;  // eos_id -1 -> 0: another section, same vocab
  EXPECT_ERROR_CONTAINS(UpdateCompiler(kl).Compile(model, &teacher),
                        "tokenizer hash mismatch");
  teacher.tokenizer = section_with_vocab(16);
  EXPECT_OK(UpdateCompiler(kl).Compile(model, &teacher));
  teacher.tokenizer.clear();
  EXPECT_OK(UpdateCompiler(kl).Compile(model, &teacher));
}

TEST(UpdateCompiler, TheDataPathPinsThePlanToTheCanonicalCorpus) {
  // S2 #148: --data runs the vendored stages with the model's own section,
  // writes the canonical corpus, pins the plan to it by hash and fills the
  // corpus rows; a given SDS is held to the model and pinned as is.
  seeml::testing::ScopedTempDir dir;
  auto write = [&](const std::string& name, const std::string& bytes) {
    const std::string path = dir.File(name);
    std::ofstream f(path, std::ios::binary);
    f << bytes;
    return path;
  };
  // A byte vocabulary (id = byte) with one special EOS token (id 256).
  auto section = [] {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    u32(0x4B544D53); u32(1); u32(1); u32(2); u32(0); u32(257); u32(0); u32(1);
    u32(256); u32(0xFFFFFFFFu); u32(0xFFFFFFFFu); u32(0xFFFFFFFFu); u32(0);
    for (int v = 0; v < 256; ++v) {
      b.push_back(1); b.push_back(0); b.push_back(static_cast<uint8_t>(v));
    }
    const std::string eos = "<eos>";
    b.push_back(static_cast<uint8_t>(eos.size())); b.push_back(0);
    b.insert(b.end(), eos.begin(), eos.end());
    u32(256);
    return b;
  }();
  const int64_t vocab = 257, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 43);
  model.tokenizer = section;
  // 40 bytes, one document: 8 windows of 5 ids, no remainder.
  const std::string text = "abcdefghij" "klmnopqrst" "uvwxyz0123" "456789ABCD";
  UpdateConfig cfg = BaseConfig(batch);
  cfg.default_steps = 8;
  cfg.data_path = write("corpus.txt", text);
  auto compiled = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(compiled);
  const PlanHeader h = HeaderOf(*compiled);
  EXPECT_NE(h.data_flags & kDataFlagCorpusPinned, 0u);
  EXPECT_FALSE(compiled->corpus_from_sds);
  EXPECT_EQ(compiled->corpus_sds.size(), 64u + 8 * 5 * 4);
  EXPECT_EQ(h.corpus_hash, ContentHash64(compiled->corpus_sds.data(),
                                         compiled->corpus_sds.size()));
  EXPECT_EQ(h.tokenizer_hash, model.TokenizerHash());
  EXPECT_STR_CONTAINS(compiled->corpus_manifest, "\"windows\": 8");
  const std::string screen = compiled->screens.frontend.Json(0);
  EXPECT_STR_CONTAINS(screen, "\"corpus_tokens\": 40");
  EXPECT_STR_CONTAINS(screen, "\"corpus_distinct_ids\": \"40 of 257\"");
  EXPECT_STR_CONTAINS(screen, "\"corpus_targets\": \"32 of 32\"");
  // batch 16 rows = 4 sequences per step; 8 records: 2 batches per epoch,
  // 8 steps cover 4 epochs.
  EXPECT_STR_CONTAINS(screen, "\"batches_per_epoch\": 2");
  EXPECT_STR_CONTAINS(screen, "\"epochs_covered\": 4");
  bool data_phase = false;
  for (const auto& t : compiled->pass_timings) data_phase |= t.name == "data";
  EXPECT_TRUE(data_phase);

  // The same corpus given as an SDS: pinned to the same hash, nothing to
  // write; and the pin differs from a plan compiled without data.
  const std::string sds_path = write(
      "corpus.sds", std::string(compiled->corpus_sds.begin(),
                                compiled->corpus_sds.end()));
  cfg.data_path = sds_path;
  auto again = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(again);
  EXPECT_TRUE(again->corpus_from_sds);
  EXPECT_EQ(HeaderOf(*again).corpus_hash, h.corpus_hash);
  cfg.data_path.clear();
  auto plain = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(plain);
  EXPECT_EQ(HeaderOf(*plain).corpus_hash, 0u);
  EXPECT_EQ(HeaderOf(*plain).data_flags & kDataFlagCorpusPinned, 0u);

  // The compiled split (S3 #149): two compiles of the same corpus and seed
  // name the same held-out set; the header carries seed, fraction and the
  // indices' hash; a split without a corpus is refused.
  cfg.data_path = sds_path;
  cfg.val_frac = 0.25f;
  cfg.split_seed = 3;
  auto split_a = UpdateCompiler(cfg).Compile(model);
  auto split_b = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(split_a);
  ASSERT_OK(split_b);
  const PlanHeader hs = HeaderOf(*split_a);
  EXPECT_NE(hs.data_flags & kDataFlagSplit, 0u);
  EXPECT_EQ(hs.split_seed, 3u);
  EXPECT_EQ(hs.val_frac_bits, ValFracBits(0.25f));
  EXPECT_EQ(hs.val_split_hash, SplitHash(HeldOutIndices(8, 3, 0.25f)));
  EXPECT_EQ(hs.val_split_hash, HeaderOf(*split_b).val_split_hash);
  EXPECT_EQ(split_a->plan, split_b->plan);
  EXPECT_STR_CONTAINS(split_a->screens.frontend.Json(0),
                      "\"validation\": \"2 of 8 records held out, seed 3, split ");
  // The epoch rows count the 6 training records: 1.5 batches per epoch,
  // 8 steps x 4 sequences / 6 records = 5.333 epochs (not the 2 / 4 of the
  // whole corpus).
  EXPECT_STR_CONTAINS(split_a->screens.frontend.Json(0),
                      "\"batches_per_epoch\": 1.5");
  EXPECT_STR_CONTAINS(split_a->screens.frontend.Json(0),
                      "\"epochs_covered\": 5.333");
  cfg.split_seed = 4;
  auto split_c = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(split_c);
  EXPECT_NE(HeaderOf(*split_c).val_split_hash, hs.val_split_hash);
  cfg.data_path.clear();
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(model),
                        "needs the corpus");
  cfg.data_path = sds_path;
  cfg.val_frac = 1.5f;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(model), "val_frac must lie");
  cfg.val_frac = 0.0f;
  cfg.split_seed = 0;

  // Refusals, by name.
  cfg.data_path = write("corpus.csv", text);
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(model),
                        "unknown data format");
  cfg.data_path = write("corpus.txt", text);
  SmfModel sectionless = model;
  sectionless.tokenizer.clear();
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(sectionless),
                        "carries no tokenizer section");
  UpdateConfig mlp_cfg = BaseConfig(kBatch);
  mlp_cfg.data_path = cfg.data_path;
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(mlp_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "a feature model takes no raw text");
  // An SDS encoded by another tokenizer, and one carrying a loss mask.
  std::vector<int32_t> ids(8 * 5, 1);
  auto foreign = seeml::update_rt::Dataset::FromTokens(ids, 8, seq, {}, 0x999);
  ASSERT_OK(foreign);
  ASSERT_OK(foreign->SaveToFile(dir.File("foreign.sds")));
  cfg.data_path = dir.File("foreign.sds");
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(model), "another tokenizer");
  std::vector<uint8_t> mask(8 * 4, 1);
  auto masked = seeml::update_rt::Dataset::FromTokens(ids, 8, seq, mask,
                                                      model.TokenizerHash());
  ASSERT_OK(masked);
  ASSERT_OK(masked->SaveToFile(dir.File("masked.sds")));
  cfg.data_path = dir.File("masked.sds");
  // A masked corpus compiles the masked cross-entropy (S3 #149): the flag
  // in the header, opcodes 51 / 52 in place of 11 / 12 in every program,
  // and the loss objective is the cross-entropy alone.
  auto masked_plan = UpdateCompiler(cfg).Compile(model);
  ASSERT_OK(masked_plan);
  EXPECT_NE(HeaderOf(*masked_plan).data_flags & kDataFlagMaskedLoss, 0u);
  {
    const auto all = TrainProgramOf(*masked_plan);
    EXPECT_EQ(CountOpcode(all, OpCode::kSoftmaxXEntFwd), 0u);
    EXPECT_EQ(CountOpcode(all, OpCode::kSoftmaxXEntBwd), 0u);
    EXPECT_GT(CountOpcode(all, OpCode::kSoftmaxXEntMaskedFwd), 0u);
    EXPECT_GT(CountOpcode(all, OpCode::kSoftmaxXEntMaskedBwd), 0u);
  }
  EXPECT_STR_CONTAINS(masked_plan->screens.frontend.Json(0),
                      "\"corpus_targets\": \"32 of 32\"");
  // The mask is the cross-entropy's alone: a feature model under MSE (which
  // topology admits) with --masked-loss is refused by the driver.
  UpdateConfig mse_masked = BaseConfig(kBatch);
  mse_masked.masked_loss = true;
  mse_masked.loss = LossKind::kMse;
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(mse_masked).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "cross-entropy alone");
  // --masked-loss without data: the flag and the masked opcodes, no pin.
  UpdateConfig flagged_cfg = cfg;
  flagged_cfg.data_path.clear();
  flagged_cfg.masked_loss = true;
  auto flagged = UpdateCompiler(flagged_cfg).Compile(model);
  ASSERT_OK(flagged);
  EXPECT_NE(HeaderOf(*flagged).data_flags & kDataFlagMaskedLoss, 0u);
  EXPECT_EQ(HeaderOf(*flagged).data_flags & kDataFlagCorpusPinned, 0u);
  // A record wider than the model's seq_len, and an id past the table.
  auto wide = seeml::update_rt::Dataset::FromTokens(std::vector<int32_t>(8 * 6, 1),
                                                    8, seq + 1);
  ASSERT_OK(wide);
  ASSERT_OK(wide->SaveToFile(dir.File("wide.sds")));
  cfg.data_path = dir.File("wide.sds");
  EXPECT_ERROR_CONTAINS(UpdateCompiler(cfg).Compile(model), "-token records");
  ids[7] = 257;
  auto oob = seeml::update_rt::Dataset::FromTokens(ids, 8, seq);
  ASSERT_OK(oob);
  ASSERT_OK(oob->SaveToFile(dir.File("oob.sds")));
  cfg.data_path = dir.File("oob.sds");
  EXPECT_ERROR(UpdateCompiler(cfg).Compile(model));

  // --masked-loss over a corpus that carries no mask would pin the plan to
  // the one corpus the device must refuse: a compile-time refusal.
  UpdateConfig masked_plain = cfg;
  masked_plain.data_path = sds_path;
  masked_plain.masked_loss = true;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(masked_plain).Compile(model),
                        "would refuse its own corpus");
  // ... and on a feature model no corpus could carry one at all.
  UpdateConfig masked_feature = BaseConfig(kBatch);
  masked_feature.masked_loss = true;
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(masked_feature).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "token models alone");

  // A feature model takes its corpus as a pre-tokenized .sds (feature
  // rows), pinned by the file's hash like a token corpus; the labels the
  // corpus carries are held to the labels the loss stages, and the device's
  // own feeder rule runs over the finished header before the plan ships.
  auto rows = seeml::testing::MakeClassificationData(16, kInDim, 7);
  ASSERT_OK(rows);
  ASSERT_OK(rows->SaveToFile(dir.File("rows.sds")));
  UpdateConfig feature_cfg = BaseConfig(kBatch);
  feature_cfg.data_path = dir.File("rows.sds");
  auto pinned = UpdateCompiler(feature_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3));
  ASSERT_OK(pinned);
  {
    std::ifstream f(dir.File("rows.sds"), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    EXPECT_NE(HeaderOf(*pinned).corpus_hash, 0u);
    EXPECT_EQ(HeaderOf(*pinned).corpus_hash,
              ContentHash64(reinterpret_cast<const uint8_t*>(bytes.data()),
                            bytes.size()));
    EXPECT_NE(HeaderOf(*pinned).data_flags & kDataFlagCorpusPinned, 0u);
  }
  feature_cfg.loss = LossKind::kMse;  // dense targets over class labels
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(feature_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "carries class-index labels, but the loss stages dense f32 labels");
  auto dense = seeml::testing::MakeRegressionData(16, kInDim, kOutDim + 1, 7);
  ASSERT_OK(dense);
  ASSERT_OK(dense->SaveToFile(dir.File("dense.sds")));
  feature_cfg.data_path = dir.File("dense.sds");
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(feature_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "dense labels are 4 wide; the model's output is 3");
  auto unlabeled = seeml::testing::MakeUnlabeledData(16, kInDim, 7);
  ASSERT_OK(unlabeled);
  ASSERT_OK(unlabeled->SaveToFile(dir.File("unlabeled.sds")));
  feature_cfg.loss = LossKind::kSoftmaxXEnt;
  feature_cfg.data_path = dir.File("unlabeled.sds");
  EXPECT_ERROR_CONTAINS(
      UpdateCompiler(feature_cfg).Compile(MakeMlp(kInDim, kHidden, kOutDim, 3)),
      "carries no labels, but the loss stages class-index labels");
}

TEST(UpdateCompiler, TheAdapterScaleIsTheRecipesChoice) {
  // S5 #151: lora_scaling picks s = alpha / r (LoRA's own, the default) or
  // rank-stabilized s = alpha / sqrt(r). The default is untouched; the
  // other is a different plan with a different resolved recipe; at rank 1
  // the two scales are the same number and so the same program.
  LoRASpec spec;
  spec.rank = 8;
  spec.alpha = 16.0f;
  EXPECT_EQ(spec.Scale(), 2.0f);
  spec.scaling = LoraScaling::kAlphaOverSqrtRank;
  EXPECT_EQ(spec.Scale(), 16.0f / std::sqrt(8.0f));
  spec.rank = 4;
  EXPECT_EQ(spec.Scale(), 8.0f);

  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 3);
  UpdateConfig config = BaseConfig(kBatch);
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain, UpdateCompiler(config).Compile(model));
  UpdateConfig rs = config;
  rs.lora.scaling = LoraScaling::kAlphaOverSqrtRank;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate stabilized, UpdateCompiler(rs).Compile(model));
  EXPECT_NE(plain.recipe_hash, stabilized.recipe_hash);
  EXPECT_NE(HeaderOf(plain).plan_hash, HeaderOf(stabilized).plan_hash);
  EXPECT_EQ(plain.train_instruction_count, stabilized.train_instruction_count);

  UpdateConfig one = config;
  one.lora.rank = 1;
  UpdateConfig one_rs = one;
  one_rs.lora.scaling = LoraScaling::kAlphaOverSqrtRank;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate a, UpdateCompiler(one).Compile(model));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate b, UpdateCompiler(one_rs).Compile(model));
  EXPECT_NE(a.recipe_hash, b.recipe_hash);  // the recipe still says which
  const PlanHeader ha = HeaderOf(a), hb = HeaderOf(b);
  ASSERT_EQ(ha.train_instr_count, hb.train_instr_count);
  EXPECT_EQ(std::memcmp(a.plan.data() + ha.train_instr_offset,
                        b.plan.data() + hb.train_instr_offset,
                        ha.train_instr_count * sizeof(UpdateInstruction)),
            0);
}

TEST(UpdateCompiler, ThePlanCarriesTheManifestAndTheLedger) {
  // Plan v23 (S4 #150 / S5 #151 / S6 #152): the metadata section holds the
  // resolved recipe's hash and one ledger line per pass; an exact plan's
  // ledger names no reordered arithmetic, a relaxed plan's names every
  // relaxed instruction.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 3);
  auto decode = [](const CompiledUpdate& c) {
    DecodedMetadata meta;
    const PlanHeader h = HeaderOf(c);
    if ((h.data_flags & kDataFlagMetadata) == 0)
      ADD_FAILURE("the plan carries no metadata flag");
    if (h.version != kSeeuVersion) ADD_FAILURE("the plan is not at kSeeuVersion");
    const auto why = PlanMetadataViolation(h, c.plan.data(), c.plan.size(), &meta);
    if (why) ADD_FAILURE(*why);
    return meta;
  };
  ASSERT_OK_AND_ASSIGN(CompiledUpdate exact,
                       UpdateCompiler(BaseConfig(kBatch)).Compile(model));
  const DecodedMetadata em = decode(exact);
  ASSERT_TRUE(em.present);
  EXPECT_EQ(em.meta.recipe_hash, exact.recipe_hash);
  EXPECT_NE(em.meta.recipe_hash, 0u);
  EXPECT_EQ(em.meta.export_flags, 0u);  // an in-memory model: no report
  ASSERT_TRUE(em.ledger.size() >= 4);
  std::vector<std::string> passes;
  for (const LedgerEntry& e : em.ledger) {
    EXPECT_EQ(e.arithmetic, kLedgerExact);
    EXPECT_EQ(e.instructions, 0u);
    EXPECT_TRUE(e.why[0] != '\0');
    passes.emplace_back(e.pass);
  }
  for (const char* want : {"lora-graft", "autodiff", "optimizer", "dce"})
    EXPECT_TRUE(std::find(passes.begin(), passes.end(), want) != passes.end());
  EXPECT_EQ(exact.ledger.size(), em.ledger.size());

  // The same request spelled another way resolves to the same plan: an MLP
  // has no attention, so --attention tiled decides nothing, and a memory
  // budget the plan fits changes no byte.
  UpdateConfig respelled = BaseConfig(kBatch);
  respelled.attention = AttentionKind::kTiled;
  respelled.memory_budget_bytes = uint64_t{1} << 30;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate same,
                       UpdateCompiler(respelled).Compile(model));
  EXPECT_EQ(same.recipe_hash, exact.recipe_hash);
  EXPECT_TRUE(same.plan == exact.plan);
  // Another recipe is another hash, and so another plan.
  UpdateConfig other = BaseConfig(kBatch);
  other.optimizer.lr *= 2.0f;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate moved, UpdateCompiler(other).Compile(model));
  EXPECT_NE(moved.recipe_hash, exact.recipe_hash);

  // Relaxed: the precision assignment is the one reordering entry, and it
  // accounts for every relaxed instruction.
  UpdateConfig relaxed_cfg = BaseConfig(kBatch);
  relaxed_cfg.precision = Precision::kCertifiedBf16;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate relaxed,
                       UpdateCompiler(relaxed_cfg).Compile(model));
  ASSERT_TRUE(relaxed.relaxed_gemms > 0);
  const DecodedMetadata rm = decode(relaxed);
  uint64_t reordered = 0, reordering_entries = 0;
  for (const LedgerEntry& e : rm.ledger)
    if (e.arithmetic == kLedgerReordered) {
      ++reordering_entries;
      reordered += e.instructions;
      EXPECT_EQ(std::string(e.pass), "precision");
    }
  EXPECT_EQ(reordering_entries, 1u);
  EXPECT_EQ(reordered, relaxed.relaxed_gemms + relaxed.relaxed_attention);

  // What the exporter proved rides in the section.
  UpdateConfig reported = BaseConfig(kBatch);
  reported.export_report = {.present = true,
                            .report_hash = 0xFEED,
                            .tokenizer_parity = true,
                            .tokenizer_parity_ids = 1234,
                            .logits_parity = true,
                            .logits_parity_max_delta = 6.4e-5f};
  ASSERT_OK_AND_ASSIGN(CompiledUpdate with_report,
                       UpdateCompiler(reported).Compile(model));
  const DecodedMetadata wm = decode(with_report);
  EXPECT_EQ(wm.meta.export_flags, kExportReportPresent | kExportTokenizerParity |
                                      kExportLogitsParity);
  EXPECT_EQ(wm.meta.export_report_hash, 0xFEEDu);
  EXPECT_EQ(wm.meta.tokenizer_parity_ids, 1234u);
  EXPECT_EQ(wm.meta.logits_parity_max_delta, 6.4e-5f);
  EXPECT_EQ(with_report.recipe_hash, exact.recipe_hash);  // not a recipe field

  // A report built in code holds the reader's rules (ExportReportViolation):
  // a parity "checked" over zero ids passes the loader, and the committed
  // model's own report would then be one ReadExportReport refuses.
  UpdateConfig unreadable = reported;
  unreadable.export_report.tokenizer_parity_ids = 0;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(unreadable).Compile(model),
                        "config export_report: the tokenizer parity is marked "
                        "checked over zero ids");
  unreadable = reported;
  unreadable.export_report.report_hash = 0;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(unreadable).Compile(model),
                        "config export_report: the report carries no hash");
  unreadable = reported;
  unreadable.export_report.logits_parity_max_delta = std::nanf("");
  EXPECT_ERROR_CONTAINS(UpdateCompiler(unreadable).Compile(model),
                        "config export_report: logits_parity_max_delta");
}

TEST(UpdateCompiler, TheRecipesBoundsHoldForAConfigBuiltInCode) {
  // S4 #150, review N18: the rank was unbounded on the library path too,
  // and K x r overflowing the arena arithmetic emitted an invalid plan.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 3);
  UpdateConfig wide = BaseConfig(kBatch);
  wide.lora.rank = 4097;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(wide).Compile(model),
                        "config lora rank 4097 is outside [1, 4096]");
  UpdateConfig empty = BaseConfig(kBatch);
  empty.lora.target_filters = {""};
  EXPECT_ERROR_CONTAINS(UpdateCompiler(empty).Compile(model),
                        "config an adapter target filter is empty");
}

TEST(UpdateCompiler, ANonFiniteWeightIsRefusedBeforeAnyPass) {
  // CR1 #142 F12 / F34: a NaN weight passed the quantization review (max
  // skips NaN) and reached the int8 pack's cast — undefined behavior. The
  // compile refuses the model by tensor name and element before any pass;
  // under UBSan this is the acceptance (the cast is never reached).
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  SmfTensor* w = nullptr;
  for (SmfTensor& t : model.tensors)
    if (t.is_const && t.dims.size() == 2) {
      w = &t;
      break;
    }
  ASSERT_TRUE(w != nullptr);
  const float nan = std::bit_cast<float>(0x7FC00000u);
  std::memcpy(w->data.data() + 3 * sizeof(float), &nan, sizeof(nan));
  const auto r = UpdateCompiler(BaseConfig(kBatch)).Compile(model);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().find("'" + w->name + "'") != std::string::npos);
  EXPECT_TRUE(r.error().find("element 3") != std::string::npos);
  // The teacher is swept too: an infinity in it is refused by role.
  SmfModel student = MakeMlp(kInDim, kHidden, kOutDim, 6);
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 7);
  SmfTensor* tw = nullptr;
  for (SmfTensor& t : teacher.tensors)
    if (t.is_const && t.dims.size() == 2) {
      tw = &t;
      break;
    }
  ASSERT_TRUE(tw != nullptr);
  const float inf = INFINITY;
  std::memcpy(tw->data.data(), &inf, sizeof(inf));
  UpdateConfig config = BaseConfig(kBatch);
  config.loss = LossKind::kKLDistill;
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(student, &teacher),
                        "teacher tensor");
}

TEST(UpdateCompiler, ConfigFloatsAreFiniteAndInRange) {
  // CR1 #142 N26: the CLI bounded its flags; the library path bounded
  // nothing, so a NaN learning rate or an unbounded temperature compiled.
  SmfModel model = MakeMlp(kInDim, kHidden, kOutDim, 1);
  const float nan = std::bit_cast<float>(0x7FC00000u);
  struct Case {
    const char* name;
    void (*set)(UpdateConfig&, float);
    float bad;
  };
  const Case cases[] = {
      {"lr", [](UpdateConfig& c, float v) { c.optimizer.lr = v; }, nan},
      {"lr", [](UpdateConfig& c, float v) { c.optimizer.lr = v; }, 0.0f},
      {"beta1", [](UpdateConfig& c, float v) { c.optimizer.beta1 = v; }, 1.0f},
      {"beta2", [](UpdateConfig& c, float v) { c.optimizer.beta2 = v; }, nan},
      {"eps", [](UpdateConfig& c, float v) { c.optimizer.eps = v; }, 0.0f},
      {"weight_decay",
       [](UpdateConfig& c, float v) { c.optimizer.weight_decay = v; }, -0.01f},
      {"clip_norm", [](UpdateConfig& c, float v) { c.optimizer.clip_norm = v; },
       nan},
      {"clip_norm", [](UpdateConfig& c, float v) { c.optimizer.clip_norm = v; },
       -1.0f},
      {"min_lr_factor",
       [](UpdateConfig& c, float v) { c.optimizer.min_lr_factor = v; }, 1.5f},
      {"lora.alpha", [](UpdateConfig& c, float v) { c.lora.alpha = v; }, 0.0f},
  };
  for (const Case& c : cases) {
    UpdateConfig config = BaseConfig(kBatch);
    c.set(config, c.bad);
    EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model), c.name);
  }
  // The temperature and the distillation weight are checked where they
  // apply — under a KL loss — with the bounds the validator and the CLI
  // share; under cross-entropy a NaN temperature is unused, not a refusal.
  SmfModel teacher = MakeMlp(kInDim, 14, kOutDim, 7);
  for (const float t : {0.0f, -2.0f, nan, 1e-4f, 1e4f}) {
    UpdateConfig config = BaseConfig(kBatch);
    config.loss = LossKind::kKLDistill;
    config.temperature = t;
    EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model, &teacher),
                          "temperature");
  }
  UpdateConfig xent = BaseConfig(kBatch);
  xent.temperature = nan;
  EXPECT_OK(UpdateCompiler(xent).Compile(model));
}

}  // namespace
