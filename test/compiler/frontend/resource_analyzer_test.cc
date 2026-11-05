// =============================================================================
// Resource analyzer tests: the static training-footprint lower bound, local
// memory detection, and the fail-fast gate in UpdateCompiler::Compile.
// =============================================================================

#include <cstdint>

#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/accountant/resource_analyzer.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"

namespace {

using namespace seeml::update;
using seeml::testing::MakeMlp;

constexpr int64_t kBatch = 16;

TEST(ResourceAnalyzer, FootprintCountsWeightsAndWhatTheBackwardReads) {
  // MakeMlp(4, 8, 2): w1[4x8] b1[8] w2[8x2] b2[2], ops mm1/ab1/relu1/mm2/ab2.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);

  EXPECT_EQ(fp.weight_bytes, (4 * 8 + 8 + 8 * 2 + 2) * sizeof(float));
  // The backward reads ab1's output (relu's input, differentiated since w1
  // is adapted), relu1's output (mm2's input, whose w2 is adapted) and the
  // model output the loss differentiates: [16x8], [16x8], [16x2]. The two
  // MatMul outputs feed bias adds, which read nothing of their forward, so
  // they die there and are not counted (N29).
  EXPECT_EQ(fp.activation_bytes,
            (2 * kBatch * 8 + kBatch * 2) * sizeof(float));
  // Rematerialization would recompute relu1's output from ab1's, which
  // the backward holds anyway.
  EXPECT_EQ(fp.recomputable_bytes, kBatch * 8 * sizeof(float));
  EXPECT_EQ(fp.total_bytes(), fp.weight_bytes + fp.activation_bytes);
}

TEST(ResourceAnalyzer, LayerNormCachesRowStatisticsWhereItsBackwardRuns) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 8}, .is_const = false});
  model.tensors.push_back(
      {.name = "w", .dims = {8, 8}, .is_const = true, .byte_size = 256});
  model.tensors.push_back(
      {.name = "gamma", .dims = {8}, .is_const = true, .byte_size = 32});
  model.tensors.push_back(
      {.name = "beta", .dims = {8}, .is_const = true, .byte_size = 32});
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "h"});
  model.ops.push_back(
      {SmfOpKind::kLayerNorm, "ln", {"h", "gamma", "beta"}, "y"});

  TrainingFootprint fp = EstimateTrainingFootprint(model, /*batch=*/4);
  EXPECT_EQ(fp.weight_bytes, 320u);
  // w is adapted, so h is differentiated: the norm's backward runs, reads
  // h [4x8] and its per-row mean/rstd (2 f32 per row); y is the output.
  EXPECT_EQ(fp.activation_bytes, (2 * 4 * 8 + 2 * 4) * sizeof(float));
  // Adapt nothing and no backward rule runs past the loss: h dies at the
  // norm, the statistics with it, and only the output is live.
  const std::vector<std::string> none = {"no-such-tensor"};
  EXPECT_EQ(EstimateTrainingFootprint(model, 4, none).activation_bytes,
            4 * 8 * sizeof(float));
}

TEST(ResourceAnalyzer, TheTrainingBoundCountsOnlyWhatTheBackwardReads) {
  // One attention block, written out: x -> q/k/v = x@wq/wk/wv ->
  // a = attn(rope(q), rope(k), v) -> x1 = x + a@wo -> n = rms_norm(x1, g)
  // -> m = silu(n) * n -> y = m@w2. Two sequences of 2 rows, 2 heads.
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.seq_len = 2;
  model.tensors.push_back({.name = "x", .dims = {-1, 8}, .is_const = false});
  for (const char* w : {"wq", "wk", "wv", "wo", "w2"})
    model.tensors.push_back(
        {.name = w, .dims = {8, 8}, .is_const = true, .byte_size = 256});
  model.tensors.push_back(
      {.name = "g", .dims = {8}, .is_const = true, .byte_size = 32});
  model.ops.push_back({SmfOpKind::kMatMul, "mm_q", {"x", "wq"}, "q"});
  model.ops.push_back({SmfOpKind::kRope, "rope_q", {"q"}, "qr", 2});
  model.ops.push_back({SmfOpKind::kMatMul, "mm_k", {"x", "wk"}, "k"});
  model.ops.push_back({SmfOpKind::kRope, "rope_k", {"k"}, "kr", 2});
  model.ops.push_back({SmfOpKind::kMatMul, "mm_v", {"x", "wv"}, "v"});
  model.ops.push_back(
      {SmfOpKind::kAttention, "attn", {"qr", "kr", "v"}, "a", 2});
  model.ops.push_back({SmfOpKind::kMatMul, "mm_o", {"a", "wo"}, "o"});
  model.ops.push_back({SmfOpKind::kAdd, "res", {"x", "o"}, "x1"});
  model.ops.push_back({SmfOpKind::kRmsNorm, "rms", {"x1", "g"}, "n"});
  model.ops.push_back({SmfOpKind::kSilu, "silu", {"n"}, "s"});
  model.ops.push_back({SmfOpKind::kMul, "gate", {"s", "n"}, "m"});
  model.ops.push_back({SmfOpKind::kMatMul, "mm_2", {"m", "w2"}, "y"});
  constexpr uint64_t kRow = 8 * sizeof(float);  // one [1x8] f32 row
  constexpr int64_t rows = 4;

  // Every weight adapted: the backward reads qr, kr and v (attention),
  // a (mm_o's input), x1 (the norm's input), n (silu's input and the
  // product's second factor), s (its first), m (mm_2's input) and y — nine
  // row blocks — plus the norm's rstd. The rotations' inputs q and k and
  // the residual add's input o die in the forward.
  TrainingFootprint all = EstimateTrainingFootprint(model, rows);
  EXPECT_EQ(all.weight_bytes, 5 * 256 + 32u);
  EXPECT_EQ(all.activation_bytes, 9 * rows * kRow + rows * sizeof(float));
  // Rematerialization recomputes n from x1 (held for the norm's backward),
  // s from n and m from s and n: three row blocks, never the GEMM outputs.
  EXPECT_EQ(all.recomputable_bytes, 3 * rows * kRow);
  // The probability cache [rows*heads, seq] and the tiled family's stats.
  EXPECT_EQ(all.probs_cache_bytes, rows * 2 * 2 * sizeof(float));
  EXPECT_EQ(all.attention_stats_bytes,
            rows * 2 * kAttnStatsWidth * sizeof(float));

  // Adapt w2 alone: no adjoint flows above mm_2, so the backward reads m
  // (for dA) and the output, nothing else — no norm statistics, no
  // probability cache — and m's factors are not held, so recomputing m
  // would keep something new: nothing is recomputable.
  const std::vector<std::string> only_w2 = {"w2"};
  TrainingFootprint head = EstimateTrainingFootprint(model, rows, only_w2);
  EXPECT_EQ(head.activation_bytes, 2 * rows * kRow);
  EXPECT_EQ(head.recomputable_bytes, 0u);
  EXPECT_EQ(head.probs_cache_bytes, 0u);
  EXPECT_EQ(head.attention_stats_bytes, 0u);
  EXPECT_EQ(head.weight_bytes, all.weight_bytes);
}

TEST(ResourceAnalyzer, UnresolvableWidthStaysALowerBound) {
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 4}, .is_const = false});
  // "w" is missing entirely: the estimate must not invent bytes for it.
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);
  EXPECT_EQ(fp.weight_bytes, 0u);
  EXPECT_EQ(fp.activation_bytes, 0u);
}

TEST(ResourceAnalyzer, DetectsLocalMemory) {
  EXPECT_TRUE(DetectLocalMemoryBytes() > 0);
}

TEST(ResourceAnalyzer, CheckPassesWithinBudget) {
  TrainingFootprint fp{.weight_bytes = 1024, .activation_bytes = 1024};
  EXPECT_OK(CheckTrainableLocally(fp, 4096));
  // Budget 0 = detect physical memory; a 2 KiB footprint always fits.
  EXPECT_OK(CheckTrainableLocally(fp, 0));
}

TEST(ResourceAnalyzer, CheckRejectsBeyondBudget) {
  TrainingFootprint fp{.weight_bytes = 4096, .activation_bytes = 4096};
  EXPECT_ERROR_CONTAINS(CheckTrainableLocally(fp, 1024),
                        "too big to train locally");
}

TEST(ResourceAnalyzer, CompilerFailsFastOnOversizedModel) {
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  UpdateConfig config;
  config.batch = kBatch;
  config.memory_budget_bytes = 256;  // below the model's proven lower bound
  EXPECT_ERROR_CONTAINS(UpdateCompiler(config).Compile(model),
                        "too big to train locally");
}

TEST(ResourceAnalyzer, CompilerAcceptsModelWithinBudget) {
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  UpdateConfig config;
  config.batch = kBatch;
  config.memory_budget_bytes = 64ull << 20;  // 64 MiB
  ASSERT_OK(UpdateCompiler(config).Compile(model));
}

TEST(ResourceAnalyzer, FrozenForwardEstimateChargesPeakNotSum) {
  // A teacher under distillation has no backward: its transients die at
  // their single reader and the binder reuses their slots, so the honest
  // lower bound is the single widest live activation — summing them (the
  // training model) over-counts and can refuse a feasible compile.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  const TrainingFootprint train = EstimateTrainingFootprint(model, kBatch);
  const TrainingFootprint frozen =
      EstimateFrozenForwardFootprint(model, kBatch);
  EXPECT_EQ(frozen.weight_bytes, train.weight_bytes);
  // Peak term: the widest activation is [16 x 8] f32.
  EXPECT_EQ(frozen.activation_bytes, kBatch * 8 * sizeof(float));
  EXPECT_TRUE(frozen.activation_bytes < train.activation_bytes);
}

// --- The merged-delta term of the step-0 gate (E2, #81 — finding #8) ---------

TEST(ResourceAnalyzer, DeltaSegmentCountsExactlyWhatLoraWillAdapt) {
  // MakeMlp(4, 8, 2): w1[4x8] and w2[8x2] are MatMul weights; the biases
  // are consumed by AddBias and can never be adapted.
  SmfModel model = MakeMlp(4, 8, 2, /*seed=*/1);
  EXPECT_EQ(EstimateLoraDeltaBytes(model, {}),
            (4 * 8 + 8 * 2) * sizeof(float));
  const std::vector<std::string> only_w2 = {"w2"};
  EXPECT_EQ(EstimateLoraDeltaBytes(model, only_w2), 8 * 2 * sizeof(float));
  const std::vector<std::string> nothing = {"no-such-tensor"};
  EXPECT_EQ(EstimateLoraDeltaBytes(model, nothing), 0u);

  // The grafter's rule: a weight read any other way — here w2 doubling as
  // a MatMul's LEFT operand — is not adapted, so it must not be charged.
  SmfModel tied = model;
  tied.ops.push_back({.kind = SmfOpKind::kMatMul,
                      .name = "odd",
                      .inputs = {"w2", "w2"},
                      .output = "odd_out"});
  EXPECT_EQ(EstimateLoraDeltaBytes(tied, {}), 4 * 8 * sizeof(float));
}

TEST(ResourceAnalyzer, TheEarlyGateSeesTheDeltasAQuantizedBaseHides) {
  // Under --quantize-base the weights count at 1/4, but the merged deltas
  // are full-size f32 — 4x the weights they patch. A budget between the
  // two estimates used to pass step 0 and fail only at the final gate,
  // after all the work; now it is refused up front, and names the term.
  SmfModel model = MakeMlp(64, 256, 32, /*seed=*/2);
  UpdateConfig config;
  config.batch = kBatch;
  config.quantize_base = true;
  TrainingFootprint fp = EstimateTrainingFootprint(model, kBatch);
  fp.weight_bytes /= 4;
  const uint64_t without_deltas = fp.total_bytes();
  fp.delta_bytes = EstimateLoraDeltaBytes(model, {});
  ASSERT_GT(fp.delta_bytes, fp.weight_bytes);
  EXPECT_EQ(fp.total_bytes(), without_deltas + fp.delta_bytes);

  config.memory_budget_bytes = without_deltas + fp.delta_bytes / 2;
  auto refused = UpdateCompiler(config).Compile(model);
  EXPECT_ERROR_CONTAINS(refused, "too big to train locally");
  EXPECT_ERROR_CONTAINS(refused, "merged LoRA deltas");
  // A lower bound still: what it admits, the exact final gate may refuse,
  // but it never refuses what fits.
  ASSERT_OK_AND_ASSIGN(CompiledUpdate roomy, [&] {
    config.memory_budget_bytes = 64ull << 20;
    return UpdateCompiler(config).Compile(model);
  }());
  EXPECT_LE(fp.total_bytes(), roomy.arena_size + roomy.plan.size());
}

TEST(ResourceAnalyzer, TheGateCountsTheBaseAtItsStoredWidth) {
  // --bf16-base stores the eligible weights at half width, --quantize-base
  // at a quarter; the early gate used to count the bf16 base at full f32
  // (F03) and refuse what the exact gate would pass.
  SmfModel model = MakeMlp(64, 256, 32, /*seed=*/2);
  const uint64_t f32 = EstimateTrainingFootprint(model, kBatch).weight_bytes;
  UpdateConfig config;
  config.batch = kBatch;
  config.memory_budget_bytes = 64ull << 20;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain, UpdateCompiler(config).Compile(model));
  EXPECT_EQ(plain.estimate_weight_bytes, f32);
  config.bf16_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate half, UpdateCompiler(config).Compile(model));
  EXPECT_EQ(half.estimate_weight_bytes, f32 / 2);
  config.bf16_base = false;
  config.quantize_base = true;
  ASSERT_OK_AND_ASSIGN(CompiledUpdate quarter,
                       UpdateCompiler(config).Compile(model));
  EXPECT_EQ(quarter.estimate_weight_bytes, f32 / 4);
}

TEST(ResourceAnalyzer, TheEstimateNeverExceedsTheExactFootprint) {
  // N29 (S1 #147), pinned: the step-0 bound is a lower bound on the arena
  // plus the plan under every policy, so a budget the exact gate admits is
  // never refused up front — at the exact footprint the compile succeeds,
  // one byte under it the FINAL gate refuses, naming the exact numbers.
  const int64_t vocab = 16, dim = 8, heads = 2, seq = 4, ffn = 16, batch = 16;
  SmfModel model =
      seeml::testing::MakeTinyTokenDecoder(vocab, dim, heads, seq, ffn, 41);
  UpdateConfig base;
  base.batch = batch;
  base.loss = LossKind::kSoftmaxXEnt;
  base.lora.rank = 2;
  base.memory_budget_bytes = 256ull << 20;
  auto compiled_with = [&](auto adjust) {
    UpdateConfig config = base;
    adjust(config);
    return UpdateCompiler(config).Compile(model);
  };
  ASSERT_OK_AND_ASSIGN(CompiledUpdate plain, compiled_with([](UpdateConfig& c) {
                         c.remat = RematKind::kOff;
                       }));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate remat, compiled_with([](UpdateConfig& c) {
                         c.remat = RematKind::kOn;
                       }));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate int8, compiled_with([](UpdateConfig& c) {
                         c.remat = RematKind::kOff;
                         c.quantize_base = true;
                       }));
  ASSERT_OK_AND_ASSIGN(CompiledUpdate tiled, compiled_with([](UpdateConfig& c) {
                         c.remat = RematKind::kOff;
                         c.attention = AttentionKind::kTiled;
                       }));
  for (const CompiledUpdate* c : {&plain, &remat, &int8, &tiled}) {
    EXPECT_GT(c->estimate_bytes, 0u);
    EXPECT_LE(c->estimate_bytes, c->arena_size + c->plan.size());
  }
  // Under --remat the bound drops the recomputable activations, and the
  // plan really did recompute them.
  EXPECT_TRUE(remat.remat_applied);
  EXPECT_LT(remat.estimate_activation_bytes, plain.estimate_activation_bytes);
  EXPECT_EQ(remat.estimate_weight_bytes, plain.estimate_weight_bytes);

  for (const auto& [policy, exact] :
       {std::pair{RematKind::kOff, plain.arena_size + plain.plan.size()},
        std::pair{RematKind::kOn, remat.arena_size + remat.plan.size()}}) {
    UpdateConfig config = base;
    config.remat = policy;
    config.memory_budget_bytes = exact;
    EXPECT_OK(UpdateCompiler(config).Compile(model));
    config.memory_budget_bytes = exact - 1;
    const auto refused = UpdateCompiler(config).Compile(model);
    ASSERT_FALSE(refused.has_value());
    EXPECT_STR_CONTAINS(refused.error(), "cannot run locally");
    EXPECT_TRUE(refused.error().find("too big to train") == std::string::npos);
  }
}

}  // namespace
