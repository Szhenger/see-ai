// =============================================================================
// analysis/pass_manager tests: PassManager (ordering, the per-pass
// Block::verify gate, the timings) and the DCE sweep.
// =============================================================================

#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "compiler/analysis/effects.h"
#include "compiler/analysis/update_passes.h"
#include "compiler/frontend/operator/op_builder.h"
#include "compiler/frontend/representation/sir.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;
namespace sir = seeml::sir;

size_t CountOps(sir::Block& block, std::string_view mnemonic) {
  size_t n = 0;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == mnemonic) ++n;
  });
  return n;
}

TEST(PassManager, RunsPassesInRegistrationOrder) {
  sir::Block block;
  std::vector<std::string> ran;

  PassManager pm;
  pm.Add("first", [&](sir::Block&) -> std::expected<void, std::string> {
    ran.push_back("first");
    return {};
  });
  pm.Add("second", [&](sir::Block&) -> std::expected<void, std::string> {
    ran.push_back("second");
    return {};
  });

  ASSERT_TRUE(pm.Run(block).has_value());
  ASSERT_EQ(ran.size(), 2u);
  EXPECT_EQ(ran[0], "first");
  EXPECT_EQ(ran[1], "second");
}

TEST(PassManager, PassErrorsPropagateVerbatim) {
  sir::Block block;
  PassManager pm;
  pm.Add("doomed", [](sir::Block&) -> std::expected<void, std::string> {
    return std::unexpected("DoomedPass: reasons");
  });
  auto r = pm.Run(block);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), "DoomedPass: reasons");
}

TEST(PassManager, NamesThePassThatViolatesInvariants) {
  sir::Block block;
  sir::Operation* fill = block.appendOp("sc_low.fill");
  fill->addResult("x", sir::DataType::F32, sir::Shape{2});

  PassManager pm;
  pm.Add("fine",
         [](sir::Block&) -> std::expected<void, std::string> { return {}; });
  // Corrupts the block by duplicating an existing value id — the invariant
  // gate must attribute the violation to THIS pass, not fail later.
  pm.Add("corruptor", [](sir::Block& b) -> std::expected<void, std::string> {
    sir::Operation* dup = b.appendOp("sc_low.fill");
    dup->addResult("x", sir::DataType::F32, sir::Shape{2});
    return {};
  });

  auto r = pm.Run(block);
  ASSERT_FALSE(r.has_value());
  EXPECT_NE(r.error().find("after pass 'corruptor'"), std::string::npos);
  EXPECT_NE(r.error().find("duplicate value id"), std::string::npos);
}

TEST(DeadCodeElimination, CompactsALargeInterleavedGraphInOneSweep) {
  // Mark-and-compact (E5, #84): thousands of dead chains interleaved with
  // live ops, each dead chain a fan-in (its tail reads two dead values, so
  // the private use counts must reach zero from both sides). One sweep
  // removes all of them, the survivors keep their order, and the block
  // still verifies — use-list symmetry included.
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{4});
  std::unordered_set<const sir::Value*> roots;
  std::vector<std::string> live_order;
  const size_t groups = 3000;
  for (size_t g = 0; g < groups; ++g) {
    const std::string id = std::to_string(g);
    sir::Operation* live = block.appendOp("sc_high.relu");
    live->addOperand(x);
    roots.insert(
        live->addResult("live." + id, sir::DataType::F32, sir::Shape{4}));
    live_order.push_back("live." + id);

    sir::Operation* d0 = block.appendOp("sc_high.gelu");
    d0->addOperand(x);
    sir::Value* v0 =
        d0->addResult("d0." + id, sir::DataType::F32, sir::Shape{4});
    sir::Operation* d1 = block.appendOp("sc_high.silu");
    d1->addOperand(v0);
    sir::Value* v1 =
        d1->addResult("d1." + id, sir::DataType::F32, sir::Shape{4});
    sir::Operation* d2 = block.appendOp("sc_high.add");
    d2->addOperand(v0);  // v0 has two dead users; v1 one
    d2->addOperand(v1);
    d2->addResult("d2." + id, sir::DataType::F32, sir::Shape{4});
  }
  ASSERT_OK_AND_ASSIGN(size_t removed, DeadCodeElimination().Run(block, roots));
  EXPECT_EQ(removed, 3 * groups);
  ASSERT_EQ(block.numOps(), groups);
  size_t i = 0;
  block.walk([&](sir::Operation* op) {
    EXPECT_EQ(std::string(op->result(0)->id()), live_order[i++]);
  });
  EXPECT_EQ(x->users().size(), groups);  // the dead readers of x are gone
  ASSERT_OK(block.verify());
}

TEST(PassManager, RecordsEveryCompletedPassWithItsWallTime) {
  sir::Block block;
  block.addArgument(sir::DataType::F32, sir::Shape{2});
  PassManager pm;
  pm.Add("first",
         [](sir::Block&) -> std::expected<void, std::string> { return {}; });
  pm.Add("second", [](sir::Block& b) -> std::expected<void, std::string> {
    b.appendOp("sc_mem.param")
        ->addResult("p", sir::DataType::F32, sir::Shape{2});
    return {};
  });
  pm.Add("failing", [](sir::Block&) -> std::expected<void, std::string> {
    return std::unexpected("boom");
  });
  EXPECT_ERROR_CONTAINS(pm.Run(block), "boom");
  ASSERT_EQ(pm.timings().size(), 2u);  // the failing pass completed nothing
  EXPECT_EQ(pm.timings()[0].name, "first");
  EXPECT_EQ(pm.timings()[0].ops_after, 0u);
  EXPECT_EQ(pm.timings()[1].name, "second");
  EXPECT_EQ(pm.timings()[1].ops_after, 1u);
  for (const PassTiming& t : pm.timings()) EXPECT_TRUE(t.ms >= 0.0);
}

TEST(DeadCodeElimination, SweepsDeadChainsKeepsRootsUsesAndEffects) {
  sir::Block block;
  sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 4});

  // Live: rooted result.
  sir::Operation* live = block.appendOp("sc_high.relu");
  live->addOperand(x);
  sir::Value* rooted =
      live->addResult("live.y", sir::DataType::F32, sir::Shape{2, 4});

  // Dead chain: a -> b, b unused and unrooted; the backward sweep must
  // remove b first and then the now-useless a in the same run.
  sir::Operation* a = block.appendOp("sc_high.gelu");
  a->addOperand(x);
  sir::Value* av = a->addResult("dead.a", sir::DataType::F32, sir::Shape{2, 4});
  sir::Operation* b = block.appendOp("sc_high.silu");
  b->addOperand(av);
  b->addResult("dead.b", sir::DataType::F32, sir::Shape{2, 4});

  // Effectful with an unused result: must survive on the effect alone.
  sir::Operation* step = block.appendOp("sc_low.sgd_step");
  step->addOperand(rooted);
  step->addResult("step.tick", sir::DataType::F32, sir::Shape{});

  // Storage declaration with no users: kept — the binder holds its name.
  sir::Operation* mem = block.appendOp("sc_mem.param");
  mem->addResult("orphan.param", sir::DataType::F32, sir::Shape{4});

  std::unordered_set<const sir::Value*> roots{rooted};
  ASSERT_OK_AND_ASSIGN(size_t removed, DeadCodeElimination().Run(block, roots));
  EXPECT_EQ(removed, 2u);
  EXPECT_EQ(CountOps(block, "sc_high.gelu"), 0u);
  EXPECT_EQ(CountOps(block, "sc_high.silu"), 0u);
  EXPECT_EQ(CountOps(block, "sc_high.relu"), 1u);
  EXPECT_EQ(CountOps(block, "sc_low.sgd_step"), 1u);
  EXPECT_EQ(CountOps(block, "sc_mem.param"), 1u);
  ASSERT_OK(block.verify());

  // Idempotent: a second sweep finds nothing.
  ASSERT_OK_AND_ASSIGN(size_t again, DeadCodeElimination().Run(block, roots));
  EXPECT_EQ(again, 0u);
}

TEST(InPlaceEffects, EveryRowSurvivesDeadCodeElimination) {
  // effects.h is the one list of ops that rewrite a value they are handed
  // (review 2026-11-25: DCE and the addend fuser each kept a copy). An op
  // on it has no reader and no root and is still kept — for every row.
  for (const seeml::update::InPlaceEffect& e : seeml::update::kInPlaceEffects) {
    sir::Block block;
    sir::Value* x = block.addArgument(sir::DataType::F32, sir::Shape{2, 4});
    sir::Operation* op = block.appendOp(std::string(e.mnemonic));
    for (int i = 0; i < 5; ++i) op->addOperand(x);
    ASSERT_OK_AND_ASSIGN(size_t removed, DeadCodeElimination().Run(block, {}));
    EXPECT_EQ(removed, 0u);
    EXPECT_EQ(CountOps(block, std::string(e.mnemonic)), 1u);
    EXPECT_TRUE(seeml::update::MutatesInPlace(*op));
  }
  sir::Block block;
  sir::Operation* pure = block.appendOp("sc_high.relu");
  EXPECT_FALSE(seeml::update::MutatesInPlace(*pure));
}

TEST(InPlaceEffects, AnOpRewritesTheOperandsItsRowNamesAndReadsTheRest) {
  sir::Block block;
  auto value = [&](const char* id) {
    return block.appendOp("sc_mem.param")
        ->addResult(id, sir::DataType::F32, sir::Shape{2, 4});
  };
  sir::Value *p = value("p"), *g = value("g"), *m = value("m"), *v = value("v"),
             *other_p = value("p2"), *other_g = value("g2");
  sir::Operation* step = block.appendOp("sc_low.adamw_step");
  for (sir::Value* operand : {p, g, m, v}) step->addOperand(operand);
  // The step moves the parameter and both moments; the gradient is read.
  EXPECT_TRUE(seeml::update::RewrittenInPlace(p));
  EXPECT_TRUE(seeml::update::RewrittenInPlace(m));
  EXPECT_TRUE(seeml::update::RewrittenInPlace(v));
  EXPECT_FALSE(seeml::update::RewrittenInPlace(g));
  sir::Operation* sgd = block.appendOp("sc_low.sgd_step");
  sgd->addOperand(other_p);
  sgd->addOperand(other_g);
  EXPECT_TRUE(seeml::update::RewrittenInPlace(other_p));
  EXPECT_FALSE(seeml::update::RewrittenInPlace(other_g));
  // A clip scales the gradient where it lies: now it is rewritten too.
  block.appendOp("sc_low.clip_norm")->addOperand(other_g);
  EXPECT_TRUE(seeml::update::RewrittenInPlace(other_g));
}

TEST(InPlaceEffects, AChainIsNotFoldedAcrossARewriteOfWhatItReads) {
  // The chain fuser moves every read of a chain to its tail. It refused
  // operands that are storage declarations and called gradients single-
  // assignment — but a clip rewrites one in place, and a chain whose head
  // read it before the clip would read it after. Latent (no plan has an
  // elementwise reader of a clipped gradient); closed by the shared rule.
  auto build = [](bool clip) {
    auto block = std::make_unique<sir::Block>();
    sir::Value* g = block->addArgument(sir::DataType::F32, sir::Shape{2, 4});
    sir::Operation* head = block->appendOp("sc_high.relu");
    head->addOperand(g);
    sir::Value* a = head->addResult("a", sir::DataType::F32, sir::Shape{2, 4});
    if (clip) block->appendOp("sc_low.clip_norm")->addOperand(g);
    sir::Operation* tail = block->appendOp("sc_high.gelu");
    tail->addOperand(a);
    tail->addResult("b", sir::DataType::F32, sir::Shape{2, 4});
    return block;
  };
  auto plain = build(false);
  ASSERT_OK_AND_ASSIGN(seeml::update::ChainFusion folded,
                       seeml::update::ElementwiseChainFuser().Run(*plain, {}));
  EXPECT_EQ(folded.fused_chains, 1u);
  auto clipped = build(true);
  ASSERT_OK_AND_ASSIGN(
      seeml::update::ChainFusion kept,
      seeml::update::ElementwiseChainFuser().Run(*clipped, {}));
  EXPECT_EQ(kept.fused_chains, 0u);
}

}  // namespace
