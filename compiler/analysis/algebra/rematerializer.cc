#include "compiler/analysis/algebra/rematerializer.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "compiler/diagnostics/updating/error.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

namespace {

/// The cost class: ops that cost one pass over their output. A GEMM, an
/// attention op or a loss never qualifies; nor does an add the addend
/// fuser folded into a GEMM (E10, "gemm_addend"), which lowers as one.
bool Recomputable(const sir::Operation& op) {
  if (op.numResults() == 0 || op.hasAttribute("gemm_addend")) return false;
  if (op.hasAttribute("fused_stages")) return true;  // a kFusedMap chain
  const std::string_view m = op.mnemonic();
  return m == "sc_high.rms_norm" || m == "sc_high.layer_norm" ||
         m == "sc_high.silu" || m == "sc_high.relu" || m == "sc_high.gelu" ||
         m == "sc_high.mul" || m == "sc_high.add" || m == "sc_high.scale" ||
         m == "sc_high.add_bias" || m == "sc_high.rope" ||
         m == "sc_high.embedding";
}

uint64_t Bytes(const sir::Value* v) {
  return v->shape().byteSize(v->dtype());
}

}  // namespace

std::expected<Rematerialization, std::string> Rematerializer::Run(
    sir::Block& block, const std::vector<sir::Operation*>& primal_ops,
    const std::unordered_set<const sir::Value*>& pinned) {
  const std::unordered_set<const sir::Operation*> primal(primal_ops.begin(),
                                                         primal_ops.end());
  // Program positions. Originals take their index; a copy inserted before
  // an anchor takes the midpoint between its two neighbours, so the order
  // tests below (first backward reader, last reader of an operand) hold
  // for copies and originals alike without renumbering.
  std::vector<sir::Operation*> ops;
  ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { ops.push_back(op); });
  std::unordered_map<const sir::Operation*, double> pos;
  pos.reserve(ops.size() * 2);
  for (size_t i = 0; i < ops.size(); ++i) pos[ops[i]] = static_cast<double>(i);
  auto position = [&](const sir::Operation* op) -> std::expected<double, std::string> {
    if (auto it = pos.find(op); it != pos.end()) return it->second;
    return updating::Error(updating::kRematerializer,
                           "reader outside the program: " + op->toString());
  };
  // Storage (arguments, weights, parameters, optimizer state) and pinned
  // values are resident regardless: recomputing from them keeps nothing.
  auto fixed = [&](const sir::Value* v) {
    return v->isBlockArgument() || v->definingOp()->isMemoryOp() ||
           pinned.contains(v);
  };

  Rematerialization out;
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    sir::Operation* op = *it;
    if (!primal.contains(op) || !Recomputable(*op)) continue;
    bool any_pinned = false;
    for (const auto& r : op->results()) any_pinned |= pinned.contains(r.get());
    if (any_pinned) continue;

    // The first reader outside the snapshot (an original backward op or a
    // copy placed earlier in this sweep) anchors the recompute; the bytes
    // freed are the results it reads.
    sir::Operation* anchor = nullptr;
    double anchor_pos = std::numeric_limits<double>::infinity();
    uint64_t gain = 0;
    size_t read_results = 0;
    for (const auto& r : op->results()) {
      bool backward_read = false;
      for (sir::Operation* u : r->users()) {
        if (primal.contains(u)) continue;
        backward_read = true;
        auto p = position(u);
        if (!p) return std::unexpected(p.error());
        if (*p < anchor_pos) {
          anchor_pos = *p;
          anchor = u;
        }
      }
      if (backward_read) {
        gain += Bytes(r.get());
        ++read_results;
      }
    }
    if (!anchor) continue;

    // What the recompute would newly keep: a transient operand whose last
    // reader precedes the anchor would live until it — unless its own
    // producer is a cheap primal op that can be recomputed in turn, in
    // which case it costs what *that* keeps (the sweep reaches the
    // producer next and takes the same decision at the same point: the
    // embedding gather under the first norm, whose operands are all
    // storage, is free, so the norm is a clean win rather than a swap).
    std::function<uint64_t(const sir::Value*, int)> keep_cost =
        [&](const sir::Value* v, int depth) -> uint64_t {
      if (fixed(v)) return 0;
      double last = -1.0;
      for (const sir::Operation* u : v->users())
        last = std::max(last, pos.at(u));
      if (last >= anchor_pos) return 0;  // live there regardless
      const uint64_t bytes = Bytes(v);
      const sir::Operation* producer = v->definingOp();
      if (depth == 0 || !primal.contains(producer) || !Recomputable(*producer))
        return bytes;
      uint64_t chain = 0;
      for (const sir::Value* u : producer->operands()) {
        chain += keep_cost(u, depth - 1);
        if (chain >= bytes) return bytes;
      }
      return chain;
    };
    uint64_t cost = 0;
    for (const sir::Value* v : op->operands()) cost += keep_cost(v, 4);
    if (cost >= gain) {
      ++out.declined;
      continue;
    }

    // The copy: the op as it stands, results renamed.
    auto clone = std::make_unique<sir::Operation>(std::string(op->mnemonic()));
    for (const auto& [key, value] : op->attributes())
      clone->setAttribute(key, value);
    for (sir::Value* v : op->operands()) clone->addOperand(v);
    std::vector<std::pair<sir::Value*, sir::Value*>> renamed;
    renamed.reserve(op->numResults());
    for (const auto& r : op->results())
      renamed.emplace_back(
          r.get(), clone->addResult(std::string(r->id()) + ".remat",
                                    r->dtype(), r->shape()));

    // Right before the anchor: after the op that precedes it now.
    const auto current = block.operations();
    const auto at = std::find_if(
        current.begin(), current.end(),
        [&](const std::unique_ptr<sir::Operation>& p) { return p.get() == anchor; });
    if (at == current.begin() || at == current.end())
      return updating::Error(updating::kRematerializer,
                             "backward reader has no predecessor: " +
                                 anchor->toString());
    sir::Operation* pred = std::prev(at)->get();
    sir::Operation* copy = clone.get();
    std::vector<std::unique_ptr<sir::Operation>> inserted;
    inserted.push_back(std::move(clone));
    block.insertOpsAfter(pred, std::move(inserted));
    pos[copy] = (pos.at(pred) + anchor_pos) / 2.0;

    // Every reader outside the snapshot now reads the copy. (The user list
    // is copied first: setOperand rewrites it.)
    for (const auto& [original, fresh] : renamed) {
      std::vector<sir::Operation*> readers(original->users().begin(),
                                           original->users().end());
      for (sir::Operation* u : readers) {
        if (primal.contains(u)) continue;
        for (size_t i = 0; i < u->numOperands(); ++i)
          if (u->operand(i) == original) u->setOperand(i, fresh);
      }
    }
    ++out.recomputed_ops;
    out.recomputed_values += read_results;
    out.bytes_estimated += gain - cost;
  }
  return out;
}

}  // namespace seeml::update
