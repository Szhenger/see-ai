#include "compiler/frontend/accountant/resource_analyzer.h"

#include "source/contract/bounds.h"
#include "source/identity/host_memory.h"
#include "source/plan/instruction.h"

#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#include "compiler/diagnostics/tokenizing/error.h"
#include "compiler/frontend/topology/sema.h"

namespace seeml::update {

namespace {

// SatAdd / SatMul: the shared saturating helpers (source/contract/bounds.h).

std::string FormatMiB(uint64_t bytes) {
  // Round up without the additive form: bytes near UINT64_MAX (exactly what
  // SatAdd/SatMul saturate to) would wrap and print "0 MiB".
  const uint64_t whole = bytes >> 20;
  const uint64_t frac = bytes & ((uint64_t{1} << 20) - 1);
  return std::to_string(whole + (frac != 0 ? 1 : 0)) + " MiB";
}

}  // namespace

uint64_t TrainingFootprint::total_bytes() const {
  return SatAdd(SatAdd(SatAdd(weight_bytes, activation_bytes), delta_bytes),
                probs_cache_bytes);
}

TrainingFootprint& TrainingFootprint::operator+=(const TrainingFootprint& o) {
  weight_bytes = SatAdd(weight_bytes, o.weight_bytes);
  activation_bytes = SatAdd(activation_bytes, o.activation_bytes);
  delta_bytes = SatAdd(delta_bytes, o.delta_bytes);
  probs_cache_bytes = SatAdd(probs_cache_bytes, o.probs_cache_bytes);
  attention_stats_bytes =
      SatAdd(attention_stats_bytes, o.attention_stats_bytes);
  return *this;
}

namespace {

// The LoRA grafter's eligibility rule is topology's one reading of it
// (sema::AdaptedWeights); the delta estimate and the activation bound
// below both turn on which products the grafter will adapt.
using sema::AdaptedWeights;

/// Shared walk behind both estimators. `sum_activations` selects the
/// training model (the forward values the backward reads are all alive
/// when it begins — sum them) or the frozen-forward model (no backward
/// consumes anything, so transients die at their single reader and the
/// arena binder reuses their slots — the honest lower bound is the single
/// widest live term). `adapted` names the weights LoRA adapts (the
/// training walk alone reads it).
TrainingFootprint EstimateFootprintImpl(
    const SmfModel& model, int64_t batch, bool sum_activations,
    const std::unordered_set<std::string_view>& adapted) {
  TrainingFootprint fp;
  uint64_t peak_activation = 0;
  auto charge = [&](uint64_t term) {
    if (sum_activations)
      fp.activation_bytes = SatAdd(fp.activation_bytes, term);
    else
      peak_activation = std::max(peak_activation, term);
  };
  const uint64_t rows = batch > 0 ? static_cast<uint64_t>(batch) : 0;

  // Frozen weights: one resident copy each (they at least fill rodata).
  // Track each tensor's last dimension as its "width" to seed propagation;
  // the reader has already validated dims, so widths are positive (or the
  // dynamic batch dim, which never appears last).
  std::unordered_map<std::string_view, const SmfTensor*> tensors;
  std::unordered_map<std::string_view, uint64_t> width;
  std::unordered_map<std::string_view, uint64_t> row_count;
  tensors.reserve(model.tensors.size());
  for (const SmfTensor& t : model.tensors) {
    tensors[t.name] = &t;
    if (t.is_const) fp.weight_bytes = SatAdd(fp.weight_bytes, t.byte_size);
    if (!t.dims.empty() && t.dims.back() > 0)
      width[t.name] = static_cast<uint64_t>(t.dims.back());
    // Row seed: a constant tensor keeps its declared row count (the parser
    // sizes a MatMul from its actual LHS, which may be a const with rows !=
    // batch); anything else is served at the compiled batch. Over-counting
    // rows would break the lower-bound guarantee rejections rely on.
    row_count[t.name] =
        t.is_const ? (t.dims.size() >= 2 && t.dims.front() > 0
                          ? static_cast<uint64_t>(t.dims.front())
                          : 1)
                   : rows;
  }

  // The training walk charges a forward value only where the backward
  // reads it, and the VJP rules (analysis/calculus/autodiff.cc) say where:
  // an adapted MatMul's activation input (dA = Xᵀ·dt, emitted for every
  // trainable adapter), each operand of an activation-by-activation
  // product for the other's adjoint, attention's V where Q or K is
  // differentiated and its Q or K for the other's adjoint, the input of a
  // norm or activation function and each factor of an elementwise product
  // where the other side is differentiated — and the model output the loss
  // differentiates. A residual add, a bias add and a rotation read nothing
  // of their forward (their adjoint is the incoming one, reduced or
  // rotated), so their inputs die in the forward. Whether a rule is
  // emitted at all follows needs_grad forward from the adapted products,
  // as autodiff propagates it. Every value charged is alive when the
  // backward begins, which is what makes the sum a lower bound on the
  // arena. `recomputable` marks the charged values rematerialization
  // rebuilds there instead of keeping — a norm's, an activation function's,
  // a product's or the embedding gather's output whose operands the
  // backward holds anyway (read, pinned IO, rodata, or recomputable
  // themselves: the pass's own rule, that a recompute may keep nothing new).
  std::unordered_set<std::string_view> needs_grad, read, recomputable;
  auto is_const = [&](std::string_view name) {
    auto it = tensors.find(name);
    return it != tensors.end() && it->second->is_const;
  };
  auto grad = [&](std::string_view name) { return needs_grad.contains(name); };
  if (sum_activations) {
    for (const SmfOp& op : model.ops) {
      bool out_grad = false;
      for (const std::string& in : op.inputs) out_grad = out_grad || grad(in);
      const bool matmul =
          op.kind == SmfOpKind::kMatMul && op.inputs.size() == 2;
      const bool adapted_weight = matmul && adapted.contains(op.inputs[1]);
      out_grad = out_grad || adapted_weight;
      switch (op.kind) {
        case SmfOpKind::kMatMul:
          if (!matmul) break;
          if (adapted_weight ||
              (!is_const(op.inputs[1]) && grad(op.inputs[1])))
            read.insert(op.inputs[0]);
          if (!is_const(op.inputs[1]) && grad(op.inputs[0]))
            read.insert(op.inputs[1]);
          break;
        case SmfOpKind::kAttention:
          if (op.inputs.size() != 3) break;
          if (grad(op.inputs[0]) || grad(op.inputs[1]))
            read.insert(op.inputs[2]);
          if (grad(op.inputs[0])) read.insert(op.inputs[1]);
          if (grad(op.inputs[1])) read.insert(op.inputs[0]);
          break;
        case SmfOpKind::kLayerNorm:
        case SmfOpKind::kRmsNorm:
        case SmfOpKind::kRelu:
        case SmfOpKind::kGelu:
        case SmfOpKind::kSilu:
          if (!op.inputs.empty() && grad(op.inputs[0]))
            read.insert(op.inputs[0]);
          break;
        case SmfOpKind::kMul:
          if (op.inputs.size() != 2) break;
          if (grad(op.inputs[1])) read.insert(op.inputs[0]);
          if (grad(op.inputs[0])) read.insert(op.inputs[1]);
          break;
        default:  // kAdd, kAddBias, kRope, kEmbedding: nothing of the forward
          break;
      }
      if (out_grad) needs_grad.insert(op.output);
    }
    read.insert(model.output_name);
  }
  auto kept_anyway = [&](std::string_view name) {
    return read.contains(name) || recomputable.contains(name) ||
           is_const(name) || name == model.input_name;
  };

  // Forward activations: propagate output widths and row counts with the
  // parser's shape rules. Every activation is [rows, width] f32. An
  // unresolvable width contributes zero — the estimate must stay a lower
  // bound.
  for (const SmfOp& op : model.ops) {
    uint64_t w = 0;
    uint64_t r = rows;
    if (!op.inputs.empty())
      if (auto it = row_count.find(op.inputs[0]); it != row_count.end())
        r = it->second;
    // The training walk: whether this op's backward rule runs at all (its
    // output needs a gradient), which decides whether its side caches —
    // the norms' row statistics, attention's probabilities — live past
    // the forward.
    const bool backward_runs = !sum_activations || grad(op.output);
    switch (op.kind) {
      case SmfOpKind::kMatMul:
      case SmfOpKind::kEmbedding: {
        // Both produce [rows, dims[1] of the second operand]: the matmul's
        // B width, or the embedding table's dim.
        if (op.inputs.size() != 2) break;
        auto it = tensors.find(op.inputs[1]);
        if (it != tensors.end() && it->second->dims.size() == 2 &&
            it->second->dims[1] > 0)
          w = static_cast<uint64_t>(it->second->dims[1]);
        break;
      }
      case SmfOpKind::kAddBias:
      case SmfOpKind::kRelu:
      case SmfOpKind::kGelu:
      case SmfOpKind::kSilu:
      case SmfOpKind::kMul:
      case SmfOpKind::kLayerNorm:
      case SmfOpKind::kAdd:
      case SmfOpKind::kRmsNorm:
      case SmfOpKind::kRope:
      case SmfOpKind::kAttention: {
        if (op.inputs.empty()) break;
        if (auto it = width.find(op.inputs[0]); it != width.end())
          w = it->second;
        // LayerNorm additionally caches per-row mean/rstd for the backward
        // kernel (two f32 per row); RMSNorm caches rstd (one).
        if (op.kind == SmfOpKind::kLayerNorm && backward_runs)
          charge(SatMul(2 * sizeof(float), r));
        if (op.kind == SmfOpKind::kRmsNorm && backward_runs)
          charge(SatMul(sizeof(float), r));
        // Attention caches the probability matrix P[B,H,S,S] — flattened
        // [rows * heads, seq_len] — for the backward primitives; usually
        // the dominant transformer activation. heads/seq_len of zero mean
        // an invalid model the parser will reject; contribute nothing here
        // so the estimate stays a lower bound.
        // It is counted on its own (probs_cache_bytes) in the summing
        // walk — the training footprint, where the compiler decides
        // whether to keep it — and as an activation at the peak walk (a
        // frozen forward never holds it past the op).
        if (op.kind == SmfOpKind::kAttention && op.attr0 > 0 &&
            model.seq_len > 0 && backward_runs) {
          const uint64_t rows_heads = SatMul(r, op.attr0);
          const uint64_t cache =
              SatMul(SatMul(rows_heads, model.seq_len), sizeof(float));
          if (sum_activations) {
            fp.probs_cache_bytes = SatAdd(fp.probs_cache_bytes, cache);
            fp.attention_stats_bytes = SatAdd(
                fp.attention_stats_bytes,
                SatMul(SatMul(rows_heads, kAttnStatsWidth), sizeof(float)));
          } else {
            charge(cache);
          }
        }
        break;
      }
    }
    if (w == 0) continue;
    width[op.output] = w;
    row_count[op.output] = r;
    if (sum_activations && !read.contains(op.output)) continue;
    const uint64_t bytes = SatMul(SatMul(r, w), sizeof(float));
    charge(bytes);
    if (!sum_activations) continue;
    bool cheap = false;
    switch (op.kind) {
      case SmfOpKind::kLayerNorm:
      case SmfOpKind::kRmsNorm:
      case SmfOpKind::kRelu:
      case SmfOpKind::kGelu:
      case SmfOpKind::kSilu:
      case SmfOpKind::kMul:
      case SmfOpKind::kEmbedding:
        cheap = true;
        break;
      default:  // a GEMM, attention, a rotation, an add: never recomputed
        break;
    }
    if (!cheap) continue;
    bool free_to_recompute = true;
    for (const std::string& in : op.inputs)
      free_to_recompute = free_to_recompute && kept_anyway(in);
    if (!free_to_recompute) continue;
    recomputable.insert(op.output);
    fp.recomputable_bytes = SatAdd(fp.recomputable_bytes, bytes);
  }
  if (!sum_activations) fp.activation_bytes = peak_activation;
  return fp;
}

}  // namespace

TrainingFootprint EstimateTrainingFootprint(
    const SmfModel& model, int64_t batch,
    std::span<const std::string> target_filters) {
  return EstimateFootprintImpl(model, batch, /*sum_activations=*/true,
                               AdaptedWeights(model, target_filters));
}

TrainingFootprint EstimateFrozenForwardFootprint(const SmfModel& model,
                                                 int64_t batch) {
  return EstimateFootprintImpl(model, batch, /*sum_activations=*/false, {});
}

uint64_t EstimateLoraDeltaBytes(const SmfModel& model,
                                std::span<const std::string> target_filters) {
  const std::unordered_set<std::string_view> adapted =
      AdaptedWeights(model, target_filters);
  uint64_t bytes = 0;
  for (const SmfTensor& t : model.tensors)
    if (adapted.contains(t.name))
      bytes = SatAdd(bytes, t.byte_size);  // the delta is f32, like the file
  return bytes;
}

uint64_t DetectLocalMemoryBytes() {
  // One query, shared with the runtime's load (source/identity/host_memory.h).
  return DetectHostMemoryBytes();
}

std::expected<void, std::string> CheckTrainableLocally(
    const TrainingFootprint& footprint, uint64_t budget_bytes) {
  const uint64_t budget =
      budget_bytes != 0 ? budget_bytes : DetectLocalMemoryBytes();
  if (budget == 0) return {};  // cannot prove infeasibility — do not reject
  const uint64_t need = footprint.total_bytes();
  if (need <= budget) return {};
  return seeml::diag::tokenizing::IngressError(
      "model is too big to train locally: weights " +
      FormatMiB(footprint.weight_bytes) + " + activations " +
      FormatMiB(footprint.activation_bytes) + " + merged LoRA deltas " +
      FormatMiB(footprint.delta_bytes) + " + attention caches " +
      FormatMiB(footprint.probs_cache_bytes) + " need at least " +
      FormatMiB(need) + ", but the local memory budget is " +
      FormatMiB(budget));
}

std::expected<void, std::string> CheckPlanFitsLocally(uint64_t arena_bytes,
                                                      uint64_t plan_bytes,
                                                      uint64_t budget_bytes) {
  const uint64_t budget =
      budget_bytes != 0 ? budget_bytes : DetectLocalMemoryBytes();
  if (budget == 0) return {};  // cannot prove infeasibility — do not reject
  const uint64_t need = SatAdd(arena_bytes, plan_bytes);
  if (need <= budget) return {};
  return seeml::diag::tokenizing::IngressError(
      "compiled update cannot run locally: arena " + FormatMiB(arena_bytes) +
      " + plan " + FormatMiB(plan_bytes) + " need " + FormatMiB(need) +
      ", but the local memory budget is " + FormatMiB(budget));
}

}  // namespace seeml::update
