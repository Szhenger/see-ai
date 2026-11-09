#include "compiler/frontend/topology/sema.h"

#include <bit>
#include <cmath>
#include <string_view>
#include <unordered_set>

#include "compiler/diagnostics/parsing/error.h"
#include "source/contract/bounds.h"

namespace seeml::update::sema {

namespace sir = seeml::sir;
namespace parsing = seeml::diag::parsing;

std::expected<void, std::string> CheckGraph(const SmfModel& model) {
  // Every name an op may produce, for distinguishing "consumed before
  // produced" (a topological-order violation) from "does not exist" (a
  // resolution error named by VerifyModel, which knows tensor constness).
  std::unordered_set<std::string_view> all_outputs;
  all_outputs.reserve(model.ops.size());
  for (const SmfOp& op : model.ops) all_outputs.insert(op.output);

  // Names bound so far, in op order: tensors and the input up front, then
  // each op's output as it appears.
  std::unordered_set<std::string_view> bound;
  bound.reserve(model.tensors.size() + model.ops.size() + 1);
  for (const SmfTensor& t : model.tensors) bound.insert(t.name);
  bound.insert(model.input_name);

  // The graph input is I/O by definition. A constant tensor carrying the
  // input's name would be silently shadowed by the batch input (the resolver
  // binds the input first and never materializes the weight), compiling to
  // wrong math instead of an error — reject the contradiction here.
  for (const SmfTensor& t : model.tensors)
    if (t.is_const && t.name == model.input_name)
      return parsing::Error("model input '" + model.input_name +
                            "' is declared as a constant tensor");

  for (const SmfOp& op : model.ops) {
    for (const std::string& in : op.inputs)
      if (!bound.contains(in) && all_outputs.contains(in))
        return parsing::OpError(
            "op", op.name,
            "consumes '" + in +
                "' before it is produced (the op list must be topologically "
                "ordered)");
    if (!bound.insert(op.output).second)
      return parsing::OpError("op", op.name,
                              "output '" + op.output +
                                  "' redefines an existing value");
  }

  // Some ops derive companion value ids from their output (LayerNorm's
  // "<output>.mean"/".rstd", RMSNorm's ".rstd", Attention's ".probs"); a
  // model name equal to one of them would collide in the SIR value table
  // far from either offender, so reject it here.
  for (const SmfOp& op : model.ops) {
    const char* suffixes[2] = {nullptr, nullptr};
    switch (op.kind) {
      case SmfOpKind::kLayerNorm:
        suffixes[0] = ".mean";
        suffixes[1] = ".rstd";
        break;
      case SmfOpKind::kRmsNorm:   suffixes[0] = ".rstd"; break;
      case SmfOpKind::kAttention: suffixes[0] = ".probs"; break;
      default: continue;
    }
    for (const char* suffix : suffixes) {
      if (suffix == nullptr) continue;
      const std::string derived = op.output + suffix;
      if (bound.contains(derived))
        return parsing::OpError(
            "op", op.name,
            "output '" + op.output + "' derives companion id '" + derived +
                "', which collides with an existing name");
    }
  }

  if (!all_outputs.contains(model.output_name))
    return parsing::Error("model output '" + model.output_name +
                          "' was never produced by an operation");

  // Token-input discipline (SMF v4): a rank-1 dynamic ({-1}) non-const
  // input declares i32 token ids. That declaration and kEmbedding must
  // appear together; every consumer of the input must be an embedding (any
  // other op would read the ids as f32 features); and every embedding must
  // read the graph input directly (a lookup over computed activations has
  // no meaning here).
  const SmfTensor* input_tensor = nullptr;
  for (const SmfTensor& t : model.tensors)
    if (t.name == model.input_name) input_tensor = &t;
  const bool token_input = input_tensor && !input_tensor->is_const &&
                           input_tensor->dims.size() == 1 &&
                           input_tensor->dims[0] == -1;
  bool any_embedding = false;
  for (const SmfOp& op : model.ops) {
    if (op.kind == SmfOpKind::kEmbedding) {
      any_embedding = true;
      if (op.inputs.empty() || op.inputs[0] != model.input_name)
        return parsing::OpError(
            "Embedding", op.name,
            "must read the graph input's token ids directly");
      // The table must be a frozen weight: lowering emits it as a rodata
      // ref, so a computed-activation table (rank-2 F32, which autodiff's
      // needs-grad sweep skips) would compile clean and then fail every
      // load at the validator's rodata check — a model bug blamed on the
      // plan. Reject it here.
      const SmfTensor* table =
          op.inputs.size() >= 2 ? model.FindTensor(op.inputs[1]) : nullptr;
      if (!table || !table->is_const)
        return parsing::OpError("Embedding", op.name,
                                "table must be a frozen weight tensor");
    } else if (token_input) {
      for (const std::string& in : op.inputs)
        if (in == model.input_name)
          return parsing::OpError(
              "op", op.name,
              "consumes the token input as floats; only Embedding may read "
              "it");
    }
  }
  if (any_embedding && !token_input)
    return parsing::Error(
        "Embedding requires the model input to be a rank-1 dynamic ({-1}) "
        "token tensor");
  if (token_input && !any_embedding)
    return parsing::Error(
        "a token input requires at least one Embedding consumer");
  return {};
}

namespace {

// --- The per-kind shape rules, over shapes alone ------------------------------

std::expected<void, std::string> CheckMatMul(const SmfOp& op,
                                             const sir::Shape& x,
                                             const sir::Shape& w) {
  if (x.dims.size() != 2 || w.dims.size() != 2)
    return parsing::OpError("MatMul", op.name, "operands must be rank-2");
  const int64_t k_x = x.dims.at(1);
  const int64_t k_w = w.dims.at(0);
  if (k_x != k_w)
    return parsing::OpError("MatMul", op.name,
                            "inner dimensions disagree (" +
                                std::to_string(k_x) + " vs " +
                                std::to_string(k_w) + ")");
  return {};
}

std::expected<void, std::string> CheckAddBias(const SmfOp& op,
                                              const sir::Shape& x,
                                              const sir::Shape& b) {
  // Rank-2 x is a lowering requirement, not a preference: sc_high.add_bias
  // lowering reads dims 0/1 as rows/cols, so a rank-1 input aborts the
  // compiler and a rank-3 input compiles to wrong math.
  if (x.dims.size() != 2)
    return parsing::OpError("AddBias", op.name, "input must be rank-2");
  if (b.dims.size() != 1 || b.dims[0] != x.dims.back())
    return parsing::OpError("AddBias", op.name,
                            "bias width does not match its input");
  return {};
}

std::expected<void, std::string> CheckSameShape(const char* unit,
                                                const SmfOp& op,
                                                const sir::Shape& x,
                                                const sir::Shape& y) {
  if (x != y) return parsing::OpError(unit, op.name, "operand shapes disagree");
  return {};
}

std::expected<void, std::string> CheckLayerNorm(const SmfOp& op,
                                                const sir::Shape& x,
                                                const sir::Shape& gamma,
                                                const sir::Shape& beta) {
  if (x.dims.size() != 2)
    return parsing::OpError("LayerNorm", op.name, "input must be rank-2");
  const int64_t d = x.dims.back();
  for (const sir::Shape* affine : {&gamma, &beta})
    if (affine->dims.size() != 1 || affine->dims[0] != d)
      return parsing::OpError("LayerNorm", op.name,
                              "gamma/beta width does not match input");
  return {};
}

std::expected<void, std::string> CheckRmsNorm(const SmfOp& op,
                                              const sir::Shape& x,
                                              const sir::Shape& gamma) {
  if (x.dims.size() != 2)
    return parsing::OpError("RmsNorm", op.name, "input must be rank-2");
  if (gamma.dims.size() != 1 || gamma.dims[0] != x.dims.back())
    return parsing::OpError("RmsNorm", op.name,
                            "gamma width does not match input");
  return {};
}

/// Shared sequence-geometry gate for the ops that interpret rows as
/// positions: rows split into sequences of seq_len, and heads divide the
/// width. `even_head` additionally requires an even head width (RoPE pairs).
std::expected<void, std::string> CheckSeqGeometry(const char* unit,
                                                  const SmfOp& op,
                                                  const sir::Shape& x,
                                                  uint32_t heads,
                                                  uint64_t seq_len,
                                                  bool even_head) {
  if (x.dims.size() != 2)
    return parsing::OpError(unit, op.name, "input must be rank-2");
  if (heads == 0)
    return parsing::OpError(unit, op.name,
                            "num_heads (attr0) must be positive");
  if (seq_len == 0)
    return parsing::OpError(unit, op.name,
                            "model declares no seq_len, which this op needs");
  // seq_len is untrusted u64 from the file; a value past INT64_MAX would
  // wrap negative through the int64 casts below (rows % -1 == 0 accepts
  // everything) and reach the parser as a negative shape dim.
  if (seq_len > static_cast<uint64_t>(INT64_MAX))
    return parsing::OpError(unit, op.name,
                            "seq_len does not fit a signed 64-bit dimension");
  const int64_t rows = x.dims.at(0);
  const int64_t width = x.dims.at(1);
  if (rows % static_cast<int64_t>(seq_len) != 0)
    return parsing::OpError(
        unit, op.name,
        "batch rows (" + std::to_string(rows) +
            ") are not a whole number of sequences of seq_len " +
            std::to_string(seq_len));
  if (width % static_cast<int64_t>(heads) != 0)
    return parsing::OpError(unit, op.name,
                            "width " + std::to_string(width) +
                                " is not divisible by num_heads " +
                                std::to_string(heads));
  if (even_head && (width / static_cast<int64_t>(heads)) % 2 != 0)
    return parsing::OpError(unit, op.name,
                            "head width must be even for the rotation pairs");
  return {};
}

std::expected<void, std::string> CheckRope(const SmfOp& op,
                                           const sir::Shape& x, uint32_t heads,
                                           uint64_t seq_len,
                                           uint32_t base_bits) {
  if (auto ok = CheckSeqGeometry("Rope", op, x, heads, seq_len,
                                 /*even_head=*/true);
      !ok)
    return ok;
  // The base feeds pow(base, -2i/d): anything non-finite, non-positive, or
  // <= 1 makes the rotation angles degenerate (or NaN) rather than a
  // different positional encoding.
  const float base = RopeBaseOf(base_bits);
  if (!std::isfinite(base) || base <= 1.0f)
    return parsing::OpError("Rope", op.name,
                            "rotary base (attr1) must be a finite float > 1");
  return {};
}

std::expected<void, std::string> CheckEmbedding(const SmfOp& op,
                                                bool tokens_are_the_input,
                                                const sir::Shape& tokens,
                                                const sir::Shape& table) {
  if (!tokens_are_the_input || tokens.dims.size() != 1)
    return parsing::OpError("Embedding", op.name,
                            "tokens must be the rank-1 i32 graph input");
  if (table.dims.size() != 2)
    return parsing::OpError("Embedding", op.name,
                            "table must be a rank-2 [vocab, dim] weight");
  return {};
}

std::expected<void, std::string> CheckAttention(const SmfOp& op,
                                                const sir::Shape& q,
                                                const sir::Shape& k,
                                                const sir::Shape& v,
                                                uint32_t heads,
                                                uint64_t seq_len) {
  if (q != k || q != v)
    return parsing::OpError("Attention", op.name, "q/k/v shapes disagree");
  return CheckSeqGeometry("Attention", op, q, heads, seq_len,
                          /*even_head=*/false);
}

std::expected<void, std::string> CheckArity(const char* unit, const SmfOp& op,
                                            size_t n, const char* what) {
  if (op.inputs.size() == n) return {};
  return parsing::OpError(unit, op.name,
                          std::string("needs ") + std::to_string(n) +
                              " input" + (n == 1 ? "" : "s") + what);
}

/// f32 bytes of a value of `shape`, saturating.
uint64_t BytesOf(const sir::Shape& shape) {
  uint64_t n = 1;
  for (int64_t d : shape.dims) n = SatMul(n, d > 0 ? static_cast<uint64_t>(d) : 0);
  return SatMul(n, sizeof(float));
}

}  // namespace

float RopeBaseOf(uint32_t base_bits) {
  return base_bits == 0 ? kSmfDefaultRopeBase : std::bit_cast<float>(base_bits);
}

std::expected<Topology, std::string> VerifyModel(const SmfModel& model,
                                                 int64_t batch) {
  if (batch < 1)
    return parsing::Error("batch must be at least 1, got " +
                          std::to_string(batch));
  // Whole-graph checks first: after this, every op introduces a fresh name,
  // names resolve in topological order, and the model output is known to be
  // producible — the shape walk can fail only on per-op grounds.
  if (auto graph_ok = CheckGraph(model); !graph_ok)
    return std::unexpected(graph_ok.error());

  Topology topo;
  const SmfTensor* in_tensor = model.FindTensor(model.input_name);
  if (!in_tensor || in_tensor->dims.empty())
    return parsing::Error("model input '" + model.input_name +
                          "' is not a tensor of the model with a shape");
  // A rank-1 dynamic ({-1}) input declares i32 token ids (SMF v4);
  // otherwise the reader restricts the dynamic marker to the leading dim
  // and the width is the static last dim. A non-positive width would flow
  // into the SIR as a dynamic dim, bind zero-byte slots, and seal a plan
  // the runtime rejects — a model bug blamed on the plan.
  const int64_t in_dim = in_tensor->dims.back();
  topo.token_input = in_tensor->dims.size() == 1 && in_dim == -1;
  if (!topo.token_input && in_dim <= 0)
    return parsing::Error("model input '" + model.input_name +
                          "' width must be static and positive");
  topo.input_width = topo.token_input ? 0 : in_dim;

  // The shape of every name computation may read: the input, every
  // constant tensor at its declared dims, then each op's output in order.
  topo.shapes.reserve(model.tensors.size() + model.ops.size() + 1);
  topo.shapes[model.input_name] = topo.token_input
                                      ? sir::Shape{batch}
                                      : sir::Shape{batch, in_dim};
  for (const SmfTensor& t : model.tensors)
    if (t.is_const) topo.shapes[t.name] = sir::Shape(t.dims);

  // Resolves an operand's shape the way computation's resolver will bind
  // the value: the input, an earlier op output, or a constant tensor
  // materialized as a weight. Anything else is the resolver's error,
  // raised here with its words so the stage that owns the rule owns the
  // diagnostic.
  auto resolve = [&](std::string_view name)
      -> std::expected<const sir::Shape*, std::string> {
    if (auto it = topo.shapes.find(std::string(name)); it != topo.shapes.end())
      return &it->second;
    const SmfTensor* t = model.FindTensor(name);
    if (!t)
      return parsing::Error("'" + std::string(name) +
                            "' is not a tensor of the model");
    return parsing::Error("'" + std::string(name) +
                          "' is not a constant tensor in the model");
  };
  auto emit = [&](const SmfOp& op, sir::Shape shape) {
    topo.activation_bytes = SatAdd(topo.activation_bytes, BytesOf(shape));
    ++topo.activations;
    topo.shapes[op.output] = std::move(shape);
  };

  for (const SmfOp& op : model.ops) {
    switch (op.kind) {
      case SmfOpKind::kMatMul: {
        if (auto ok = CheckArity("MatMul", op, 2, ""); !ok) return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto w = resolve(op.inputs[1]);
        if (!w) return std::unexpected(w.error());
        if (auto ok = CheckMatMul(op, **x, **w); !ok) return std::unexpected(ok.error());
        // Rows come from the actual LHS, not the batch: x may be a constant
        // with its own row count, and lowering sizes the GEMM's M from it.
        emit(op, sir::Shape{(*x)->dims.at(0), (*w)->dims.at(1)});
        break;
      }
      case SmfOpKind::kAddBias: {
        if (auto ok = CheckArity("AddBias", op, 2, ""); !ok) return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto b = resolve(op.inputs[1]);
        if (!b) return std::unexpected(b.error());
        if (auto ok = CheckAddBias(op, **x, **b); !ok) return std::unexpected(ok.error());
        emit(op, **x);
        break;
      }
      case SmfOpKind::kRelu:
      case SmfOpKind::kGelu:
      case SmfOpKind::kSilu: {
        if (op.inputs.size() != 1)
          return parsing::Error("'" + op.name + "' needs 1 input");
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        emit(op, **x);
        break;
      }
      case SmfOpKind::kMul:
      case SmfOpKind::kAdd: {
        const char* unit = op.kind == SmfOpKind::kMul ? "Mul" : "Add";
        if (auto ok = CheckArity(unit, op, 2, ""); !ok) return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto y = resolve(op.inputs[1]);
        if (!y) return std::unexpected(y.error());
        if (auto ok = CheckSameShape(unit, op, **x, **y); !ok) return std::unexpected(ok.error());
        emit(op, **x);
        break;
      }
      case SmfOpKind::kEmbedding: {
        if (auto ok = CheckArity("Embedding", op, 2, " (tokens, table)"); !ok)
          return std::unexpected(ok.error());
        auto tokens = resolve(op.inputs[0]);
        if (!tokens) return std::unexpected(tokens.error());
        auto table = resolve(op.inputs[1]);
        if (!table) return std::unexpected(table.error());
        if (auto ok = CheckEmbedding(op, topo.token_input &&
                                             op.inputs[0] == model.input_name,
                                     **tokens, **table);
            !ok)
          return std::unexpected(ok.error());
        ++topo.embedding_ops;
        topo.vocabulary = static_cast<uint64_t>((*table)->dims.at(0));
        emit(op, sir::Shape{(*tokens)->dims.at(0), (*table)->dims.at(1)});
        break;
      }
      case SmfOpKind::kRmsNorm: {
        if (auto ok = CheckArity("RmsNorm", op, 2, " (x, gamma)"); !ok)
          return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto gamma = resolve(op.inputs[1]);
        if (!gamma) return std::unexpected(gamma.error());
        if (auto ok = CheckRmsNorm(op, **x, **gamma); !ok) return std::unexpected(ok.error());
        emit(op, **x);
        break;
      }
      case SmfOpKind::kRope: {
        if (auto ok = CheckArity("Rope", op, 1, ""); !ok) return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        if (auto ok = CheckRope(op, **x, op.attr0, model.seq_len, op.attr1); !ok)
          return std::unexpected(ok.error());
        emit(op, **x);
        break;
      }
      case SmfOpKind::kAttention: {
        if (auto ok = CheckArity("Attention", op, 3, " (q, k, v)"); !ok)
          return std::unexpected(ok.error());
        auto q = resolve(op.inputs[0]);
        if (!q) return std::unexpected(q.error());
        auto k = resolve(op.inputs[1]);
        if (!k) return std::unexpected(k.error());
        auto v = resolve(op.inputs[2]);
        if (!v) return std::unexpected(v.error());
        if (auto ok = CheckAttention(op, **q, **k, **v, op.attr0, model.seq_len); !ok)
          return std::unexpected(ok.error());
        ++topo.attention_ops;
        emit(op, **q);
        break;
      }
      case SmfOpKind::kLayerNorm: {
        if (auto ok = CheckArity("LayerNorm", op, 3, " (x, gamma, beta)"); !ok)
          return std::unexpected(ok.error());
        auto x = resolve(op.inputs[0]);
        if (!x) return std::unexpected(x.error());
        auto gamma = resolve(op.inputs[1]);
        if (!gamma) return std::unexpected(gamma.error());
        auto beta = resolve(op.inputs[2]);
        if (!beta) return std::unexpected(beta.error());
        if (auto ok = CheckLayerNorm(op, **x, **gamma, **beta); !ok)
          return std::unexpected(ok.error());
        emit(op, **x);
        break;
      }
      default:
        // The reader bounds kinds per file version, so this is unreachable
        // from a loaded file; an in-memory model with a kind this stage
        // does not know must still get a diagnostic naming the op.
        return parsing::OpError("Op", op.name,
                                "unsupported op kind " +
                                    std::to_string(static_cast<int>(op.kind)));
    }
  }

  // CheckGraph proved the output is an op's output, so it has a shape.
  // Loss grafting consumes [rows, classes] logits and reads dims[1]: a
  // rank-1 output (reachable via an activation of a rank-1 constant) is a
  // diagnostic, not an out_of_range crash.
  topo.output = topo.shapes.at(model.output_name);
  if (topo.output.dims.size() != 2)
    return parsing::Error("model output '" + model.output_name +
                          "' must be rank-2 [batch, classes], got rank " +
                          std::to_string(topo.output.dims.size()));
  if (topo.output.dims.at(1) <= 0)
    return parsing::Error("model output '" + model.output_name +
                          "' width must be static and positive");
  return topo;
}

std::unordered_set<std::string_view> AdaptedWeights(
    const SmfModel& model, std::span<const std::string> target_filters) {
  std::unordered_map<std::string_view, bool> weight_only;  // name -> eligible
  for (const SmfOp& op : model.ops) {
    const bool matmul = op.kind == SmfOpKind::kMatMul && op.inputs.size() == 2;
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      const bool as_weight = matmul && i == 1 && op.inputs[0] != op.inputs[1];
      auto [it, fresh] = weight_only.try_emplace(op.inputs[i], as_weight);
      if (!fresh) it->second = it->second && as_weight;
    }
  }
  std::unordered_set<std::string_view> adapted;
  for (const SmfTensor& t : model.tensors) {
    if (!t.is_const || t.dims.size() != 2) continue;
    auto it = weight_only.find(t.name);
    if (it == weight_only.end() || !it->second) continue;
    if (!target_filters.empty()) {
      bool matched = false;
      for (const std::string& f : target_filters)
        if (t.name.find(f) != std::string::npos) matched = true;
      if (!matched) continue;
    }
    adapted.insert(t.name);
  }
  return adapted;
}

std::expected<void, std::string> VerifyCrossInputs(
    const SmfModel& student, const Topology& st, const SmfModel* teacher,
    const Topology* tt, const CrossInputs& inputs) {
  // The teacher's namespace is reserved (S1 #147, F06): a student value
  // named "t::x" would share its SIR id with the teacher's "x", and the
  // grafter — which treats every `t::` value as the frozen teacher — would
  // skip a student weight named that way while the accountant still
  // counted its delta.
  auto reserved = [](std::string_view name) {
    return name.starts_with(kTeacherPrefix);
  };
  for (const SmfTensor& t : student.tensors)
    if (reserved(t.name))
      return parsing::Error("tensor '" + t.name + "' uses the teacher's "
                            "reserved namespace '" +
                            std::string(kTeacherPrefix) + "'");
  for (const SmfOp& op : student.ops)
    if (reserved(op.output) || reserved(op.name))
      return parsing::OpError("op", op.name,
                              "uses the teacher's reserved namespace '" +
                                  std::string(kTeacherPrefix) + "'");

  // The output's rows are the batch (F68): the loss consumes one row per
  // label the feeder stages, so a MatMul over a constant LHS with its own
  // row count would read labels past the slot.
  if (st.output.dims.at(0) != inputs.batch)
    return parsing::Error(
        "model output '" + student.output_name + "' has " +
        std::to_string(st.output.dims.at(0)) + " rows, but the compiled "
        "batch is " + std::to_string(inputs.batch) +
        ": the loss consumes one row per label");

  // A token model's labels are the next tokens — classes, never dense
  // targets (N19): an MSE plan over one compiled to a corpus contract no
  // dataset could satisfy.
  if (st.token_input && inputs.loss == LossKind::kMse)
    return parsing::Error(
        "--loss mse needs dense f32 labels, and a token model's labels are "
        "next-token classes; use xent (or kl / xent+kl with a teacher)");

  // Every adapter filter names at least one eligible weight: a filter that
  // matches nothing compiles a plan with fewer adapters than asked for.
  for (const std::string& f : inputs.target_filters) {
    const std::string one[] = {f};
    if (AdaptedWeights(student, one).empty())
      return parsing::Error("--targets '" + f +
                            "' matches no weight LoRA can adapt (a constant "
                            "2-D tensor read only as a MatMul's weight)");
  }

  if (!teacher) return {};
  // The teacher shares the student's input: the same kind and, for
  // features, the same width (its forward reads the same block argument).
  if (tt->token_input != st.token_input ||
      (!st.token_input && tt->input_width != st.input_width))
    return parsing::Error("teacher input dimensionality mismatch");
  // A sequential teacher must agree on seq_len (F02): the causal mask and
  // the rotary positions follow each model's own declaration, so a
  // disagreement is a distillation target computed over other positions —
  // with no diagnostic, until now.
  if (teacher->seq_len != 0 && student.seq_len != 0 &&
      teacher->seq_len != student.seq_len)
    return parsing::Error("teacher seq_len " + std::to_string(teacher->seq_len) +
                          " disagrees with the student's " +
                          std::to_string(student.seq_len));
  // The teacher's logits are compared row for row with the student's.
  if (tt->output.dims.at(1) != st.output.dims.at(1))
    return parsing::Error("teacher output dimensionality mismatch");
  if (tt->output.dims.at(0) != st.output.dims.at(0))
    return parsing::Error("teacher output has " +
                          std::to_string(tt->output.dims.at(0)) +
                          " rows against the student's " +
                          std::to_string(st.output.dims.at(0)));
  return {};
}

}  // namespace seeml::update::sema
