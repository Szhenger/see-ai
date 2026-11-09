#include "compiler/frontend/computation/parser.h"

#include <bit>
#include <vector>

#include "compiler/diagnostics/parsing/error.h"
#include "compiler/frontend/computation/value_resolver.h"
#include "compiler/frontend/topology/sema.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace parsing = seeml::diag::parsing;

std::expected<sir::Value*, std::string> BuildForward(
    sir::Block& block, const SmfModel& model, const std::string& prefix,
    sir::Value* input, int64_t batch, GraphBuild& build) {
  // Topology first, to completion (S1 #147): every op's arity, operand
  // existence and constness, and shape rule are proven over the tensor
  // table before any SIR exists, and the shape of every value is decided
  // there. What follows is a mechanical translation that reads those
  // shapes and recomputes none — one shape rule per op kind, in one place.
  auto topology = sema::VerifyModel(model, batch);
  if (!topology) return std::unexpected(topology.error());
  const sema::Topology& topo = *topology;
  auto shape_of = [&](const SmfOp& op) -> const sir::Shape& {
    return topo.shapes.at(op.output);
  };

  ValueResolver resolver(block, model, prefix, build);
  resolver.Bind(model.input_name, input);

  for (const SmfOp& op : model.ops) {
    // Operands resolve in order; topology proved each name is the input,
    // an earlier output or a constant tensor, so Resolve cannot fail here
    // — the check stays as the computation's own invariant.
    std::vector<sir::Value*> in;
    in.reserve(op.inputs.size());
    for (const std::string& name : op.inputs) {
      auto v = resolver.Resolve(name);
      if (!v) return std::unexpected(v.error());
      in.push_back(*v);
    }
    const sir::Shape& out_shape = shape_of(op);
    auto emit = [&](const char* mnemonic) {
      sir::Operation* o = block.appendOp(mnemonic);
      for (sir::Value* v : in) o->addOperand(v);
      return o;
    };
    auto bind_result = [&](sir::Operation* o) {
      resolver.Bind(op.output,
                    o->addResult(prefix + op.output, sir::DataType::F32,
                                 out_shape));
    };
    switch (op.kind) {
      case SmfOpKind::kMatMul:
        bind_result(emit("sc_high.matmul"));
        break;
      case SmfOpKind::kAddBias:
        bind_result(emit("sc_high.add_bias"));
        break;
      case SmfOpKind::kRelu:
        bind_result(emit("sc_high.relu"));
        break;
      case SmfOpKind::kGelu:
        bind_result(emit("sc_high.gelu"));
        break;
      case SmfOpKind::kSilu:
        bind_result(emit("sc_high.silu"));
        break;
      case SmfOpKind::kMul:
        bind_result(emit("sc_high.mul"));
        break;
      case SmfOpKind::kEmbedding:
        bind_result(emit("sc_high.embedding"));
        break;
      case SmfOpKind::kAdd:
        bind_result(emit("sc_high.add"));
        break;
      case SmfOpKind::kRmsNorm: {
        sir::Operation* rn = emit("sc_high.rms_norm");
        // SMF v6 attr2: the model's epsilon. Default-valued ones set no
        // attribute, so a v5 model's SIR (and plan) is what it always was.
        if (op.attr2 != 0) rn->setAttribute("eps", std::bit_cast<float>(op.attr2));
        bind_result(rn);
        // Row statistic cached for the backward kernel.
        rn->addResult(prefix + op.output + ".rstd", sir::DataType::F32,
                      sir::Shape{out_shape.dims.at(0)});
        break;
      }
      case SmfOpKind::kRope: {
        sir::Operation* r = emit("sc_high.rope");
        r->setAttribute("heads", static_cast<int64_t>(op.attr0));
        r->setAttribute("seq", static_cast<int64_t>(model.seq_len));
        // v5 attr1: the rotary base, lowered into kRopeFwd's out[2] and
        // copied onto the backward by autodiff. Pre-v5 files mean 10000.
        r->setAttribute("base", sema::RopeBaseOf(op.attr1));
        bind_result(r);
        break;
      }
      case SmfOpKind::kAttention: {
        const int64_t rows = out_shape.dims.at(0);
        const auto heads = static_cast<int64_t>(op.attr0);
        const auto seq = static_cast<int64_t>(model.seq_len);
        sir::Operation* at = emit("sc_high.attention");
        at->setAttribute("heads", heads);
        at->setAttribute("seq", seq);
        bind_result(at);
        // Probability matrix P[B,H,S,S] flattened [B*H*S, S] = [rows*H, S],
        // cached for the backward primitives.
        at->addResult(prefix + op.output + ".probs", sir::DataType::F32,
                      sir::Shape{rows * heads, seq});
        break;
      }
      case SmfOpKind::kLayerNorm: {
        const int64_t rows = out_shape.dims.at(0);
        sir::Operation* ln = emit("sc_high.layer_norm");
        if (op.attr2 != 0) ln->setAttribute("eps", std::bit_cast<float>(op.attr2));
        bind_result(ln);
        // Row statistics cached for the backward kernel.
        ln->addResult(prefix + op.output + ".mean", sir::DataType::F32,
                      sir::Shape{rows});
        ln->addResult(prefix + op.output + ".rstd", sir::DataType::F32,
                      sir::Shape{rows});
        break;
      }
      default:
        // Topology refused every kind it does not know; unreachable.
        return parsing::OpError("Op", op.name,
                                "unsupported op kind " +
                                    std::to_string(static_cast<int>(op.kind)));
    }
  }

  // Topology proved the output is an op's output and rank-2; every op
  // binds its output on success — this lookup cannot fail, and is kept as
  // a defensive invariant, not a reachable error path.
  sir::Value* out = resolver.Lookup(model.output_name);
  if (!out)
    return parsing::Error("model output '" + model.output_name +
                          "' was never produced");
  return out;
}

}  // namespace seeml::update
