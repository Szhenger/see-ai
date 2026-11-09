#ifndef SEEML_COMPILER_FRONTEND_TOPOLOGY_SEMA_H_
#define SEEML_COMPILER_FRONTEND_TOPOLOGY_SEMA_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "compiler/frontend/representation/sir.h"
#include "source/language/model_format.h"
#include "source/plan/config.h"

// =============================================================================
// Topology — the frontend's verification stage (S1 #147, seeai.md §3). It
// runs to completion over the decoded model BEFORE computation builds any
// SIR, and answers "is this model well-formed, and what shape is every
// value?" from the tensor table alone:
//
//   graph   CheckGraph: every op output introduces a fresh name, every name
//           an op consumes is produced earlier (the topological-order
//           contract the single forward pass relies on), the declared output
//           is an op's output, and the token-input discipline holds.
//   shapes  VerifyModel: one shape rule per op kind, applied in op order to
//           the tensor table and the batch — rank, inner dimension, affine
//           width, sequence geometry — producing the shape of every value.
//           Computation (parser.cc) reads these shapes; it recomputes none.
//   cross   VerifyCrossInputs: what no single model can know — the student
//           against the teacher (input kind and width, seq_len, output
//           shape), the loss against the input kind, the adapter filters
//           against the weights they must match, and the teacher's `t::`
//           namespace kept out of the student.
//
// Operand existence and constness are decided here too (a name that is
// neither a tensor nor an earlier output, a weight that is not constant):
// after VerifyModel, computation can fail only on its own invariants.
// =============================================================================

namespace seeml::update::sema {

/// Whole-graph validation (see banner). Cheap — two O(tensors + ops) passes
/// over string_views into the model.
[[nodiscard]] std::expected<void, std::string> CheckGraph(
    const SmfModel& model);

/// The verified model: the shape of every value computation will emit,
/// keyed by its SMF name (the input, every op output), and what the walk
/// learned on the way.
struct Topology {
  std::unordered_map<std::string, seeml::sir::Shape> shapes;
  seeml::sir::Shape output;        // the declared output's shape
  bool token_input = false;        // rank-1 dynamic input: i32 token ids
  int64_t input_width = 0;         // features per row (0 for a token input)
  uint64_t vocabulary = 0;         // rows of the embedding table (0 = none)
  uint64_t embedding_ops = 0;
  uint64_t attention_ops = 0;
  // Activations computation will emit, with their f32 bytes at this batch
  // — the values the SIR graph holds, before any pass.
  uint64_t activations = 0;
  uint64_t activation_bytes = 0;
};

/// Verifies `model` at `batch` and infers every value's shape (see
/// banner). The input is shaped {batch} for a token model and
/// {batch, width} for a feature model; constant tensors take their declared
/// dims. Fails with the Parser unit on the first violation, naming the op.
[[nodiscard]] std::expected<Topology, std::string> VerifyModel(
    const SmfModel& model, int64_t batch);

/// The LoRA grafter's eligibility rule, read off the SMF op list: a
/// constant 2-D tensor consumed only as the weight (second) operand of
/// two-input MatMuls whose operands differ, named by one of
/// `target_filters` (empty = all). The accountant's delta and activation
/// bounds and the cross-input check share this one reading of the rule.
std::unordered_set<std::string_view> AdaptedWeights(
    const SmfModel& model, std::span<const std::string> target_filters);

/// What the cross-input check compares the models against.
struct CrossInputs {
  int64_t batch = 0;
  LossKind loss = LossKind::kSoftmaxXEnt;
  std::span<const std::string> target_filters = {};
};

/// The teacher's value-id namespace. A student name with this prefix would
/// collide with the teacher's values in the SIR and hide from the grafter,
/// which skips `t::` as frozen — so it is reserved.
inline constexpr std::string_view kTeacherPrefix = "t::";

/// Cross-input checks (see banner): the student's output rows must be the
/// batch; a token model takes no MSE loss; every adapter filter must match
/// an eligible weight; no student name may start with `t::`; a teacher
/// must share the input kind and width, the seq_len of a sequential
/// student, and the output shape. `teacher` and `teacher_topology` are
/// null without distillation.
[[nodiscard]] std::expected<void, std::string> VerifyCrossInputs(
    const SmfModel& student, const Topology& student_topology,
    const SmfModel* teacher, const Topology* teacher_topology,
    const CrossInputs& inputs);

/// The rotary base a kRope op's attr1 encodes (kSmfDefaultRopeBase for 0).
float RopeBaseOf(uint32_t base_bits);

}  // namespace seeml::update::sema

#endif  // SEEML_COMPILER_FRONTEND_TOPOLOGY_SEMA_H_
