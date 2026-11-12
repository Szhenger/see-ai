#include "compiler/analysis/statistics/precision.h"

namespace seeml::update {

RelaxedCounts AssignRelaxedPrecision(std::vector<UpdateInstruction>& train) {
  RelaxedCounts counts;
  for (UpdateInstruction& ins : train) {
    const auto op = static_cast<OpCode>(ins.opcode);
    const bool frozen_gemm =
        (op == OpCode::kGemmNN || op == OpCode::kGemmNT ||
         op == OpCode::kGemmNNQ8 || op == OpCode::kGemmNTQ8 ||
         op == OpCode::kGemmNNBF16 || op == OpCode::kGemmNTBF16) &&
        IsRodataRef(ins.in[1]);
    const bool tiled_attention =
        op == OpCode::kAttnFwdTiled || op == OpCode::kAttnDQTiled ||
        op == OpCode::kAttnDKTiled || op == OpCode::kAttnDVTiled;
    if (!frozen_gemm && !tiled_attention) continue;
    ins.flags |= kFlagRelaxed;
    ++(frozen_gemm ? counts.gemms : counts.attention);
  }
  return counts;
}

}  // namespace seeml::update
