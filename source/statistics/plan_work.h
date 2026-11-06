#ifndef SEEML_SOURCE_STATISTICS_PLAN_WORK_H_
#define SEEML_SOURCE_STATISTICS_PLAN_WORK_H_

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "source/contract/bounds.h"
#include "source/plan/update_types.h"

// =============================================================================
// The arithmetic work of a plan, read from its own instructions — the one
// FLOP definition the compiler's backend screen, the device's runtime screen
// and seeml-bench share (S0 #146). GEMM work is seeml-bench's historical
// definition, unchanged, because its stored baselines and the nightly gate's
// utilization figures are in that unit; attention work is reported beside
// it, never folded into it.
// =============================================================================

namespace seeml::update {

namespace plan_work_detail {
inline uint64_t SatMul(uint64_t a, uint64_t b) {
  uint64_t out = 0;
  return MulOk(a, b, &out) ? out : UINT64_MAX;
}
inline uint64_t SatAdd(uint64_t a, uint64_t b) {
  return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}
/// B·H·S·S·d — one dense [S, d] × [d, S] (or [S, S] × [S, d]) product per
/// (batch, head), in multiply-adds.
inline uint64_t AttnMacs(uint64_t B, uint64_t S, uint64_t H, uint64_t d) {
  return SatMul(SatMul(SatMul(SatMul(B, H), S), S), d);
}
}  // namespace plan_work_detail

/// 2·M·N·K over the eight GEMM opcodes of `program` (out[0..2] = M, N, K)
/// — seeml-bench's gemm_flops_per_step, the unit its baselines carry.
inline uint64_t GemmFlops(std::span<const UpdateInstruction> program) {
  using plan_work_detail::SatAdd;
  using plan_work_detail::SatMul;
  uint64_t flops = 0;
  for (const UpdateInstruction& ins : program) {
    switch (static_cast<OpCode>(ins.opcode)) {
      case OpCode::kGemmNN:
      case OpCode::kGemmNT:
      case OpCode::kGemmTN:
      case OpCode::kGemmAccNN:
      case OpCode::kGemmNNQ8:
      case OpCode::kGemmNTQ8:
      case OpCode::kGemmNNBF16:
      case OpCode::kGemmNTBF16:
        flops = SatAdd(flops,
                       SatMul(SatMul(SatMul(2, ins.out[0]), ins.out[1]),
                              ins.out[2]));
        break;
      default:
        break;
    }
  }
  return flops;
}

/// The model FLOPs of `program`'s attention instructions: the dense
/// products, 2 FLOPs per multiply-add — QKᵀ and PV in the forward
/// (4·B·H·S²·d), dP, dV, dQ and dK in the backward (8·B·H·S²·d) — the same
/// in both families. Causal masking is not discounted, and the tiled
/// family's recomputation of P is not counted: this is the work the model
/// needs, not the work a kernel happens to do.
inline uint64_t AttentionFlops(std::span<const UpdateInstruction> program) {
  using plan_work_detail::AttnMacs;
  using plan_work_detail::SatAdd;
  using plan_work_detail::SatMul;
  auto hi = [](uint64_t w) { return w >> 32; };
  auto lo = [](uint64_t w) { return w & 0xFFFFFFFFu; };
  uint64_t flops = 0;
  for (const UpdateInstruction& ins : program) {
    uint64_t products = 0, macs = 0;
    switch (static_cast<OpCode>(ins.opcode)) {
      case OpCode::kAttnFwd:  // out[1] = B|S, out[2] = H|d
      case OpCode::kAttnFwdTiled:
        products = 2;
        macs = AttnMacs(hi(ins.out[1]), lo(ins.out[1]), hi(ins.out[2]),
                        lo(ins.out[2]));
        break;
      case OpCode::kAttnDP:  // out[0] = B|S, out[1] = H|d
      case OpCode::kAttnDV:
      case OpCode::kAttnDQ:
      case OpCode::kAttnDK:
        products = 1;
        macs = AttnMacs(hi(ins.out[0]), lo(ins.out[0]), hi(ins.out[1]),
                        lo(ins.out[1]));
        break;
      case OpCode::kAttnDQTiled:  // dP and dQ: out[2] = B|S|H|d, 16 bits each
      case OpCode::kAttnDKTiled:  // dK
      case OpCode::kAttnDVTiled: {  // dV
        const AttnGeometry g = UnpackAttnGeometry(ins.out[2]);
        products =
            static_cast<OpCode>(ins.opcode) == OpCode::kAttnDQTiled ? 2 : 1;
        macs = AttnMacs(g.B, g.S, g.H, g.d);
        break;
      }
      default:
        break;
    }
    if (products) flops = SatAdd(flops, SatMul(SatMul(2, products), macs));
  }
  return flops;
}

/// The arithmetic of one optimizer step of a plan.
struct PlanWork {
  uint64_t micro_batches = 1;          // G: train-program passes per step
  uint64_t train_gemm_flops = 0;       // one pass of the train program
  uint64_t train_attention_flops = 0;  // one pass of the train program
  uint64_t step_gemm_flops = 0;        // the step program (normally none)

  /// GEMM FLOPs per optimizer step: G train passes plus the step program.
  uint64_t GemmFlopsPerStep() const {
    return plan_work_detail::SatAdd(
        plan_work_detail::SatMul(train_gemm_flops, micro_batches),
        step_gemm_flops);
  }
  uint64_t AttentionFlopsPerStep() const {
    return plan_work_detail::SatMul(train_attention_flops, micro_batches);
  }
  uint64_t FlopsPerStep() const {
    return plan_work_detail::SatAdd(GemmFlopsPerStep(),
                                    AttentionFlopsPerStep());
  }
};

/// The work of `plan` (a sealed .seeu blob) from its own train and step
/// instructions; all zero when the header's sections do not lie inside the
/// blob (callers measure plans the loader or the compiler already proved).
inline PlanWork MeasurePlanWork(std::span<const uint8_t> plan) {
  PlanWork work;
  if (plan.size() < sizeof(PlanHeader)) return work;
  PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  auto decode = [&](uint64_t offset, uint64_t count,
                    std::vector<UpdateInstruction>* out) {
    uint64_t bytes = 0;
    if (!MulOk(count, sizeof(UpdateInstruction), &bytes) ||
        !RangeOk(offset, bytes, plan.size()))
      return false;
    out->resize(static_cast<size_t>(count));
    if (bytes) std::memcpy(out->data(), plan.data() + offset, bytes);
    return true;
  };
  std::vector<UpdateInstruction> train, step;
  if (!decode(h.train_instr_offset, h.train_instr_count, &train) ||
      !decode(h.step_instr_offset, h.step_instr_count, &step))
    return work;
  work.micro_batches = h.grad_accum_steps > 1 ? h.grad_accum_steps : 1;
  work.train_gemm_flops = GemmFlops(train);
  work.train_attention_flops = AttentionFlops(train);
  work.step_gemm_flops = GemmFlops(step);
  return work;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_STATISTICS_PLAN_WORK_H_
