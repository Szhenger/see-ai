#ifndef SEEML_RUNTIME_LOADER_CONTRACT_H_
#define SEEML_RUNTIME_LOADER_CONTRACT_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "runtime/pipeline/dataset.h"
#include "source/plan/update_types.h"

// =============================================================================
// Subsystem usage contracts — the engine's verification layer, mirroring
// compiler/driver/contract.h. Each runtime subsystem guards its own
// invariants (the verifier's per-instruction sweep, storage's
// checkpoint binding, the feeder's format checks); these contracts check
// the *seams*: that the engine hands each subsystem what it requires and
// that what the compiler's driver promised at emit time still holds at
// load time. One verifier per boundary the engine crosses:
//
//   VerifyPlanContract      at load      — the header's section table, arena
//                           relations, and I/O slots are self-consistent
//                           with the plan blob, the rodata section's
//                           offset is aligned (every rodata ref is relative
//                           to it; CR2 #143 N05), and the optimizer
//                           hyperparameters are finite and in range (CR1
//                           #142)
//   VerifyExecutorContract  after decode — every instruction of every
//                           program is bounds-proven (verifier/), no
//                           program writes the staged input or label slot
//                           (the feeder's; CR2 #143 N03), every quantized
//                           GEMM's column scales are finite positive floats
//                           (read from rodata; CR1 #142 F33), and every
//                           emit entry targets the arena, so Execute() may
//                           dispatch blindly
//   VerifyRelaxedPermit     after decode — a plan that carries relaxed
//                           instructions is the one plan this binary was
//                           built to run under a granted numerics
//                           certificate, or the caller is a measuring
//                           instrument that said so (F2 #206)
//   VerifyFeederContract    before train — the dataset's geometry matches
//                           the compiled plan's batch, input, and label
//                           slots
//   WellFormedDiagnostic    on any error — the message is attributable to a
//                           unit registered in runtime/diagnostics/
//
// Contract violations mean a corrupt or foreign artifact, or a misused
// subsystem — the engine reports them under its own unit.
// =============================================================================

namespace seeml::update_rt {

/// True iff `message` reads "<unit>: <nonempty message>" for a unit
/// registered in one of the four runtime diagnostics process modules.
[[nodiscard]] bool WellFormedDiagnostic(std::string_view message);

/// The plan header is self-consistent with a blob of `plan_size` bytes:
/// no section size overflows, every section lies inside the blob, the
/// arena dominates its persistent segment and image, and the input, label,
/// and loss I/O slots lie inside the mutable arena.
[[nodiscard]] std::expected<void, std::string> VerifyPlanContract(
    const seeml::update::PlanHeader& header, uint64_t plan_size);

/// The decoded programs are fit for blind dispatch: every operand ref of
/// every instruction passes the validator against its address space, and
/// every emit entry's delta range lies inside the arena.
[[nodiscard]] std::expected<void, std::string> VerifyExecutorContract(
    std::span<const seeml::update::UpdateInstruction> train,
    std::span<const seeml::update::UpdateInstruction> merge,
    std::span<const seeml::update::UpdateInstruction> eval,
    std::span<const seeml::update::UpdateInstruction> step,
    std::span<const seeml::update::EmitEntry> emit_table,
    std::span<const uint8_t> rodata, const seeml::update::PlanHeader& header);

/// Which relaxed plans a load admits (F2 #206). Relaxed arithmetic
/// (kFlagRelaxed: the bf16 tensor-op GEMMs, relaxed tiled attention) ships
/// only under a granted numerics certificate, and a certificate cannot sit
/// inside the plan it hashes — so the binding lives in the binary: the build
/// path that checked the certificate compiles the certified plan's seal
/// (PlanHeader::plan_hash) into the runner, and the loader holds every plan
/// that reaches it — embedded, mapped from disk, or named by --plan — to it.
/// The default admits no relaxed plan; an exact plan loads under any permit.
class RelaxedPlanPermit {
 public:
  /// No relaxed plan: any kFlagRelaxed instruction refuses the load.
  constexpr RelaxedPlanPermit() = default;
  /// The one relaxed plan a granted certificate was checked for when the
  /// calling binary was built, by its seal. Zero is "none was": the value
  /// of a runner built without SEEML_CERTIFIED_PLAN_HASH.
  [[nodiscard]] static constexpr RelaxedPlanPermit CertifiedPlan(
      uint64_t plan_hash) {
    RelaxedPlanPermit permit;
    permit.certified_plan_hash_ = plan_hash;
    return permit;
  }
  /// NOT a certificate — a measurement and testing permission: any relaxed
  /// plan loads. For the instruments that must run a relaxed plan before a
  /// certificate can exist (the certifier's probe, the bench, the test
  /// suites that prove the relaxed kernels). The generated runner never
  /// passes it and has no flag that could.
  [[nodiscard]] static constexpr RelaxedPlanPermit UncertifiedForMeasurement() {
    RelaxedPlanPermit permit;
    permit.uncertified_for_measurement_ = true;
    return permit;
  }
  constexpr uint64_t certified_plan_hash() const {
    return certified_plan_hash_;
  }
  constexpr bool uncertified_for_measurement() const {
    return uncertified_for_measurement_;
  }

 private:
  uint64_t certified_plan_hash_ = 0;
  bool uncertified_for_measurement_ = false;
};

/// The relaxed rule (F2 #206): a plan with at least one kFlagRelaxed
/// instruction in any program loads only when `permit` names its seal —
/// `header.plan_hash`, which the caller has already verified against the
/// plan's bytes — or is the measurement permission. An exact plan passes
/// under every permit.
[[nodiscard]] std::expected<void, std::string> VerifyRelaxedPermit(
    const seeml::update::PlanHeader& header,
    std::span<const seeml::update::UpdateInstruction> train,
    std::span<const seeml::update::UpdateInstruction> merge,
    std::span<const seeml::update::UpdateInstruction> eval,
    std::span<const seeml::update::UpdateInstruction> step,
    RelaxedPlanPermit permit);

/// The dataset can feed this plan: batch x input_dim matches the input
/// slot, the label kind and per-batch width match the label slot, and
/// class labels (when used) all index inside the softmax width. For token
/// plans (header.input_kind == 1) the corpus must be a token corpus whose
/// record length equals the plan's seq_len, and every token id must index
/// inside BOTH the narrowest embedding table (`vocab_bound`) and — via the
/// derived next-token labels — the narrowest softmax width.
[[nodiscard]] std::expected<void, std::string> VerifyFeederContract(
    const seeml::update::PlanHeader& header, Dataset& data,
    uint64_t num_classes, uint64_t vocab_bound);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_LOADER_CONTRACT_H_
