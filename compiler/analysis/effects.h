#ifndef SEEML_COMPILER_ANALYSIS_EFFECTS_H_
#define SEEML_COMPILER_ANALYSIS_EFFECTS_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// The in-place effects of the analysis IR, in one place.
//
// SIR is single-assignment for values an op PRODUCES; the ops below also
// rewrite a value they are HANDED — a gradient clipped where it lies, a
// parameter stepped, an accumulator folded into or zeroed, the stats row the
// tiled dQ pass leaves its deltas in. Nothing in the IR's structure says so:
// the op has no result for the bytes it changes. Three passes must know
// anyway (dead-code elimination, which may not drop an op for having no
// reader; the addend fuser, which may not move a read across a write; the
// chain fuser, which moves every read of a chain to its tail), and until
// review 2026-11-25 two of them kept their own copy of the list ("Mirrors
// DeadCodeElimination's effectful set") while the third trusted a comment
// ("gradients are single-assignment" — a clip rewrites one in place).
//
// This table is the list, with the one fact the copies lacked: WHICH
// operands each op rewrites. A new in-place op is one row here.
// =============================================================================

namespace seeml::update {

/// One op that rewrites storage in place, and the operands it rewrites (bit
/// i set = operand i). The others it only reads.
struct InPlaceEffect {
  std::string_view mnemonic;
  uint32_t rewrites;
};

inline constexpr InPlaceEffect kInPlaceEffects[] = {
    {"sc_low.clip_norm", 1u << 0},     // (g): g scaled where it lies
    {"sc_low.sqnorm_accum", 1u << 1},  // (g, sqnorm): sqnorm += |g|^2
    {"sc_low.clip_global", 1u << 0},   // (g, sqnorm): g scaled; sqnorm read
    {"sc_low.sgd_step", 1u << 0},      // (p, g): p stepped
    // (p, g, m, v[, scales]): everything but the gradient.
    {"sc_low.adamw_step", (1u << 0) | (1u << 2) | (1u << 3) | (1u << 4)},
    {"sc_low.gemm_acc", 1u << 2},       // (a, b, c): c += alpha * a @ b
    {"sc_low.accumulate", 1u << 0},     // (acc, g): acc += g
    {"sc_low.zero", 1u << 0},           // (x): x = 0
    {"sc_low.attn_dq_tiled", 1u << 4},  // (q, k, v, dO, stats): the deltas
};

/// The row for `op`, or nullptr when it rewrites nothing in place.
[[nodiscard]] inline const InPlaceEffect* InPlaceEffectOf(
    const sir::Operation& op) {
  const std::string_view m = op.mnemonic();
  for (const InPlaceEffect& e : kInPlaceEffects)
    if (e.mnemonic == m) return &e;
  return nullptr;
}

/// Whether executing `op` changes bytes of a value it was handed.
[[nodiscard]] inline bool MutatesInPlace(const sir::Operation& op) {
  return InPlaceEffectOf(op) != nullptr;
}

/// Whether `op` rewrites its operand `index` (as opposed to reading it).
[[nodiscard]] inline bool RewritesOperand(const sir::Operation& op,
                                          size_t index) {
  const InPlaceEffect* e = InPlaceEffectOf(op);
  return e != nullptr && index < 32 && ((e->rewrites >> index) & 1u) != 0;
}

/// Whether some op of the program rewrites `v` in place: a read of `v`
/// moved across that op would see other bytes.
[[nodiscard]] inline bool RewrittenInPlace(const sir::Value* v) {
  for (const sir::Operation* user : v->users())
    for (size_t i = 0; i < user->numOperands(); ++i)
      if (user->operand(i) == v && RewritesOperand(*user, i)) return true;
  return false;
}

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_EFFECTS_H_
