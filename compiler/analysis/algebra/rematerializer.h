#ifndef SEEML_COMPILER_ANALYSIS_ALGEBRA_REMATERIALIZER_H_
#define SEEML_COMPILER_ANALYSIS_ALGEBRA_REMATERIALIZER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <unordered_set>
#include <vector>

#include "compiler/frontend/representation/sir.h"

// =============================================================================
// Rematerializer — roadmap Phase 2b (F5, #133): recompute instead of store.
//
// The arena binder frees a value after its last reader (LinearScanTransients:
// birth to last use in program order), so every forward activation the
// backward reads stays resident across the whole forward — on a decoder at
// 512 tokens about 98 % of the arena is that transient segment. This pass
// drops the activations whose recompute is cheap from that set: for a
// forward op F in the primal snapshot whose results the backward reads, it
// re-emits F — same mnemonic, same attributes, same operands, in the same
// place relative to its own operands — right before the FIRST backward
// instruction that needs a result, and points every backward reader at the
// copy. The original's last reader is then in the forward, the binder frees
// it there, and the recomputed floats are bit-identical to the ones the
// non-remat plan kept, because the kernel runs the same expression over the
// same inputs. The eval program is the untouched primal snapshot.
//
// What qualifies is a cost class and a liveness rule, not a list of
// tensors. The class: ops that cost one pass over their output — the norms
// (`sc_high.rms_norm` / `layer_norm`), the activations and elementwise
// arithmetic (`silu` / `relu` / `gelu` / `mul` / `add` / `scale` /
// `add_bias`), a folded elementwise chain (kFusedMap tail), `rope`, the
// embedding gather — and never a GEMM, attention or a loss, nor an add that
// the addend fuser folded into a GEMM. The rule: recomputing F at point P
// keeps F's operands alive until P, so F is recomputed only when the bytes
// it frees exceed the bytes it would newly keep — an operand already read
// by the backward at or after P costs nothing. On the decoder that admits
// the RMSNorm outputs (x is read by the norm's own VJP), silu(gate) (gate
// by silu's VJP) and the SwiGLU product (gate and up by the mul's VJP),
// and declines the RoPE outputs: the pre-rotation Q and K have no backward
// reader, so recomputing the rotation only swaps which tensor stays live.
// An operand that is not live at P but whose own producer is a cheap primal
// op costs what that producer would keep, to a small depth — the first
// norm's input is the embedding gather over storage, free to recompute, so
// the norm is a clean win rather than a swap.
//
// Candidates are visited in reverse program order, so a consumer's copy
// (the SwiGLU product's) is in place before its producer (the silu) is
// considered, and the producer's copy lands before it: a chain recomputes
// as a chain, and nothing it reads is kept for it.
// =============================================================================

namespace seeml::update {

struct Rematerialization {
  size_t recomputed_ops = 0;     // forward ops re-emitted in the backward
  size_t recomputed_values = 0;  // their results the backward now reads anew
  // Candidates whose recompute would keep at least as many bytes alive as
  // it frees (the RoPE outputs on a decoder).
  size_t declined = 0;
  // The pass's own estimate of what it freed: the dropped values' bytes
  // less the operands newly kept. The binder's rescan is the real number.
  uint64_t bytes_estimated = 0;
};

class Rematerializer {
 public:
  /// `primal_ops` is the forward snapshot (the eval program): only its ops
  /// are candidates, and a reader outside it is a backward reader. `pinned`
  /// values (the loss, parameter gradients) are read outside the program
  /// and are never recomputed nor counted as freeable.
  [[nodiscard]] std::expected<Rematerialization, std::string> Run(
      seeml::sir::Block& block,
      const std::vector<seeml::sir::Operation*>& primal_ops,
      const std::unordered_set<const seeml::sir::Value*>& pinned);
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_ANALYSIS_ALGEBRA_REMATERIALIZER_H_
