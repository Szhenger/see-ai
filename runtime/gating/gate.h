#ifndef SEEML_RUNTIME_GATING_GATE_H_
#define SEEML_RUNTIME_GATING_GATE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// =============================================================================
// Gating (S7 #153, seeai.md §6): what the gate weighs and keeps. The source
// model's held-out score (measured once, by the first run, and carried
// through every checkpoint), the best evaluated state (E9 #92: a copy of the
// persistent segment, its step and its scores), and patience (evaluations
// since the best improved). The engine evaluates; this decides what an
// evaluation means. The state round-trips through a checkpoint (SEKP v5,
// runtime/storage/checkpoint.h) field for field.
// =============================================================================

namespace seeml::update_rt {

struct GateState {
  // The source model's score: the gate's "before".
  bool has_val_initial = false;
  bool val_initial_has_accuracy = false;
  float val_initial_loss = 0.0f;
  float val_initial_accuracy = 0.0f;
  // The best evaluated state.
  bool has_best = false;
  uint64_t best_step = 0;
  float best_loss = 0.0f;
  float best_accuracy = 0.0f;
  uint32_t stale_evals = 0;  // evaluations since the best last improved
  std::vector<uint8_t> best_state;

  /// A new plan: nothing measured, nothing kept.
  void Reset();
  /// `persistent` (its first `bytes`) becomes the best state, scored at
  /// `step`; patience restarts.
  void Snapshot(const uint8_t* persistent, size_t bytes, uint64_t step,
                float loss, float accuracy);
  /// One periodic evaluation: a strictly lower loss is the new best; anything
  /// else is one more stale evaluation. True when that exhausts `patience`
  /// (0 = never).
  [[nodiscard]] bool Observe(const uint8_t* persistent, size_t bytes,
                             uint64_t step, float loss, float accuracy,
                             uint32_t patience);
  /// The best state back into `persistent` when it is not the one there
  /// (scored at `step`). True when it restored.
  bool RestoreBest(uint8_t* persistent, uint64_t step) const;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_GATING_GATE_H_
