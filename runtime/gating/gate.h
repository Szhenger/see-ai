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
// evaluation means. The state round-trips through a checkpoint (SEKP,
// runtime/storage/checkpoint.h) field for field — and a resume re-measures
// the two losses in it rather than trusting the file (CR4 #221).
//
// Without a validation set the gate weighs the training loss instead: the
// mean over the run's first window of steps against the mean over its last.
// LossWindows is that state. It is the RUN's, not one Train call's: it too
// rides in every checkpoint, so a resumed run gates on the whole run.
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

/// The train-loss gate's two windows (CR4 #221). `window` steps each — a
/// fifth of the run, at most kMax, at least one — decided when the run
/// starts and kept across a resume. The first window is a running f64 sum
/// in step order; the last is a ring of the most recent `window` losses,
/// summed oldest first: the sums an uninterrupted run forms, bit for bit.
struct LossWindows {
  static constexpr uint32_t kMax = 20;
  uint32_t window = 0;  // 0 = no run has started
  uint64_t count = 0;   // losses observed
  double first_sum = 0.0;
  float last[kMax] = {};

  bool live() const { return window != 0; }
  /// A run of `run_steps` steps begins: nothing observed.
  void Start(uint64_t run_steps);
  /// The next step's training loss.
  void Observe(float loss);
  /// Mean over the first window so far (0 when nothing was observed).
  float FirstMean() const;
  /// Mean over the last min(count, window) losses (FirstMean() when none).
  float LastMean() const;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_GATING_GATE_H_
