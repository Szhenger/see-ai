#include "runtime/gating/gate.h"

#include <cstring>

namespace seeml::update_rt {

void GateState::Reset() {
  has_val_initial = val_initial_has_accuracy = false;
  val_initial_loss = val_initial_accuracy = 0.0f;
  has_best = false;
  best_step = 0;
  best_loss = best_accuracy = 0.0f;
  stale_evals = 0;
  best_state.clear();
}

void GateState::Snapshot(const uint8_t* persistent, size_t bytes,
                         uint64_t step, float loss, float accuracy) {
  best_state.assign(persistent, persistent + bytes);
  has_best = true;
  best_step = step;
  best_loss = loss;
  best_accuracy = accuracy;
  stale_evals = 0;
}

bool GateState::Observe(const uint8_t* persistent, size_t bytes,
                        uint64_t step, float loss, float accuracy,
                        uint32_t patience) {
  if (loss < best_loss) {
    Snapshot(persistent, bytes, step, loss, accuracy);
    return false;
  }
  ++stale_evals;
  return patience > 0 && stale_evals >= patience;
}

bool GateState::RestoreBest(uint8_t* persistent, uint64_t step) const {
  if (best_step == step) return false;
  std::memcpy(persistent, best_state.data(), best_state.size());
  return true;
}

}  // namespace seeml::update_rt
