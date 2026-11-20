#include "runtime/gating/gate.h"

#include <algorithm>
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

void GateState::Snapshot(const uint8_t* persistent, size_t bytes, uint64_t step,
                         float loss, float accuracy) {
  best_state.assign(persistent, persistent + bytes);
  has_best = true;
  best_step = step;
  best_loss = loss;
  best_accuracy = accuracy;
  stale_evals = 0;
}

bool GateState::Observe(const uint8_t* persistent, size_t bytes, uint64_t step,
                        float loss, float accuracy, uint32_t patience) {
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

void LossWindows::Start(uint64_t run_steps) {
  *this = LossWindows{};
  window = static_cast<uint32_t>(
      std::max<uint64_t>(1, std::min<uint64_t>(kMax, run_steps / 5)));
}

void LossWindows::Observe(float loss) {
  if (count < window) first_sum += loss;
  last[count % window] = loss;
  ++count;
}

float LossWindows::FirstMean() const {
  const uint64_t n = std::min<uint64_t>(count, window);
  return n ? static_cast<float>(first_sum / static_cast<double>(n)) : 0.0f;
}

float LossWindows::LastMean() const {
  const uint64_t n = std::min<uint64_t>(count, window);
  if (n == 0) return FirstMean();
  double sum = 0.0;
  for (uint64_t i = count - n; i < count; ++i)  // oldest first
    sum += last[i % window];
  return static_cast<float>(sum / static_cast<double>(n));
}

}  // namespace seeml::update_rt
