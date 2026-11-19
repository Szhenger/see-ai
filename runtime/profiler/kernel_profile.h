#ifndef SEEML_RUNTIME_PROFILER_KERNEL_PROFILE_H_
#define SEEML_RUNTIME_PROFILER_KERNEL_PROFILE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

// The Metal backend's two diagnostic profiles, the profiler's since #202
// item 3. The backend decides when to measure — it serializes the stream
// per instruction for the profile and reads each command buffer's GPU
// timestamps — and hands every number here; these classes hold and print.
// Plain C++, so the tables are testable without a GPU.

namespace seeml::update_rt {

/// SEEML_METAL_PROFILE=1: GPU time per kernel and shape (one command buffer
/// per instruction), CPU-resident instructions by wall clock, and the
/// forward and backward passes counted at the loss opcodes (F9 #140: a run
/// with validation on is 7:1 forward).
class KernelProfile {
 public:
  void CountPass(bool forward) { ++(forward ? fwd_passes_ : bwd_passes_); }
  void Add(const std::string& key, double seconds) {
    Row& row = rows_[key];
    row.seconds += seconds;
    row.count++;
  }

  /// The table, slowest key first; `lowrank_*` is what the low-rank
  /// epilogue fold decided on the run's programs.
  void Print(std::FILE* out, uint64_t lowrank_fused,
             uint64_t lowrank_dropped) const {
    std::vector<std::pair<std::string, Row>> rows(rows_.begin(), rows_.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return a.second.seconds > b.second.seconds;
    });
    double total = 0;
    for (const auto& r : rows) total += r.second.seconds;
    std::fprintf(out,
                 "metal profile: %.3f s over %zu keys; %llu forward pass(es), "
                 "%llu backward pass(es) (counted at the loss opcodes)\n",
                 total, rows.size(),
                 static_cast<unsigned long long>(fwd_passes_),
                 static_cast<unsigned long long>(bwd_passes_));
    if (lowrank_fused)
      std::fprintf(out,
                   "  low-rank epilogue: %llu fused dispatch(es), %llu with C's "
                   "store dropped (k_gemm_*_lr; SEEML_METAL_UNFUSED_LOWRANK=1 "
                   "keeps the pairs apart)\n",
                   static_cast<unsigned long long>(lowrank_fused),
                   static_cast<unsigned long long>(lowrank_dropped));
    for (const auto& r : rows)
      std::fprintf(out, "  %8.3f s  %5.1f%%  %7llu x %8.1f us  %s\n",
                   r.second.seconds,
                   total > 0 ? 100.0 * r.second.seconds / total : 0.0,
                   static_cast<unsigned long long>(r.second.count),
                   1e6 * r.second.seconds / static_cast<double>(r.second.count),
                   r.first.c_str());
  }

 private:
  struct Row {
    double seconds = 0;
    uint64_t count = 0;
  };
  std::map<std::string, Row> rows_;
  uint64_t fwd_passes_ = 0, bwd_passes_ = 0;
};

/// SEEML_METAL_SPANS=1 (F8 #139): per command buffer, the GPU's busy span,
/// the CPU wall from the first encode to completion, and the dispatch
/// count — the time outside kernels is the difference, measured without
/// serializing the stream the way the profile does.
class SpanLog {
 public:
  void Add(uint64_t dispatches, double gpu_seconds, double wall_seconds) {
    rows_.push_back({dispatches, gpu_seconds, wall_seconds});
  }

  void Print(std::FILE* out, uint64_t lowrank_fused,
             uint64_t lowrank_dropped) const {
    double gpu = 0, wall = 0;
    uint64_t dispatches = 0;
    for (const Row& s : rows_) {
      gpu += s.gpu_seconds;
      wall += s.wall_seconds;
      dispatches += s.dispatches;
    }
    std::fprintf(out,
                 "metal spans: %zu command buffer(s), %llu dispatch(es); GPU "
                 "busy %.3f s, CPU wall %.3f s, outside the GPU %.3f s "
                 "(%.1f%% of the wall; %.2f us per dispatch)\n",
                 rows_.size(), static_cast<unsigned long long>(dispatches), gpu,
                 wall,
                 wall - gpu, wall > 0 ? 100.0 * (wall - gpu) / wall : 0.0,
                 dispatches
                     ? 1e6 * (wall - gpu) / static_cast<double>(dispatches)
                     : 0.0);
    if (lowrank_fused)
      std::fprintf(out,
                   "  low-rank epilogue: %llu fused dispatch(es), %llu with C's "
                   "store dropped\n",
                   static_cast<unsigned long long>(lowrank_fused),
                   static_cast<unsigned long long>(lowrank_dropped));
    // Per buffer size: the train, eval and merge programs are told apart by
    // their dispatch counts (one buffer per ExecuteRange).
    std::map<uint64_t, std::array<double, 3>> by_size;  // n, gpu, wall
    for (const Row& s : rows_) {
      auto& v = by_size[s.dispatches];
      v[0] += 1; v[1] += s.gpu_seconds; v[2] += s.wall_seconds;
    }
    for (const auto& [n, v] : by_size)
      std::fprintf(out,
                   "  %7llu dispatches x %4.0f: GPU %8.3f ms, wall %8.3f ms, "
                   "outside %7.3f ms (%.1f%%) per buffer\n",
                   static_cast<unsigned long long>(n), v[0], 1e3 * v[1] / v[0],
                   1e3 * v[2] / v[0], 1e3 * (v[2] - v[1]) / v[0],
                   v[2] > 0 ? 100.0 * (v[2] - v[1]) / v[2] : 0.0);
  }

 private:
  struct Row {
    uint64_t dispatches;
    double gpu_seconds, wall_seconds;
  };
  std::vector<Row> rows_;
};

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_PROFILER_KERNEL_PROFILE_H_
