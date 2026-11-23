#include <algorithm>
#include <cmath>
#include <iterator>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Optimizer family: gradient conditioning and the in-place parameter steps.
// The clip's norm reduces with one partial per chunk, combined in chunk
// order, so clipping decisions are bitwise-deterministic at any thread count.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

/// ||g||_2^2 with one f64 partial per chunk, combined in chunk order — the
/// reduction ClipNorm has always run, shared with the fused steps and the
/// global clip's accumulation.
double L2Squared(const float* g, size_t n) {
  const size_t grain = kGrainCheap;
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(n, grain, [&](size_t b, size_t e, size_t chunk) {
    double sq = 0.0;
    for (size_t i = b; i < e; ++i)
      sq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
    partials[chunk] = sq;
  });
  double sq = 0.0;
  const size_t chunks = up::ParallelChunkCount(n, grain);
  for (size_t c = 0; c < chunks; ++c) sq += partials[c];
  return sq;
}

double L2Norm(const float* g, size_t n) { return std::sqrt(L2Squared(g, n)); }

/// The factor ClipNorm multiplies g by, or exactly 1 where it leaves g
/// alone (multiplying by 1.0f is the identity on every float, so the fused
/// steps need no second code path). False for a non-finite norm.
bool ClipScale(const float* g, size_t n, float max_norm, float* scale) {
  *scale = 1.0f;
  if (!(max_norm > 0.0f)) return true;
  const double norm = L2Norm(g, n);
  if (!std::isfinite(norm)) return false;
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) return true;
  *scale = static_cast<float>(static_cast<double>(max_norm) / norm);
  return true;
}

}  // namespace

bool ClipNorm(float* g, size_t n, float max_norm) {
  // CR1 #142 F09: the standalone clip refuses what the fused steps refuse —
  // a cap that is not a finite positive float, and a non-finite norm —
  // leaving g as it was; the backend maps false to the shared verdict and
  // the engine stops before the step. Before, a NaN norm scaled every
  // element to NaN and a zero cap scaled them to zero.
  if (!std::isfinite(max_norm) || !(max_norm > 0.0f)) return false;
  const double norm = L2Norm(g, n);
  if (!std::isfinite(norm)) return false;
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) return true;
  const float s = static_cast<float>(static_cast<double>(max_norm) / norm);
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) g[i] *= s;
  });
  return true;
}

void SqNormAccum(const float* g, size_t n, float* acc) {
  acc[0] = static_cast<float>(static_cast<double>(acc[0]) + L2Squared(g, n));
}

bool ClipGlobal(float* g, size_t n, const float* acc, float max_norm) {
  if (!std::isfinite(max_norm) || !(max_norm > 0.0f)) return false;
  const double norm = std::sqrt(static_cast<double>(acc[0]));
  if (!std::isfinite(norm)) return false;
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) return true;
  const float s = static_cast<float>(static_cast<double>(max_norm) / norm);
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) g[i] *= s;
  });
  return true;
}

// The fused-clip steps round g[i] * s to f32 in a statement of its own
// before anything consumes it: that product is the float ClipNorm would
// have stored, and it must not be contracted into the step's arithmetic
// (an fma(g, s, ...) keeps bits the stored gradient never had).
bool SgdStep(float* p, const float* g, size_t n, float lr, float weight_decay,
             float clip_norm) {
  float s = 1.0f;
  if (!ClipScale(g, n, clip_norm, &s)) return false;
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float gi = g[i] * s;
      p[i] -= lr * (gi + weight_decay * p[i]);
    }
  });
  return true;
}

bool AdamWStep(float* SEEML_RESTRICT p, const float* SEEML_RESTRICT g,
               float* SEEML_RESTRICT m, float* SEEML_RESTRICT v, size_t n,
               float lr, float beta1, float beta2, float eps,
               float weight_decay, uint64_t step, float clip_norm) {
  float s = 1.0f;
  if (!ClipScale(g, n, clip_norm, &s)) return false;
  // Hoist everything that is per-step, not per-element: the bias-correction
  // divides become two multiplies, and the (1-beta) blend factors are
  // computed once instead of n times.
  const float inv_bc1 =
      1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
  const float inv_bc2 =
      1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
  const float om_b1 = 1.0f - beta1;
  const float om_b2 = 1.0f - beta2;
  up::ParallelFor(n, kGrainMath, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const float gi = g[i] * s;
      const float mi = beta1 * m[i] + om_b1 * gi;
      const float vi = beta2 * v[i] + om_b2 * gi * gi;
      m[i] = mi;
      v[i] = vi;
      const float m_hat = mi * inv_bc1;
      const float v_hat = vi * inv_bc2;
      p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
    }
  });
  return true;
}

// 8-bit moments (plan v27, S5 #192). One block of kMomentBlock values is
// the unit of work: requantizing needs the block's new absmax, so a block
// is computed whole, by one chunk, and the blocks are independent — the
// bytes cannot depend on the worker count. Two passes over the blocks: the
// first proves every new moment finite and every stored scale a finite
// non-negative float and writes nothing (the fused clip's "refused,
// touching nothing"); the second runs AdamWStep's expressions on the
// dequantized moments, moves p, and requantizes.
MomentStep AdamW8Step(float* SEEML_RESTRICT p, const float* SEEML_RESTRICT g,
                      int8_t* SEEML_RESTRICT m8, uint8_t* SEEML_RESTRICT v8,
                      float* SEEML_RESTRICT scales, size_t n, float lr,
                      float beta1, float beta2, float eps, float weight_decay,
                      uint64_t step, float clip_norm) {
  float s = 1.0f;
  if (!ClipScale(g, n, clip_norm, &s)) return MomentStep::kNormRefused;
  const float inv_bc1 =
      1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
  const float inv_bc2 =
      1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
  const float om_b1 = 1.0f - beta1;
  const float om_b2 = 1.0f - beta2;
  const size_t blocks = up::MomentBlocks(n);
  float* m_scale = scales;
  float* v_scale = scales + blocks;
  const size_t grain = std::max<size_t>(1, kGrainMath / up::kMomentBlock);
  // One verdict per chunk, combined after: no shared write in the body.
  bool finite[up::kMaxParallelChunks];
  std::fill(std::begin(finite), std::end(finite), true);
  up::ParallelFor(blocks, grain, [&](size_t b0, size_t b1, size_t chunk) {
    for (size_t b = b0; b < b1 && finite[chunk]; ++b) {
      const float ms = m_scale[b], vs = v_scale[b];
      if (!std::isfinite(ms) || !std::isfinite(vs) || ms < 0.0f || vs < 0.0f) {
        finite[chunk] = false;
        break;
      }
      const float dm = ms / up::kMomentMLevels;
      const size_t e = std::min<size_t>(n, (b + 1) * up::kMomentBlock);
      for (size_t i = b * up::kMomentBlock; i < e; ++i) {
        const float gi = g[i] * s;
        const float m0 = static_cast<float>(m8[i]) * dm;
        const float v0 = up::MomentValueV(v8[i], vs);
        const float mi = beta1 * m0 + om_b1 * gi;
        const float vi = beta2 * v0 + om_b2 * gi * gi;
        if (!std::isfinite(mi) || !std::isfinite(vi)) {
          finite[chunk] = false;
          break;
        }
      }
    }
  });
  const size_t chunks = up::ParallelChunkCount(blocks, grain);
  for (size_t c = 0; c < chunks; ++c)
    if (!finite[c]) return MomentStep::kMomentRefused;
  up::ParallelFor(blocks, grain, [&](size_t b0, size_t b1, size_t) {
    float mloc[up::kMomentBlock], vloc[up::kMomentBlock];
    for (size_t b = b0; b < b1; ++b) {
      const float dm = m_scale[b] / up::kMomentMLevels;
      const float vs = v_scale[b];
      const size_t base = b * up::kMomentBlock;
      const size_t len = std::min<size_t>(n, base + up::kMomentBlock) - base;
      float amax_m = 0.0f, max_v = 0.0f;
      for (size_t j = 0; j < len; ++j) {
        const size_t i = base + j;
        const float gi = g[i] * s;
        const float m0 = static_cast<float>(m8[i]) * dm;
        const float v0 = up::MomentValueV(v8[i], vs);
        const float mi = beta1 * m0 + om_b1 * gi;
        const float vi = beta2 * v0 + om_b2 * gi * gi;
        mloc[j] = mi;
        vloc[j] = vi;
        amax_m = std::max(amax_m, std::fabs(mi));
        max_v = std::max(max_v, vi);
        const float m_hat = mi * inv_bc1;
        const float v_hat = vi * inv_bc2;
        p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p[i]);
      }
      if (amax_m < up::kMomentScaleFloor) {
        m_scale[b] = 0.0f;
        for (size_t j = 0; j < len; ++j) m8[base + j] = 0;
      } else {
        m_scale[b] = amax_m;
        const float d = amax_m / up::kMomentMLevels;
        for (size_t j = 0; j < len; ++j)
          m8[base + j] = up::MomentCodeM(mloc[j], d);
      }
      if (max_v < up::kMomentScaleFloor) {
        v_scale[b] = 0.0f;
        for (size_t j = 0; j < len; ++j) v8[base + j] = 0;
      } else {
        v_scale[b] = max_v;
        for (size_t j = 0; j < len; ++j)
          v8[base + j] = up::MomentCodeV(vloc[j], max_v);
      }
    }
  });
  return MomentStep::kDone;
}

}  // namespace seeml::update_rt::kernels
