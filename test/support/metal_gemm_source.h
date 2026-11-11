#ifndef SEEML_TEST_SUPPORT_METAL_GEMM_SOURCE_H_
#define SEEML_TEST_SUPPORT_METAL_GEMM_SOURCE_H_

#include <string>

#include "compiler/backend/architecture/host_arch.h"

// =============================================================================
// The Metal source the G1a hardware harness (runtime/executor/metal_gemm.mm)
// JIT-compiles. It was the compiler's kernel emitter until the runtime took
// ownership of the kernel library; nothing ships it, so it is test support
// (S6 #152).
// =============================================================================

namespace seeml::testing {

/// The .metal translation unit: seeml_matmul (C = A@B), seeml_matmul_nt
/// (C = A@B^T), seeml_matmul_tn (C = A^T@B), and seeml_gemm_acc
/// (C += alpha * A@B), all f32, all tiled per `tiling`.
std::string MetalGemmSource(const seeml::update::GpuTiling& tiling);

}  // namespace seeml::testing

#endif  // SEEML_TEST_SUPPORT_METAL_GEMM_SOURCE_H_
