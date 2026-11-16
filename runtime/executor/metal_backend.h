#ifndef SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_
#define SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "runtime/executor/backend.h"

// =============================================================================
// The Metal backend's factory (roadmap Project 5, G1b). Two translation
// units define these symbols: metal_backend.mm on Apple platforms (the real
// backend — zero-copy arena residency, batched command encoding, the GPU
// kernel library) and metal_backend_stub.cc everywhere else, where Metal
// does not exist. No other translation unit references a Metal symbol, so
// a package still builds with `c++` alone on Linux and Windows; the Metal
// path is an `#ifdef __APPLE__` plus a run-time device check, never a
// build requirement.
// =============================================================================

namespace seeml::update_rt {

/// Whether this runtime carries the Metal backend AND a device exists.
bool MetalBackendAvailable();

/// What the build host's Metal device is (S6 #152, the GPU half of the target
/// description): its name, the highest Apple GPU family it supports, whether it
/// reports the Metal 4 family, its recommended working set and threadgroup
/// memory. Empty (`present` false) where no device or no Metal build exists.
/// The compiler states it on the backend screen; no plan byte depends on it.
struct MetalDeviceInfo {
  bool present = false;
  std::string name;
  int apple_family = 0;  // N of MTLGPUFamilyAppleN, 0 = none reported
  bool metal4 = false;
  uint64_t working_set_bytes = 0;
  uint64_t threadgroup_memory_bytes = 0;
};
MetalDeviceInfo DescribeMetalDevice();

/// Builds the Metal backend, JIT-compiling the kernel library for the
/// default device. Fails with a one-line reason where unavailable.
[[nodiscard]] std::expected<std::unique_ptr<ExecutorBackend>, std::string>
CreateMetalBackend();

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_EXECUTOR_METAL_BACKEND_H_
