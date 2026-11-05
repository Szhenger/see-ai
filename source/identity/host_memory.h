#ifndef SEEML_SOURCE_IDENTITY_HOST_MEMORY_H_
#define SEEML_SOURCE_IDENTITY_HOST_MEMORY_H_

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

// =============================================================================
// The host's physical memory, in bytes, or 0 where the platform does not say.
// One query, shared by the compiler's resource contract (does the training
// footprint fit this machine) and the runtime's load (an arena the host
// cannot hold is refused before the allocation is attempted — the executing
// fuzzer, CR3 #144 F22, reached aligned_alloc with a header asking for
// exabytes). Header-only, so the vendored package carries it as it is.
// =============================================================================

namespace seeml::update {

inline uint64_t DetectHostMemoryBytes() {
#if defined(__APPLE__)
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) return 0;
  return mem;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page_size <= 0) return 0;
  const uint64_t p = static_cast<uint64_t>(pages);
  const uint64_t s = static_cast<uint64_t>(page_size);
  return p > UINT64_MAX / s ? UINT64_MAX : p * s;  // saturating
#endif
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_IDENTITY_HOST_MEMORY_H_
