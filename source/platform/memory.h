#ifndef SEEML_SOURCE_PLATFORM_MEMORY_H_
#define SEEML_SOURCE_PLATFORM_MEMORY_H_

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

// =============================================================================
// The platform seam's memory queries (S7 #153, seeai.md §6: source/platform/).
// The host's physical memory — shared by the compiler's resource contract
// (does the training footprint fit this machine) and the loader (an arena
// the host cannot hold is refused before the allocation; the executing
// fuzzer, CR3 #144 F22, reached aligned_alloc with a header asking for
// exabytes) — and the process's peak resident set, the measured side of the
// runtime screen's "RSS beside arena" (S0 #146). Header-only, so the
// vendored package carries it as it is.
// =============================================================================

namespace seeml::update {

/// The host's physical memory in bytes, or 0 where the platform does not say.
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

/// The process's peak resident set size in bytes, or 0 where the platform
/// does not say. getrusage's ru_maxrss is in bytes on macOS and in
/// kilobytes on Linux — the one unit trap here.
inline uint64_t PeakRssBytes() {
#if defined(_WIN32)
  return 0;
#else
  struct rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<uint64_t>(ru.ru_maxrss);
#else
  return static_cast<uint64_t>(ru.ru_maxrss) * 1024u;
#endif
#endif
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLATFORM_MEMORY_H_
