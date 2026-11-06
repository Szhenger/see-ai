#ifndef SEEML_SOURCE_STATISTICS_PROCESS_H_
#define SEEML_SOURCE_STATISTICS_PROCESS_H_

#include <cstdint>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

// =============================================================================
// What the operating system observed of this process — the measured side of
// the runtime screen's "RSS beside arena" (S0 #146). Moved from seeml-bench,
// which has reported peak_rss_bytes since P5, so the device's runner and the
// bench read the same number the same way.
// =============================================================================

namespace seeml::update {

/// The process's peak resident set size in bytes, or 0 where the platform
/// does not say. getrusage's ru_maxrss is in bytes on macOS and in
/// kilobytes on Linux — the one unit trap here.
inline uint64_t PeakRssBytes() {
#if defined(_WIN32)
  return 0;
#else
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<uint64_t>(ru.ru_maxrss);
#else
  return static_cast<uint64_t>(ru.ru_maxrss) * 1024u;
#endif
#endif
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_STATISTICS_PROCESS_H_
