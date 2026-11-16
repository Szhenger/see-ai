#ifndef SEEML_SOURCE_PLATFORM_DURABLE_H_
#define SEEML_SOURCE_PLATFORM_DURABLE_H_

#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

// =============================================================================
// The platform seam's durable-replace primitives (S7 #153). The storage
// role's discipline (runtime/storage/durable_io.cc: write a sidecar, fsync,
// replace, persist the replace) is the same on every OS; what each step is
// made of is not. POSIX now; the Win32 storage proof is the follow-up §9
// names.
// =============================================================================

namespace seeml::platform {

/// The directory holding `path`, as the replace landed in it.
inline std::string ParentDirectory(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".")
                                    : path.substr(0, slash == 0 ? 1 : slash);
}

/// Persists a rename into `path`'s directory: fsyncs the directory. Best
/// effort — some filesystems reject a directory fsync — so it reports
/// whether one happened rather than failing.
inline bool SyncParentDirectory(const std::string& path) {
#ifndef _WIN32
  const int dfd = ::open(ParentDirectory(path).c_str(), O_RDONLY);
  if (dfd < 0) return false;
  const bool synced = ::fsync(dfd) == 0;
  ::close(dfd);
  return synced;
#else
  (void)path;
  return false;
#endif
}

}  // namespace seeml::platform

#endif  // SEEML_SOURCE_PLATFORM_DURABLE_H_
