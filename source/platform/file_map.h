#ifndef SEEML_SOURCE_PLATFORM_FILE_MAP_H_
#define SEEML_SOURCE_PLATFORM_FILE_MAP_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// =============================================================================
// The platform seam's file mapping (S7 #153): a move-only mapped region,
// unmapped on Release or destruction, and the one call that maps a file —
// read-only (the source model the eval program reads in place) or private
// copy-on-write (the plan beside the binary, #158). POSIX; a Win32 port maps
// through CreateFileMapping behind the same API.
// =============================================================================

namespace seeml::platform {

class FileMapping {
 public:
  FileMapping() = default;
  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping&) = delete;
  FileMapping(FileMapping&& other) noexcept
      : data(std::exchange(other.data, nullptr)),
        bytes(std::exchange(other.bytes, 0)) {}
  FileMapping& operator=(FileMapping&& other) noexcept {
    if (this != &other) {
      Release();
      data = std::exchange(other.data, nullptr);
      bytes = std::exchange(other.bytes, 0);
    }
    return *this;
  }
  ~FileMapping() { Release(); }
  void Release() {
    if (data) ::munmap(const_cast<uint8_t*>(data), bytes);
    data = nullptr;
    bytes = 0;
  }

  const uint8_t* data = nullptr;
  size_t bytes = 0;
};

enum class MapAccess {
  kReadOnly,     // PROT_READ, MAP_PRIVATE
  kCopyOnWrite,  // PROT_READ | PROT_WRITE, MAP_PRIVATE: writes stay private
};

/// Which step failed, and errno at it — the caller words the diagnostic.
struct MapFailure {
  enum Stage { kOpen, kSize, kMap } stage;
  int err;
};

/// Maps `path` whole. `require_regular` refuses anything but a regular file
/// at the size step; an empty file is always refused there.
[[nodiscard]] inline std::expected<FileMapping, MapFailure> MapFile(
    const std::string& path, MapAccess access, bool require_regular) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return std::unexpected(MapFailure{MapFailure::kOpen, errno});
  struct stat st {};
  if (::fstat(fd, &st) != 0 || (require_regular && !S_ISREG(st.st_mode)) ||
      st.st_size <= 0) {
    const int err = errno;
    ::close(fd);
    return std::unexpected(MapFailure{MapFailure::kSize, err});
  }
  const auto bytes = static_cast<size_t>(st.st_size);
  void* map = ::mmap(nullptr, bytes,
                     access == MapAccess::kCopyOnWrite ? PROT_READ | PROT_WRITE
                                                       : PROT_READ,
                     MAP_PRIVATE, fd, 0);
  const int err = errno;
  ::close(fd);
  if (map == MAP_FAILED)
    return std::unexpected(MapFailure{MapFailure::kMap, err});
  FileMapping m;
  m.data = static_cast<const uint8_t*>(map);
  m.bytes = bytes;
  return m;
}

}  // namespace seeml::platform

#endif  // SEEML_SOURCE_PLATFORM_FILE_MAP_H_
