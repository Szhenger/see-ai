#ifndef SEEML_SOURCE_PLATFORM_DURABLE_H_
#define SEEML_SOURCE_PLATFORM_DURABLE_H_

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>
#else
#include <io.h>
#include <windows.h>
#endif

// =============================================================================
// The platform seam's durable-replace primitives (S7 #153, #202 item 4). The
// storage role's discipline (runtime/storage/durable_io.cc: write a sidecar,
// fsync, replace, persist the replace; the commit lock; the in-place edit)
// is the same on every OS; what each step is made of is not, and lives here.
// The POSIX bodies are the ones CI runs; the Win32 bodies are the ones the
// storage writer carried, moved unchanged — no CI job builds Windows, and
// the Win32 storage proof is the follow-up §9 names.
// =============================================================================

namespace seeml::platform {

/// An open file: a POSIX descriptor, or a Win32 HANDLE as an integer.
using FileHandle = intptr_t;
inline constexpr FileHandle kNoFile = -1;

/// This process's id (the sidecar names carry it).
inline uint64_t ProcessId() {
#ifndef _WIN32
  return static_cast<uint64_t>(::getpid());
#else
  return static_cast<uint64_t>(::GetCurrentProcessId());
#endif
}

/// Creates (or truncates) `path` for writing; kNoFile on failure.
inline FileHandle CreateForWrite(const std::string& path) {
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  return fd < 0 ? kNoFile : static_cast<FileHandle>(fd);
#else
  HANDLE h = ::CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  return h == INVALID_HANDLE_VALUE ? kNoFile : reinterpret_cast<FileHandle>(h);
#endif
}

/// One write call of `n` > 0 bytes: the bytes written (> 0), or -1. On -1,
/// `*retry` says the call was interrupted before writing anything (POSIX
/// EINTR) and the caller should simply call again; anything else is a
/// failed write — a write that took nothing included, on both platforms, so
/// a caller looping to `n` cannot spin on it.
inline int64_t WriteSome(FileHandle f, const uint8_t* data, size_t n,
                         bool* retry) {
  *retry = false;
#ifndef _WIN32
  const ssize_t got = ::write(static_cast<int>(f), data, n);
  if (got <= 0) {
    *retry = got < 0 && errno == EINTR;
    return -1;
  }
  return static_cast<int64_t>(got);
#else
  const DWORD want = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
  DWORD got = 0;
  if (!::WriteFile(reinterpret_cast<HANDLE>(f), data, want, &got, nullptr) ||
      got == 0)
    return -1;
  return static_cast<int64_t>(got);
#endif
}

/// Forces the file's data to stable storage: fsync / FlushFileBuffers.
[[nodiscard]] inline bool SyncFile(FileHandle f) {
#ifndef _WIN32
  return ::fsync(static_cast<int>(f)) == 0;
#else
  return ::FlushFileBuffers(reinterpret_cast<HANDLE>(f)) != 0;
#endif
}

[[nodiscard]] inline bool CloseFile(FileHandle f) {
#ifndef _WIN32
  return ::close(static_cast<int>(f)) == 0;
#else
  return ::CloseHandle(reinterpret_cast<HANDLE>(f)) != 0;
#endif
}

inline void RemoveFile(const std::string& path) {
#ifndef _WIN32
  ::unlink(path.c_str());
#else
  ::DeleteFileA(path.c_str());
#endif
}

/// Atomically installs `from` as `to`, replacing an existing `to`. POSIX
/// rename does; CRT rename on Windows refuses an existing destination, so
/// Win32 uses MoveFileEx (and writes the move through). Not "ReplaceFile":
/// <windows.h> defines that name as a macro (ReplaceFileA / ReplaceFileW).
[[nodiscard]] inline bool AtomicReplace(const std::string& from,
                                        const std::string& to) {
#ifndef _WIN32
  return std::rename(from.c_str(), to.c_str()) == 0;
#else
  return ::MoveFileExA(from.c_str(), to.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#endif
}

/// Creates (or truncates) `path` as a stdio stream for writing; nullptr on
/// failure. fopen(path, "w") asks the OS for mode 0666 and leaves the
/// result to the process's umask — group- and world-writable under a
/// permissive one (CWE-732; CodeQL cpp/world-writable-file-creation). This
/// states the mode CreateForWrite states, 0644, and the umask can only
/// narrow it. Bytes are written as given (binary) on every platform.
[[nodiscard]] inline std::FILE* OpenStreamForWrite(const std::string& path) {
#ifndef _WIN32
  const FileHandle fd = CreateForWrite(path);
  if (fd == kNoFile) return nullptr;
  std::FILE* f = ::fdopen(static_cast<int>(fd), "wb");
  if (f == nullptr) ::close(static_cast<int>(fd));
  return f;
#else
  return std::fopen(path.c_str(), "wb");
#endif
}

/// SyncFile for a stdio stream (already fflush'ed by the caller).
[[nodiscard]] inline bool SyncStream(std::FILE* f) {
#ifndef _WIN32
  return ::fsync(::fileno(f)) == 0;
#else
  return ::FlushFileBuffers(
             reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(f)))) != 0;
#endif
}

/// A 64-bit absolute seek. CRT fseek takes a 32-bit long on Windows: a
/// commit target with weight ranges past 2 GiB would wrap the offset and
/// patch the wrong bytes (the bounds check upstream runs on the untruncated
/// value, so nothing trips).
[[nodiscard]] inline bool SeekStream(std::FILE* f, uint64_t offset) {
#ifndef _WIN32
  return ::fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#else
  return ::_fseeki64(f, static_cast<long long>(offset), SEEK_SET) == 0;
#endif
}

/// An exclusive, non-blocking, crash-safe lock on `lock_path` (created if
/// absent): flock on POSIX; on Win32 a share-mode-0 handle, whose second
/// open fails with a sharing violation. Process death releases either, so a
/// lock is never stale.
enum class LockResult { kLocked, kHeldElsewhere, kCannotOpen };
inline LockResult TryLockExclusive(const std::string& lock_path,
                                   FileHandle* out) {
  *out = kNoFile;
#ifndef _WIN32
  const int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0) return LockResult::kCannotOpen;
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return LockResult::kHeldElsewhere;
  }
  *out = static_cast<FileHandle>(fd);
#else
  HANDLE h =
      ::CreateFileA(lock_path.c_str(), GENERIC_WRITE, /*share=*/0, nullptr,
                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return ::GetLastError() == ERROR_SHARING_VIOLATION
               ? LockResult::kHeldElsewhere
               : LockResult::kCannotOpen;
  *out = reinterpret_cast<FileHandle>(h);
#endif
  return LockResult::kLocked;
}

/// Releases a lock TryLockExclusive took (closing drops it).
inline void ReleaseLock(FileHandle f) {
  if (f != kNoFile) (void)CloseFile(f);
}

/// The directory holding `path`, as the replace landed in it.
inline std::string ParentDirectory(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".")
                                    : path.substr(0, slash == 0 ? 1 : slash);
}

/// Persists a rename into `path`'s directory: fsyncs the directory. Best
/// effort — some filesystems reject a directory fsync — so it reports
/// whether one happened rather than failing.
[[nodiscard]] inline bool SyncParentDirectory(const std::string& path) {
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

/// The names of the entries of directory `dir`, "." and ".." left out;
/// empty when the directory cannot be read (a caller sweeping leftovers
/// then simply finds none).
inline std::vector<std::string> ListDirectory(const std::string& dir) {
  std::vector<std::string> names;
#ifndef _WIN32
  if (DIR* d = ::opendir(dir.c_str())) {
    while (const dirent* e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name != "." && name != "..") names.push_back(name);
    }
    ::closedir(d);
  }
#else
  WIN32_FIND_DATAA found;
  HANDLE h = ::FindFirstFileA((dir + "\\*").c_str(), &found);
  if (h != INVALID_HANDLE_VALUE) {
    do {
      const std::string name = found.cFileName;
      if (name != "." && name != "..") names.push_back(name);
    } while (::FindNextFileA(h, &found));
    ::FindClose(h);
  }
#endif
  return names;
}

/// Whether process `pid` may be running. False only when the OS says no
/// such process exists: a caller collecting a dead writer's leftovers must
/// never take a live writer's, so every doubt (no permission to signal it,
/// an unexpected error) answers true.
inline bool ProcessMayBeAlive(uint64_t pid) {
#ifndef _WIN32
  if (pid == 0 || pid > static_cast<uint64_t>(INT32_MAX)) return true;
  return ::kill(static_cast<pid_t>(pid), 0) == 0 || errno != ESRCH;
#else
  if (pid == 0 || pid > 0xFFFFFFFFull) return true;
  HANDLE h = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                           static_cast<DWORD>(pid));
  if (!h) return ::GetLastError() != ERROR_INVALID_PARAMETER;
  DWORD code = 0;
  const bool alive = !::GetExitCodeProcess(h, &code) || code == STILL_ACTIVE;
  ::CloseHandle(h);
  return alive;
#endif
}

}  // namespace seeml::platform

#endif  // SEEML_SOURCE_PLATFORM_DURABLE_H_
