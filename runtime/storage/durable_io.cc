#include "runtime/storage/durable_io.h"

#include "source/platform/durable.h"

#include <span>

#include "runtime/diagnostics/persisting/error.h"
#include "source/identity/hash.h"
#include "source/parallel/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

/// Per-writer-unique sidecar name in the destination's directory (rename
/// atomicity requires same-filesystem). A fixed ".tmp" would let two
/// concurrent writers to one path interleave on a shared inode and rename
/// a mixed-content file into place; pid + a process-local counter keeps
/// every writer on its own sidecar. Crash leftovers are inert — nothing
/// ever reads a sidecar it did not just create.
std::string UniqueTmpPath(const std::string& path) {
  static std::atomic<uint64_t> counter{0};
  const unsigned long pid = static_cast<unsigned long>(platform::ProcessId());
  return path + ".tmp." + std::to_string(pid) + "." +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

// The injected write fault (CR4 #145 F08): tests only, consumed once.
std::atomic<int> g_write_fault{0};

}  // namespace

uint64_t CollectDeadSidecars(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const std::string dir = platform::ParentDirectory(path);
  const std::string prefix =
      (slash == std::string::npos ? path : path.substr(slash + 1)) + ".tmp.";
  auto digits = [](const std::string& s, size_t from, size_t to) {
    if (from >= to || to - from > 19) return false;
    for (size_t i = from; i < to; ++i)
      if (s[i] < '0' || s[i] > '9') return false;
    return true;
  };
  const uint64_t self = platform::ProcessId();
  uint64_t removed = 0;
  for (const std::string& name : platform::ListDirectory(dir)) {
    if (!name.starts_with(prefix)) continue;
    // Exactly <pid>.<n>, both decimal: UniqueTmpPath's shape and no other.
    const size_t pid_at = prefix.size();
    const size_t dot = name.find('.', pid_at);
    if (dot == std::string::npos || !digits(name, pid_at, dot) ||
        !digits(name, dot + 1, name.size()))
      continue;
    const uint64_t pid = std::stoull(name.substr(pid_at, dot - pid_at));
    if (pid == self || platform::ProcessMayBeAlive(pid)) continue;
    platform::RemoveFile(slash == std::string::npos ? name : dir + "/" + name);
    ++removed;
  }
  return removed;
}

void FailNextDurableWriteForTesting(int errno_value) {
  g_write_fault.store(errno_value, std::memory_order_relaxed);
}

std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::initializer_list<ByteSpan> parts) {
  return WriteFileDurable(
      path, std::span<const ByteSpan>(parts.begin(), parts.size()));
}

std::expected<void, std::string> WriteFileDurable(
    const std::string& path, std::span<const ByteSpan> parts,
    DurableWriteStats* stats) {
  const std::string tmp = UniqueTmpPath(path);
  DurableWriteStats cost;
  // Every OS call is the platform seam's (#202 item 4); the discipline is
  // this function's: every byte's write status checked (no buffer whose
  // final flush nobody sees), the data synced before the replace, a replace
  // that installs over an existing file, and the replace itself persisted.
  const platform::FileHandle f = platform::CreateForWrite(tmp);
  if (f == platform::kNoFile)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot write '" + tmp + "'");
  // Every failure past this point removes the sidecar (CR4 #145 F08): a
  // partial temp file beside the destination is a stray no later writer
  // reuses and a reader can mistake for a checkpoint.
  auto fail = [&](bool open_file, const std::string& what) {
    if (open_file) (void)platform::CloseFile(f);
    platform::RemoveFile(tmp);
    return diag::persisting::Error(diag::persisting::kDurableIo, what);
  };
  for (const ByteSpan& part : parts) {
    size_t written = 0;
    while (written < part.size) {
      int64_t n;
      bool retry = false;
      if (const int fault =
              g_write_fault.exchange(0, std::memory_order_relaxed);
          fault != 0) {
        n = -1;  // the injected full disk (an injected EINTR retries)
        retry = fault == EINTR;
      } else {
        n = platform::WriteSome(f, part.data + written, part.size - written,
                                &retry);
      }
      if (n < 0) {
        // A signal landing mid-write (checkpoints are large and SIGINT-era
        // stop flags are common) is a retry, not a failed update.
        if (retry) continue;
        return fail(true, "short write to '" + tmp + "'");
      }
      written += static_cast<size_t>(n);
    }
    cost.bytes += written;
  }
  if (!platform::SyncFile(f))
    return fail(true, "fsync of '" + tmp + "' failed");
  ++cost.fsyncs;
  if (!platform::CloseFile(f))
    return fail(false, "close of '" + tmp + "' failed");
  if (!platform::AtomicReplace(tmp, path))
    return fail(false, "atomic rename to '" + path + "' failed");
  // Persist the rename itself (best effort: some filesystems reject a
  // directory fsync; Win32 writes the move through instead).
  if (platform::SyncParentDirectory(path)) ++cost.fsyncs;
  if (stats) *stats = cost;
  return {};
}

std::expected<void, std::string> WriteFileDurable(const std::string& path,
                                                  const uint8_t* data,
                                                  size_t size) {
  return WriteFileDurable(path, {ByteSpan{data, size}});
}

std::expected<std::vector<uint8_t>, std::string> ReadFileBytes(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot stat '" + path + "'");
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  if (!bytes.empty() && !f.read(reinterpret_cast<char*>(bytes.data()),
                                static_cast<std::streamsize>(bytes.size())))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot read '" + path + "'");
  return bytes;
}

std::expected<uint64_t, std::string> HashFileContent(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot stat '" + path + "'");
  f.seekg(0);
  const size_t size = static_cast<size_t>(end);

  // Mirror ContentHash64's structure exactly (source/identity/hash.h): the
  // same chunk geometry — a pure function of the size — the same per-chunk
  // striped digests folded in chunk order, the same final size mix. The
  // only difference is that each chunk is read from disk on demand, so the
  // peak resident set is one chunk, not one file.
  const size_t grain = up::ParallelChunkGrain(size, up::kContentHashChunk);
  const size_t chunks = up::ParallelChunkCount(size, up::kContentHashChunk);
  std::vector<uint8_t> buf(chunks <= 1 ? size : grain);

  uint64_t h = up::kFnvOffsetBasis;
  const size_t folds = chunks ? chunks : 1;
  for (size_t c = 0; c < folds; ++c) {
    const size_t begin = c * grain;
    const size_t n = chunks <= 1 ? size : std::min(grain, size - begin);
    if (n != 0 && !f.read(reinterpret_cast<char*>(buf.data()),
                          static_cast<std::streamsize>(n)))
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot read '" + path + "'");
    h = up::FnvMixWord(h, up::StripedFnv1a64(buf.data(), n));
  }
  return up::FnvMixWord(h, size);
}

// --- CommitLock --------------------------------------------------------------

std::expected<CommitLock, std::string> CommitLock::Acquire(
    const std::string& target_path) {
  CommitLock lock;
  const std::string lock_path = target_path + ".lock";
  // The platform seam's exclusive lock (#202 item 4): crash-safe, never
  // stale — process death releases it.
  platform::FileHandle h = platform::kNoFile;
  switch (platform::TryLockExclusive(lock_path, &h)) {
    case platform::LockResult::kLocked:
      break;
    case platform::LockResult::kHeldElsewhere:
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "another update is committing to '" +
                                         target_path +
                                         "' — refusing to race it");
    case platform::LockResult::kCannotOpen:
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot open lock '" + lock_path + "'");
  }
  lock.handle_ = h;
  return lock;
}

CommitLock::CommitLock(CommitLock&& o) noexcept : handle_(o.handle_) {
  o.handle_ = -1;
}

CommitLock& CommitLock::operator=(CommitLock&& o) noexcept {
  if (this != &o) {
    platform::ReleaseLock(handle_);
    handle_ = o.handle_;
    o.handle_ = -1;
  }
  return *this;
}

CommitLock::~CommitLock() { platform::ReleaseLock(handle_); }

// --- DurableFileEdit ---------------------------------------------------------

std::expected<DurableFileEdit, std::string> DurableFileEdit::Begin(
    const std::string& source, const std::string& dest) {
  DurableFileEdit edit;
  edit.tmp_ = dest + ".tmp";
  edit.dest_ = dest;

  {
    std::ifstream in(source, std::ios::binary);
    if (!in)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot open '" + source + "'");
    std::ofstream out(edit.tmp_, std::ios::binary | std::ios::trunc);
    if (!out)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot write '" + edit.tmp_ + "'");
    std::vector<char> buf(1u << 20);
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      const std::streamsize got = in.gcount();
      if (got <= 0) break;
      out.write(buf.data(), got);
      if (!out)
        return diag::persisting::Error(diag::persisting::kDurableIo,
                                       "short write to '" + edit.tmp_ + "'");
      edit.size_ += static_cast<uint64_t>(got);
    }
    if (in.bad())
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot read '" + source + "'");
    // The final sub-buffer chunk may still sit in the filebuf; the
    // destructor's flush swallows failure, which would leave a silently
    // truncated sidecar that passes every size_-based bounds check (size_
    // was counted from the SOURCE reads). Flush and close explicitly —
    // close sets failbit when the flush fails.
    out.flush();
    out.close();
    if (!out)
      return diag::persisting::Error(diag::persisting::kDurableIo,
                                     "cannot flush '" + edit.tmp_ + "'");
  }

  edit.f_ = std::fopen(edit.tmp_.c_str(), "r+b");
  if (!edit.f_)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot reopen '" + edit.tmp_ + "'");
  return edit;
}

DurableFileEdit::DurableFileEdit(DurableFileEdit&& o) noexcept
    : f_(o.f_),
      tmp_(std::move(o.tmp_)),
      dest_(std::move(o.dest_)),
      size_(o.size_),
      committed_(o.committed_) {
  o.f_ = nullptr;
  o.committed_ = true;  // the moved-from shell owns nothing to discard
}

DurableFileEdit& DurableFileEdit::operator=(DurableFileEdit&& o) noexcept {
  if (this != &o) {
    CloseAndDiscard();
    f_ = o.f_;
    tmp_ = std::move(o.tmp_);
    dest_ = std::move(o.dest_);
    size_ = o.size_;
    committed_ = o.committed_;
    o.f_ = nullptr;
    o.committed_ = true;
  }
  return *this;
}

DurableFileEdit::~DurableFileEdit() { CloseAndDiscard(); }

void DurableFileEdit::CloseAndDiscard() {
  if (f_) {
    std::fclose(f_);
    f_ = nullptr;
  }
  if (!committed_ && !tmp_.empty()) std::remove(tmp_.c_str());
}

// Seeks go through platform::SeekStream: 64-bit on every platform.
std::expected<void, std::string> DurableFileEdit::ReadAt(uint64_t offset,
                                                         uint8_t* buf,
                                                         size_t n) {
  if (!f_ || offset > size_ || n > size_ - offset)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "edit read outside '" + tmp_ + "'");
  if (!platform::SeekStream(f_, offset) || std::fread(buf, 1, n, f_) != n)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot read '" + tmp_ + "'");
  return {};
}

std::expected<void, std::string> DurableFileEdit::WriteAt(uint64_t offset,
                                                          const uint8_t* buf,
                                                          size_t n) {
  if (!f_ || offset > size_ || n > size_ - offset)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "edit write outside '" + tmp_ + "'");
  if (!platform::SeekStream(f_, offset) || std::fwrite(buf, 1, n, f_) != n)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "short write to '" + tmp_ + "'");
  return {};
}

std::expected<void, std::string> DurableFileEdit::Commit() {
  if (!f_)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "commit of a closed edit");
  if (std::fflush(f_) != 0)
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "cannot flush '" + tmp_ + "'");
  // Same durability discipline as WriteFileDurable on both platforms: data
  // fsync (FlushFileBuffers) before the rename, and a rename that can
  // replace an existing destination — CRT rename cannot on Windows, which
  // would fail every repeat commit (the exact defect WriteFileDurable's
  // Win32 branch documents).
  if (!platform::SyncStream(f_))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "fsync of '" + tmp_ + "' failed");
  std::fclose(f_);
  f_ = nullptr;
  if (!platform::AtomicReplace(tmp_, dest_))
    return diag::persisting::Error(diag::persisting::kDurableIo,
                                   "atomic rename to '" + dest_ + "' failed");
  committed_ = true;
  (void)platform::SyncParentDirectory(dest_);  // best effort
  return {};
}

}  // namespace seeml::update_rt
