#ifndef SEEML_TEST_SUPPORT_CHILD_PROCESS_H_
#define SEEML_TEST_SUPPORT_CHILD_PROCESS_H_

// =============================================================================
// A test body run in a forked child under a file-size limit (CR4 #145): the
// kernel fails a write, or kills the writer, partway through a file — the
// disk that fills and the process that dies mid-save, which the storage
// contract must survive — with no fault seam in the code under test. The
// child ends with `_exit`: no destructor of the parent's state runs (its
// temp directories stay the parent's), no buffered output is flushed twice,
// and no core is dumped. POSIX only.
//
// The body reports through its return value alone: an EXPECT_* or ASSERT_*
// inside it records in the child's copy of the registry, which the parent
// never sees. And only the forking thread lives on in the child, so a body
// must not reach the parallel-for pool (ParallelFor, a multi-chunk
// ContentHash64): a worker the parent had would never answer.
// =============================================================================

#ifndef _WIN32

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <functional>

#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace seeml::testing {

/// What the kernel does to the write that would grow a file past the limit.
enum class PastTheLimit {
  kFail,  // SIGXFSZ ignored: the write fails with EFBIG — a full disk
  kKill,  // SIGXFSZ's default action: the writer dies where it stands
};

/// Runs `body` in a forked child whose files may not grow past `limit`
/// bytes, and returns the child's wait status: it exits with what `body`
/// returns (0 when the child's own checks held), with 124 when `body` threw,
/// or dies of SIGXFSZ. -1 when the child could not be forked or waited for.
inline int RunUnderFileSizeLimit(uint64_t limit, PastTheLimit past,
                                 const std::function<int()>& body) {
  std::fflush(nullptr);  // nothing buffered is written twice
  const pid_t pid = ::fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    const struct rlimit no_core = {0, 0};
    ::setrlimit(RLIMIT_CORE, &no_core);
    std::signal(SIGXFSZ, past == PastTheLimit::kFail ? SIG_IGN : SIG_DFL);
    struct rlimit fsize;
    if (::getrlimit(RLIMIT_FSIZE, &fsize) != 0) ::_exit(125);
    fsize.rlim_cur = static_cast<rlim_t>(limit);
    if (::setrlimit(RLIMIT_FSIZE, &fsize) != 0) ::_exit(126);
    // An exception must not leave the child: unwinding would run the test's
    // destructors here (a ScopedTempDir removing the parent's directory),
    // and the runner would go on to run the rest of the suite in the child.
    int code = 124;
    try {
      code = body();
    } catch (...) {
    }
    ::_exit(code);
  }
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) return -1;
  return status;
}

}  // namespace seeml::testing

#endif  // _WIN32

#endif  // SEEML_TEST_SUPPORT_CHILD_PROCESS_H_
