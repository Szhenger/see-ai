# source/platform — the OS seam

S7 (#153) creates the platform seam that `seeai.md` §6 calls for. It is
header-only, so the package vendors it as it is. It is POSIX now, keeping
the Win32 branches the tree already had.

- **`file_map.h`:** `FileMapping`, a move-only mapped region, and `MapFile`.
  The engine uses it to map the source model read-only and the plan beside
  the binary copy-on-write.
- **`memory.h`:** `DetectHostMemoryBytes`, the host's physical memory, used
  by the compiler's resource contract and the loader's arena refusal.
  `PeakRssBytes`, the process's resident peak, used by the runtime screen and
  seeml-bench.
- **`durable.h`:** `ParentDirectory` and `SyncParentDirectory`, which persist
  a replace in its directory. Storage's two writers use them.

Still direct OS calls elsewhere: the sidecar writes, fsyncs and renames
inside `runtime/storage/durable_io.cc`, and its advisory lock. They are
interleaved with the fault injection the CR4 tests drive. A Win32 port moves
them behind this seam, together with the storage commit proof that §9 names
as a follow-up.
