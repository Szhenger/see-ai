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
- **`durable.h`:** every OS call the storage role makes (#202 item 4).
  - The sidecar write: `CreateForWrite`, `WriteSome`, `SyncFile`,
    `CloseFile`, `RemoveFile`.
  - The replace: `ReplaceFile`, `ParentDirectory`, `SyncParentDirectory`.
  - The in-place edit: `SyncStream`, `SeekStream` (64-bit).
  - The commit lock: `TryLockExclusive`, `ReleaseLock`.
  - `ProcessId`.

`runtime/storage/durable_io.cc` keeps the discipline (write, sync, replace,
persist the replace) and the CR4 fault injection, which sits just before
`WriteSome`. It makes no direct OS call.

The POSIX bodies are the ones CI runs. The Win32 bodies are the ones the
storage writer carried, moved unchanged, but no CI job builds Windows. A
Win32 port proves them together with the storage commit proof that §9
names as a follow-up.
