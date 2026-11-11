// =============================================================================
// storage/ unit tests: the durable write path (gather writes, atomic
// replacement, whole-file reads) and the SEKP checkpoint container (round
// trip, plan binding, and the corruption/truncation rejections that must
// fire before a byte reaches the arena).
// =============================================================================

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

#include "runtime/storage/checkpoint.h"
#include "runtime/storage/durable_io.h"
#include "source/identity/hash.h"
#include "runtime/dispatcher/contract.h"
#include "test/framework/seetest.h"
#include "test/support/child_process.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update_rt;
using seeml::testing::ScopedTempDir;

TEST(DurableIo, RoundTripsSingleAndGatherWrites) {
  ScopedTempDir dir;
  const std::vector<uint8_t> payload{1, 2, 3, 4, 5};
  ASSERT_OK(WriteFileDurable(dir.File("one.bin"), payload.data(),
                             payload.size()));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> back,
                       ReadFileBytes(dir.File("one.bin")));
  EXPECT_TRUE(back == payload);

  // Gather form: header + payload written as one durable concatenation.
  const std::vector<uint8_t> head{9, 9};
  ASSERT_OK(WriteFileDurable(
      dir.File("two.bin"),
      {ByteSpan{head.data(), head.size()},
       ByteSpan{payload.data(), payload.size()}}));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> both,
                       ReadFileBytes(dir.File("two.bin")));
  const std::vector<uint8_t> want{9, 9, 1, 2, 3, 4, 5};
  EXPECT_TRUE(both == want);
}

TEST(DurableIo, ReplacesExistingFilesAtomically) {
  ScopedTempDir dir;
  const std::vector<uint8_t> old_bytes(128, 0xAA);
  const std::vector<uint8_t> new_bytes{7};
  ASSERT_OK(WriteFileDurable(dir.File("f.bin"), old_bytes.data(),
                             old_bytes.size()));
  ASSERT_OK(WriteFileDurable(dir.File("f.bin"), new_bytes.data(),
                             new_bytes.size()));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> back,
                       ReadFileBytes(dir.File("f.bin")));
  EXPECT_TRUE(back == new_bytes);  // fully replaced, not appended or torn
}

TEST(DurableIo, ReadReportsMissingFilesAsWellFormedDiagnostics) {
  ScopedTempDir dir;
  const auto r = ReadFileBytes(dir.File("absent.bin"));
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
}

// =============================================================================
// Checkpoints
// =============================================================================

constexpr uint64_t kPlanHash = 0x5EEA11ULL;
constexpr uint64_t kBytes = 256;

std::vector<uint8_t> Segment() {
  std::vector<uint8_t> seg(kBytes);
  std::iota(seg.begin(), seg.end(), 0);
  return seg;
}

TEST(DurableIo, StreamingFileHashMatchesContentHash64) {
  ScopedTempDir dir;
  // Cross the parallel hash's 1 MiB chunk boundary so the streamed fold
  // reproduces the multi-chunk geometry, not just the single-chunk case.
  std::vector<uint8_t> bytes((1u << 20) * 2 + 12345);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>((i * 131) ^ (i >> 7));
  const std::string path = dir.File("blob.bin");
  ASSERT_OK(WriteFileDurable(path, bytes.data(), bytes.size()));

  ASSERT_OK_AND_ASSIGN(uint64_t streamed, HashFileContent(path));
  EXPECT_EQ(streamed, seeml::update::ContentHash64(bytes.data(), bytes.size()));

  // Empty file: identical to hashing an empty buffer.
  const std::string empty = dir.File("empty.bin");
  ASSERT_OK(WriteFileDurable(empty, nullptr, 0));
  ASSERT_OK_AND_ASSIGN(uint64_t empty_hash, HashFileContent(empty));
  EXPECT_EQ(empty_hash, seeml::update::ContentHash64(nullptr, 0));
}

TEST(DurableIo, ConcurrentWritersToOnePathYieldOneCompleteFile) {
  // Bounded nondeterminism: two writers racing on the same destination may
  // land in either order, but the surviving file must be ONE writer's
  // complete payload — never an interleaving. (Per-writer-unique sidecar
  // names are what rule out the shared-inode mix.)
  ScopedTempDir dir;
  const std::string path = dir.File("contended.bin");
  const std::vector<uint8_t> a(256 * 1024, 0xAA);
  const std::vector<uint8_t> b(256 * 1024, 0xBB);

  for (int round = 0; round < 8; ++round) {
    // Both writers must succeed every round (CR3 #144 F27: with the results
    // discarded, a file left by an earlier round satisfied the content check
    // even when every later write failed).
    std::expected<void, std::string> ra, rb;
    std::thread ta([&] { ra = WriteFileDurable(path, {{a.data(), a.size()}}); });
    std::thread tb([&] { rb = WriteFileDurable(path, {{b.data(), b.size()}}); });
    ta.join();
    tb.join();
    ASSERT_OK(ra);
    ASSERT_OK(rb);
    ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> got, ReadFileBytes(path));
    EXPECT_TRUE(got == a || got == b);
  }
}

TEST(DurableIo, CommitLockIsExclusivePerTarget) {
  ScopedTempDir dir;
  const std::string target = dir.File("model.smf");
  auto first = CommitLock::Acquire(target);
  ASSERT_TRUE(first.has_value());
  // Second committer to the same target is refused with a diagnostic.
  auto second = CommitLock::Acquire(target);
  ASSERT_FALSE(second.has_value());
  EXPECT_STR_CONTAINS(second.error(), "another update is committing");
  EXPECT_TRUE(WellFormedDiagnostic(second.error()));
  // A different target is independent.
  EXPECT_TRUE(CommitLock::Acquire(dir.File("other.smf")).has_value());
  // Releasing the first lock frees the target.
  first = CommitLock::Acquire(dir.File("third.smf"));
  EXPECT_TRUE(CommitLock::Acquire(target).has_value());
}

TEST(DurableIo, DurableFileEditPatchesACopyAndDiscardsOnAbort) {
  ScopedTempDir dir;
  const std::string src = dir.File("src.bin");
  const std::string dst = dir.File("dst.bin");
  std::vector<uint8_t> bytes(256);
  std::iota(bytes.begin(), bytes.end(), 0);
  ASSERT_OK(WriteFileDurable(src, bytes.data(), bytes.size()));

  {
    // Abort path: destruction without Commit leaves no destination and no
    // sidecar behind.
    ASSERT_OK_AND_ASSIGN(auto edit, DurableFileEdit::Begin(src, dst));
    EXPECT_EQ(edit.size(), bytes.size());
    const uint8_t patch[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    ASSERT_OK(edit.WriteAt(16, patch, sizeof(patch)));
  }
  EXPECT_FALSE(std::ifstream(dst).good());
  EXPECT_FALSE(std::ifstream(dst + ".tmp").good());

  {
    ASSERT_OK_AND_ASSIGN(auto edit, DurableFileEdit::Begin(src, dst));
    // Bounds are enforced on the sidecar's size.
    uint8_t two[2] = {0, 0};
    EXPECT_ERROR(edit.ReadAt(bytes.size(), two, 1));
    EXPECT_ERROR(edit.WriteAt(bytes.size() - 1, two, 2));
    const uint8_t patch[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    ASSERT_OK(edit.WriteAt(16, patch, sizeof(patch)));
    ASSERT_OK(edit.Commit());
  }
  ASSERT_OK_AND_ASSIGN(auto committed, ReadFileBytes(dst));
  std::vector<uint8_t> expect = bytes;
  expect[16] = 0xAA; expect[17] = 0xBB; expect[18] = 0xCC; expect[19] = 0xDD;
  EXPECT_TRUE(committed == expect);
  // The source is untouched.
  ASSERT_OK_AND_ASSIGN(auto src_bytes, ReadFileBytes(src));
  EXPECT_TRUE(src_bytes == bytes);
}

TEST(Checkpoint, RoundTripsTheSegmentAndStep) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, /*step=*/7,
                               seg.data(), kBytes));

  std::vector<uint8_t> restored(kBytes, 0xFF);
  ASSERT_OK_AND_ASSIGN(uint64_t step,
                       LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash,
                                          kBytes, restored.data()));
  EXPECT_EQ(step, 7u);
  EXPECT_TRUE(restored == seg);
}

TEST(Checkpoint, RejectsForeignPlansWithoutTouchingTheArena) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, 3, seg.data(),
                               kBytes));

  std::vector<uint8_t> dst(kBytes, 0xFF);
  const auto r =
      LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash + 1, kBytes,
                         dst.data());
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(WellFormedDiagnostic(r.error()));
  for (uint8_t b : dst) EXPECT_EQ(b, 0xFF);  // dst untouched on failure

  // A layout mismatch (wrong persistent size) is equally foreign.
  EXPECT_ERROR(LoadCheckpointFile(dir.File("c.ckpt"), kPlanHash, kBytes / 2,
                                  dst.data()));
}

TEST(Checkpoint, RejectsBitFlippedAndTruncatedFiles) {
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("c.ckpt"), kPlanHash, 3, seg.data(),
                               kBytes));
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> raw,
                       ReadFileBytes(dir.File("c.ckpt")));

  std::vector<uint8_t> dst(kBytes);

  // Flip one payload byte.
  std::vector<uint8_t> flipped = raw;
  flipped.back() ^= 0x01;
  {
    std::ofstream f(dir.File("flipped.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(flipped.data()),
            static_cast<std::streamsize>(flipped.size()));
  }
  EXPECT_ERROR(LoadCheckpointFile(dir.File("flipped.ckpt"), kPlanHash,
                                  kBytes, dst.data()));

  // Cut the payload short.
  {
    std::ofstream f(dir.File("short.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size() - 16));
  }
  EXPECT_ERROR(LoadCheckpointFile(dir.File("short.ckpt"), kPlanHash, kBytes,
                                  dst.data()));
}

TEST(Checkpoint, TheV5RecordRoundTripsWithItsBestPayload) {
  // SEKP v5 (E9, #92): the run binding, the source model's score and the
  // best state travel with the segment; the best payload is hashed and
  // verified like the main one, and a peek reads the binding alone.
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  std::vector<uint8_t> best = seg;
  for (auto& b : best) b ^= 0x5A;
  CheckpointRecord record;
  record.step = 12;
  record.horizon_steps = 30;
  record.has_binding = true;
  record.shuffle_origin = 0x1234;
  record.train_samples = 100;
  record.val_samples = 11;
  record.has_val_initial = true;
  record.has_accuracy = true;
  record.val_initial_loss = 0.75f;
  record.val_initial_accuracy = 0.5f;
  record.has_best = true;
  record.best_step = 9;
  record.best_loss = 0.25f;
  record.best_accuracy = 0.875f;
  record.stale_evals = 3;
  record.best_payload = best;
  ASSERT_OK(SaveCheckpointRecord(dir.File("v5.ckpt"), kPlanHash, record,
                                 seg.data(), kBytes));

  std::vector<uint8_t> dst(kBytes, 0xFF);
  ASSERT_OK_AND_ASSIGN(
      CheckpointRecord got,
      LoadCheckpointRecord(dir.File("v5.ckpt"), kPlanHash, kBytes, dst.data()));
  EXPECT_TRUE(dst == seg);
  EXPECT_EQ(got.step, 12u);
  EXPECT_EQ(got.horizon_steps, 30u);
  EXPECT_TRUE(got.has_binding);
  EXPECT_EQ(got.shuffle_origin, 0x1234u);
  EXPECT_EQ(got.train_samples, 100u);
  EXPECT_EQ(got.val_samples, 11u);
  EXPECT_TRUE(got.has_val_initial);
  EXPECT_TRUE(got.has_accuracy);
  EXPECT_EQ(got.val_initial_loss, 0.75f);
  EXPECT_EQ(got.val_initial_accuracy, 0.5f);
  EXPECT_TRUE(got.has_best);
  EXPECT_EQ(got.best_step, 9u);
  EXPECT_EQ(got.best_loss, 0.25f);
  EXPECT_EQ(got.best_accuracy, 0.875f);
  EXPECT_EQ(got.stale_evals, 3u);
  EXPECT_TRUE(got.best_payload == best);
  EXPECT_EQ(got.plan_hash, kPlanHash);

  ASSERT_OK_AND_ASSIGN(CheckpointRecord peek,
                       PeekCheckpointRecord(dir.File("v5.ckpt")));
  // The plan the file is bound to: a driver tells a checkpoint the engine
  // will resume from one it will refuse as another plan's.
  EXPECT_EQ(peek.plan_hash, kPlanHash);
  EXPECT_TRUE(peek.has_binding);
  EXPECT_EQ(peek.shuffle_origin, 0x1234u);
  EXPECT_EQ(peek.train_samples, 100u);
  EXPECT_EQ(peek.val_samples, 11u);
  EXPECT_EQ(peek.step, 12u);

  // A flipped byte in the BEST payload is corruption too — and it is
  // caught before the main payload reaches the arena.
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> raw,
                       ReadFileBytes(dir.File("v5.ckpt")));
  EXPECT_EQ(raw.size(), 40u + 8 + 64 + 2 * kBytes);
  raw.back() ^= 0x01;
  {
    std::ofstream f(dir.File("bad-best.ckpt"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size()));
  }
  std::fill(dst.begin(), dst.end(), 0xFF);
  EXPECT_ERROR(LoadCheckpointRecord(dir.File("bad-best.ckpt"), kPlanHash,
                                    kBytes, dst.data()));
  EXPECT_TRUE(std::all_of(dst.begin(), dst.end(),
                          [](uint8_t b) { return b == 0xFF; }));

  // Without a best payload the file is header + one segment, and the
  // v3/v4-shaped call reads it (step and horizon) as before.
  record.best_payload.clear();
  ASSERT_OK(SaveCheckpointRecord(dir.File("lean.ckpt"), kPlanHash, record,
                                 seg.data(), kBytes));
  ASSERT_OK_AND_ASSIGN(raw, ReadFileBytes(dir.File("lean.ckpt")));
  EXPECT_EQ(raw.size(), 40u + 8 + 64 + kBytes);
  uint64_t horizon = 0;
  ASSERT_OK_AND_ASSIGN(uint64_t step,
                       LoadCheckpointFile(dir.File("lean.ckpt"), kPlanHash,
                                          kBytes, dst.data(), &horizon));
  EXPECT_EQ(step, 12u);
  EXPECT_EQ(horizon, 30u);
  EXPECT_TRUE(dst == seg);
}

TEST(DurableIo, AWriteReportsWhatItCostOnDisk) {
  // CR4 #145: the storage role's screen line reads these — the bytes a
  // durable write put on disk, and the fsyncs that succeeded: the file's,
  // then its directory's, which a filesystem may refuse.
  ScopedTempDir dir;
  const std::vector<uint8_t> head{1, 2, 3};
  const std::vector<uint8_t> body(1000, 7);
  const ByteSpan parts[] = {{head.data(), head.size()},
                            {body.data(), body.size()}};
  DurableWriteStats stats;
  ASSERT_OK(WriteFileDurable(dir.File("x.bin"),
                             std::span<const ByteSpan>(parts), &stats));
  EXPECT_EQ(stats.bytes, uint64_t{1003});
  EXPECT_EQ(stats.bytes,
            static_cast<uint64_t>(std::filesystem::file_size(dir.File("x.bin"))));
  EXPECT_GE(stats.fsyncs, uint64_t{1});
  EXPECT_LE(stats.fsyncs, uint64_t{2});
  // A checkpoint's cost is the file it wrote, header and payload.
  const std::vector<uint8_t> segment(256, 3);
  CheckpointRecord record;
  record.step = 5;
  DurableWriteStats saved;
  ASSERT_OK(SaveCheckpointRecord(dir.File("c.ckpt"), 77, record,
                                 segment.data(), segment.size(), &saved));
  EXPECT_EQ(saved.bytes, static_cast<uint64_t>(
                             std::filesystem::file_size(dir.File("c.ckpt"))));
  EXPECT_GT(saved.bytes, uint64_t{segment.size()});
  EXPECT_GE(saved.fsyncs, uint64_t{1});
}

TEST(DurableIo, AFailedWriteLeavesNoSidecarAndTheOldFile) {
  // CR4 #145 F08: every failure path removes the partial temp file and the
  // destination keeps its previous bytes. The write fault is injected (a
  // full disk, ENOSPC) — the one failure a test can ask for by name.
  ScopedTempDir dir;
  const std::string path = dir.File("state.bin");
  const std::vector<uint8_t> old = {1, 2, 3, 4};
  ASSERT_OK(WriteFileDurable(path, old.data(), old.size()));
  const std::vector<uint8_t> next(4096, 0xAB);
  FailNextDurableWriteForTesting(ENOSPC);
  EXPECT_ERROR_CONTAINS(WriteFileDurable(path, next.data(), next.size()),
                        "short write");
  ASSERT_OK_AND_ASSIGN(auto bytes, ReadFileBytes(path));
  EXPECT_TRUE(bytes == old);
  size_t entries = 0;
  for (const auto& e : std::filesystem::directory_iterator(dir.path())) {
    ++entries;
    EXPECT_EQ(e.path().filename().string(), std::string("state.bin"));
  }
  EXPECT_EQ(entries, 1u);
  // Disarmed by its one use: the next write succeeds and replaces the file.
  ASSERT_OK(WriteFileDurable(path, next.data(), next.size()));
  ASSERT_OK_AND_ASSIGN(auto replaced, ReadFileBytes(path));
  EXPECT_TRUE(replaced == next);
}

#ifndef _WIN32
// The entries of `dir` whose names begin with `prefix`, and their sizes.
std::vector<uint64_t> SizesOf(const std::filesystem::path& dir,
                              const std::string& prefix) {
  std::vector<uint64_t> out;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().filename().string().rfind(prefix, 0) == 0)
      out.push_back(static_cast<uint64_t>(e.file_size()));
  return out;
}

TEST(DurableIo, AWriteThatFailsMidwayLeavesNoSidecarAndTheOldFile) {
  // CR4 #145: F08's injection fails the first write; here the disk fills
  // partway through the file. The child's files may not grow past 16 KiB
  // of the 64 KiB being written, and the kernel fails the write that
  // crosses (EFBIG). The same holds: the old file, and nothing beside it.
  using seeml::testing::PastTheLimit;
  ScopedTempDir dir;
  const std::string path = dir.File("state.bin");
  const std::vector<uint8_t> old = {1, 2, 3, 4};
  ASSERT_OK(WriteFileDurable(path, old.data(), old.size()));
  const std::vector<uint8_t> next(64 << 10, 0xCD);
  const int status = seeml::testing::RunUnderFileSizeLimit(
      16 << 10, PastTheLimit::kFail, [&] {
        auto r = WriteFileDurable(path, next.data(), next.size());
        if (r) return 1;
        if (r.error().find("short write") == std::string::npos) return 2;
        auto bytes = ReadFileBytes(path);
        if (!bytes || *bytes != old) return 3;
        if (SizesOf(dir.path(), "") != std::vector<uint64_t>{old.size()})
          return 4;
        return 0;
      });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(Checkpoint, AWriterKilledMidSaveLeavesTheLastCheckpointWhole) {
  // CR4 #145, kill mid-save: the process dies partway through a checkpoint
  // (SIGXFSZ's default action at the child's 64 KiB file-size limit). The
  // rename never ran, so the destination is the last checkpoint, whole —
  // it loads, with its step and its bytes — and the next save goes
  // through. The dead writer's sidecar stays behind, short of the full
  // file: the inert leftover the durable writer documents (a pid-unique
  // name nothing reads), and the proof the writer died mid-file.
  using seeml::testing::PastTheLimit;
  ScopedTempDir dir;
  const std::string path = dir.File("run.ckpt");
  const uint64_t hash = 0x5EEC;
  const std::vector<uint8_t> last(4096, 1);
  ASSERT_OK(SaveCheckpointFile(path, hash, 3, last.data(), last.size()));
  const std::vector<uint8_t> next(1 << 20, 2);
  const int status = seeml::testing::RunUnderFileSizeLimit(
      64 << 10, PastTheLimit::kKill, [&] {
        (void)SaveCheckpointFile(path, hash, 9, next.data(), next.size());
        return 0;  // reached only if the writer survived
      });
  ASSERT_TRUE(WIFSIGNALED(status));
  EXPECT_EQ(WTERMSIG(status), SIGXFSZ);
  std::vector<uint8_t> back(last.size());
  ASSERT_OK_AND_ASSIGN(uint64_t step, LoadCheckpointFile(path, hash,
                                                         last.size(),
                                                         back.data()));
  EXPECT_EQ(step, uint64_t{3});
  EXPECT_TRUE(back == last);
  const std::vector<uint64_t> sidecars = SizesOf(dir.path(), "run.ckpt.tmp.");
  ASSERT_EQ(sidecars.size(), size_t{1});
  EXPECT_LE(sidecars[0], uint64_t{64 << 10});
  ASSERT_OK(SaveCheckpointFile(path, hash, 9, next.data(), next.size()));
  std::vector<uint8_t> now(next.size());
  ASSERT_OK_AND_ASSIGN(uint64_t step2, LoadCheckpointFile(path, hash,
                                                          next.size(),
                                                          now.data()));
  EXPECT_EQ(step2, uint64_t{9});
  EXPECT_TRUE(now == next);
}
#endif  // _WIN32

TEST(Checkpoint, TheBindingTravelsOnlyWhenSaidTo) {
  // CR4 #145 F71: has_binding was never written, and every v5 file read as
  // bound — one saved through the v3/v4-shaped call (zero binding fields)
  // made a resume refuse any real seed or split. The bit round-trips both
  // ways, through the load and the peek.
  ScopedTempDir dir;
  const std::vector<uint8_t> seg = Segment();
  ASSERT_OK(SaveCheckpointFile(dir.File("plain.ckpt"), kPlanHash, 3,
                               seg.data(), kBytes, 10));
  std::vector<uint8_t> dst(kBytes, 0);
  ASSERT_OK_AND_ASSIGN(CheckpointRecord plain,
                       LoadCheckpointRecord(dir.File("plain.ckpt"), kPlanHash,
                                            kBytes, dst.data()));
  EXPECT_FALSE(plain.has_binding);
  ASSERT_OK_AND_ASSIGN(CheckpointRecord plain_peek,
                       PeekCheckpointRecord(dir.File("plain.ckpt")));
  EXPECT_FALSE(plain_peek.has_binding);
  CheckpointRecord bound;
  bound.step = 3;
  bound.has_binding = true;
  bound.shuffle_origin = 77;
  bound.train_samples = 90;
  bound.val_samples = 10;
  ASSERT_OK(SaveCheckpointRecord(dir.File("bound.ckpt"), kPlanHash, bound,
                                 seg.data(), kBytes));
  ASSERT_OK_AND_ASSIGN(CheckpointRecord got,
                       LoadCheckpointRecord(dir.File("bound.ckpt"), kPlanHash,
                                            kBytes, dst.data()));
  EXPECT_TRUE(got.has_binding);
  EXPECT_EQ(got.shuffle_origin, 77u);
  ASSERT_OK_AND_ASSIGN(CheckpointRecord peek,
                       PeekCheckpointRecord(dir.File("bound.ckpt")));
  EXPECT_TRUE(peek.has_binding);
  // The fields without the flag are not a binding.
  CheckpointRecord unsaid = bound;
  unsaid.has_binding = false;
  ASSERT_OK(SaveCheckpointRecord(dir.File("unsaid.ckpt"), kPlanHash, unsaid,
                                 seg.data(), kBytes));
  ASSERT_OK_AND_ASSIGN(CheckpointRecord u,
                       PeekCheckpointRecord(dir.File("unsaid.ckpt")));
  EXPECT_FALSE(u.has_binding);
}

}  // namespace
