#include "runtime/storage/checkpoint.h"

#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <vector>

#include "runtime/storage/checkpoint_format.h"
#include "runtime/storage/durable_io.h"
#include "runtime/diagnostics/persisting/error.h"
#include "source/identity/hash.h"

namespace seeml::update_rt {

namespace up = seeml::update;

namespace {

std::unexpected<std::string> Bad(const std::string& what) {
  return diag::persisting::Error(diag::persisting::kCheckpoint, what);
}

/// The v6 header as it lies in the file: the four parts, packed.
#pragma pack(push, 1)
struct FileHeader {
  CkptHeader h;
  CkptHeaderV4Tail v4;
  CkptHeaderV5Tail v5;
  CkptHeaderV6Tail v6;
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == kCkptHeaderBytes,
              "the v6 SEKP header is 240 bytes");
static_assert(sizeof(CheckpointRecord::last_losses) ==
                  kCkptWindowMax * sizeof(float),
              "the record's ring is the header's");

/// The integrity hash (CR4 #221): ContentHash64 over the whole header with
/// the hash field zero. The payload hashes are header fields, so this one
/// word chains the payloads in.
uint64_t HeaderHash(FileHeader header) {
  header.v6.header_hash = 0;
  return up::ContentHash64(reinterpret_cast<const uint8_t*>(&header),
                           sizeof(header));
}

bool Finite(uint32_t f32_bits) {
  return std::isfinite(std::bit_cast<float>(f32_bits));
}

/// Every header field against its bound (CR4 #221), before any is used:
/// the field a refusal names, and why; nullopt when all hold. What depends
/// on the run — the plan, the corpus, the backend, the patience — is the
/// loader's and the engine's to hold; these are the rules of the file
/// alone, and a save holds its own header to them too, so the writer
/// cannot produce a file its reader refuses.
std::optional<std::string> OutOfBounds(const FileHeader& f) {
  const uint32_t flags = f.v5.flags;
  auto field = [](const char* name, const std::string& why) {
    return std::optional<std::string>("checkpoint field '" + std::string(name) +
                                      "' is out of bounds: " + why);
  };
  if ((flags & ~kCkptKnownFlags) != 0)
    return field("flags", "a bit outside the known set");
  if ((flags & kCkptHasBestPayload) != 0 && (flags & kCkptHasBest) == 0)
    return field("flags", "a best payload without a best state");
  if ((flags & kCkptHasAccuracy) != 0 && (flags & kCkptHasValInitial) == 0)
    return field("flags", "an accuracy without a validation score");
  // A horizon of 0 is a file saved outside any run (no schedule recorded);
  // a run's checkpoint never stands past its own horizon.
  if (f.v4.horizon_steps != 0 && f.h.step > f.v4.horizon_steps)
    return field("step", "step " + std::to_string(f.h.step) +
                             " is past the run's horizon " +
                             std::to_string(f.v4.horizon_steps));
  if ((flags & kCkptHasValInitial) != 0) {
    if (!Finite(f.v5.val_initial_loss_bits))
      return field("val_initial_loss", "not finite");
    if (!Finite(f.v5.val_initial_accuracy_bits))
      return field("val_initial_accuracy", "not finite");
  }
  if ((flags & kCkptHasBest) != 0) {
    if (f.v5.best_step > f.h.step)
      return field("best_step", "the best state is later than the step");
    // The best payload rides along exactly when the best state is not the
    // main payload (a save at a new best writes it once).
    if (((flags & kCkptHasBestPayload) != 0) != (f.v5.best_step != f.h.step))
      return field("best_step",
                   (flags & kCkptHasBestPayload) != 0
                       ? "a best payload at the step the main payload holds"
                       : "a best state at another step with no best payload");
    if (!Finite(f.v5.best_loss_bits)) return field("best_loss", "not finite");
    if (!Finite(f.v5.best_accuracy_bits))
      return field("best_accuracy", "not finite");
  }
  // One evaluation per step at most, so the count cannot pass the step.
  if (f.v5.stale_evals > f.h.step)
    return field("stale_evals", "more stale evaluations than steps");
  if (f.v6.backend > kCkptBackendMetal)
    return field("backend", "not a known backend");
  if ((flags & kCkptHasWindows) != 0) {
    if (f.v6.window == 0 || f.v6.window > kCkptWindowMax)
      return field("window", "not 1.." + std::to_string(kCkptWindowMax));
    if (f.v6.window_count > f.h.step)
      return field("window_count", "more training losses than steps");
    if (!std::isfinite(std::bit_cast<double>(f.v6.first_window_sum_bits)))
      return field("first_window_sum", "not finite");
    for (const uint32_t bits : f.v6.last_losses_bits)
      if (!Finite(bits)) return field("last_losses", "not finite");
  } else if (f.v6.window != 0 || f.v6.window_count != 0) {
    return field("window", "window state without the flag that carries it");
  }
  return std::nullopt;
}

/// Reads and verifies the header: magic and version, then the integrity
/// hash — before any other field is looked at — then the field bounds and
/// the file's length, which must be exactly the header and its payload(s).
std::expected<FileHeader, std::string> ReadHeader(std::ifstream& f,
                                                  const std::string& path) {
  FileHeader header;
  constexpr size_t kPrefix = sizeof(header.h.magic) + sizeof(header.h.version);
  f.read(reinterpret_cast<char*>(&header), kPrefix);
  if (!f || header.h.magic != kCkptMagic)
    return Bad("not a checkpoint: '" + path + "'");
  if (header.h.version != kCkptVersion) {
    const std::string versions = "checkpoint '" + path + "' is format v" +
                                 std::to_string(header.h.version) +
                                 "; this runtime reads v" +
                                 std::to_string(kCkptVersion);
    if (header.h.version > kCkptVersion)
      return Bad(versions + " — it was written by a newer runtime");
    return Bad(versions +
               " — an older checkpoint cannot carry the corpus, split and "
               "header bindings a resume is held to; it is refused, not "
               "restarted over: run again without --resume (or remove the "
               "file) to start the run afresh");
  }
  f.read(reinterpret_cast<char*>(&header) + kPrefix,
         static_cast<std::streamsize>(sizeof(header) - kPrefix));
  if (!f) return Bad("truncated checkpoint (the header is cut short)");
  if (HeaderHash(header) != header.v6.header_hash)
    return Bad(
        "checkpoint header is corrupt (its integrity hash does not match)");
  if (auto why = OutOfBounds(header)) return Bad(*why);
  // The length the header states, against the file's: header + payload,
  // plus the best payload when one is flagged.
  const uint64_t payloads =
      (header.v5.flags & kCkptHasBestPayload) != 0 ? 2 : 1;
  uint64_t want = 0;
  if (__builtin_mul_overflow(header.h.persistent_size, payloads, &want) ||
      __builtin_add_overflow(want, kCkptHeaderBytes, &want))
    return Bad(
        "checkpoint field 'persistent_size' is out of bounds: the payload "
        "length overflows");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0) return Bad("cannot size checkpoint '" + path + "'");
  if (static_cast<uint64_t>(end) != want)
    return Bad(std::string(static_cast<uint64_t>(end) < want
                               ? "truncated checkpoint"
                               : "checkpoint with trailing bytes") +
               " (" + std::to_string(end) + " bytes; its header states " +
               std::to_string(want) + ")");
  f.seekg(static_cast<std::streamoff>(kCkptHeaderBytes));
  return header;
}

/// Everything the header carries, as a record (no payload).
CheckpointRecord RecordOf(const FileHeader& f) {
  const uint32_t flags = f.v5.flags;
  CheckpointRecord record;
  record.step = f.h.step;
  record.horizon_steps = f.v4.horizon_steps;
  record.plan_hash = f.h.plan_hash;
  record.has_binding = (flags & kCkptHasBinding) != 0;
  record.shuffle_origin = f.v5.shuffle_origin;
  record.train_samples = f.v5.train_samples;
  record.val_samples = f.v5.val_samples;
  record.has_val_initial = (flags & kCkptHasValInitial) != 0;
  record.has_accuracy = (flags & kCkptHasAccuracy) != 0;
  record.val_initial_loss = std::bit_cast<float>(f.v5.val_initial_loss_bits);
  record.val_initial_accuracy =
      std::bit_cast<float>(f.v5.val_initial_accuracy_bits);
  record.best_step = f.v5.best_step;
  record.best_loss = std::bit_cast<float>(f.v5.best_loss_bits);
  record.best_accuracy = std::bit_cast<float>(f.v5.best_accuracy_bits);
  record.stale_evals = f.v5.stale_evals;
  record.has_best = (flags & kCkptHasBest) != 0;
  record.corpus_hash = f.v6.corpus_hash;
  record.split_hash = f.v6.split_hash;
  record.backend = f.v6.backend;
  record.has_windows = (flags & kCkptHasWindows) != 0;
  record.window = f.v6.window;
  record.window_count = f.v6.window_count;
  record.first_window_sum = std::bit_cast<double>(f.v6.first_window_sum_bits);
  for (uint32_t i = 0; i < kCkptWindowMax; ++i)
    record.last_losses[i] = std::bit_cast<float>(f.v6.last_losses_bits[i]);
  return record;
}

}  // namespace

bool CheckpointExists(const std::string& path) {
  errno = 0;
  if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    return true;
  }
  return errno != ENOENT && errno != ENOTDIR;
}

std::expected<void, std::string> SaveCheckpointRecord(
    const std::string& path, uint64_t plan_hash, const CheckpointRecord& record,
    const uint8_t* persistent, uint64_t persistent_size,
    DurableWriteStats* stats) {
  FileHeader header;
  CkptHeader& h = header.h;
  CkptHeaderV5Tail& v5 = header.v5;
  CkptHeaderV6Tail& v6 = header.v6;
  h.plan_hash = plan_hash;
  h.step = record.step;
  h.persistent_size = persistent_size;
  h.payload_hash = up::ContentHash64(persistent, persistent_size);
  header.v4.horizon_steps = record.horizon_steps;
  v5.shuffle_origin = record.shuffle_origin;
  v5.train_samples = record.train_samples;
  v5.val_samples = record.val_samples;
  v5.best_step = record.best_step;
  v5.stale_evals = record.stale_evals;
  v5.val_initial_loss_bits = std::bit_cast<uint32_t>(record.val_initial_loss);
  v5.val_initial_accuracy_bits =
      std::bit_cast<uint32_t>(record.val_initial_accuracy);
  v5.best_loss_bits = std::bit_cast<uint32_t>(record.best_loss);
  v5.best_accuracy_bits = std::bit_cast<uint32_t>(record.best_accuracy);
  if (record.has_val_initial) v5.flags |= kCkptHasValInitial;
  if (record.has_accuracy) v5.flags |= kCkptHasAccuracy;
  if (record.has_best) v5.flags |= kCkptHasBest;
  if (record.has_binding) v5.flags |= kCkptHasBinding;
  const bool best = record.has_best && !record.best_payload.empty();
  if (best) {
    if (record.best_payload.size() != persistent_size)
      return Bad("best payload is not the persistent segment's size");
    v5.flags |= kCkptHasBestPayload;
    v5.best_payload_hash =
        up::ContentHash64(record.best_payload.data(), persistent_size);
  }
  v6.corpus_hash = record.corpus_hash;
  v6.split_hash = record.split_hash;
  v6.backend = record.backend;
  if (record.has_windows) {
    v5.flags |= kCkptHasWindows;
    v6.window = record.window;
    v6.window_count = record.window_count;
    v6.first_window_sum_bits = std::bit_cast<uint64_t>(record.first_window_sum);
    for (uint32_t i = 0; i < kCkptWindowMax; ++i)
      v6.last_losses_bits[i] = std::bit_cast<uint32_t>(record.last_losses[i]);
  }
  // The writer holds its header to the reader's bounds: a file this call
  // writes is one the load accepts.
  if (auto why = OutOfBounds(header)) return Bad("refusing to save: " + *why);
  v6.header_hash = HeaderHash(header);

  // Durable: a checkpoint that can vanish in a power cut is not a checkpoint.
  // Gather-write the header and the segment(s) straight from where they
  // live — no concatenated staging blob, so periodic checkpointing costs no
  // extra allocation or copy of the (potentially large) state.
  // The sidecars a writer of this path left when it was killed mid-save
  // (CR4 #221) are a full checkpoint's size each; collected here, at the
  // next save of the same path, and only a dead process's.
  (void)CollectDeadSidecars(path);
  std::vector<ByteSpan> parts{
      ByteSpan{reinterpret_cast<const uint8_t*>(&header), sizeof(header)},
      ByteSpan{persistent, persistent_size}};
  if (best)
    parts.push_back(ByteSpan{record.best_payload.data(), persistent_size});
  return WriteFileDurable(path, std::span<const ByteSpan>(parts), stats);
}

std::expected<CheckpointRecord, std::string> LoadCheckpointRecord(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Bad("no checkpoint at '" + path + "'");
  auto header = ReadHeader(f, path);
  if (!header) return std::unexpected(header.error());
  const CkptHeader& h = header->h;
  const CkptHeaderV5Tail& v5 = header->v5;
  // Binding: a checkpoint carries adapter and optimizer state laid out by
  // one specific plan. Resuming it under any other plan is silent corruption.
  if (h.plan_hash != plan_hash)
    return Bad("checkpoint belongs to a different plan");
  if (h.persistent_size != persistent_size)
    return Bad("checkpoint incompatible with plan");
  std::vector<uint8_t> payload(h.persistent_size);
  f.read(reinterpret_cast<char*>(payload.data()),
         static_cast<std::streamsize>(payload.size()));
  if (!f) return Bad("truncated checkpoint");
  if (up::ContentHash64(payload.data(), payload.size()) != h.payload_hash)
    return Bad("checkpoint payload is corrupt");

  CheckpointRecord record = RecordOf(*header);
  if (v5.flags & kCkptHasBestPayload) {
    record.best_payload.resize(h.persistent_size);
    f.read(reinterpret_cast<char*>(record.best_payload.data()),
           static_cast<std::streamsize>(record.best_payload.size()));
    if (!f) return Bad("truncated checkpoint (best payload)");
    if (up::ContentHash64(record.best_payload.data(),
                          record.best_payload.size()) != v5.best_payload_hash)
      return Bad("checkpoint best payload is corrupt");
  }
  // Nothing reaches the arena until every check above has passed.
  std::memcpy(dst, payload.data(), payload.size());
  return record;
}

std::expected<CheckpointRecord, std::string> PeekCheckpointRecord(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Bad("no checkpoint at '" + path + "'");
  auto header = ReadHeader(f, path);
  if (!header) return std::unexpected(header.error());
  return RecordOf(*header);
}

std::expected<void, std::string> SaveCheckpointFile(const std::string& path,
                                                    uint64_t plan_hash,
                                                    uint64_t step,
                                                    const uint8_t* persistent,
                                                    uint64_t persistent_size,
                                                    uint64_t horizon_steps) {
  CheckpointRecord record;
  record.step = step;
  record.horizon_steps = horizon_steps;
  return SaveCheckpointRecord(path, plan_hash, record, persistent,
                              persistent_size);
}

std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst, uint64_t* horizon_steps) {
  auto record = LoadCheckpointRecord(path, plan_hash, persistent_size, dst);
  if (!record) return std::unexpected(record.error());
  if (horizon_steps) *horizon_steps = record->horizon_steps;
  return record->step;
}

}  // namespace seeml::update_rt
