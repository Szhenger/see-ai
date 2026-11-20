#ifndef SEEML_RUNTIME_STORAGE_CHECKPOINT_H_
#define SEEML_RUNTIME_STORAGE_CHECKPOINT_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "runtime/storage/durable_io.h"

// =============================================================================
// SEKP — the SeeAI checkpoint container.
//
// A checkpoint is the arena's persistent segment (LoRA adapters + optimizer
// moments) plus the training step, hash-bound to the exact plan that laid the
// segment out. Layout (little-endian), 240 header bytes then the payloads:
//   u32 magic "SEKP"; u32 version (6; older files are refused by version)
//   u64 plan_hash        must equal the plan's PlanHeader::plan_hash
//   u64 step             1-indexed AdamW timestep at save
//   u64 persistent_size  byte length of each payload
//   u64 payload_hash     ContentHash64 of the payload
//   u64 horizon_steps    the run's LR-schedule horizon (see below)
//   v5 tail              64 bytes: the run binding, the source model's
//                        validation score, the best state's step / score,
//                        the best payload's hash, the patience counter
//   v6 tail              128 bytes: the corpus and split identity, the
//                        backend, the train-loss windows, the header hash
//   payload              the arena's persistent segment
//   best payload         optional: the best evaluated segment
//
// The horizon (E8 #91) is the step count the interrupted run was annealing
// over. A resume with no explicit step count trains exactly the remainder
// to it, on the same schedule — so the resumed run's learning rates, and
// therefore its bits, equal the uninterrupted run's.
//
// The v5 tail (E9, #92) makes a resume honest in two more ways. The run
// binding — the shuffle stream's origin and the sizes of the two halves of
// the train/validation split — lets a resume refuse a different seed or
// split, which would otherwise silently move the boundary and train on the
// first run's validation rows. The stored validation score of the SOURCE
// model lets the gate keep comparing the whole update against it. And the
// best payload is the state the update will commit (the best evaluated one,
// not the last), so it must survive an interruption exactly as the last one
// does.
//
// A checkpoint is an INPUT (CR4 #221), held like every other: assumed
// malformed. It has no key, so nothing in it can be authenticated against
// someone who can rewrite the file; what can be done is done —
//   * the header hash covers every header byte (and, through the payload
//     hashes, the payloads) and is checked before any other field is read:
//     corruption and truncation are refusals;
//   * every field is held to a bound before use, the refusal naming it;
//   * the file is exactly header + payload(s): a short or a long one is
//     refused;
//   * the corpus and split identity, and the backend, ride along for the
//     engine to compare with the run's (UpdateEngine::Train), which also
//     RE-MEASURES the stored losses rather than trusting them.
//
// Saves are durable (fsync + atomic rename via durable_io) and collect the
// sidecars dead writers of the same path left behind; loads verify
// everything above before a single byte reaches the arena.
// =============================================================================

namespace seeml::update_rt {

/// Everything a checkpoint carries besides the segment itself.
struct CheckpointRecord {
  uint64_t step = 0;
  uint64_t horizon_steps = 0;
  // The plan the file is bound to, as a load or a peek read it. A save takes
  // the plan's hash as its own argument and does not read this.
  uint64_t plan_hash = 0;
  // v5 — absent (all zero / false) from a v3/v4 file.
  bool has_binding = false;
  uint64_t shuffle_origin = 0;
  uint64_t train_samples = 0;
  uint64_t val_samples = 0;
  bool has_val_initial = false;
  bool has_accuracy = false;
  float val_initial_loss = 0.0f;
  float val_initial_accuracy = 0.0f;
  bool has_best = false;  // best_* are live (a tracked run)
  uint64_t best_step = 0;
  float best_loss = 0.0f;
  float best_accuracy = 0.0f;
  uint32_t stale_evals = 0;
  // The best payload (persistent_size bytes) when the best state differs
  // from the main payload; empty when it is the main payload, or untracked.
  std::vector<uint8_t> best_payload;
  // v6 (CR4 #221). The corpus and the split the run trained on (live with
  // has_binding), the backend whose bits the losses are (kCkptBackend*), and
  // the train-loss gate's windows (checkpoint_format.h).
  uint64_t corpus_hash = 0;
  uint64_t split_hash = 0;
  uint32_t backend = 0;
  bool has_windows = false;
  uint32_t window = 0;
  uint64_t window_count = 0;
  double first_window_sum = 0.0;
  float last_losses[20] = {};
};

/// Durably writes a checkpoint: header + `persistent` payload, gather-written
/// straight from the arena, plus `record`'s best payload when it has one.
/// `stats`, when given, receives what the write cost on disk.
[[nodiscard]] std::expected<void, std::string> SaveCheckpointRecord(
    const std::string& path, uint64_t plan_hash, const CheckpointRecord& record,
    const uint8_t* persistent, uint64_t persistent_size,
    DurableWriteStats* stats = nullptr);

/// Whether there is a file at `path` to resume from. False only when the
/// path names nothing (ENOENT / ENOTDIR): a file that exists and cannot be
/// opened is still a checkpoint a strict resume must refuse, not "nothing
/// to resume".
[[nodiscard]] bool CheckpointExists(const std::string& path);

/// Loads and fully verifies a checkpoint bound to `plan_hash`, copying its
/// main payload (which must be exactly `persistent_size` bytes) into `dst`
/// and returning everything else it carries. `dst` is untouched on any
/// error. The order of the refusals: magic and version, the header hash,
/// the field bounds, the file's length, the plan binding, the payload
/// hashes.
[[nodiscard]] std::expected<CheckpointRecord, std::string> LoadCheckpointRecord(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst);

/// Reads only the header of a checkpoint (no payload is read or verified;
/// the header is, exactly as a load verifies it — version, hash, bounds,
/// length): what a driver needs to refuse a mismatched seed or split before
/// anything else happens — and the plan hash the file is bound to, so the
/// driver can refuse another plan's checkpoint by name. A file that is not
/// a v6 checkpoint is an error; one saved without a binding yields
/// has_binding == false. The record's best_payload is left empty.
[[nodiscard]] std::expected<CheckpointRecord, std::string> PeekCheckpointRecord(
    const std::string& path);

/// The v3/v4-shaped calls: a step, a horizon, one payload.
[[nodiscard]] std::expected<void, std::string> SaveCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t step,
    const uint8_t* persistent, uint64_t persistent_size,
    uint64_t horizon_steps = 0);
[[nodiscard]] std::expected<uint64_t, std::string> LoadCheckpointFile(
    const std::string& path, uint64_t plan_hash, uint64_t persistent_size,
    uint8_t* dst, uint64_t* horizon_steps = nullptr);

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_STORAGE_CHECKPOINT_H_
