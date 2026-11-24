#ifndef SEEML_RUNTIME_STORAGE_CHECKPOINT_FORMAT_H_
#define SEEML_RUNTIME_STORAGE_CHECKPOINT_FORMAT_H_

#include <cstdint>

// =============================================================================
// The SEKP byte layout, in a header of its own (P6, #86) so the ABI manifest
// can publish it: until now these lived in checkpoint.cc's anonymous
// namespace, where no machine could check a second reader against them.
// The format is documented in checkpoint.h and docs/formats.md.
// =============================================================================

namespace seeml::update_rt {

inline constexpr uint32_t kCkptMagic = 0x504B4553;  // "SEKP"
// v3: payload_hash uses ContentHash64 — the deterministic parallel identity
// hash — instead of serial byte-at-a-time Fnv1a64. The persistent segment
// (parameters + AdamW moments) is the largest thing the storage role hashes,
// every checkpoint_every steps; the buffer is immutable for the duration of
// the call, so the chunked hash is race-free. v2 checkpoints are rejected
// by the version gate (resume restarts from the plan's initial state).
// v4: the header grows a trailing u64, the run's LR-schedule horizon (E8).
// Additive — v3 files are still read, their horizon 0 (the plan default).
// v5 (E9, #92): a second tail binds the run the segment belongs to — the
// shuffle stream and the train/validation split — records the source
// model's validation score so a resumed run's gate still measures the
// whole update against it, and carries the BEST evaluated state (its step,
// loss and a second persistent-segment payload) so the state that will be
// committed survives an interruption. v3/v4 files still load: no binding
// (nothing to refuse), no stored score, no best state.
// v6 (CR4 #221): the checkpoint is an input and is held as one. A third
// tail carries an integrity hash over the WHOLE header (checked before any
// other field is read), the identity of the corpus and of the split the
// run trained on, the backend whose bits the stored losses are, and the
// train-loss gate's windows (so a run without a validation split gates on
// the whole run across a resume). Every field is bounded before use. v3-v5
// files are REFUSED, by version: they cannot carry the bindings, and a
// resume that silently restarted instead would overwrite them.
inline constexpr uint32_t kCkptVersion = 6;
inline constexpr uint32_t kCkptOldestReadable = 6;

#pragma pack(push, 1)
struct CkptHeader {
  uint32_t magic = kCkptMagic;
  uint32_t version = kCkptVersion;
  uint64_t plan_hash = 0;
  uint64_t step = 0;
  uint64_t persistent_size = 0;
  uint64_t payload_hash = 0;
};
/// What v4 appends to the v3 header.
struct CkptHeaderV4Tail {
  uint64_t horizon_steps = 0;
};
/// What v5 appends after the v4 tail. Floats travel as their f32 bits.
/// The best payload (persistent_size bytes, hashed like the main one)
/// follows the main payload when kCkptHasBestPayload is set.
inline constexpr uint32_t kCkptHasValInitial = 1u << 0;
inline constexpr uint32_t kCkptHasAccuracy = 1u << 1;
inline constexpr uint32_t kCkptHasBestPayload = 1u << 2;
inline constexpr uint32_t kCkptHasBest = 1u << 3;  // best_* fields are live
// CR4 #145 F71: the run binding (shuffle_origin, train_samples,
// val_samples) is live. Before this bit every v5 file read as bound —
// one saved through the v3/v4-shaped call, whose binding fields are zero,
// made a resume refuse any real seed or split. A v5 file written before
// the bit existed reads as unbound.
inline constexpr uint32_t kCkptHasBinding = 1u << 4;
// v6: the train-loss windows of the v6 tail are live (a run trained).
inline constexpr uint32_t kCkptHasWindows = 1u << 5;
inline constexpr uint32_t kCkptKnownFlags = (1u << 6) - 1;
struct CkptHeaderV5Tail {
  uint64_t shuffle_origin = 0;  // Dataset::shuffle_origin(); 0 = unshuffled
  uint64_t train_samples = 0;   // the training set's size after the split
  uint64_t val_samples = 0;     // the validation set's size; 0 = no split
  uint64_t best_step = 0;       // step of the best payload (0 = the start)
  uint64_t best_payload_hash = 0;
  uint32_t flags = 0;
  uint32_t val_initial_loss_bits = 0;
  uint32_t val_initial_accuracy_bits = 0;
  uint32_t best_loss_bits = 0;
  uint32_t best_accuracy_bits = 0;
  uint32_t stale_evals = 0;  // evaluations since the best, for patience
};
/// What v6 appends after the v5 tail (CR4 #221).
/// The train-loss gate's windows: the gate without a validation split
/// compares the mean loss of the run's first `window` steps with the mean of
/// its last `window`. `window_count` losses have been seen; the first
/// window's running sum travels as f64 bits (the sum the uninterrupted run
/// holds, so the mean is its bits), the last window as a ring of f32 bits
/// indexed by (loss index % window).
inline constexpr uint32_t kCkptWindowMax = 20;
/// The backend the run executes on: the stored losses are that backend's
/// bits, and a resume re-measures them bit for bit. 0 = unstated (a file
/// saved outside any run).
inline constexpr uint32_t kCkptBackendUnstated = 0;
inline constexpr uint32_t kCkptBackendCpu = 1;
inline constexpr uint32_t kCkptBackendMetal = 2;
struct CkptHeaderV6Tail {
  uint64_t corpus_hash = 0;   // Dataset::content_hash(); 0 = an unnamed corpus
  uint64_t split_hash = 0;    // Dataset::split_hash(); 0 = no compiled split
  uint64_t window_count = 0;  // training losses the windows have seen
  uint64_t first_window_sum_bits = 0;  // f64: sum of the first window so far
  uint32_t backend = kCkptBackendUnstated;
  uint32_t window = 0;  // the window length, 1..kCkptWindowMax; 0 = none
  uint32_t last_losses_bits[kCkptWindowMax] = {};
  // ContentHash64 of the whole header — CkptHeader, the v4, v5 and v6 tails
  // — with this field zero. The payload hashes are header fields, so the one
  // hash chains the payloads in. Integrity, not authenticity: there is no
  // key, and whoever can rewrite the file can rewrite the hash.
  uint64_t header_hash = 0;
};
#pragma pack(pop)

/// The v6 header: the four parts as they lie in the file.
inline constexpr uint64_t kCkptHeaderBytes = 40 + 8 + 64 + 128;

static_assert(sizeof(CkptHeaderV6Tail) == 128, "v6 appends 128 bytes");
static_assert(sizeof(CkptHeader) == 40, "the v3 SEKP header is 40 bytes");
static_assert(sizeof(CkptHeaderV4Tail) == 8, "v4 appends one u64");
static_assert(sizeof(CkptHeaderV5Tail) == 64, "v5 appends 64 bytes");

}  // namespace seeml::update_rt

#endif  // SEEML_RUNTIME_STORAGE_CHECKPOINT_FORMAT_H_
