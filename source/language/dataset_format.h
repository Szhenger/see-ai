#ifndef SEEML_SOURCE_LANGUAGE_DATASET_FORMAT_H_
#define SEEML_SOURCE_LANGUAGE_DATASET_FORMAT_H_

#include <bit>
#include <cstddef>
#include <cstdint>

// =============================================================================
// SDS — SeeAI Dataset format: the on-disk constants (S2 #148 moved them here
// from runtime/pipeline/dataset.h, so the compiler's data path can write a
// canonical corpus with the one statement of the format the device reads).
//
// Layout (little-endian):
//   u32 magic "SDS1"; u32 version (1, 2 or 3)
//   u64 num_samples; u64 input_dim
//   u32 label_kind (0 = none, 1 = class index i32, 2 = dense f32)
//   u32 input_kind (v2: 0 = f32 feature rows, 1 = i32 token records;
//                   the word was header padding in v1, always 0)
//   u64 label_dim (dense labels only)
//   v3 appends: u64 tokenizer_hash (the SMF tokenizer section the ids came
//               from; 0 = the corpus names none); u32 flags (bit 0 = token
//               records carry a loss mask); u32 reserved (0); u64 reserved (0)
//   records[num_samples]:
//     input_kind 0:  f32 input[input_dim], then the label (i32 class,
//                    f32[label_dim] dense, nothing for kind 0)
//     input_kind 1:  i32 tokens[input_dim + 1] and NO stored label — the
//                    record is one sequence; inputs are tokens[0..S) and
//                    the class labels are the shifted view tokens[1..S],
//                    derived at serving time (label_kind must be 1); under
//                    the mask flag, then u8 mask[S] padded to 4 bytes (1 =
//                    the next token is a loss target, 0 = the feeder stages
//                    the label kSdsIgnoreLabel the masked loss ignores;
//                    every record keeps at least one target)
//   The sample section fills the file exactly; a writer emits the lowest
//   version that carries its content.
// =============================================================================

namespace seeml::update {

inline constexpr uint32_t kSdsMagic = 0x31534453;  // "SDS1" little-endian
inline constexpr uint32_t kSdsVersion = 3;
inline constexpr uint32_t kSdsMinVersion = 1;
inline constexpr uint64_t kSdsHeaderBytes = 40;  // v1 and v2
inline constexpr uint64_t kSdsHeaderBytesV3 = 64;
inline constexpr uint32_t kSdsFlagMasked = 1u << 0;
inline constexpr uint32_t kSdsFlagsMask = kSdsFlagMasked;
inline constexpr uint64_t SdsHeaderBytes(uint32_t version) {
  return version >= 3 ? kSdsHeaderBytesV3 : kSdsHeaderBytes;
}
inline constexpr uint64_t SdsMaskBytes(uint64_t seq) {
  return (seq + 3) & ~uint64_t{3};
}
// label_kind: 0 none, 1 class index (i32), 2 dense f32[label_dim].
inline constexpr uint32_t kSdsLabelKindMax = 2;
/// The staged label of a masked-out position.
inline constexpr int32_t kSdsIgnoreLabel = -1;

static_assert(std::endian::native == std::endian::little,
              "SDS serialization assumes a little-endian host.");

/// The version a token corpus is written at: v3 when it names its tokenizer
/// or carries a mask, else the v2 bytes every earlier writer produced.
inline constexpr uint32_t SdsTokenVersion(uint64_t tokenizer_hash,
                                          bool masked) {
  return (tokenizer_hash != 0 || masked) ? 3u : 2u;
}

/// Writes a token corpus — the header, then `records` records of `seq + 1`
/// ids each from `ids` and, when `masked`, each record's `seq` mask bytes
/// (from `mask`, records x seq) padded to 4 — through `put(const void*,
/// size_t)`: byte for byte the same whether `put` appends to a buffer (the
/// compiler's ingest, whose bytes the plan is pinned to) or writes a stream
/// (Dataset::SaveToFile). The one statement of the token record layout.
template <class Put>
void WriteSdsTokenCorpus(Put&& put, uint64_t records, uint64_t seq,
                         uint64_t tokenizer_hash, bool masked,
                         const int32_t* ids, const uint8_t* mask) {
  const uint32_t version = SdsTokenVersion(tokenizer_hash, masked);
  const uint32_t label_kind = 1, input_kind = 1, zero32 = 0;
  const uint32_t flags = masked ? kSdsFlagMasked : 0;
  const uint64_t label_dim = 0, zero64 = 0;
  put(&kSdsMagic, 4);
  put(&version, 4);
  put(&records, 8);
  put(&seq, 8);
  put(&label_kind, 4);
  put(&input_kind, 4);
  put(&label_dim, 8);
  if (version >= 3) {
    put(&tokenizer_hash, 8);
    put(&flags, 4);
    put(&zero32, 4);
    put(&zero64, 8);
  }
  const uint64_t rec = seq + 1;
  if (!masked) {
    // One contiguous i32 block; labels are derived, never stored.
    put(ids, static_cast<size_t>(records * rec * sizeof(int32_t)));
    return;
  }
  const uint8_t pad[4] = {0, 0, 0, 0};
  const uint64_t pad_bytes = SdsMaskBytes(seq) - seq;
  for (uint64_t r = 0; r < records; ++r) {
    put(ids + r * rec, static_cast<size_t>(rec * sizeof(int32_t)));
    put(mask + r * seq, static_cast<size_t>(seq));
    if (pad_bytes != 0) put(pad, static_cast<size_t>(pad_bytes));
  }
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_LANGUAGE_DATASET_FORMAT_H_
