#ifndef SEEML_SOURCE_TOKENIZER_TOKENIZER_FORMAT_H_
#define SEEML_SOURCE_TOKENIZER_TOKENIZER_FORMAT_H_

#include <bit>
#include <cstdint>

// =============================================================================
// SMTK — the SMF tokenizer section (S2 #148). The bytes an SMF v7 file
// places at its tokenizer_offset: one self-describing blob the exporter
// writes from a checkpoint's own tokenizer files and the frontend's
// tokenizer is instantiated from — and from nothing else (no --tokenizer
// flag exists). Its ContentHash64 is the tokenizer identity every corpus is
// bound to (SDS v3 tokenizer_hash, plan v22 tokenizer_hash).
//
// Layout (little-endian):
//   u32 magic "SMTK"      u32 version (1)
//   u32 kind              1 = byte-level BPE (the GPT-2 family)
//   u32 pre_tokenizer     1 = the GPT-2 split pattern
//                         2 = Digits(individual) then the GPT-2 pattern
//   u32 flags             bit 0 = add_prefix_space (refused: unsupported)
//                         bit 1 = NFC normalizer (refused: unsupported)
//   u32 vocab_size        u32 num_merges        u32 num_special
//   i32 eos_id  i32 bos_id  i32 unk_id  i32 pad_id   (-1 = none)
//   u32 chat_template_len; bytes chat_template     (Jinja source, verbatim)
//   vocab[vocab_size]:    u16 len; bytes           the token's BYTES — the
//                         byte-level string mapped back through the GPT-2
//                         byte↔unicode table, so the core never sees it
//   merges[num_merges]:   u32 a, u32 b, u32 c      in rank order; c is the
//                         id whose bytes are a's followed by b's
//   special[num_special]: u32 id                   split before the pattern,
//                         longest match first
//
// The core's tokenizer (byte_level_bpe.h) decodes this and refuses, by
// name, every construct it does not implement; the exporter refuses to
// write any it would not read. Standard library only, like the rest of
// source/: the section is vendored with the package so the device's
// pipeline can re-run the encoder over its own data.
// =============================================================================

namespace seeml::tokenizer {

inline constexpr uint32_t kSmtkMagic = 0x4B544D53;  // "SMTK" little-endian
inline constexpr uint32_t kSmtkVersion = 1;

enum class TokenizerKind : uint32_t { kByteLevelBpe = 1 };
enum class PreTokenizer : uint32_t { kGpt2 = 1, kDigitsThenGpt2 = 2 };

inline constexpr uint32_t kSmtkFlagAddPrefixSpace = 1u << 0;
inline constexpr uint32_t kSmtkFlagNfc = 1u << 1;
inline constexpr uint32_t kSmtkFlagsKnown = kSmtkFlagAddPrefixSpace | kSmtkFlagNfc;

// The fixed-width head before the chat template: 7 u32 + 4 i32 + 1 u32.
inline constexpr uint64_t kSmtkHeadBytes = 4 * 12;

static_assert(std::endian::native == std::endian::little,
              "the tokenizer section is read by memcpy of host integers");

}  // namespace seeml::tokenizer

#endif  // SEEML_SOURCE_TOKENIZER_TOKENIZER_FORMAT_H_
