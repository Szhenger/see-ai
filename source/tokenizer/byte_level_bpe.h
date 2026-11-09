#ifndef SEEML_SOURCE_TOKENIZER_BYTE_LEVEL_BPE_H_
#define SEEML_SOURCE_TOKENIZER_BYTE_LEVEL_BPE_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "source/tokenizer/tokenizer_format.h"

// =============================================================================
// The byte-level BPE encoder (S2 #148): raw bytes in, token ids out, from
// an SMTK section alone. It reproduces the reference (Hugging Face
// `tokenizers`) for the constructs the section may carry — the GPT-2 split
// pattern, optionally preceded by the Digits(individual) pre-tokenizer,
// special tokens split first by longest match, byte-level BPE with ranked
// merges — and is proven against it at export on a golden adversarial
// corpus (tool/export_model.py --tokenizer-parity, seeml-tokenize).
//
// A pure deterministic function of the section and the bytes: no locale, no
// allocation beyond the output, the same ids on the build host and on the
// device, which re-runs it over its own data. Standard library only.
// =============================================================================

namespace seeml::tokenizer {

struct TokenizerInfo {
  TokenizerKind kind = TokenizerKind::kByteLevelBpe;
  PreTokenizer pre_tokenizer = PreTokenizer::kGpt2;
  uint32_t vocab_size = 0;
  uint32_t num_merges = 0;
  uint32_t num_special = 0;
  int32_t eos_id = -1;
  int32_t bos_id = -1;
  int32_t unk_id = -1;
  int32_t pad_id = -1;
  std::string chat_template;  // Jinja source, verbatim; empty = none
};

class ByteLevelBpe {
 public:
  /// Decodes and validates an SMTK section: the magic and version, a kind
  /// and pre-tokenizer this encoder implements, no flag it does not, every
  /// merge over known ids whose result is the concatenation it names, every
  /// special id in range. Refuses by name.
  [[nodiscard]] static std::expected<ByteLevelBpe, std::string> Load(
      std::span<const uint8_t> section);

  const TokenizerInfo& info() const { return info_; }
  /// The token's bytes.
  std::string_view token(uint32_t id) const { return tokens_[id]; }
  /// Whether `id` is a special token.
  bool is_special(uint32_t id) const;

  /// Encodes `text` (bytes; valid UTF-8 for the pre-tokenizer's character
  /// classes to mean what they mean) and appends the ids to `out`. A byte
  /// with no single-byte token in the vocabulary — one that cannot occur in
  /// valid UTF-8, or a control character the sanitizer removes — is dropped,
  /// as the reference drops a character its vocabulary lacks (the parity
  /// corpus holds both to it); the ingest path counts such bytes first.
  [[nodiscard]] std::expected<void, std::string> Encode(
      std::string_view text, std::vector<int32_t>& out) const;

  /// The pre-tokenizer alone: the byte ranges of `text`'s pieces, in order
  /// (special tokens are pieces of their own). For tests and the screen.
  std::vector<std::pair<size_t, size_t>> PreTokenize(
      std::string_view text) const;

 private:
  ByteLevelBpe() = default;

  // The BPE over one pre-token's bytes.
  std::expected<void, std::string> EncodePiece(std::string_view piece,
                                               size_t offset,
                                               std::vector<int32_t>& out) const;
  // The GPT-2 pattern over `text`, pieces appended as (begin, end).
  void SplitGpt2(std::string_view text, size_t base,
                 std::vector<std::pair<size_t, size_t>>& pieces) const;

  TokenizerInfo info_;
  std::vector<std::string> tokens_;                     // id -> bytes
  std::unordered_map<std::string_view, uint32_t> ids_;  // bytes -> id
  int32_t byte_token_[256];                             // -1 = none
  // (a << 32 | b) -> rank << 32 | c
  std::unordered_map<uint64_t, uint64_t> merges_;
  std::vector<uint32_t> special_;                       // longest first
};

/// ContentHash64 of a section: the tokenizer identity (SDS v3, plan v22).
uint64_t SectionHash(std::span<const uint8_t> section);

}  // namespace seeml::tokenizer

#endif  // SEEML_SOURCE_TOKENIZER_BYTE_LEVEL_BPE_H_
