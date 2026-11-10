#ifndef SEEML_SOURCE_TOKENIZER_TEXT_INGEST_H_
#define SEEML_SOURCE_TOKENIZER_TEXT_INGEST_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "source/tokenizer/byte_level_bpe.h"

// =============================================================================
// The training-data ingress (S2 #148, frontend/tokenization/README.md):
// raw text in, a canonical SDS corpus out, by pure deterministic stages the
// package vendors so the device's pipeline can re-run them on its own data:
//
//   documents   one per record: the whole file (a .txt is one document) or
//               one per line of a .jsonl ({"text": ...})
//   sanitize    invalid UTF-8 bytes, control characters other than tab,
//               newline and carriage return, and the bidirectional override
//               and isolate characters are removed; a document empty after
//               that is dropped. In one-document mode the sanitizer only
//               counts (the exporter's path encodes the bytes as they are,
//               and the two must agree id for id).
//   deduplicate exact, by the sanitized bytes' hash; later twins dropped
//   encode      the section's byte-level BPE
//   pack        documents joined by EOS (the section's eos_id), cut into
//               (S + 1)-id windows at stride S + 1, the remainder dropped;
//               never padded (every position is a loss target); windows
//               that straddle a document boundary are counted
//
// Every drop is recorded by index and reason in the manifest; the screen
// prints the counts. What this does not defend against, and says: well-
// formed poisoned data — the gate and a held-out set are that defense.
// =============================================================================

namespace seeml::tokenizer {

struct IngestOptions {
  uint64_t seq_len = 0;          // S: records are S + 1 ids
  bool one_document = false;     // a .txt: the file is one document, no EOS,
                                 // sanitizer counts only (bit-exact with the
                                 // exporter's --text-corpus)
};

/// One input document: plain text (every position a loss target), or a
/// prompt / response pair (S3 #149) the pipeline renders through the ChatML
/// template — `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
/// then `{response}<|im_end|>\n` — with the response side's ids the only
/// targets. The section must name both markers as special tokens.
struct Document {
  std::string text;      // the plain text, or the prompt
  std::string response;  // the response (chat documents)
  bool chat = false;
};

/// One document's sanitizer counts.
struct SanitizeCounts {
  uint64_t invalid_utf8_bytes = 0;
  uint64_t control_chars = 0;
  uint64_t bidi_controls = 0;
};

struct DropRecord {
  uint64_t index = 0;        // the document's index in the input
  std::string reason;        // "empty", "duplicate of N", "degenerate"
};

/// The manifest: what came in, what was kept, and why not.
struct IngestManifest {
  uint64_t documents_in = 0;
  uint64_t documents_kept = 0;
  uint64_t chat_documents = 0;      // prompt / response pairs among them
  uint64_t dropped_zero_target_windows = 0;  // masked windows with no target
  uint64_t target_tokens = 0;       // positions the loss counts (masked)
  uint64_t total_tokens = 0;        // positions in the records (windows x S)
  uint64_t dropped_empty = 0;
  uint64_t dropped_duplicate = 0;
  uint64_t dropped_degenerate = 0;  // fewer than two ids: no target
  SanitizeCounts sanitized;         // summed over the documents
  uint64_t tokens = 0;              // ids after packing's EOS joins
  uint64_t windows = 0;             // records written
  uint64_t remainder_tokens = 0;    // ids the last partial window dropped
  uint64_t straddling_windows = 0;  // windows holding a document boundary
  uint64_t distinct_ids = 0;
  std::vector<DropRecord> drops;
};

struct IngestResult {
  std::vector<uint8_t> sds;   // the canonical corpus (SDS v3 with the
                              // tokenizer hash, or v2 when unbound)
  std::vector<int32_t> ids;   // the packed ids the records hold (windows ×
                              // (S + 1)), for a caller that serves them
                              // without reading the SDS back
  std::vector<uint8_t> mask;  // windows × S target flags (masked corpora;
                              // empty when every position is a target)
  bool masked = false;
  IngestManifest manifest;
  uint64_t corpus_hash = 0;   // ContentHash64 of `sds`
  uint64_t tokenizer_hash = 0;
};

/// Removes what the sanitizer removes and counts it; `one_document` mode
/// counts without removing.
std::string Sanitize(std::string_view text, bool count_only,
                     SanitizeCounts* counts);

/// Splits a .jsonl file into its documents: each line a JSON object with a
/// "text" string, or a "prompt" and a "response" string (JSON string escapes
/// decoded); a line that is neither is refused by line number.
std::expected<std::vector<Document>, std::string> DocumentsFromJsonl(
    std::string_view jsonl);

/// Ingests `documents` through the stages above with `tokenizer`, whose
/// section hash is `tokenizer_hash` (0 = the corpus names none). Refuses a
/// multi-document corpus whose section names no EOS, a seq_len of zero, a
/// chat document under a section without the ChatML markers, and a corpus
/// that yields no complete window. A corpus with a chat document is
/// masked: the SDS carries the mask and windows with no target are dropped.
std::expected<IngestResult, std::string> Ingest(
    const ByteLevelBpe& tokenizer, uint64_t tokenizer_hash,
    const std::vector<Document>& documents, const IngestOptions& options);

/// The manifest as JSON (one object; the drops as an array).
std::string ManifestJson(const IngestManifest& m);

}  // namespace seeml::tokenizer

#endif  // SEEML_SOURCE_TOKENIZER_TEXT_INGEST_H_
