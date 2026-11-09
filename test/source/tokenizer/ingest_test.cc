// =============================================================================
// The training-data ingress (S2 #148): the sanitizer's counts and removals,
// the .jsonl document reader's refusals by line, and the pipeline on a
// hand-sized corpus — drops by index and reason, the EOS joins, the
// window cut, the straddle count, and the canonical SDS the runtime's
// loader reads back.
// =============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/pipeline/dataset.h"
#include "source/tokenizer/byte_level_bpe.h"
#include "source/tokenizer/text_ingest.h"
#include "test/framework/seetest.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::tokenizer;

/// A byte-vocabulary section (id = byte value) with one special EOS token.
std::vector<uint8_t> ByteSectionWithEos() {
  std::vector<uint8_t> b;
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u16 = [&](uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  const std::string eos = "<eos>";
  u32(0x4B544D53);
  u32(1);    // version
  u32(1);    // byte-level BPE
  u32(2);    // Digits then GPT-2
  u32(0);    // flags
  u32(257);  // vocab: 256 bytes + EOS
  u32(0);    // merges
  u32(1);    // specials
  u32(256);  // eos
  u32(static_cast<uint32_t>(-1));
  u32(static_cast<uint32_t>(-1));
  u32(static_cast<uint32_t>(-1));
  u32(0);    // no chat template
  for (int v = 0; v < 256; ++v) {
    u16(1);
    b.push_back(static_cast<uint8_t>(v));
  }
  u16(static_cast<uint16_t>(eos.size()));
  b.insert(b.end(), eos.begin(), eos.end());
  u32(256);  // the special
  return b;
}

ByteLevelBpe Tokenizer() {
  static const std::vector<uint8_t> section = ByteSectionWithEos();
  auto t = ByteLevelBpe::Load(section);
  if (!t) ADD_FAILURE(t.error());
  return std::move(*t);
}

TEST(Ingest, TheSanitizerRemovesAndCountsOrOnlyCounts) {
  const std::string text = std::string("a\x01" "b") + "\xff" + "c" +
                           "\xe2\x80\xae" + "d\tn";
  SanitizeCounts c;
  EXPECT_EQ(Sanitize(text, false, &c), "abcd\tn");
  EXPECT_EQ(c.control_chars, 1u);
  EXPECT_EQ(c.invalid_utf8_bytes, 1u);
  EXPECT_EQ(c.bidi_controls, 1u);
  SanitizeCounts k;
  EXPECT_EQ(Sanitize(text, true, &k), text);
  EXPECT_EQ(k.control_chars, 1u);
  EXPECT_EQ(k.invalid_utf8_bytes, 1u);
  EXPECT_EQ(k.bidi_controls, 1u);
  // Overlong and surrogate encodings are invalid bytes, not characters;
  // DEL and C1 controls are controls; tab / newline / CR stay.
  SanitizeCounts o;
  EXPECT_EQ(Sanitize(std::string("\xc0\x80") + "\xed\xa0\x80" + "\x7f" +
                         "\xc2\x85" + "\r\n",
                     false, &o),
            "\r\n");
  EXPECT_EQ(o.invalid_utf8_bytes, 5u);
  EXPECT_EQ(o.control_chars, 2u);
}

TEST(Ingest, JsonlDocumentsAreTextFieldsAndEverythingElseIsRefusedByLine) {
  auto docs = DocumentsFromJsonl(
      "{\"text\": \"hello\\nworld\", \"id\": 3, \"ok\": true}\n"
      "\n"
      "  {\"source\":\"x\",\"text\":\"caf\\u00e9 \\ud83d\\ude00\"}\n");
  ASSERT_OK(docs);
  ASSERT_EQ(docs->size(), 2u);
  EXPECT_EQ((*docs)[0], "hello\nworld");
  EXPECT_EQ((*docs)[1], std::string("caf\xc3\xa9 \xf0\x9f\x98\x80"));
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"id\": 1}\n"),
                        "line 1: no \"text\" field");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\"}\nx\n"),
                        "line 2: not a JSON object");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": {\"a\": 1}}\n"),
                        "line 1: nested values");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\", \"text\": \"b\"}\n"),
                        "line 1: two \"text\" fields");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\\q\"}\n"),
                        "line 1: malformed string value");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\"\n"),
                        "line 1: truncated object");
}

TEST(Ingest, ThePipelineDropsJoinsCutsAndCountsStraddles) {
  const ByteLevelBpe tok = Tokenizer();
  IngestOptions o;
  o.seq_len = 3;  // records of 4 ids
  // a b c d | <eos> e f g | h <eos> i j   (12 ids, 3 windows, 0 remainder)
  const std::vector<std::string> docs = {"abcd", "abcd", "", "efgh", "ij"};
  auto r = Ingest(tok, 0x1234, docs, o);
  ASSERT_OK(r);
  const IngestManifest& m = r->manifest;
  EXPECT_EQ(m.documents_in, 5u);
  EXPECT_EQ(m.documents_kept, 3u);
  EXPECT_EQ(m.dropped_duplicate, 1u);
  EXPECT_EQ(m.dropped_empty, 1u);
  EXPECT_EQ(m.dropped_degenerate, 0u);
  ASSERT_EQ(m.drops.size(), 2u);
  EXPECT_EQ(m.drops[0].index, 1u);
  EXPECT_EQ(m.drops[0].reason, "duplicate of 0");
  EXPECT_EQ(m.drops[1].index, 2u);
  EXPECT_EQ(m.drops[1].reason, "empty");
  EXPECT_EQ(m.tokens, 12u);
  EXPECT_EQ(m.windows, 3u);
  EXPECT_EQ(m.remainder_tokens, 0u);
  // Window 1 begins with the EOS (a new document, not a straddle); window
  // 2 holds one inside it.
  EXPECT_EQ(m.straddling_windows, 1u);
  EXPECT_EQ(m.distinct_ids, 11u);
  EXPECT_EQ(r->tokenizer_hash, 0x1234u);
  EXPECT_NE(r->corpus_hash, 0u);
  // The canonical SDS: v3 (it names its tokenizer), 64-byte header, the
  // 12 ids, nothing else — and the runtime's loader reads it back.
  EXPECT_EQ(r->sds.size(), 64u + 12 * 4);
  uint32_t version = 0;
  std::memcpy(&version, r->sds.data() + 4, 4);
  EXPECT_EQ(version, 3u);
  const std::string json = ManifestJson(m);
  EXPECT_STR_CONTAINS(json, "\"dropped_duplicate\": 1");
  EXPECT_STR_CONTAINS(json, "\"straddling_windows\": 1");
  EXPECT_STR_CONTAINS(json, "{\"index\": 1, \"reason\": \"duplicate of 0\"}");

  seeml::testing::ScopedTempDir dir;
  const std::string path = dir.File("corpus.sds");
  {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_TRUE(f != nullptr);
    std::fwrite(r->sds.data(), 1, r->sds.size(), f);
    std::fclose(f);
  }
  auto ds = seeml::update_rt::Dataset::LoadFromFile(path);
  ASSERT_OK(ds);
  EXPECT_EQ(ds->num_samples(), 3u);
  EXPECT_EQ(ds->input_kind(), 1u);
  EXPECT_EQ(ds->tokenizer_hash(), 0x1234u);
  EXPECT_FALSE(ds->masked());
  const std::array<int32_t, 12> expect = {'a', 'b', 'c', 'd', 256, 'e',
                                          'f', 'g', 'h', 256, 'i', 'j'};
  EXPECT_EQ(std::memcmp(r->sds.data() + 64, expect.data(), 48), 0);

  // Unbound (hash 0): v2, the 40-byte header; the same ids.
  auto r2 = Ingest(tok, 0, docs, o);
  ASSERT_OK(r2);
  EXPECT_EQ(r2->sds.size(), 40u + 12 * 4);
  EXPECT_EQ(std::memcmp(r2->sds.data() + 40, expect.data(), 48), 0);

  // One-document mode: the whole file, the sanitizer only counting, no
  // EOS joins, no deduplication (the exporter's --text-corpus path).
  IngestOptions one;
  one.seq_len = 2;
  one.one_document = true;
  auto r3 = Ingest(tok, 0x1234, {std::string("ab\x01" "cde")}, one);
  ASSERT_OK(r3);
  EXPECT_EQ(r3->manifest.tokens, 6u);
  EXPECT_EQ(r3->manifest.sanitized.control_chars, 1u);
  EXPECT_EQ(r3->manifest.windows, 2u);
  EXPECT_EQ(r3->manifest.drops.size(), 0u);

  // Refusals, by name.
  IngestOptions zero;
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, docs, zero), "seq_len is zero");
  IngestOptions wide;
  wide.seq_len = 100;
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, docs, wide),
                        "12 ids is fewer than one record of 101");
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, {}, o), "no documents");
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, {"a", "b"}, one),
                        "one-document mode takes one document");
}

}  // namespace
