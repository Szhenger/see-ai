// =============================================================================
// The training-data ingress (S2 #148, S3 #149): the sanitizer's counts and
// removals, the .jsonl document reader's refusals by line, the pipeline on a
// hand-sized corpus — drops by index and reason, the EOS joins, the window
// cut, the straddle count, and the canonical SDS the runtime's loader reads
// back — and the prompt / response path: the ChatML render, the response-
// only mask, the zero-target windows dropped, the masked SDS.
// =============================================================================

#include <array>
#include <cstdint>
#include <cstdio>
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

/// A byte-vocabulary section (id = byte value) with the given special
/// tokens (ids 256, 257, ...); the first is the EOS.
std::vector<uint8_t> ByteSection(const std::vector<std::string>& specials) {
  std::vector<uint8_t> b;
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u16 = [&](uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  u32(0x4B544D53);
  u32(1);    // version
  u32(1);    // byte-level BPE
  u32(2);    // Digits then GPT-2
  u32(0);    // flags
  u32(static_cast<uint32_t>(256 + specials.size()));
  u32(0);    // merges
  u32(static_cast<uint32_t>(specials.size()));
  u32(256);  // eos: the first special
  u32(static_cast<uint32_t>(-1));
  u32(static_cast<uint32_t>(-1));
  u32(static_cast<uint32_t>(-1));
  u32(0);    // no chat template
  for (int v = 0; v < 256; ++v) {
    u16(1);
    b.push_back(static_cast<uint8_t>(v));
  }
  for (const std::string& s : specials) {
    u16(static_cast<uint16_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  }
  for (uint32_t i = 0; i < specials.size(); ++i) u32(256 + i);
  return b;
}

ByteLevelBpe Tokenizer(const std::vector<std::string>& specials = {"<eos>"}) {
  const std::vector<uint8_t> section = ByteSection(specials);
  auto t = ByteLevelBpe::Load(section);
  if (!t) ADD_FAILURE(t.error());
  return std::move(*t);
}

std::vector<Document> Plain(const std::vector<std::string>& texts) {
  std::vector<Document> docs;
  for (const std::string& t : texts) docs.push_back({t, {}, false});
  return docs;
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

TEST(Ingest, JsonlDocumentsAreTextOrPromptResponseAndTheRestIsRefusedByLine) {
  auto docs = DocumentsFromJsonl(
      "{\"text\": \"hello\\nworld\", \"id\": 3, \"ok\": true}\n"
      "\n"
      "  {\"source\":\"x\",\"text\":\"caf\\u00e9 \\ud83d\\ude00\"}\n"
      "{\"prompt\": \"2+2?\", \"response\": \"4\"}\n");
  ASSERT_OK(docs);
  ASSERT_EQ(docs->size(), 3u);
  EXPECT_EQ((*docs)[0].text, "hello\nworld");
  EXPECT_FALSE((*docs)[0].chat);
  EXPECT_EQ((*docs)[1].text, std::string("caf\xc3\xa9 \xf0\x9f\x98\x80"));
  EXPECT_TRUE((*docs)[2].chat);
  EXPECT_EQ((*docs)[2].text, "2+2?");
  EXPECT_EQ((*docs)[2].response, "4");
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
  // One record per line, members comma-separated: a second object on the
  // line, a stray comma or a missing comma is a refusal, never a record
  // that silently vanishes from the canonical corpus.
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\"} {\"text\": \"b\"}\n"),
                        "line 1: content after the object");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\"}, x\n"),
                        "line 1: content after the object");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\" \"x\": 1}\n"),
                        "line 1: expected ',' or '}'");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"text\": \"a\",}\n"),
                        "line 1: a ',' with no member after it");
  EXPECT_ERROR_CONTAINS(DocumentsFromJsonl("{\"prompt\": \"a\"}\n"),
                        "line 1: a prompt / response record needs both");
  EXPECT_ERROR_CONTAINS(
      DocumentsFromJsonl("{\"text\": \"a\", \"prompt\": \"b\", \"response\": \"c\"}\n"),
      "not both");
}

TEST(Ingest, ThePipelineDropsJoinsCutsAndCountsStraddles) {
  const ByteLevelBpe tok = Tokenizer();
  IngestOptions o;
  o.seq_len = 3;  // records of 4 ids
  // a b c d | <eos> e f g | h <eos> i j   (12 ids, 3 windows, 0 remainder)
  const std::vector<Document> docs = Plain({"abcd", "abcd", "", "efgh", "ij"});
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
  EXPECT_EQ(m.target_tokens, 9u);
  EXPECT_EQ(m.total_tokens, 9u);
  EXPECT_FALSE(r->masked);
  EXPECT_TRUE(r->mask.empty());
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
  auto r3 = Ingest(tok, 0x1234, Plain({std::string("ab\x01" "cde")}), one);
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
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, Plain({"a", "b"}), one),
                        "one-document mode takes one document");
}

TEST(Ingest, PromptResponseRecordsRenderChatMlAndMaskThePrompt) {
  // S3 #149: the response side's ids are the targets; a window whose
  // positions predict prompt ids alone is dropped; the SDS carries the mask.
  const ByteLevelBpe tok = Tokenizer({"<eos>", "<|im_start|>", "<|im_end|>"});
  IngestOptions o;
  o.seq_len = 4;  // records of 5 ids
  // Rendered: <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n (21 ids,
  // the prompt side) then yo<|im_end|>\n (4 ids, the response side): 25 ids,
  // 5 windows, of which the first four predict prompt ids only.
  std::vector<Document> docs = {{"hi", "yo", true}};
  auto r = Ingest(tok, 0x55, docs, o);
  ASSERT_OK(r);
  const IngestManifest& m = r->manifest;
  EXPECT_EQ(m.chat_documents, 1u);
  EXPECT_EQ(m.tokens, 25u);
  EXPECT_EQ(m.windows, 1u);
  EXPECT_EQ(m.dropped_zero_target_windows, 4u);
  EXPECT_EQ(m.target_tokens, 4u);
  EXPECT_EQ(m.total_tokens, 4u);
  EXPECT_TRUE(r->masked);
  ASSERT_EQ(r->ids.size(), 5u);
  const std::array<int32_t, 5> expect = {'\n', 'y', 'o', 258, '\n'};
  EXPECT_EQ(std::memcmp(r->ids.data(), expect.data(), 20), 0);
  ASSERT_EQ(r->mask.size(), 4u);
  for (uint8_t t : r->mask) EXPECT_EQ(t, 1);
  // The SDS: v3 with the mask flag, one record of 5 ids + 4 mask bytes.
  EXPECT_EQ(r->sds.size(), 64u + 20 + 4);
  uint32_t flags = 0;
  std::memcpy(&flags, r->sds.data() + 48, 4);
  EXPECT_EQ(flags, 1u);
  EXPECT_STR_CONTAINS(ManifestJson(m), "\"dropped_zero_target_windows\": 4");

  // A plain and a chat document together: the plain one's positions are
  // all targets, the EOS after it too; the mask follows each.
  // ab (2) | <eos> | <|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n (20)
  // | r<|im_end|>\n (3) = 26 ids; S = 12 -> 2 windows of 13.
  IngestOptions wide;
  wide.seq_len = 12;
  std::vector<Document> mixed = {{"ab", {}, false}, {"q", "r", true}};
  auto rm = Ingest(tok, 0x55, mixed, wide);
  ASSERT_OK(rm);
  EXPECT_EQ(rm->manifest.tokens, 26u);
  EXPECT_EQ(rm->manifest.windows, 2u);
  EXPECT_TRUE(rm->masked);
  // Window 0 predicts ids 1..12: 'b' (1), <eos> (1, after a plain document),
  // then ten prompt ids (0). Window 1 predicts ids 14..25: nine prompt ids
  // then the three response ids.
  ASSERT_EQ(rm->mask.size(), 24u);
  EXPECT_EQ(rm->mask[0], 1);
  EXPECT_EQ(rm->mask[1], 1);
  for (size_t j = 2; j < 12; ++j) EXPECT_EQ(rm->mask[j], 0);
  for (size_t j = 12; j < 21; ++j) EXPECT_EQ(rm->mask[j], 0);
  for (size_t j = 21; j < 24; ++j) EXPECT_EQ(rm->mask[j], 1);
  EXPECT_EQ(rm->manifest.target_tokens, 5u);
  EXPECT_EQ(rm->manifest.straddling_windows, 1u);

  // Refusals: a section without the markers; chat in one-document mode;
  // a response that encodes to nothing is degenerate.
  const ByteLevelBpe plain = Tokenizer();
  EXPECT_ERROR_CONTAINS(Ingest(plain, 0, docs, o),
                        "does not name as special tokens");
  IngestOptions one;
  one.seq_len = 4;
  one.one_document = true;
  EXPECT_ERROR_CONTAINS(Ingest(tok, 0, docs, one), "takes plain text");
}

}  // namespace
