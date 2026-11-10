// =============================================================================
// The core's byte-level BPE (S2 #148): the section reader's refusals, the
// GPT-2 pre-tokenizer's rules (as the reference tokenizer applies them),
// the Digits splitter, special tokens, and the merge algorithm — on a
// synthetic section small enough to reason about by hand. The parity with
// the reference on a real checkpoint is the exporter's harness
// (tool/export_model.py --tokenizer-parity, test/tool/).
// =============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "source/tokenizer/byte_level_bpe.h"
#include "test/framework/seetest.h"

namespace {

using seeml::tokenizer::ByteLevelBpe;
using seeml::tokenizer::PreTokenizer;

struct Section {
  std::vector<std::string> tokens;                   // id order
  std::vector<std::array<uint32_t, 3>> merges;       // rank order
  std::vector<uint32_t> specials;
  uint32_t pre = 2;  // Digits then GPT-2
  uint32_t flags = 0;
  uint32_t version = 1;
  uint32_t kind = 1;
  int32_t eos = -1;
  std::string chat_template;

  std::vector<uint8_t> Bytes() const {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u16 = [&](uint16_t v) {
      for (int i = 0; i < 2; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    u32(0x4B544D53);
    u32(version);
    u32(kind);
    u32(pre);
    u32(flags);
    u32(static_cast<uint32_t>(tokens.size()));
    u32(static_cast<uint32_t>(merges.size()));
    u32(static_cast<uint32_t>(specials.size()));
    u32(static_cast<uint32_t>(eos));
    u32(static_cast<uint32_t>(-1));
    u32(static_cast<uint32_t>(-1));
    u32(static_cast<uint32_t>(-1));
    u32(static_cast<uint32_t>(chat_template.size()));
    b.insert(b.end(), chat_template.begin(), chat_template.end());
    for (const std::string& t : tokens) {
      u16(static_cast<uint16_t>(t.size()));
      b.insert(b.end(), t.begin(), t.end());
    }
    for (const auto& m : merges) {
      u32(m[0]);
      u32(m[1]);
      u32(m[2]);
    }
    for (uint32_t s : specials) u32(s);
    return b;
  }
};

/// A vocabulary of every byte (id = byte value) plus the tokens listed, so
/// any text encodes; merges build the listed tokens from their halves.
Section ByteVocab(std::vector<std::pair<std::string, std::string>> merges,
                  std::vector<std::string> specials = {}) {
  Section s;
  for (int b = 0; b < 256; ++b) s.tokens.push_back(std::string(1, static_cast<char>(b)));
  auto id_of = [&](const std::string& t) -> uint32_t {
    for (uint32_t i = 0; i < s.tokens.size(); ++i)
      if (s.tokens[i] == t) return i;
    s.tokens.push_back(t);
    return static_cast<uint32_t>(s.tokens.size() - 1);
  };
  for (const auto& [a, b] : merges) {
    const uint32_t ia = id_of(a), ib = id_of(b), ic = id_of(a + b);
    s.merges.push_back({ia, ib, ic});
  }
  for (const std::string& sp : specials) s.specials.push_back(id_of(sp));
  return s;
}

std::vector<std::string> Pieces(const ByteLevelBpe& t, std::string_view text) {
  std::vector<std::string> out;
  for (const auto& [b, e] : t.PreTokenize(text)) out.emplace_back(text.substr(b, e - b));
  return out;
}

std::vector<int32_t> Ids(const ByteLevelBpe& t, std::string_view text) {
  std::vector<int32_t> ids;
  if (auto ok = t.Encode(text, ids); !ok) ADD_FAILURE(ok.error());
  return ids;
}

TEST(ByteLevelBpe, LoadsASectionAndRefusesWhatItDoesNotImplement) {
  Section s = ByteVocab({{"t", "h"}, {"th", "e"}});
  s.eos = 0;
  s.chat_template = "{{ messages }}";
  const auto bytes = s.Bytes();
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe t, ByteLevelBpe::Load(bytes));
  EXPECT_EQ(t.info().vocab_size, 258u);
  EXPECT_EQ(t.info().num_merges, 2u);
  EXPECT_EQ(t.info().eos_id, 0);
  EXPECT_EQ(t.info().chat_template, "{{ messages }}");
  EXPECT_TRUE(seeml::tokenizer::SectionHash(bytes) != 0);

  Section bad = s;
  bad.kind = 2;
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "unsupported tokenizer kind");
  bad = s;
  bad.pre = 9;
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "unsupported pre-tokenizer");
  bad = s;
  bad.flags = 2;
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "NFC normalizer");
  bad = s;
  bad.flags = 4;
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "unknown flags");
  bad = s;
  bad.version = 2;
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "unsupported version");
  bad = s;
  bad.merges[1] = {bad.merges[1][0], bad.merges[1][1], 5};  // "the" != "t"+"h"+"e"? no: c wrong
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "does not produce the concatenation");
  bad = s;
  bad.specials.push_back(999);
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(bad.Bytes()), "special id out of range");
  auto truncated = s.Bytes();
  truncated.resize(truncated.size() - 3);
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(truncated), "truncated");
  auto trailing = s.Bytes();
  trailing.push_back(0);
  EXPECT_ERROR_CONTAINS(ByteLevelBpe::Load(trailing), "trailing bytes");
}

TEST(ByteLevelBpe, TheGpt2PatternSplitsAsTheReferenceDoes) {
  // Each case is the reference tokenizer's pre-tokenization (Hugging Face
  // `tokenizers`, ByteLevel use_regex, Digits individual) on the same text.
  Section s = ByteVocab({});
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe t, ByteLevelBpe::Load(s.Bytes()));
  using V = std::vector<std::string>;
  EXPECT_TRUE(Pieces(t, "Hello world 123") == V({"Hello", " world", " ", "1", "2", "3"}));
  EXPECT_TRUE(Pieces(t, "don't") == V({"don", "'t"}));
  EXPECT_TRUE(Pieces(t, "it's") == V({"it", "'s"}));
  EXPECT_TRUE(Pieces(t, "'S 'RE") == V({"'", "S", " '", "RE"}));
  EXPECT_TRUE(Pieces(t, "  two  spaces") == V({" ", " two", " ", " spaces"}));
  EXPECT_TRUE(Pieces(t, "a   b") == V({"a", "  ", " b"}));
  EXPECT_TRUE(Pieces(t, "a   ") == V({"a", "   "}));
  EXPECT_TRUE(Pieces(t, "  \n\n  ") == V({"  \n\n  "}));
  EXPECT_TRUE(Pieces(t, "x\n\ny") == V({"x", "\n", "\n", "y"}));
  EXPECT_TRUE(Pieces(t, " \t x") == V({" \t", " x"}));
  EXPECT_TRUE(Pieces(t, "1 2") == V({"1", " ", "2"}));
  EXPECT_TRUE(Pieces(t, " 12 ab") == V({" ", "1", "2", " ab"}));
  EXPECT_TRUE(Pieces(t, "ab12cd") == V({"ab", "1", "2", "cd"}));
  EXPECT_TRUE(Pieces(t, "x1.5") == V({"x", "1", ".", "5"}));
  EXPECT_TRUE(Pieces(t, "naïve café") == V({"naïve", " café"}));
  // NBSP (U+00A0) is White_Space: a run before a letter keeps one apart.
  EXPECT_TRUE(Pieces(t, "a\xC2\xA0\xC2\xA0" "b") == V({"a", "\xC2\xA0", "\xC2\xA0", "b"}));
  // A combining mark (Mn) is not a letter: "e" then the mark as "other".
  EXPECT_TRUE(Pieces(t, "e\xCC\x81") == V({"e", "\xCC\x81"}));
  // Arabic-Indic digit (Nd) and circled one (No) are numbers: digits apart.
  EXPECT_TRUE(Pieces(t, "x\xD9\xA3" "y") == V({"x", "\xD9\xA3", "y"}));
  EXPECT_TRUE(Pieces(t, "a\xE2\x91\xA0") == V({"a", "\xE2\x91\xA0"}));
  // Without the Digits splitter, number runs stay whole.
  Section plain = ByteVocab({});
  plain.pre = 1;
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe g, ByteLevelBpe::Load(plain.Bytes()));
  EXPECT_TRUE(Pieces(g, "ab12cd 345") == V({"ab", "12", "cd", " 345"}));
  EXPECT_TRUE(Pieces(g, "1 2") == V({"1", " 2"}));
}

TEST(ByteLevelBpe, SpecialTokensSplitFirstAndMergesApplyByRank) {
  Section s = ByteVocab({{"t", "h"}, {"th", "e"}, {"e", "r"}, {"h", "e"}},
                        {"<|endoftext|>", "<|im_start|>", "<|im_start|>x"});
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe t, ByteLevelBpe::Load(s.Bytes()));
  const uint32_t th = 256, the = 257, er = 258, he = 259;
  const uint32_t eot = 260, ims = 261, imsx = 262;
  // "the": t h e -> (t,h) rank 0 first -> th, then (th,e) rank 1 -> the.
  EXPECT_TRUE(Ids(t, "the") == std::vector<int32_t>({static_cast<int32_t>(the)}));
  // "ther": the lowest rank anywhere first: (t,h)=0 -> th e r; (th,e)=1 ->
  // the r; (e,r) is gone: the + r.
  EXPECT_TRUE(Ids(t, "ther") == std::vector<int32_t>({static_cast<int32_t>(the), 'r'}));
  // "her": (h,e)=3 vs (e,r)=2: (e,r) merges first: h + er.
  EXPECT_TRUE(Ids(t, "her") == std::vector<int32_t>({'h', static_cast<int32_t>(er)}));
  (void)th;
  (void)he;
  // Specials: longest match first, in or out of context, back to back.
  EXPECT_TRUE(Ids(t, "<|endoftext|>the<|endoftext|>") ==
              std::vector<int32_t>({static_cast<int32_t>(eot), static_cast<int32_t>(the),
                                    static_cast<int32_t>(eot)}));
  EXPECT_TRUE(Ids(t, "<|im_start|>x") == std::vector<int32_t>({static_cast<int32_t>(imsx)}));
  EXPECT_TRUE(Ids(t, "<|im_start|>y") ==
              std::vector<int32_t>({static_cast<int32_t>(ims), 'y'}));
  // A near miss is ordinary text.
  EXPECT_EQ(Ids(t, "<|notatoken|>").size(), 13u);
  // A byte with no token is dropped, as the reference drops it.
  Section sparse = ByteVocab({});
  sparse.tokens.erase(sparse.tokens.begin() + 4);  // byte 0x04 gone (ids shift)
  // Rebuild without byte 4: every other byte keeps a token.
  Section nb;
  for (int b = 0; b < 256; ++b)
    if (b != 4) nb.tokens.push_back(std::string(1, static_cast<char>(b)));
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe u, ByteLevelBpe::Load(nb.Bytes()));
  EXPECT_EQ(Ids(u, std::string("a\x04" "b")).size(), 2u);
}

TEST(ByteLevelBpe, EncodeSplitSeesTheJoinAsOneString) {
  // A prompt rendered in front of its response is encoded as ONE string:
  // the GPT-2 pattern's `\s+(?!\S)` folds the template's trailing newline
  // with all but the last of the response's leading whitespace, as the
  // reference does, and the folded piece belongs to the response (the
  // token holding the response's first byte is the response's).
  Section s = ByteVocab({{"\n", " "}});
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe t, ByteLevelBpe::Load(s.Bytes()));
  const std::string prompt = "assistant\n", response = "  yes";
  const std::string whole = prompt + response;
  EXPECT_TRUE(Pieces(t, whole) ==
              std::vector<std::string>({"assistant", "\n ", " yes"}));
  std::vector<int32_t> ids;
  const size_t before = t.EncodeSplit(whole, prompt.size(), ids);
  EXPECT_EQ(before, 9u);  // "assistant"; the "\n " piece straddles the join
  EXPECT_EQ(ids.size(), 14u);
  EXPECT_EQ(ids[9], 256);  // the merged "\n "
  EXPECT_TRUE(ids == Ids(t, whole));
  // Encoded separately the two sides give one id more — the divergence
  // the one-string encoding removes.
  EXPECT_EQ(Ids(t, prompt).size() + Ids(t, response).size(), 15u);
  // A single leading whitespace character never differs; a boundary at
  // the text's end counts every id as the prompt's.
  const std::string one = prompt + " yes";
  std::vector<int32_t> a, b;
  EXPECT_EQ(t.EncodeSplit(one, prompt.size(), a), 10u);
  EXPECT_EQ(t.EncodeSplit(one, one.size(), b), Ids(t, one).size());
}

TEST(ByteLevelBpe, AStrayByteIsAnOtherCharacterAndLongRunsEncodeFast) {
  Section s = ByteVocab({{"x", "x"}, {"xx", "xx"}});
  ASSERT_OK_AND_ASSIGN(ByteLevelBpe t, ByteLevelBpe::Load(s.Bytes()));
  // A byte that begins no UTF-8 sequence is neither letter, number nor
  // space, whatever its value: 0xC3 (a letter if read as U+00C3) breaks a
  // letter run, 0xA0 (White_Space if read as U+00A0) joins the ` ?` other
  // run rather than the whitespace run.
  EXPECT_TRUE(Pieces(t, std::string("ab\xC3" "cd")) ==
              std::vector<std::string>({"ab", "\xC3", "cd"}));
  EXPECT_TRUE(Pieces(t, std::string(" \xA0" "b")) ==
              std::vector<std::string>({" \xA0", "b"}));
  // An overlong encoding is two stray bytes (one run of "others"), not the
  // letter it spells.
  EXPECT_TRUE(Pieces(t, std::string("\xC1\xA1" "b")) ==
              std::vector<std::string>({"\xC1\xA1", "b"}));
  // The merge loop is O(n log n) per piece: a 40 000-byte run of one letter
  // is one piece, merged pairwise by rank — every (x,x) at rank 0, leftmost
  // first, then every (xx,xx) at rank 1 — into 10 000 "xxxx" ids.
  const std::string run(40000, 'x');
  const std::vector<int32_t> ids = Ids(t, run);
  ASSERT_EQ(ids.size(), 10000u);
  for (int32_t id : ids) EXPECT_EQ(id, 257);
  // Leftmost-first on equal rank: "xxx" is xx + x, "xxxxx" is xxxx + x.
  EXPECT_TRUE(Ids(t, "xxx") == std::vector<int32_t>({256, 'x'}));
  EXPECT_TRUE(Ids(t, "xxxxx") == std::vector<int32_t>({257, 'x'}));
}

}  // namespace
