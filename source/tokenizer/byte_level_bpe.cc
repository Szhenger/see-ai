#include "source/tokenizer/byte_level_bpe.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>

#include "source/identity/hash.h"
#include "source/tokenizer/unicode_tables.h"

namespace seeml::tokenizer {

namespace {

// --- The section reader ---------------------------------------------------------

struct Reader {
  std::span<const uint8_t> data;
  size_t pos = 0;
  bool ok = true;

  template <typename T>
  T Read() {
    T v{};
    if (pos + sizeof(T) > data.size()) {
      ok = false;
      return v;
    }
    std::memcpy(&v, data.data() + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::string ReadBytes(size_t n) {
    if (pos + n > data.size()) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(data.data() + pos), n);
    pos += n;
    return s;
  }
};

std::unexpected<std::string> Refuse(const std::string& what) {
  return std::unexpected("tokenizer section: " + what);
}

// --- Unicode classes over the generated tables ------------------------------------

bool InRanges(std::span<const CodePointRange> table, uint32_t cp) {
  // Sorted, non-overlapping ranges: the first whose upper bound reaches cp.
  size_t lo = 0, hi = table.size();
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (table[mid].hi < cp)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < table.size() && table[lo].lo <= cp && cp <= table[lo].hi;
}
bool IsLetter(uint32_t cp) { return InRanges(kLetterRanges, cp); }
bool IsNumber(uint32_t cp) { return InRanges(kNumberRanges, cp); }
bool IsSpace(uint32_t cp) { return InRanges(kWhiteSpaceRanges, cp); }

/// What a byte that begins no well-formed UTF-8 sequence decodes as: U+FFFD,
/// a symbol (category So) that is neither letter, number nor space, so the
/// byte is an "other" character to the pattern and never joins a letter,
/// number or whitespace run. The reference, fed a string, only ever sees
/// well-formed input; the sanitizer removes invalid bytes from a .jsonl
/// corpus, and a .txt corpus and seeml-tokenize encode the bytes as they
/// are, so the rule has to hold here too. (The raw byte value would not do:
/// 0xA0 is White_Space and 0xC0-0xFF are letters in the tables.)
constexpr uint32_t kInvalidByteClass = 0xFFFD;

/// Decodes the UTF-8 scalar at `i` and returns the index past it, or
/// kInvalidByteClass and the index one byte on for a byte that begins no
/// well-formed sequence (DecodeUtf8Strict's rule).
size_t Decode(std::string_view s, size_t i, uint32_t* cp) {
  const size_t next = DecodeUtf8Strict(s, i, cp);
  if (next != 0) return next;
  *cp = kInvalidByteClass;
  return i + 1;
}

}  // namespace

size_t DecodeUtf8Strict(std::string_view s, size_t i, uint32_t* cp) {
  const auto b0 = static_cast<uint8_t>(s[i]);
  if (b0 < 0x80) {
    *cp = b0;
    return i + 1;
  }
  size_t need = 0;
  uint32_t v = 0, min = 0;
  if ((b0 & 0xE0) == 0xC0) {
    need = 1;
    v = b0 & 0x1F;
    min = 0x80;
  } else if ((b0 & 0xF0) == 0xE0) {
    need = 2;
    v = b0 & 0x0F;
    min = 0x800;
  } else if ((b0 & 0xF8) == 0xF0) {
    need = 3;
    v = b0 & 0x07;
    min = 0x10000;
  } else {
    return 0;
  }
  if (i + need >= s.size()) return 0;
  for (size_t k = 1; k <= need; ++k) {
    const auto b = static_cast<uint8_t>(s[i + k]);
    if ((b & 0xC0) != 0x80) return 0;
    v = (v << 6) | (b & 0x3F);
  }
  if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) return 0;
  *cp = v;
  return i + need + 1;
}

// --- Load ------------------------------------------------------------------------

std::expected<ByteLevelBpe, std::string> ByteLevelBpe::Load(
    std::span<const uint8_t> section) {
  // The fixed head through the format's own reader (size, magic, version);
  // what the head declares is refused here, by name.
  auto head = ReadSmtkHead(section);
  if (!head) return Refuse(head.error());
  ByteLevelBpe t;
  if (head->kind != static_cast<uint32_t>(TokenizerKind::kByteLevelBpe))
    return Refuse("unsupported tokenizer kind " + std::to_string(head->kind) +
                  " (only byte-level BPE is implemented)");
  t.info_.kind = TokenizerKind::kByteLevelBpe;
  const uint32_t pre = head->pre_tokenizer;
  if (pre != static_cast<uint32_t>(PreTokenizer::kGpt2) &&
      pre != static_cast<uint32_t>(PreTokenizer::kDigitsThenGpt2))
    return Refuse("unsupported pre-tokenizer " + std::to_string(pre));
  t.info_.pre_tokenizer = static_cast<PreTokenizer>(pre);
  const uint32_t flags = head->flags;
  if (flags & ~kSmtkFlagsKnown) return Refuse("unknown flags");
  if (flags & kSmtkFlagAddPrefixSpace)
    return Refuse("add_prefix_space is not implemented");
  if (flags & kSmtkFlagNfc) return Refuse("the NFC normalizer is not implemented");
  t.info_.vocab_size = head->vocab_size;
  t.info_.num_merges = head->num_merges;
  t.info_.num_special = head->num_special;
  t.info_.eos_id = head->eos_id;
  t.info_.bos_id = head->bos_id;
  t.info_.unk_id = head->unk_id;
  t.info_.pad_id = head->pad_id;
  Reader r{section, static_cast<size_t>(kSmtkHeadBytes)};
  t.info_.chat_template = r.ReadBytes(head->chat_template_len);
  if (!r.ok) return Refuse("truncated chat template");
  if (t.info_.vocab_size == 0) return Refuse("empty vocabulary");
  // Each token is at least three bytes on disk (a length and one byte).
  if (t.info_.vocab_size > section.size() / 3)
    return Refuse("vocabulary larger than the section");
  if (t.info_.vocab_size > static_cast<uint32_t>(INT32_MAX))
    return Refuse("vocabulary does not fit an i32 id");

  t.tokens_.reserve(t.info_.vocab_size);
  for (uint32_t id = 0; id < t.info_.vocab_size; ++id) {
    const uint16_t len = r.Read<uint16_t>();
    std::string bytes = r.ReadBytes(len);
    if (!r.ok) return Refuse("truncated vocabulary at id " + std::to_string(id));
    if (bytes.empty()) return Refuse("empty token at id " + std::to_string(id));
    t.tokens_.push_back(std::move(bytes));
  }
  t.ids_.reserve(t.tokens_.size());
  for (uint32_t id = 0; id < t.tokens_.size(); ++id) {
    if (!t.ids_.emplace(std::string_view(t.tokens_[id]), id).second)
      return Refuse("token " + std::to_string(id) + " repeats token " +
                    std::to_string(t.ids_.at(t.tokens_[id])));
  }
  for (int b = 0; b < 256; ++b) {
    const char c = static_cast<char>(b);
    auto it = t.ids_.find(std::string_view(&c, 1));
    t.byte_token_[b] = it == t.ids_.end() ? -1 : static_cast<int32_t>(it->second);
  }

  for (int32_t* id : {&t.info_.eos_id, &t.info_.bos_id, &t.info_.unk_id,
                      &t.info_.pad_id})
    if (*id < -1 || (*id >= 0 && static_cast<uint32_t>(*id) >= t.info_.vocab_size))
      return Refuse("a special id is out of range");

  if (t.info_.num_merges > section.size() / 12)
    return Refuse("more merges than the section holds");
  t.merges_.reserve(t.info_.num_merges);
  for (uint32_t rank = 0; rank < t.info_.num_merges; ++rank) {
    const uint32_t a = r.Read<uint32_t>(), b = r.Read<uint32_t>(),
                   c = r.Read<uint32_t>();
    if (!r.ok) return Refuse("truncated merges at rank " + std::to_string(rank));
    if (a >= t.info_.vocab_size || b >= t.info_.vocab_size ||
        c >= t.info_.vocab_size)
      return Refuse("merge " + std::to_string(rank) + " names an id out of range");
    if (t.tokens_[a] + t.tokens_[b] != t.tokens_[c])
      return Refuse("merge " + std::to_string(rank) + " does not produce the "
                    "concatenation of its parts");
    const uint64_t key = (uint64_t{a} << 32) | b;
    if (!t.merges_.emplace(key, (uint64_t{rank} << 32) | c).second)
      return Refuse("merge " + std::to_string(rank) + " repeats an earlier pair");
  }
  for (uint32_t i = 0; i < t.info_.num_special; ++i) {
    const uint32_t id = r.Read<uint32_t>();
    if (!r.ok) return Refuse("truncated special tokens");
    if (id >= t.info_.vocab_size) return Refuse("special id out of range");
    t.special_.push_back(id);
  }
  if (r.pos != section.size()) return Refuse("trailing bytes");
  // Longest first, so a scan takes the longest special at a position; ties
  // by id, so the order is a function of the section alone.
  std::sort(t.special_.begin(), t.special_.end(), [&](uint32_t x, uint32_t y) {
    if (t.tokens_[x].size() != t.tokens_[y].size())
      return t.tokens_[x].size() > t.tokens_[y].size();
    return x < y;
  });
  // The per-id flag Encode asks per piece, and the lead bytes the special
  // scan has to stop at: every other position costs one table load.
  t.special_flag_.assign(t.tokens_.size(), 0);
  for (uint32_t id : t.special_) {
    t.special_flag_[id] = 1;
    t.special_lead_[static_cast<uint8_t>(t.tokens_[id][0])] = true;
  }
  return t;
}

// --- Pre-tokenization ---------------------------------------------------------------

void ByteLevelBpe::SplitGpt2(
    std::string_view text, size_t base,
    std::vector<std::pair<size_t, size_t>>& pieces) const {
  // The GPT-2 pattern, its alternatives tried in the pattern's order at
  // each position:
  //   's|'t|'re|'ve|'m|'ll|'d
  //   ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+'
  //   '\s+(?!\S)' | '\s+'
  // over code points. The lookahead (?!\S) sees this piece's end as "no
  // non-space follows" — the Digits splitter hands the pattern one piece at
  // a time, and the reference's regex runs inside each piece too.
  const size_t n = text.size();
  size_t i = 0;
  auto run_of = [&](size_t from, auto&& in_class) {
    size_t k = from;
    while (k < n) {
      uint32_t d = 0;
      const size_t nn = Decode(text, k, &d);
      if (!in_class(d)) break;
      k = nn;
    }
    return k;
  };
  while (i < n) {
    const size_t start = i;
    uint32_t cp = 0;
    const size_t next = Decode(text, i, &cp);
    // Contractions: the apostrophe and one of the lowercase suffixes.
    if (cp == '\'') {
      static constexpr std::string_view kSuffixes[] = {"s", "t", "re", "ve",
                                                       "m", "ll", "d"};
      bool matched = false;
      for (std::string_view suf : kSuffixes) {
        if (text.substr(next).starts_with(suf)) {
          i = next + suf.size();
          matched = true;
          break;
        }
      }
      if (matched) {
        pieces.emplace_back(base + start, base + i);
        continue;
      }
    }
    // One optional ASCII space, then a run of letters, numbers or
    // "others" (anything that is not whitespace, a letter or a number).
    uint32_t c = cp;
    size_t after_first = next;
    if (cp == ' ' && next < n) {
      uint32_t c2 = 0;
      const size_t n2 = Decode(text, next, &c2);
      if (!IsSpace(c2)) {  // a letter, a number or an "other" follows
        c = c2;
        after_first = n2;
      }
    }
    if (IsLetter(c)) {
      i = run_of(after_first, IsLetter);
      pieces.emplace_back(base + start, base + i);
      continue;
    }
    if (IsNumber(c)) {
      i = run_of(after_first, IsNumber);
      pieces.emplace_back(base + start, base + i);
      continue;
    }
    if (!IsSpace(c)) {
      i = run_of(after_first, [](uint32_t d) {
        return !IsSpace(d) && !IsLetter(d) && !IsNumber(d);
      });
      pieces.emplace_back(base + start, base + i);
      continue;
    }
    // Whitespace: the run; when a non-space follows it, all but the last
    // whitespace character (the run must keep at least one), else the run.
    size_t k = i, last_ws_start = i;
    while (k < n) {
      uint32_t d = 0;
      const size_t nn = Decode(text, k, &d);
      if (!IsSpace(d)) break;
      last_ws_start = k;
      k = nn;
    }
    const bool followed = k < n;
    i = (followed && last_ws_start > i) ? last_ws_start : k;
    pieces.emplace_back(base + start, base + i);
  }
}

std::vector<std::pair<size_t, size_t>> ByteLevelBpe::PreTokenize(
    std::string_view text) const {
  std::vector<std::pair<size_t, size_t>> pieces;
  // 1. Special tokens split first: the longest special at each position.
  std::vector<std::pair<size_t, size_t>> spans;  // the text between them
  std::vector<std::pair<size_t, size_t>> specials;
  size_t i = 0, span_start = 0;
  while (i < text.size()) {
    // Only a byte some special token starts with is worth the scan.
    if (!special_lead_[static_cast<uint8_t>(text[i])]) {
      ++i;
      continue;
    }
    bool hit = false;
    for (uint32_t id : special_) {
      const std::string& s = tokens_[id];
      if (text.substr(i).starts_with(s)) {
        if (i > span_start) spans.emplace_back(span_start, i);
        specials.emplace_back(i, i + s.size());
        i += s.size();
        span_start = i;
        hit = true;
        break;
      }
    }
    if (!hit) ++i;
  }
  if (text.size() > span_start) spans.emplace_back(span_start, text.size());
  // 2. Each span: when the section says so, every number character is a
  //    piece of its own (Digits, individual) and the pattern runs over what
  //    lies between; else the pattern runs over the span.
  auto gpt2 = [&](size_t b, size_t e) {
    SplitGpt2(text.substr(b, e - b), b, pieces);
  };
  for (const auto& [b, e] : spans) {
    if (info_.pre_tokenizer == PreTokenizer::kGpt2) {
      gpt2(b, e);
      continue;
    }
    size_t run_start = b, k = b;
    while (k < e) {
      uint32_t cp = 0;
      const size_t nn = Decode(text, k, &cp);
      if (IsNumber(cp)) {
        if (k > run_start) gpt2(run_start, k);
        pieces.emplace_back(k, nn);
        run_start = nn;
      }
      k = nn;
    }
    if (e > run_start) gpt2(run_start, e);
  }
  // 3. The specials back in text order.
  pieces.insert(pieces.end(), specials.begin(), specials.end());
  std::sort(pieces.begin(), pieces.end());
  return pieces;
}

// --- Encoding ---------------------------------------------------------------------

void ByteLevelBpe::EncodePiece(std::string_view piece,
                               std::vector<int32_t>& out) const {
  // Initial symbols: one per byte. A byte with no single-byte token is
  // dropped, as the reference drops a character its vocabulary lacks
  // (byte_fallback off, no unknown token); such a byte cannot occur in
  // valid UTF-8 or is a control character the sanitizer removes.
  std::vector<uint32_t> ids;
  ids.reserve(piece.size());
  for (unsigned char b : piece) {
    const int32_t id = byte_token_[b];
    if (id >= 0) ids.push_back(static_cast<uint32_t>(id));
  }
  // Merge the lowest-ranked adjacent pair until none applies — the
  // reference's algorithm; an equal rank twice over is the first pair.
  // The symbols form a doubly linked list and the candidate pairs a
  // min-heap keyed by (rank, position), so a piece of n bytes costs
  // O(n log n) however long a run the pattern hands over (a paragraph
  // without spaces, a line of separators) instead of a rescan of every
  // pair per merge. A heap entry is stale once either of its symbols was
  // merged away or the pair at its position changed; it is checked against
  // the list before it is applied, the standard lazy deletion.
  const size_t n = ids.size();
  if (n < 2) {
    for (uint32_t id : ids) out.push_back(static_cast<int32_t>(id));
    return;
  }
  constexpr size_t kNone = std::numeric_limits<size_t>::max();
  std::vector<size_t> prev(n), next(n);
  for (size_t k = 0; k < n; ++k) {
    prev[k] = k == 0 ? kNone : k - 1;
    next[k] = k + 1 < n ? k + 1 : kNone;
  }
  struct Candidate {
    uint64_t ranked;  // rank << 32 | merged id, as merges_ stores it
    size_t at;        // the left symbol's position
    uint32_t left, right;  // the pair the entry was made for
    bool operator>(const Candidate& o) const {
      return ranked != o.ranked ? ranked > o.ranked : at > o.at;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>
      heap;
  auto offer = [&](size_t at) {
    const size_t nx = next[at];
    if (nx == kNone) return;
    auto it = merges_.find((uint64_t{ids[at]} << 32) | ids[nx]);
    if (it != merges_.end()) heap.push({it->second, at, ids[at], ids[nx]});
  };
  for (size_t k = 0; k + 1 < n; ++k) offer(k);
  std::vector<bool> alive(n, true);
  while (!heap.empty()) {
    const Candidate c = heap.top();
    heap.pop();
    const size_t nx = next[c.at];
    if (!alive[c.at] || nx == kNone || ids[c.at] != c.left || ids[nx] != c.right)
      continue;  // stale
    ids[c.at] = static_cast<uint32_t>(c.ranked & 0xFFFFFFFFu);
    alive[nx] = false;
    next[c.at] = next[nx];
    if (next[nx] != kNone) prev[next[nx]] = c.at;
    if (prev[c.at] != kNone) offer(prev[c.at]);
    offer(c.at);
  }
  for (size_t k = 0; k != kNone; k = next[k])
    out.push_back(static_cast<int32_t>(ids[k]));
}

std::expected<void, std::string> ByteLevelBpe::Encode(
    std::string_view text, std::vector<int32_t>& out) const {
  EncodeSplit(text, text.size(), out);
  return {};
}

size_t ByteLevelBpe::EncodeSplit(std::string_view text, size_t boundary,
                                 std::vector<int32_t>& out) const {
  const size_t base = out.size();
  size_t before = 0;
  for (const auto& [b, e] : PreTokenize(text)) {
    const std::string_view piece = text.substr(b, e - b);
    auto it = ids_.find(piece);
    if (it != ids_.end() && is_special(it->second))
      out.push_back(static_cast<int32_t>(it->second));
    else
      EncodePiece(piece, out);
    if (e <= boundary) before = out.size() - base;
  }
  return before;
}

uint64_t SectionHash(std::span<const uint8_t> section) {
  return seeml::update::ContentHash64(section.data(), section.size());
}

}  // namespace seeml::tokenizer
