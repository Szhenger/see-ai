#include "source/tokenizer/text_ingest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "source/identity/hash.h"
#include "source/language/dataset_format.h"

namespace seeml::tokenizer {

namespace {

bool IsControl(uint32_t cp) {
  // Cc minus tab, newline and carriage return.
  if (cp == '\t' || cp == '\n' || cp == '\r') return false;
  return cp < 0x20 || (cp >= 0x7F && cp <= 0x9F);
}

bool IsBidiControl(uint32_t cp) {
  // The explicit directional embeddings, overrides and isolates: LRE, RLE,
  // PDF, LRO, RLO (U+202A..202E) and LRI, RLI, FSI, PDI (U+2066..2069).
  return (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069);
}

void AppendJsonString(std::string& out, std::string_view s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

/// Appends the UTF-8 of `cp` to `out`.
void AppendUtf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// Parses a JSON string literal starting at `i` (the opening quote), with
/// escapes decoded, into `out`; returns the index past the closing quote
/// or 0 on a malformed literal.
size_t ParseJsonString(std::string_view s, size_t i, std::string& out) {
  if (i >= s.size() || s[i] != '"') return 0;
  ++i;
  while (i < s.size()) {
    const char c = s[i];
    if (c == '"') return i + 1;
    if (c != '\\') {
      out += c;
      ++i;
      continue;
    }
    if (i + 1 >= s.size()) return 0;
    const char e = s[i + 1];
    i += 2;
    switch (e) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        auto hex4 = [&](size_t at, uint32_t* v) {
          if (at + 4 > s.size()) return false;
          *v = 0;
          for (size_t k = 0; k < 4; ++k) {
            const char h = s[at + k];
            uint32_t d = 0;
            if (h >= '0' && h <= '9') d = static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') d = static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') d = static_cast<uint32_t>(h - 'A' + 10);
            else return false;
            *v = (*v << 4) | d;
          }
          return true;
        };
        uint32_t cp = 0;
        if (!hex4(i, &cp)) return 0;
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF) {  // a surrogate pair
          uint32_t lo = 0;
          if (i + 6 > s.size() || s[i] != '\\' || s[i + 1] != 'u' ||
              !hex4(i + 2, &lo) || lo < 0xDC00 || lo > 0xDFFF)
            return 0;
          i += 6;
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          return 0;
        }
        AppendUtf8(out, cp);
        break;
      }
      default:
        return 0;
    }
  }
  return 0;
}

size_t SkipSpaces(std::string_view s, size_t i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r')) ++i;
  return i;
}

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";

}  // namespace

std::string Sanitize(std::string_view text, bool count_only,
                     SanitizeCounts* counts) {
  std::string out;
  if (!count_only) out.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = 0;
    const size_t next = DecodeUtf8Strict(text, i, &cp);
    if (next == 0) {
      ++counts->invalid_utf8_bytes;
      if (count_only) out.push_back(text[i]);
      ++i;
      continue;
    }
    bool drop = false;
    if (IsControl(cp)) {
      ++counts->control_chars;
      drop = true;
    } else if (IsBidiControl(cp)) {
      ++counts->bidi_controls;
      drop = true;
    }
    if (!drop || count_only) out.append(text.data() + i, next - i);
    i = next;
  }
  return out;
}

std::expected<std::vector<Document>, std::string> DocumentsFromJsonl(
    std::string_view jsonl) {
  std::vector<Document> docs;
  size_t line_no = 0, pos = 0;
  while (pos < jsonl.size()) {
    size_t end = jsonl.find('\n', pos);
    if (end == std::string_view::npos) end = jsonl.size();
    std::string_view line = jsonl.substr(pos, end - pos);
    pos = end + 1;
    ++line_no;
    size_t i = SkipSpaces(line, 0);
    if (i >= line.size()) continue;  // a blank line is not a document
    auto refuse = [&](const char* what) {
      return std::unexpected("line " + std::to_string(line_no) + ": " + what);
    };
    if (line[i] != '{') return refuse("not a JSON object");
    i = SkipSpaces(line, i + 1);
    std::string text, prompt, response;
    bool have_text = false, have_prompt = false, have_response = false;
    while (i < line.size() && line[i] != '}') {
      std::string key;
      const size_t after_key = ParseJsonString(line, i, key);
      if (after_key == 0) return refuse("malformed object key");
      i = SkipSpaces(line, after_key);
      if (i >= line.size() || line[i] != ':') return refuse("expected ':'");
      i = SkipSpaces(line, i + 1);
      if (i >= line.size()) return refuse("truncated object");
      if (line[i] == '"') {
        std::string value;
        const size_t after_value = ParseJsonString(line, i, value);
        if (after_value == 0) return refuse("malformed string value");
        i = after_value;
        if (key == "text") {
          if (have_text) return refuse("two \"text\" fields");
          text = std::move(value);
          have_text = true;
        } else if (key == "prompt") {
          if (have_prompt) return refuse("two \"prompt\" fields");
          prompt = std::move(value);
          have_prompt = true;
        } else if (key == "response") {
          if (have_response) return refuse("two \"response\" fields");
          response = std::move(value);
          have_response = true;
        }
      } else {
        // Any other JSON value: skip to the next top-level ',' or '}' —
        // numbers, booleans, null; nested objects or arrays are refused,
        // since the record format carries strings only.
        if (line[i] == '{' || line[i] == '[')
          return refuse("nested values are not part of a text record");
        while (i < line.size() && line[i] != ',' && line[i] != '}') ++i;
      }
      // Members are comma-separated: a value followed by anything but ','
      // or '}' is a malformed object, not a record to read around.
      i = SkipSpaces(line, i);
      if (i >= line.size()) return refuse("truncated object");
      if (line[i] == ',') {
        i = SkipSpaces(line, i + 1);
        if (i >= line.size() || line[i] == '}')
          return refuse("a ',' with no member after it");
      } else if (line[i] != '}') {
        return refuse("expected ',' or '}' after a value");
      }
    }
    if (i >= line.size()) return refuse("truncated object");
    // One record per line: anything after the object (a second object, a
    // stray comma, text) would otherwise vanish from the canonical corpus
    // without a drop entry in the manifest.
    i = SkipSpaces(line, i + 1);
    if (i < line.size()) return refuse("content after the object (one record per line)");
    if (have_text && (have_prompt || have_response))
      return refuse("a record is {\"text\"} or {\"prompt\", \"response\"}, "
                    "not both");
    if (have_prompt != have_response)
      return refuse("a prompt / response record needs both fields");
    if (!have_text && !have_prompt)
      return refuse("no \"text\" field (nor a \"prompt\" / \"response\" pair)");
    Document d;
    if (have_text) {
      d.text = std::move(text);
    } else {
      d.text = std::move(prompt);
      d.response = std::move(response);
      d.chat = true;
    }
    docs.push_back(std::move(d));
  }
  return docs;
}

std::expected<IngestResult, std::string> Ingest(
    const ByteLevelBpe& tokenizer, uint64_t tokenizer_hash,
    const std::vector<Document>& documents, const IngestOptions& options) {
  if (options.seq_len == 0) return std::unexpected("ingest: seq_len is zero");
  if (documents.empty()) return std::unexpected("ingest: no documents");
  if (options.one_document && documents.size() != 1)
    return std::unexpected("ingest: one-document mode takes one document");
  const int32_t eos = tokenizer.info().eos_id;
  if (!options.one_document && documents.size() > 1 && eos < 0)
    return std::unexpected(
        "ingest: the tokenizer names no EOS token, which joining documents "
        "needs");
  bool any_chat = false;
  for (const Document& d : documents) any_chat |= d.chat;
  if (any_chat) {
    if (options.one_document)
      return std::unexpected("ingest: one-document mode takes plain text");
    // The ChatML markers must be special tokens of the section, or the
    // template's text would be BPE-encoded as ordinary characters.
    bool im_start = false, im_end = false;
    for (uint32_t id = 0; id < tokenizer.info().vocab_size; ++id) {
      if (!tokenizer.is_special(id)) continue;
      if (tokenizer.token(id) == kImStart) im_start = true;
      if (tokenizer.token(id) == kImEnd) im_end = true;
    }
    if (!im_start || !im_end)
      return std::unexpected(
          "ingest: prompt / response records are rendered through the ChatML "
          "template, whose markers <|im_start|> and <|im_end|> the tokenizer "
          "section does not name as special tokens");
  }

  IngestResult result;
  IngestManifest& m = result.manifest;
  m.documents_in = documents.size();
  result.tokenizer_hash = tokenizer_hash;

  // Sanitize, deduplicate, encode: one pass over the documents, in order.
  // `target` marks, per id, whether predicting it is a loss target: every
  // id of a plain document, the response side of a chat document, and the
  // EOS that joins documents when the document before it was plain.
  std::unordered_map<uint64_t, uint64_t> seen;  // hash -> first index
  std::vector<int32_t> ids;                     // the packed stream
  std::vector<uint8_t> target;
  std::vector<uint64_t> boundaries;             // ids index of each EOS join
  bool previous_plain = true;
  for (uint64_t d = 0; d < documents.size(); ++d) {
    const Document& doc = documents[d];
    SanitizeCounts counts;
    const std::string clean = Sanitize(doc.text, options.one_document, &counts);
    const std::string clean_response =
        doc.chat ? Sanitize(doc.response, false, &counts) : std::string();
    m.sanitized.invalid_utf8_bytes += counts.invalid_utf8_bytes;
    m.sanitized.control_chars += counts.control_chars;
    m.sanitized.bidi_controls += counts.bidi_controls;
    if (clean.empty() && clean_response.empty()) {
      ++m.dropped_empty;
      m.drops.push_back({d, "empty"});
      continue;
    }
    if (!options.one_document) {
      std::string key = clean;
      if (doc.chat) {
        key += '\x1F';
        key += clean_response;
      }
      const uint64_t h = seeml::update::ContentHash64(
          reinterpret_cast<const uint8_t*>(key.data()), key.size());
      auto [it, fresh] = seen.try_emplace(h, d);
      if (!fresh) {
        ++m.dropped_duplicate;
        m.drops.push_back({d, "duplicate of " + std::to_string(it->second)});
        continue;
      }
    }
    std::vector<int32_t> doc_ids;
    size_t prompt_ids = 0;
    if (doc.chat) {
      // The rendered record is encoded as ONE string, so the pre-tokenizer
      // sees the join between the template's "assistant\n" and the
      // response as the reference does (a response that opens with two or
      // more whitespace characters folds the newline with all but the last
      // of them); the prompt's ids are those of the pieces that end at or
      // before the join, a straddling piece being the response's — the
      // rule an assistant-token mask follows.
      const std::string prompt_side = std::string(kImStart) + "user\n" + clean +
                                      std::string(kImEnd) + "\n" +
                                      std::string(kImStart) + "assistant\n";
      const std::string rendered =
          prompt_side + clean_response + std::string(kImEnd) + "\n";
      prompt_ids = tokenizer.EncodeSplit(rendered, prompt_side.size(), doc_ids);
      if (doc_ids.size() == prompt_ids) {
        ++m.dropped_degenerate;
        m.drops.push_back({d, "degenerate: no response ids"});
        continue;
      }
      ++m.chat_documents;
    } else {
      if (auto ok = tokenizer.Encode(clean, doc_ids); !ok)
        return std::unexpected("ingest: document " + std::to_string(d) + ": " +
                               ok.error());
      if (doc_ids.empty()) {
        ++m.dropped_degenerate;
        m.drops.push_back({d, "degenerate: no ids"});
        continue;
      }
    }
    if (!ids.empty()) {
      ids.push_back(eos);
      target.push_back(previous_plain ? 1 : 0);
      boundaries.push_back(ids.size() - 1);
    }
    ids.insert(ids.end(), doc_ids.begin(), doc_ids.end());
    for (size_t i = 0; i < doc_ids.size(); ++i)
      target.push_back(doc.chat && i < prompt_ids ? 0 : 1);
    previous_plain = !doc.chat;
    ++m.documents_kept;
  }
  m.tokens = ids.size();
  if (ids.size() < options.seq_len + 1)
    return std::unexpected("ingest: " + std::to_string(ids.size()) +
                           " ids is fewer than one record of " +
                           std::to_string(options.seq_len + 1));

  // Pack: consecutive non-overlapping windows of S + 1 ids. The mask of a
  // window holds, per position j, whether the next id (j + 1) is a target;
  // under a mask, a window with no target trains nothing and is dropped.
  const uint64_t S = options.seq_len, rec = S + 1;
  const uint64_t windows = ids.size() / rec;
  m.remainder_tokens = ids.size() - windows * rec;
  const bool masked = any_chat;
  std::vector<int32_t> kept_ids;
  std::vector<uint8_t> kept_mask;
  kept_ids.reserve(windows * rec);
  if (masked) kept_mask.reserve(windows * S);
  {
    size_t b = 0;
    for (uint64_t w = 0; w < windows; ++w) {
      const uint64_t lo = w * rec, hi = lo + rec;  // [lo, hi)
      uint64_t targets = 0;
      for (uint64_t j = 0; j < S; ++j) targets += target[lo + j + 1];
      bool straddles = false;
      while (b < boundaries.size() && boundaries[b] < hi) {
        // An EOS strictly inside the window (not its first id) is a
        // boundary the window spans; an EOS as the window's first id means
        // the window begins a new document.
        if (boundaries[b] > lo) straddles = true;
        ++b;
      }
      if (masked && targets == 0) {
        ++m.dropped_zero_target_windows;
        continue;
      }
      if (straddles) ++m.straddling_windows;
      kept_ids.insert(kept_ids.end(), ids.begin() + static_cast<ptrdiff_t>(lo),
                      ids.begin() + static_cast<ptrdiff_t>(hi));
      if (masked)
        for (uint64_t j = 0; j < S; ++j) kept_mask.push_back(target[lo + j + 1]);
      m.target_tokens += masked ? targets : S;
      ++m.windows;
    }
  }
  if (m.windows == 0)
    return std::unexpected("ingest: every window was dropped (no loss target)");
  m.total_tokens = m.windows * S;
  {
    std::unordered_set<int32_t> distinct(kept_ids.begin(), kept_ids.end());
    m.distinct_ids = distinct.size();
  }

  // The canonical SDS, through the format's one token-corpus writer
  // (source/language/dataset_format.h): v3 when the corpus names its
  // tokenizer or carries a mask, else v2 — the bytes the plan is pinned to.
  using namespace seeml::update;
  std::vector<uint8_t>& sds = result.sds;
  sds.reserve(static_cast<size_t>(
      SdsHeaderBytes(SdsTokenVersion(tokenizer_hash, masked)) +
      m.windows * (rec * sizeof(int32_t) + (masked ? SdsMaskBytes(S) : 0))));
  auto put = [&](const void* p, size_t n) {
    const auto* bytes = static_cast<const uint8_t*>(p);
    sds.insert(sds.end(), bytes, bytes + n);
  };
  WriteSdsTokenCorpus(put, m.windows, S, tokenizer_hash, masked,
                      kept_ids.data(), kept_mask.data());
  result.corpus_hash = ContentHash64(sds.data(), sds.size());
  result.ids = std::move(kept_ids);
  result.mask = std::move(kept_mask);
  result.masked = masked;
  return result;
}

std::string ManifestJson(const IngestManifest& m) {
  std::string s = "{\n";
  auto field = [&](const char* k, uint64_t v) {
    s += "  \"";
    s += k;
    s += "\": " + std::to_string(v) + ",\n";
  };
  field("documents_in", m.documents_in);
  field("documents_kept", m.documents_kept);
  field("chat_documents", m.chat_documents);
  field("dropped_empty", m.dropped_empty);
  field("dropped_duplicate", m.dropped_duplicate);
  field("dropped_degenerate", m.dropped_degenerate);
  field("dropped_zero_target_windows", m.dropped_zero_target_windows);
  field("invalid_utf8_bytes", m.sanitized.invalid_utf8_bytes);
  field("control_chars", m.sanitized.control_chars);
  field("bidi_controls", m.sanitized.bidi_controls);
  field("tokens", m.tokens);
  field("windows", m.windows);
  field("remainder_tokens", m.remainder_tokens);
  field("straddling_windows", m.straddling_windows);
  field("target_tokens", m.target_tokens);
  field("total_tokens", m.total_tokens);
  field("distinct_ids", m.distinct_ids);
  s += "  \"drops\": [";
  for (size_t i = 0; i < m.drops.size(); ++i) {
    s += i ? ",\n    {" : "\n    {";
    s += "\"index\": " + std::to_string(m.drops[i].index) + ", \"reason\": ";
    AppendJsonString(s, m.drops[i].reason);
    s += "}";
  }
  s += m.drops.empty() ? "]\n}\n" : "\n  ]\n}\n";
  return s;
}

}  // namespace seeml::tokenizer
