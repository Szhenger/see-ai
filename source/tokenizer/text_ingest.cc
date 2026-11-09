#include "source/tokenizer/text_ingest.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "source/identity/hash.h"
#include "source/language/dataset_format.h"

namespace seeml::tokenizer {

namespace {

/// Decodes the UTF-8 scalar at `i` into `cp`, returning the index past it,
/// or 0 when the bytes at `i` are not a well-formed sequence (overlong
/// forms, surrogates and values past U+10FFFF included).
size_t DecodeStrict(std::string_view s, size_t i, uint32_t* cp) {
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

}  // namespace

std::string Sanitize(std::string_view text, bool count_only,
                     SanitizeCounts* counts) {
  std::string out;
  if (!count_only) out.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = 0;
    const size_t next = DecodeStrict(text, i, &cp);
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

std::expected<std::vector<std::string>, std::string> DocumentsFromJsonl(
    std::string_view jsonl) {
  std::vector<std::string> docs;
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
    std::string text;
    bool have_text = false;
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
        }
      } else {
        // Any other JSON value: skip to the next top-level ',' or '}' —
        // numbers, booleans, null; nested objects or arrays are refused,
        // since the record format carries strings only.
        if (line[i] == '{' || line[i] == '[')
          return refuse("nested values are not part of a text record");
        while (i < line.size() && line[i] != ',' && line[i] != '}') ++i;
      }
      i = SkipSpaces(line, i);
      if (i < line.size() && line[i] == ',') i = SkipSpaces(line, i + 1);
    }
    if (i >= line.size()) return refuse("truncated object");
    if (!have_text) return refuse("no \"text\" field");
    docs.push_back(std::move(text));
  }
  return docs;
}

std::expected<IngestResult, std::string> Ingest(
    const ByteLevelBpe& tokenizer, uint64_t tokenizer_hash,
    const std::vector<std::string>& documents, const IngestOptions& options) {
  if (options.seq_len == 0) return std::unexpected("ingest: seq_len is zero");
  if (documents.empty()) return std::unexpected("ingest: no documents");
  if (options.one_document && documents.size() != 1)
    return std::unexpected("ingest: one-document mode takes one document");
  const int32_t eos = tokenizer.info().eos_id;
  if (!options.one_document && documents.size() > 1 && eos < 0)
    return std::unexpected(
        "ingest: the tokenizer names no EOS token, which joining documents "
        "needs");

  IngestResult result;
  IngestManifest& m = result.manifest;
  m.documents_in = documents.size();
  result.tokenizer_hash = tokenizer_hash;

  // Sanitize, deduplicate, encode: one pass over the documents, in order.
  std::unordered_map<uint64_t, uint64_t> seen;  // hash -> first index
  std::vector<int32_t> ids;                     // the packed stream
  std::vector<uint64_t> boundaries;             // ids index of each EOS join
  for (uint64_t d = 0; d < documents.size(); ++d) {
    SanitizeCounts counts;
    const std::string clean =
        Sanitize(documents[d], options.one_document, &counts);
    m.sanitized.invalid_utf8_bytes += counts.invalid_utf8_bytes;
    m.sanitized.control_chars += counts.control_chars;
    m.sanitized.bidi_controls += counts.bidi_controls;
    if (clean.empty()) {
      ++m.dropped_empty;
      m.drops.push_back({d, "empty"});
      continue;
    }
    if (!options.one_document) {
      const uint64_t h = seeml::update::ContentHash64(
          reinterpret_cast<const uint8_t*>(clean.data()), clean.size());
      auto [it, fresh] = seen.try_emplace(h, d);
      if (!fresh) {
        ++m.dropped_duplicate;
        m.drops.push_back({d, "duplicate of " + std::to_string(it->second)});
        continue;
      }
    }
    std::vector<int32_t> doc_ids;
    if (auto ok = tokenizer.Encode(clean, doc_ids); !ok)
      return std::unexpected("ingest: document " + std::to_string(d) + ": " +
                             ok.error());
    if (doc_ids.empty()) {
      ++m.dropped_degenerate;
      m.drops.push_back({d, "degenerate: no ids"});
      continue;
    }
    if (!ids.empty()) {
      ids.push_back(eos);
      boundaries.push_back(ids.size() - 1);
    }
    ids.insert(ids.end(), doc_ids.begin(), doc_ids.end());
    ++m.documents_kept;
  }
  m.tokens = ids.size();
  if (ids.size() < options.seq_len + 1)
    return std::unexpected("ingest: " + std::to_string(ids.size()) +
                           " ids is fewer than one record of " +
                           std::to_string(options.seq_len + 1));

  // Pack: consecutive non-overlapping windows of S + 1 ids.
  const uint64_t rec = options.seq_len + 1;
  const uint64_t windows = ids.size() / rec;
  m.windows = windows;
  m.remainder_tokens = ids.size() - windows * rec;
  {
    size_t b = 0;
    for (uint64_t w = 0; w < windows; ++w) {
      const uint64_t lo = w * rec, hi = lo + rec;  // [lo, hi)
      bool straddles = false;
      while (b < boundaries.size() && boundaries[b] < hi) {
        // An EOS strictly inside the window (not its first id) is a
        // boundary the window spans; an EOS as the window's first id means
        // the window begins a new document.
        if (boundaries[b] > lo) straddles = true;
        ++b;
      }
      if (straddles) ++m.straddling_windows;
    }
  }
  {
    std::unordered_set<int32_t> distinct;
    for (uint64_t i = 0; i < windows * rec; ++i) distinct.insert(ids[i]);
    m.distinct_ids = distinct.size();
  }

  // The canonical SDS: v3 when the corpus names its tokenizer, else v2.
  using namespace seeml::update;
  const uint32_t version = tokenizer_hash != 0 ? 3 : 2;
  std::vector<uint8_t>& sds = result.sds;
  auto put = [&](const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    sds.insert(sds.end(), b, b + n);
  };
  const uint32_t label_kind = 1, input_kind = 1, zero32 = 0;
  const uint64_t num_samples = windows, input_dim = options.seq_len,
                 label_dim = 0, zero64 = 0;
  put(&kSdsMagic, 4);
  put(&version, 4);
  put(&num_samples, 8);
  put(&input_dim, 8);
  put(&label_kind, 4);
  put(&input_kind, 4);
  put(&label_dim, 8);
  if (version >= 3) {
    put(&tokenizer_hash, 8);
    put(&zero32, 4);  // flags: no mask
    put(&zero32, 4);
    put(&zero64, 8);
  }
  put(ids.data(), windows * rec * sizeof(int32_t));
  result.corpus_hash = ContentHash64(sds.data(), sds.size());
  return result;
}

std::string ManifestJson(const IngestManifest& m) {
  std::string s = "{\n";
  auto field = [&](const char* k, uint64_t v, bool last = false) {
    s += "  \"";
    s += k;
    s += "\": " + std::to_string(v) + (last ? "\n" : ",\n");
  };
  field("documents_in", m.documents_in);
  field("documents_kept", m.documents_kept);
  field("dropped_empty", m.dropped_empty);
  field("dropped_duplicate", m.dropped_duplicate);
  field("dropped_degenerate", m.dropped_degenerate);
  field("invalid_utf8_bytes", m.sanitized.invalid_utf8_bytes);
  field("control_chars", m.sanitized.control_chars);
  field("bidi_controls", m.sanitized.bidi_controls);
  field("tokens", m.tokens);
  field("windows", m.windows);
  field("remainder_tokens", m.remainder_tokens);
  field("straddling_windows", m.straddling_windows);
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
