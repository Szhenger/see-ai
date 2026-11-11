#ifndef SEEML_SOURCE_RECIPE_RECIPE_FILE_H_
#define SEEML_SOURCE_RECIPE_RECIPE_FILE_H_

#include <expected>
#include <string>
#include <string_view>

#include "source/recipe/recipe.h"

// =============================================================================
// The recipe file (S4 #150): one flat JSON object over the recipe's field
// names, e.g.
//
//   {"loss": "xent", "lora_rank": 8, "targets": ["wq", "wv"],
//    "lr": 1e-4, "lr_schedule": "cosine", "warmup": 10, "steps": 300,
//    "quantize_base": true}
//
// A purpose-built strict reader, not a JSON library: an unknown key, a
// duplicate key, a value of the wrong JSON type for its field, a nested
// value, or anything after the object is refused, naming the key. Values are
// returned as text for recipe::Apply, which holds each to its bound — the
// one place a bound is stated.
// =============================================================================

namespace seeml::update::recipe {

namespace file_detail {

struct Cursor {
  std::string_view s;
  size_t i = 0;
  void Skip() {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
      ++i;
  }
  bool Eat(char c) {
    Skip();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  /// A JSON string with the escapes a name or a path needs; \u is refused
  /// (no recipe value needs one, and half-decoding one would be worse).
  bool String(std::string* out) {
    Skip();
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\') {
        if (i >= s.size()) return false;
        const char e = s[i++];
        if (e == '"' || e == '\\' || e == '/') c = e;
        else if (e == 'n') c = '\n';
        else if (e == 't') c = '\t';
        else return false;
      } else if (static_cast<unsigned char>(c) < 0x20) {
        return false;
      }
      out->push_back(c);
    }
    if (i >= s.size()) return false;
    ++i;  // the closing quote
    return true;
  }
  /// A bare token (number, true, false) up to the next delimiter.
  std::string_view Token() {
    Skip();
    const size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
      ++i;
    return s.substr(start, i - start);
  }
};

}  // namespace file_detail

/// Parses a recipe file's text into field values. Each value's JSON type
/// must be its field's: a number for uint / float fields, a string for enum
/// fields, true / false for bool fields, an array of strings for list
/// fields.
inline std::expected<Values, std::string> ParseRecipeFile(
    std::string_view text) {
  file_detail::Cursor c{text};
  auto bad = [](const std::string& what) {
    return std::unexpected("recipe file: " + what);
  };
  if (!c.Eat('{')) return bad("not a JSON object");
  Values values;
  if (!c.Eat('}')) {
    while (true) {
      std::string key;
      if (!c.String(&key)) return bad("expected a field name in quotes");
      const Field* f = FindField(key);
      if (!f) return bad("unknown field '" + key + "'");
      if (values.contains(key)) return bad("field '" + key + "' appears twice");
      if (!c.Eat(':')) return bad("expected ':' after '" + key + "'");
      auto wrong = [&](const char* want) {
        return bad("field '" + key + "' takes " + want);
      };
      std::string value;
      c.Skip();
      const char first = c.i < c.s.size() ? c.s[c.i] : '\0';
      if (first == '{') return wrong("no nested object");
      switch (f->kind) {
        case Kind::kUInt:
        case Kind::kFloat:
          if (first == '"' || first == '[') return wrong("a number");
          value = std::string(c.Token());
          if (value.empty() || value == "true" || value == "false" ||
              value == "null")
            return wrong("a number");
          break;
        case Kind::kBool:
          value = std::string(c.Token());
          if (value != "true" && value != "false")
            return wrong("true or false");
          break;
        case Kind::kEnum:
          if (!c.String(&value)) return wrong("a string");
          break;
        case Kind::kList: {
          if (!c.Eat('[')) return wrong("an array of strings");
          bool any = false;
          while (!c.Eat(']')) {
            if (any && !c.Eat(',')) return wrong("an array of strings");
            std::string item;
            if (!c.String(&item)) return wrong("an array of strings");
            if (item.empty() || item.find(',') != std::string::npos)
              return bad("field '" + key + "' holds an empty name or a comma");
            value += (any ? "," : "") + item;
            any = true;
          }
          break;
        }
      }
      values.emplace(std::move(key), std::move(value));
      if (c.Eat(',')) continue;
      if (c.Eat('}')) break;
      return bad("expected ',' or '}'");
    }
  }
  c.Skip();
  if (c.i != c.s.size()) return bad("content after the closing '}'");
  return values;
}

}  // namespace seeml::update::recipe

#endif  // SEEML_SOURCE_RECIPE_RECIPE_FILE_H_
