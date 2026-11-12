#include "compiler/frontend/ingressor/export_report.h"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include "compiler/diagnostics/tokenizing/error.h"
#include "source/identity/hash.h"
#include "source/recipe/recipe.h"
#include "source/recipe/recipe_file.h"

namespace seeml::update {

namespace {

using diag::tokenizing::Error;

struct Field {
  enum Kind { kNumber, kString, kBool } kind;
  std::string text;
};

/// The report as a flat object over a fixed key set — the recipe file's
/// strict cursor, another vocabulary: an unknown or duplicate key, a value
/// of the wrong JSON type or anything after the object is refused.
std::expected<std::map<std::string, Field>, std::string> Parse(
    std::string_view text) {
  static constexpr struct {
    std::string_view key;
    Field::Kind kind;
  } kKeys[] = {{"schema", Field::kNumber},
               {"smf_hash", Field::kString},
               {"tokenizer_hash", Field::kString},
               {"tokenizer_parity_checked", Field::kBool},
               {"tokenizer_parity_ids", Field::kNumber},
               {"logits_parity_checked", Field::kBool},
               {"logits_parity_max_delta", Field::kNumber}};
  recipe::file_detail::Cursor c{text};
  if (!c.Eat('{')) return std::unexpected("not a JSON object");
  std::map<std::string, Field> out;
  if (!c.Eat('}')) {
    while (true) {
      std::string key;
      if (!c.String(&key)) return std::unexpected("expected a key in quotes");
      const Field::Kind* kind = nullptr;
      for (const auto& k : kKeys)
        if (k.key == key) kind = &k.kind;
      if (!kind) return std::unexpected("unknown key '" + key + "'");
      if (out.contains(key))
        return std::unexpected("key '" + key + "' appears twice");
      if (!c.Eat(':')) return std::unexpected("expected ':' after '" + key + "'");
      Field f{*kind, {}};
      if (*kind == Field::kString) {
        if (!c.String(&f.text))
          return std::unexpected("key '" + key + "' takes a string");
      } else {
        c.Skip();
        if (c.i < c.s.size() && (c.s[c.i] == '"' || c.s[c.i] == '{' ||
                                 c.s[c.i] == '['))
          return std::unexpected("key '" + key + "' takes a bare value");
        f.text = std::string(c.Token());
        const bool is_bool = f.text == "true" || f.text == "false";
        if (f.text.empty() || is_bool != (*kind == Field::kBool))
          return std::unexpected(
              "key '" + key + "' takes " +
              (*kind == Field::kBool ? "true or false" : "a number"));
      }
      out.emplace(std::move(key), std::move(f));
      if (c.Eat(',')) continue;
      if (c.Eat('}')) break;
      return std::unexpected("expected ',' or '}'");
    }
  }
  c.Skip();
  if (c.i != c.s.size()) return std::unexpected("content after the closing '}'");
  for (const auto& k : kKeys)
    if (!out.contains(std::string(k.key)))
      return std::unexpected("key '" + std::string(k.key) + "' is missing");
  return out;
}

bool ParseHex16(const std::string& s, uint64_t* out) {
  if (s.size() != 16) return false;
  uint64_t v = 0;
  for (const char ch : s) {
    uint64_t d = 0;
    if (ch >= '0' && ch <= '9') d = static_cast<uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') d = static_cast<uint64_t>(ch - 'a' + 10);
    else return false;
    v = (v << 4) | d;
  }
  *out = v;
  return true;
}

}  // namespace

std::expected<ExportReport, std::string> ReadExportReport(
    const std::string& smf_path, const SmfModel& model) {
  const std::string path = smf_path + kExportReportSuffix;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (NeedsExportReport(model))
      return Error(
          "no exporter report '" + path +
          "': the model carries a tokenizer section — an imported checkpoint "
          "— and the exporter's output is verified, not trusted; re-export "
          "it with tool/export_model.py --hf, which writes the report");
    return ExportReport{};
  }
  std::stringstream buffer;
  buffer << f.rdbuf();
  const std::string text = buffer.str();
  auto bad = [&](const std::string& why) {
    return Error("exporter report '" + path + "': " + why);
  };
  auto parsed = Parse(text);
  if (!parsed) return bad(parsed.error());
  auto& p = *parsed;
  if (p["schema"].text != "1")
    return bad("unsupported schema " + p["schema"].text);
  uint64_t smf_hash = 0, tokenizer_hash = 0;
  if (!ParseHex16(p["smf_hash"].text, &smf_hash) ||
      !ParseHex16(p["tokenizer_hash"].text, &tokenizer_hash))
    return bad("a hash is not 16 lowercase hex digits");
  // The binding: this report is about these bytes.
  if (smf_hash != model.content_hash)
    return bad("it describes another model file (smf_hash " +
               p["smf_hash"].text +
               " is not this file's): the model was changed after the export, "
               "or the report belongs to another export");
  if (tokenizer_hash != model.TokenizerHash())
    return bad("its tokenizer_hash is not the model's tokenizer section's");
  ExportReport r;
  r.present = true;
  r.report_hash = ContentHash64(
      reinterpret_cast<const uint8_t*>(text.data()), text.size());
  r.tokenizer_parity = p["tokenizer_parity_checked"].text == "true";
  r.logits_parity = p["logits_parity_checked"].text == "true";
  double delta = 0.0;
  if (!recipe::ParseU64Strict(p["tokenizer_parity_ids"].text,
                              &r.tokenizer_parity_ids))
    return bad("tokenizer_parity_ids is not a non-negative integer");
  if (!recipe::ParseF64Strict(p["logits_parity_max_delta"].text, &delta) ||
      delta < 0.0)
    return bad("logits_parity_max_delta is not a finite non-negative number");
  r.logits_parity_max_delta = static_cast<float>(delta);
  if (!std::isfinite(r.logits_parity_max_delta))
    return bad("logits_parity_max_delta is not a finite non-negative number");
  // A figure without its check, and a check without a figure, say nothing.
  if (!r.tokenizer_parity && r.tokenizer_parity_ids != 0)
    return bad("tokenizer_parity_ids is set but the parity was not checked");
  if (r.tokenizer_parity && r.tokenizer_parity_ids == 0)
    return bad("the tokenizer parity is marked checked over zero ids");
  if (!r.logits_parity && r.logits_parity_max_delta != 0.0f)
    return bad("logits_parity_max_delta is set but the parity was not checked");
  return r;
}

}  // namespace seeml::update
