#ifndef SEEML_SOURCE_STATISTICS_SCREEN_H_
#define SEEML_SOURCE_STATISTICS_SCREEN_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

// =============================================================================
// Statistics screens (S0 #146; docs/next-project/seeai.md §2). Every
// subsystem states what it did as one screen: a static table of rows — the
// report key, the console label, the kind, the unit, and the one-clause
// derivation of the number — and the values its code filled in. The console
// screen, the report's JSON, the golden blocks the tests pin and the schema
// seeml-abi publishes are four renderings of that one table, so the number a
// reader sees, the number a tool parses and the published name and unit of
// it cannot drift apart.
//
// Header-only: the compiler, the runtime on the device (the vendored package
// carries this file) and the tools render the same way.
// =============================================================================

namespace seeml::update {

/// What a row's value is. Counts and byte counts are exact integers; reals
/// are derived ratios and times; text names a decision or an identity.
enum class StatKind : uint8_t { kCount = 0, kBytes = 1, kReal = 2, kText = 3 };

inline constexpr const char* StatKindName(StatKind kind) {
  switch (kind) {
    case StatKind::kCount: return "count";
    case StatKind::kBytes: return "bytes";
    case StatKind::kReal: return "real";
    case StatKind::kText: return "text";
  }
  return "unknown";
}

/// Printed on the console only with --verbose (the report always has it).
inline constexpr uint8_t kStatVerbose = 1u << 0;
/// Varies with the host, the clock or the bytes' identity (timings, hashes,
/// host memory): left out of the golden blocks.
inline constexpr uint8_t kStatVolatile = 1u << 1;
/// May be left unset (absent from the report, skipped on the console).
inline constexpr uint8_t kStatOptional = 1u << 2;

struct StatRow {
  const char* key;         // the report key (snake_case)
  const char* label;       // the console label
  StatKind kind;
  const char* unit;        // "" for a dimensionless number; bytes carry "B"
  const char* derivation;  // one clause: where the number comes from
  uint8_t flags = 0;
  int precision = 0;       // decimals of a kReal on the console
};

struct StatColumn {
  const char* key;
  StatKind kind;
  const char* unit;
  uint8_t flags = 0;  // kStatVolatile drops the column from golden blocks
  int precision = 0;
};

/// Per-item detail (per pass, per adapter): --verbose on the console,
/// always in the report.
struct StatTable {
  const char* key;
  const char* label;
  const char* derivation;
  std::span<const StatColumn> columns;
};

struct ScreenSpec {
  const char* name;   // "frontend" | "analysis" | "backend" | "runtime"
  const char* title;  // the console heading's clause
  std::span<const StatRow> rows;
  std::span<const StatTable> tables = {};
};

using StatValue = std::variant<std::monostate, uint64_t, double, std::string>;

namespace screen_detail {

inline std::string Grouped(uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)
    s.insert(static_cast<size_t>(i), ",");
  return s;
}

/// The exact count, and above 1 KiB a binary-prefixed gloss at one decimal,
/// computed in integers so it reads the same on every host.
inline std::string Bytes(uint64_t b) {
  std::string s = Grouped(b) + " B";
  if (b < 1024) return s;
  static constexpr const char* kUnits[] = {"KiB", "MiB", "GiB", "TiB", "PiB"};
  uint64_t scale = 1024;
  int unit = 0;
  while (unit < 4 && b / 1024 >= scale) {
    scale *= 1024;
    ++unit;
  }
  const uint64_t whole = b / scale;
  const uint64_t tenths = ((b % scale) * 10 + scale / 2) / scale;
  const uint64_t w = whole + tenths / 10, t = tenths % 10;
  return s + " (" + std::to_string(w) + "." + std::to_string(t) + " " +
         kUnits[unit] + ")";
}

inline std::string Real(double v, int precision) {
  if (!std::isfinite(v)) return "n/a";
  char buf[64];
  // A value that would print as zero at the row's precision (a tiny
  // plan's predicted step) keeps three significant digits instead:
  // "0.000" would say nothing. An exact zero stays fixed.
  double half_unit = 0.5;
  for (int i = 0; i < precision; ++i) half_unit /= 10.0;
  if (v != 0.0 && std::fabs(v) < half_unit)
    std::snprintf(buf, sizeof(buf), "%.3g", v);
  else
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return buf;
}

inline std::string JsonString(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
      out += buf;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

inline std::string JsonReal(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

inline std::string WithUnit(std::string value, const char* unit) {
  if (unit && *unit) value += std::string(" ") + unit;
  return value;
}

inline std::string Display(StatKind kind, const char* unit, int precision,
                           const StatValue& v) {
  if (std::holds_alternative<std::monostate>(v)) return "—";
  switch (kind) {
    case StatKind::kCount:
      return WithUnit(Grouped(std::get<uint64_t>(v)), unit);
    case StatKind::kBytes:
      return Bytes(std::get<uint64_t>(v));
    case StatKind::kReal:
      return WithUnit(Real(std::get<double>(v), precision), unit);
    case StatKind::kText:
      return WithUnit(std::get<std::string>(v), unit);
  }
  return "?";
}

inline std::string Json(StatKind kind, const StatValue& v) {
  switch (kind) {
    case StatKind::kCount:
    case StatKind::kBytes:
      return std::to_string(std::get<uint64_t>(v));
    case StatKind::kReal:
      return JsonReal(std::get<double>(v));
    case StatKind::kText:
      return JsonString(std::get<std::string>(v));
  }
  return "null";
}

inline bool Fits(StatKind kind, const StatValue& v) {
  switch (kind) {
    case StatKind::kCount:
    case StatKind::kBytes:
      return std::holds_alternative<uint64_t>(v);
    case StatKind::kReal:
      return std::holds_alternative<double>(v);
    case StatKind::kText:
      return std::holds_alternative<std::string>(v);
  }
  return false;
}

inline std::string Padded(std::string s, size_t width) {
  // Pad by display columns, not bytes: "—" and "·" are one column each.
  size_t columns = 0;
  for (const char c : s)
    if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++columns;
  if (columns < width) s.append(width - columns, ' ');
  return s;
}

}  // namespace screen_detail

class Screen {
 public:
  explicit Screen(const ScreenSpec& spec)
      : spec_(&spec),
        values_(spec.rows.size()),
        tables_(spec.tables.size()) {}

  const ScreenSpec& spec() const { return *spec_; }

  void Set(std::string_view key, uint64_t v) { Store(key, StatValue(v)); }
  void Set(std::string_view key, double v) { Store(key, StatValue(v)); }
  void Set(std::string_view key, std::string v) {
    Store(key, StatValue(std::move(v)));
  }
  void Set(std::string_view key, const char* v) { Set(key, std::string(v)); }
  /// Any other integer type (size_t, int, uint32_t) is a count; a negative
  /// one is a bug, recorded as a problem rather than wrapped.
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool> &&
             !std::is_same_v<T, uint64_t>)
  void Set(std::string_view key, T v) {
    if constexpr (std::is_signed_v<T>) {
      if (v < 0) {
        problems_.push_back(std::string("negative count for '") +
                            std::string(key) + "'");
        return;
      }
    }
    Set(key, static_cast<uint64_t>(v));
  }

  /// One row of a detail table; the cells follow the table's columns.
  void AddRow(std::string_view table, std::vector<StatValue> cells) {
    const int t = TableIndex(table);
    if (t < 0) {
      problems_.push_back("unknown table '" + std::string(table) + "'");
      return;
    }
    const StatTable& spec = spec_->tables[static_cast<size_t>(t)];
    if (cells.size() != spec.columns.size()) {
      problems_.push_back("table '" + std::string(table) + "' row has " +
                          std::to_string(cells.size()) + " cells, not " +
                          std::to_string(spec.columns.size()));
      return;
    }
    for (size_t c = 0; c < cells.size(); ++c)
      if (!screen_detail::Fits(spec.columns[c].kind, cells[c])) {
        problems_.push_back("table '" + std::string(table) + "' column '" +
                            spec.columns[c].key + "' has the wrong kind");
        return;
      }
    tables_[static_cast<size_t>(t)].push_back(std::move(cells));
  }

  /// The value of `key`, or nullptr when the key is unknown or unset.
  const StatValue* Get(std::string_view key) const {
    const int i = RowIndex(key);
    if (i < 0 || std::holds_alternative<std::monostate>(
                     values_[static_cast<size_t>(i)]))
      return nullptr;
    return &values_[static_cast<size_t>(i)];
  }

  /// Every way the filled screen departs from its spec: unknown keys, kind
  /// mismatches, malformed table rows, and unset rows that are not
  /// optional. Empty means complete. The tests hold every screen to this.
  std::vector<std::string> Problems() const {
    std::vector<std::string> out = problems_;
    for (size_t i = 0; i < values_.size(); ++i)
      if (std::holds_alternative<std::monostate>(values_[i]) &&
          !(spec_->rows[i].flags & kStatOptional))
        out.push_back(std::string("unset row '") + spec_->rows[i].key + "'");
    return out;
  }

  /// The console screen: a heading, one line per row (label, value with its
  /// unit, the derivation clause), and with `verbose` the verbose rows and
  /// the detail tables.
  std::string Text(bool verbose) const {
    using screen_detail::Padded;
    std::string out = std::string(spec_->name) + " — " + spec_->title + "\n";
    for (size_t i = 0; i < values_.size(); ++i) {
      const StatRow& row = spec_->rows[i];
      if (std::holds_alternative<std::monostate>(values_[i])) continue;
      if ((row.flags & kStatVerbose) && !verbose) continue;
      out += "  " + Padded(row.label, 24) + " " +
             Padded(screen_detail::Display(row.kind, row.unit, row.precision,
                                           values_[i]),
                    28) +
             "  " + row.derivation + "\n";
    }
    if (!verbose) return out;
    for (size_t t = 0; t < tables_.size(); ++t) {
      if (tables_[t].empty()) continue;
      const StatTable& table = spec_->tables[t];
      out += std::string("  ") + table.label + " — " + table.derivation +
             "\n";
      for (const auto& cells : tables_[t]) {
        std::string line = "   ";
        for (size_t c = 0; c < cells.size(); ++c) {
          const StatColumn& col = table.columns[c];
          line += " " + Padded(screen_detail::Display(col.kind, col.unit,
                                                      col.precision, cells[c]),
                               c == 0 ? 28 : 14);
        }
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out += line + "\n";
      }
    }
    return out;
  }

  /// The report's object: every set row by key (counts and bytes as
  /// integers, reals at full precision, null when not finite), then every
  /// detail table as an array of objects. `indent` is the column of the
  /// opening brace's line, so the object nests inside a larger document.
  std::string Json(int indent) const {
    const std::string pad(static_cast<size_t>(indent) + 2, ' ');
    std::string out = "{";
    bool first = true;
    auto next = [&] {
      out += first ? "\n" : ",\n";
      first = false;
    };
    for (size_t i = 0; i < values_.size(); ++i) {
      if (std::holds_alternative<std::monostate>(values_[i])) continue;
      next();
      out += pad + screen_detail::JsonString(spec_->rows[i].key) + ": " +
             screen_detail::Json(spec_->rows[i].kind, values_[i]);
    }
    for (size_t t = 0; t < tables_.size(); ++t) {
      const StatTable& table = spec_->tables[t];
      next();
      out += pad + screen_detail::JsonString(table.key) + ": [";
      for (size_t r = 0; r < tables_[t].size(); ++r) {
        out += r == 0 ? "" : ", ";
        out += "{";
        for (size_t c = 0; c < table.columns.size(); ++c) {
          out += (c == 0 ? "" : ", ") +
                 screen_detail::JsonString(table.columns[c].key) + ": " +
                 screen_detail::Json(table.columns[c].kind, tables_[t][r][c]);
        }
        out += "}";
      }
      out += "]";
    }
    out += "\n" + std::string(static_cast<size_t>(indent), ' ') + "}";
    return out;
  }

  /// The golden block the tests pin: every set row that is not volatile,
  /// verbose ones included, as `key = value` in the console's formatting,
  /// then every detail table's rows without their volatile columns — a
  /// pass that changes what it does changes a line here.
  std::string Golden() const {
    std::string out = std::string("[") + spec_->name + "]\n";
    for (size_t i = 0; i < values_.size(); ++i) {
      const StatRow& row = spec_->rows[i];
      if (std::holds_alternative<std::monostate>(values_[i])) continue;
      if (row.flags & kStatVolatile) continue;
      out += std::string(row.key) + " = " +
             screen_detail::Display(row.kind, row.unit, row.precision,
                                    values_[i]) +
             "\n";
    }
    for (size_t t = 0; t < tables_.size(); ++t) {
      const StatTable& table = spec_->tables[t];
      for (size_t r = 0; r < tables_[t].size(); ++r) {
        std::string line =
            std::string(table.key) + "[" + std::to_string(r) + "] =";
        for (size_t c = 0; c < table.columns.size(); ++c) {
          const StatColumn& col = table.columns[c];
          if (col.flags & kStatVolatile) continue;
          line += " " + screen_detail::Display(col.kind, col.unit,
                                               col.precision, tables_[t][r][c]);
        }
        out += line + "\n";
      }
    }
    return out;
  }

 private:
  int RowIndex(std::string_view key) const {
    for (size_t i = 0; i < spec_->rows.size(); ++i)
      if (key == spec_->rows[i].key) return static_cast<int>(i);
    return -1;
  }
  int TableIndex(std::string_view key) const {
    for (size_t i = 0; i < spec_->tables.size(); ++i)
      if (key == spec_->tables[i].key) return static_cast<int>(i);
    return -1;
  }
  void Store(std::string_view key, StatValue v) {
    const int i = RowIndex(key);
    if (i < 0) {
      problems_.push_back("unknown row '" + std::string(key) + "'");
      return;
    }
    if (!screen_detail::Fits(spec_->rows[static_cast<size_t>(i)].kind, v)) {
      problems_.push_back("row '" + std::string(key) + "' set with the wrong kind (it is " +
                          StatKindName(spec_->rows[static_cast<size_t>(i)].kind) + ")");
      return;
    }
    values_[static_cast<size_t>(i)] = std::move(v);
  }

  const ScreenSpec* spec_;
  std::vector<StatValue> values_;
  std::vector<std::vector<std::vector<StatValue>>> tables_;
  std::vector<std::string> problems_;
};

}  // namespace seeml::update

#endif  // SEEML_SOURCE_STATISTICS_SCREEN_H_
