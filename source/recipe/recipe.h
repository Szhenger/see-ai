#ifndef SEEML_SOURCE_RECIPE_RECIPE_H_
#define SEEML_SOURCE_RECIPE_RECIPE_H_

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "source/identity/hash.h"
#include "source/plan/config.h"
#include "source/plan/instruction.h"

// =============================================================================
// The recipe (S4 #150): how a fine-tune is configured, as one declarative
// table instead of forty hand-parsed flags.
//
// Every field has a name (the recipe file's key), a flag (the command
// line's), a kind and — for every numeric one — a bound. A recipe file is a
// flat JSON object over the field names; flags override its fields; the
// result is applied to an UpdateConfig by ONE function that parses each
// value completely, holds it to its bound, and then holds the fields to each
// other (a teacher's knobs need a distillation loss, the cosine schedule's
// need the cosine schedule, a split needs the corpus, targets name
// something). Every refusal names the field and its bound.
//
// The resolved recipe — every field, defaults included, in table order — is
// a text the plan's identity can hash (ResolvedText / RecipeHash).
//
// seeml-abi publishes the table as the schema (tool/seeml/abi.json
// "recipe"); the generated runner validates its own flags with the sibling
// header run_flags.h. Header-only and dependency-free: it is vendored.
// =============================================================================

namespace seeml::update::recipe {

enum class Kind : uint8_t { kUInt, kFloat, kEnum, kBool, kList };

inline constexpr const char* KindName(Kind k) {
  switch (k) {
    case Kind::kUInt: return "uint";
    case Kind::kFloat: return "float";
    case Kind::kEnum: return "enum";
    case Kind::kBool: return "bool";
    case Kind::kList: return "list";
  }
  return "";
}

struct Field {
  const char* name;     // the recipe file's key
  const char* flag;     // the command line's flag
  Kind kind;
  // kUInt: [umin, umax]. kFloat: fmin .. fmax, each end open or closed.
  uint64_t umin = 0, umax = 0;
  double fmin = 0.0, fmax = 0.0;
  bool min_open = false, max_open = false;
  const char* choices = "";  // kEnum: '|'-separated
  const char* doc = "";
};

/// The largest LoRA rank a recipe may ask for (review N18: the flag was
/// unbounded, and K x r overflowing the arena arithmetic emitted an invalid
/// plan). Far above any rank that trains; adapters are low-rank by purpose.
inline constexpr uint64_t kLoraRankMax = 4096;
inline constexpr uint64_t kStepsMax = uint64_t{1} << 40;
inline constexpr uint64_t kMibMax = uint64_t{1} << 40;
inline constexpr uint64_t kU64Max = ~uint64_t{0};

inline constexpr Field kFields[] = {
    {"data_batch", "--data-batch", Kind::kUInt, 1, uint64_t{1} << 31, 0, 0,
     false, false, "", "rows per compiled batch (tokens for a token model)"},
    {"steps", "--steps", Kind::kUInt, 1, kStepsMax, 0, 0, false, false, "",
     "the default optimizer-step budget and the schedule's horizon"},
    {"grad_accum", "--grad-accum", Kind::kUInt, 1, 0xFFFFFFFFull, 0, 0, false,
     false, "", "micro-batches folded into one optimizer step"},
    {"loss", "--loss", Kind::kEnum, 0, 0, 0, 0, false, false,
     "xent|mse|kl|xent+kl", "the objective"},
    {"distill_weight", "--distill-weight", Kind::kFloat, 0, 0, 0.0, 1.0, false,
     false, "", "weight of the KL term under xent+kl"},
    {"temperature", "--temperature", Kind::kFloat, 0, 0, kKlTemperatureMin,
     kKlTemperatureMax, false, false, "", "distillation softmax temperature"},
    {"lora_rank", "--lora-rank", Kind::kUInt, 1, kLoraRankMax, 0, 0, false,
     false, "", "adapter rank r"},
    {"lora_alpha", "--lora-alpha", Kind::kFloat, 0, 0, 0.0, 1e6, true, false,
     "", "adapter scale numerator: the update is (alpha / r) A B"},
    {"lora_scaling", "--lora-scaling", Kind::kEnum, 0, 0, 0, 0, false, false,
     "alpha_over_r|alpha_over_sqrt_r",
     "the adapter scale: alpha / r, or rank-stabilized alpha / sqrt(r)"},
    {"lora_seed", "--lora-seed", Kind::kUInt, 0, kU64Max, 0, 0, false, false,
     "", "seed of the adapters' initial image"},
    {"targets", "--targets", Kind::kList, 0, 0, 0, 0, false, false, "",
     "substrings naming the weights to adapt; omitted = every eligible one"},
    {"optimizer", "--optimizer", Kind::kEnum, 0, 0, 0, 0, false, false,
     "adamw|sgd", "the update rule"},
    {"lr", "--lr", Kind::kFloat, 0, 0, 0.0, 10.0, true, false, "",
     "peak learning rate"},
    {"weight_decay", "--weight-decay", Kind::kFloat, 0, 0, 0.0, 10.0, false,
     false, "", "decoupled weight decay"},
    {"clip_norm", "--clip-norm", Kind::kFloat, 0, 0, 0.0, 1e6, false, false,
     "", "per-tensor gradient-norm clip; 0 = off"},
    {"lr_schedule", "--lr-schedule", Kind::kEnum, 0, 0, 0, 0, false, false,
     "const|cosine", "learning-rate schedule"},
    {"warmup", "--warmup", Kind::kUInt, 0, kStepsMax, 0, 0, false, false, "",
     "linear warmup steps (cosine only); shorter than steps"},
    {"min_lr_factor", "--min-lr-factor", Kind::kFloat, 0, 0, 0.0, 1.0, false,
     false, "", "the cosine floor as a fraction of lr (cosine only)"},
    {"allow_zero_lr", "--allow-zero-lr", Kind::kBool, 0, 0, 0, 0, false, false,
     "", "admit a cosine floor of exactly 0 (cosine only)"},
    {"quantize_base", "--quantize-base", Kind::kBool, 0, 0, 0, 0, false, false,
     "", "store frozen weights as int8 in the plan"},
    {"bf16_base", "--bf16-base", Kind::kBool, 0, 0, 0, 0, false, false, "",
     "store frozen weights as bfloat16 in the plan"},
    {"precision", "--precision", Kind::kEnum, 0, 0, 0, 0, false, false,
     "f32|certified-bf16", "arithmetic of the frozen-weight products"},
    {"attention", "--attention", Kind::kEnum, 0, 0, 0, 0, false, false,
     "auto|cached|tiled", "attention memory family"},
    {"attention_cache_budget_mib", "--attention-cache-budget-mib", Kind::kUInt,
     0, kMibMax, 0, 0, false, false, "",
     "probability-cache budget auto tiles above (attention auto only)"},
    {"remat", "--remat", Kind::kEnum, 0, 0, 0, 0, false, false, "auto|off|on",
     "rematerialization of cheap forward values in the backward"},
    {"adapter_only", "--adapter-only", Kind::kBool, 0, 0, 0, 0, false, false,
     "", "frozen weights read from the source model; the plan carries none"},
    {"validation_scores", "--validation-scores", Kind::kEnum, 0, 0, 0, 0,
     false, false, "shipped|plan",
     "which weights a narrow-base plan's eval program reads"},
    {"val_frac", "--val-frac", Kind::kFloat, 0, 0, 0.0, 1.0, true, true, "",
     "fraction of the corpus held out, compiled into the plan (needs data)"},
    {"split_seed", "--split-seed", Kind::kUInt, 0, kU64Max, 0, 0, false, false,
     "", "seed of the compiled held-out split (needs val_frac)"},
    {"masked_loss", "--masked-loss", Kind::kBool, 0, 0, 0, 0, false, false, "",
     "cross-entropy over target positions alone"},
    {"memory_budget_mib", "--memory-budget-mib", Kind::kUInt, 1, kMibMax, 0, 0,
     false, false, "", "the memory gate's budget; unset = the host's memory"},
};

inline const Field* FindField(std::string_view name) {
  for (const Field& f : kFields)
    if (name == f.name) return &f;
  return nullptr;
}

/// Field name -> the value as text, for the fields a recipe file or a flag
/// set explicitly. Booleans are "true" / "false"; lists are comma-joined.
using Values = std::map<std::string, std::string>;

// --- Strict scalar parsing ------------------------------------------------------

/// Decimal digits only: no sign, no whitespace, no overflow. strtoull alone
/// skips leading spaces and negates a "-1" into 2^64 - 1, which is how
/// `--steps " -1"` reached the plan (review F56 / F63).
inline bool ParseU64Strict(std::string_view s, uint64_t* out) {
  if (s.empty()) return false;
  uint64_t v = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
    const uint64_t d = static_cast<uint64_t>(c - '0');
    if (v > (kU64Max - d) / 10) return false;
    v = v * 10 + d;
  }
  *out = v;
  return true;
}

/// A finite decimal float, consumed completely, with no leading whitespace:
/// strtod accepts "nan", "inf" and " 1", and every range check is a
/// comparison a NaN silently passes. The decimal alphabet is checked first,
/// because strtod also takes a hex float ("0x1p-3"), skips every isspace()
/// prefix ("\r1", "\v1"), and stops at an embedded NUL as if the text ended
/// there.
inline bool ParseF64Strict(std::string_view s, double* out) {
  if (s.empty() || s.front() == '+') return false;
  for (const char c : s)
    if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-'))
      return false;
  const std::string text(s);
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (errno != 0 || end != text.c_str() + text.size() || !std::isfinite(v))
    return false;
  *out = v;
  return true;
}

// --- Diagnostics -----------------------------------------------------------------

inline std::string Number(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

/// "an integer in [1, 4096]", "a number in (0, 10]", "one of a|b".
inline std::string BoundText(const Field& f) {
  switch (f.kind) {
    case Kind::kUInt:
      if (f.umax == kU64Max && f.umin == 0) return "a non-negative integer";
      return "an integer in [" + std::to_string(f.umin) + ", " +
             std::to_string(f.umax) + "]";
    case Kind::kFloat:
      return std::string("a finite number in ") + (f.min_open ? "(" : "[") +
             Number(f.fmin) + ", " + Number(f.fmax) + (f.max_open ? ")" : "]");
    case Kind::kEnum: return std::string("one of ") + f.choices;
    case Kind::kBool: return "true or false";
    case Kind::kList: return "a comma-separated list of names";
  }
  return "";
}

/// "--lora-rank (recipe field lora_rank)".
inline std::string Named(const Field& f) {
  return std::string(f.flag) + " (recipe field " + f.name + ")";
}

inline std::unexpected<std::string> Refuse(const Field& f,
                                           std::string_view value) {
  return std::unexpected(Named(f) + " must be " + BoundText(f) + ", got '" +
                         std::string(value) + "'");
}

inline bool InChoices(const Field& f, std::string_view v) {
  std::string_view rest = f.choices;
  while (!rest.empty()) {
    const size_t bar = rest.find('|');
    const std::string_view one = rest.substr(0, bar);
    if (one == v) return true;
    if (bar == std::string_view::npos) break;
    rest.remove_prefix(bar + 1);
  }
  return false;
}

/// One value held to its field's kind and bound. `u` / `d` receive the
/// parsed number for the numeric kinds; a float field's `d` is the f32 the
/// config stores, widened, so narrowing it back is exact.
inline std::expected<void, std::string> CheckValue(const Field& f,
                                                   std::string_view v,
                                                   uint64_t* u, double* d) {
  switch (f.kind) {
    case Kind::kUInt:
      if (!ParseU64Strict(v, u) || *u < f.umin || *u > f.umax)
        return Refuse(f, v);
      return {};
    case Kind::kFloat: {
      // Held as the f32 the config stores: "0.001" is the f32 bound 1e-3f,
      // though as doubles the text is the smaller of the two.
      if (!ParseF64Strict(v, d)) return Refuse(f, v);
      const bool nonzero = *d != 0.0;
      // The f32 is the text's own nearest one (strtof, as the hand-parsed
      // flags read it), not the nearest double's narrowed: a text within
      // half a double ulp of the midpoint of two floats would round twice
      // and land on the other one — "1.0000000596046448" is 0x3f800001,
      // and 0x3f800000 through the double. ParseF64Strict has held the text
      // to a complete decimal, so strtof consumes all of it; an overflow
      // comes back infinite and the bound below refuses it.
      const float x = std::strtof(std::string(v).c_str(), nullptr);
      // One zero: a written "-0" is the zero a closed bound admits, and
      // -0.0f would be other header bits — another plan hash and another
      // recipe hash — for the recipe "0" writes.
      *d = x == 0.0f ? 0.0 : static_cast<double>(x);
      // A nonzero text that is zero or subnormal as that f32 underflowed
      // (strtof's ERANGE, which the hand-parsed flags refused): "1e-60"
      // would otherwise pass a closed lower bound of 0 as exactly 0 —
      // clip_norm off, weight_decay none — which is not what was written.
      if (std::isfinite(x) && nonzero && !std::isnormal(x))
        return std::unexpected(Refuse(f, v).error() +
                               ": it underflows the f32 the plan stores");
      const float lo = static_cast<float>(f.fmin);
      const float hi = static_cast<float>(f.fmax);
      if (!std::isfinite(x) || (f.min_open ? !(x > lo) : !(x >= lo)) ||
          (f.max_open ? !(x < hi) : !(x <= hi)))
        return Refuse(f, v);
      return {};
    }
    case Kind::kEnum:
      if (!InChoices(f, v)) return Refuse(f, v);
      return {};
    case Kind::kBool:
      if (v != "true" && v != "false") return Refuse(f, v);
      return {};
    case Kind::kList:
      return {};
  }
  return {};
}

// --- Applying a recipe -------------------------------------------------------------

/// What the recipe is applied beside: inputs that are paths, not fields.
struct Inputs {
  bool teacher = false;  // a teacher model was named
  bool data = false;     // a corpus was named
};

/// Applies `values` to `config`: each value parsed completely and held to
/// its bound, then the fields held to each other. Fields absent from
/// `values` keep `config`'s defaults. The first violation is returned,
/// naming the field and its bound — and `config` is then untouched: the
/// values are applied to a copy that replaces it only when every rule
/// holds, so a refused recipe never leaves half of itself behind (the
/// target filters cleared, say, which reads as "adapt every weight").
inline std::expected<void, std::string> Apply(const Values& values,
                                              const Inputs& inputs,
                                              UpdateConfig* config) {
  UpdateConfig c = *config;
  for (const auto& [name, text] : values) {
    const Field* f = FindField(name);
    if (!f)
      return std::unexpected("unknown recipe field '" + name + "'");
    uint64_t u = 0;
    double d = 0.0;
    if (auto ok = CheckValue(*f, text, &u, &d); !ok) return ok;
    const std::string_view n = name, v = text;
    const bool on = v == "true";
    if (n == "data_batch") c.batch = static_cast<int64_t>(u);
    else if (n == "steps") c.default_steps = u;
    else if (n == "grad_accum") c.grad_accum_steps = static_cast<uint32_t>(u);
    else if (n == "loss")
      c.loss = v == "xent"  ? LossKind::kSoftmaxXEnt
               : v == "mse" ? LossKind::kMse
               : v == "kl"  ? LossKind::kKLDistill
                            : LossKind::kXEntPlusKL;
    else if (n == "distill_weight") c.distill_weight = static_cast<float>(d);
    else if (n == "temperature") c.temperature = static_cast<float>(d);
    else if (n == "lora_rank") c.lora.rank = static_cast<int64_t>(u);
    else if (n == "lora_alpha") c.lora.alpha = static_cast<float>(d);
    else if (n == "lora_scaling")
      c.lora.scaling = v == "alpha_over_sqrt_r" ? LoraScaling::kAlphaOverSqrtRank
                                                : LoraScaling::kAlphaOverRank;
    else if (n == "lora_seed") c.lora.seed = u;
    else if (n == "targets") {
      c.lora.target_filters.clear();
      std::string_view rest = v;
      while (true) {
        const size_t comma = rest.find(',');
        const std::string_view item = rest.substr(0, comma);
        if (!item.empty()) c.lora.target_filters.emplace_back(item);
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
      }
      // Review N27: `--targets ""` (or ",") named nothing, and an empty
      // filter list means "adapt every weight" — the opposite of what a
      // user restricting the adapters asked for.
      if (c.lora.target_filters.empty())
        return std::unexpected(
            Named(*f) + " must name at least one weight, got '" + text +
            "'; omit it to adapt every eligible weight");
    } else if (n == "optimizer")
      c.optimizer.kind = v == "adamw" ? OptimizerKind::kAdamW : OptimizerKind::kSgd;
    else if (n == "lr") c.optimizer.lr = static_cast<float>(d);
    else if (n == "weight_decay") c.optimizer.weight_decay = static_cast<float>(d);
    else if (n == "clip_norm") c.optimizer.clip_norm = static_cast<float>(d);
    else if (n == "lr_schedule")
      c.optimizer.lr_schedule = v == "cosine" ? LrSchedule::kCosineWithWarmup
                                              : LrSchedule::kConstant;
    else if (n == "warmup") c.optimizer.warmup_steps = u;
    else if (n == "min_lr_factor") c.optimizer.min_lr_factor = static_cast<float>(d);
    else if (n == "allow_zero_lr") c.optimizer.allow_zero_lr = on;
    else if (n == "quantize_base") c.quantize_base = on;
    else if (n == "bf16_base") c.bf16_base = on;
    else if (n == "precision")
      c.precision = v == "f32" ? Precision::kF32 : Precision::kCertifiedBf16;
    else if (n == "attention")
      c.attention = v == "auto"     ? AttentionKind::kAuto
                    : v == "cached" ? AttentionKind::kCached
                                    : AttentionKind::kTiled;
    else if (n == "attention_cache_budget_mib")
      c.attention_cache_budget_bytes = u << 20;
    else if (n == "remat")
      c.remat = v == "auto" ? RematKind::kAuto
                : v == "on" ? RematKind::kOn
                            : RematKind::kOff;
    else if (n == "adapter_only") c.adapter_only = on;
    else if (n == "validation_scores") c.eval_on_plan = v == "plan";
    else if (n == "val_frac") c.val_frac = static_cast<float>(d);
    else if (n == "split_seed") c.split_seed = u;
    else if (n == "masked_loss") c.masked_loss = on;
    else if (n == "memory_budget_mib") c.memory_budget_bytes = u << 20;
    else
      // A field in the table with no branch above would be validated and
      // then dropped — the silent ignore the recipe exists to forbid.
      return std::unexpected("recipe field '" + name +
                             "' is in the table but Apply does not apply it");
  }

  // --- Cross-field rules: a field that cannot apply is never silently
  // ignored. `set` is "the recipe named it", whatever value it named.
  auto set = [&](const char* name) { return values.contains(name); };
  auto field = [](const char* name) -> const Field& { return *FindField(name); };
  auto needs = [&](const char* name, const std::string& what)
      -> std::unexpected<std::string> {
    return std::unexpected(Named(field(name)) + " " + what);
  };
  const bool distills = c.loss == LossKind::kKLDistill ||
                        c.loss == LossKind::kXEntPlusKL;
  // Review N15: the teacher and its two knobs were silently ignored unless
  // the loss used them.
  if (!distills) {
    if (inputs.teacher)
      return std::unexpected(
          "--teacher applies to a distillation loss only (--loss kl or "
          "xent+kl); with --loss " +
          std::string(c.loss == LossKind::kMse ? "mse" : "xent") +
          " it would be silently ignored");
    for (const char* knob : {"distill_weight", "temperature"})
      if (set(knob))
        return needs(knob, "applies to a distillation loss only (loss kl or "
                           "xent+kl)");
  }
  if (c.loss != LossKind::kXEntPlusKL && set("distill_weight"))
    return needs("distill_weight",
                 "weighs the KL term of loss xent+kl; under loss kl there is "
                 "nothing to weigh it against");
  const bool cosine = c.optimizer.lr_schedule == LrSchedule::kCosineWithWarmup;
  if (!cosine)
    for (const char* knob : {"warmup", "min_lr_factor", "allow_zero_lr"})
      if (set(knob) && !(std::string_view(knob) == "allow_zero_lr" &&
                         !c.optimizer.allow_zero_lr))
        return needs(knob, "applies to lr_schedule cosine only");
  if (cosine && c.optimizer.warmup_steps >= c.default_steps)
    return needs("warmup", "(" + std::to_string(c.optimizer.warmup_steps) +
                               " steps) must be shorter than steps (" +
                               std::to_string(c.default_steps) + ")");
  if (c.quantize_base && c.bf16_base)
    return std::unexpected(
        "--quantize-base and --bf16-base (recipe fields quantize_base, "
        "bf16_base) are mutually exclusive: one storage precision per weight");
  if (set("attention_cache_budget_mib") && c.attention != AttentionKind::kAuto)
    return needs("attention_cache_budget_mib", "applies to attention auto only");
  if (set("val_frac") && !inputs.data)
    return needs("val_frac", "compiles the validation split from the corpus: "
                             "pass --data too");
  if (set("split_seed") && !set("val_frac"))
    return needs("split_seed", "needs val_frac: it seeds the compiled split");
  *config = std::move(c);
  return {};
}

/// The bounds a config built in code (the library path, no recipe) must
/// still hold: the ones whose violation emitted an invalid plan.
inline std::optional<std::string> ConfigViolation(const UpdateConfig& c) {
  if (c.lora.rank < 1 || static_cast<uint64_t>(c.lora.rank) > kLoraRankMax)
    return "lora rank " + std::to_string(c.lora.rank) +
           " is outside [1, " + std::to_string(kLoraRankMax) + "]";
  for (const std::string& t : c.lora.target_filters)
    if (t.empty()) return std::string("an adapter target filter is empty");
  return std::nullopt;
}

// --- The resolved recipe ------------------------------------------------------------

/// Every field of `c` as `name=value` lines in table order, defaults
/// included: the same text for a recipe file and for the flags that say the
/// same thing. Floats print as their exact f32 bits.
inline std::string ResolvedText(const UpdateConfig& c) {
  auto bits = [](float v) {
    uint32_t b = 0;
    static_assert(sizeof(b) == sizeof(v));
    std::memcpy(&b, &v, sizeof(b));
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", b);
    return std::string(buf);
  };
  auto yes = [](bool b) { return std::string(b ? "true" : "false"); };
  // A recipe's budgets are whole MiB and print as that count. A config
  // built in code may hold any byte count: the bytes below a MiB are
  // appended, so two budgets that differ never share a text (and a hash).
  auto mib = [](uint64_t bytes) {
    std::string s = std::to_string(bytes >> 20);
    if (const uint64_t rest = bytes & ((uint64_t{1} << 20) - 1))
      s += "+" + std::to_string(rest) + "B";
    return s;
  };
  // The filters are a set — a weight is adapted when any of them matches —
  // so the text holds them sorted, each once: "wq,wv" and "wv,wq" compile
  // the same plan and are the same recipe.
  std::vector<std::string> filters = c.lora.target_filters;
  std::sort(filters.begin(), filters.end());
  filters.erase(std::unique(filters.begin(), filters.end()), filters.end());
  std::string targets;
  for (const std::string& t : filters)
    targets += (targets.empty() ? "" : ",") + t;
  const std::pair<const char*, std::string> lines[] = {
      {"data_batch", std::to_string(c.batch)},
      {"steps", std::to_string(c.default_steps)},
      {"grad_accum", std::to_string(c.grad_accum_steps)},
      {"loss", c.loss == LossKind::kSoftmaxXEnt ? "xent"
               : c.loss == LossKind::kMse       ? "mse"
               : c.loss == LossKind::kKLDistill ? "kl"
                                                : "xent+kl"},
      {"distill_weight", bits(c.distill_weight)},
      {"temperature", bits(c.temperature)},
      {"lora_rank", std::to_string(c.lora.rank)},
      {"lora_alpha", bits(c.lora.alpha)},
      {"lora_scaling", c.lora.scaling == LoraScaling::kAlphaOverSqrtRank
                           ? "alpha_over_sqrt_r"
                           : "alpha_over_r"},
      {"lora_seed", std::to_string(c.lora.seed)},
      {"targets", targets},
      {"optimizer", c.optimizer.kind == OptimizerKind::kAdamW ? "adamw" : "sgd"},
      {"lr", bits(c.optimizer.lr)},
      {"weight_decay", bits(c.optimizer.weight_decay)},
      {"clip_norm", bits(c.optimizer.clip_norm)},
      {"lr_schedule",
       c.optimizer.lr_schedule == LrSchedule::kCosineWithWarmup ? "cosine"
                                                                : "const"},
      {"warmup", std::to_string(c.optimizer.warmup_steps)},
      {"min_lr_factor", bits(c.optimizer.min_lr_factor)},
      {"allow_zero_lr", yes(c.optimizer.allow_zero_lr)},
      {"quantize_base", yes(c.quantize_base)},
      {"bf16_base", yes(c.bf16_base)},
      {"precision", c.precision == Precision::kF32 ? "f32" : "certified-bf16"},
      {"attention", c.attention == AttentionKind::kAuto     ? "auto"
                    : c.attention == AttentionKind::kCached ? "cached"
                                                            : "tiled"},
      {"attention_cache_budget_mib", mib(c.attention_cache_budget_bytes)},
      {"remat", c.remat == RematKind::kAuto ? "auto"
                : c.remat == RematKind::kOn ? "on"
                                            : "off"},
      {"adapter_only", yes(c.adapter_only)},
      {"validation_scores", c.eval_on_plan ? "plan" : "shipped"},
      {"val_frac", bits(c.val_frac)},
      {"split_seed", std::to_string(c.split_seed)},
      {"masked_loss", yes(c.masked_loss)},
      {"memory_budget_mib", mib(c.memory_budget_bytes)},
  };
  static_assert(std::size(lines) == std::size(kFields),
                "the resolved text names every field");
  std::string text;
  for (const auto& [name, value] : lines) {
    text += name;
    text += '=';
    text += value;
    text += '\n';
  }
  return text;
}

/// ContentHash64 of the resolved text: the recipe's identity.
inline uint64_t RecipeHash(const UpdateConfig& c) {
  const std::string text = ResolvedText(c);
  return ContentHash64(reinterpret_cast<const uint8_t*>(text.data()),
                       text.size());
}

}  // namespace seeml::update::recipe

#endif  // SEEML_SOURCE_RECIPE_RECIPE_H_
