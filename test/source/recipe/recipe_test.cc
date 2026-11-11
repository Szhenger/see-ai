// =============================================================================
// The recipe (S4 #150): strict scalar parsing, every field held to its bound
// with a diagnostic naming the field, the cross-field rules — each of the
// review's inputs (N15, N18, N20, N24, N26, N27, F56, F63) refused — the
// recipe file's strict reader, the resolved recipe's identity, and the
// runner's flag rules.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <string>

#include "source/recipe/recipe.h"
#include "source/recipe/recipe_file.h"
#include "source/recipe/run_flags.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;
using recipe::Apply;
using recipe::Inputs;
using recipe::Values;

TEST(Recipe, IntegersAreDigitsAlone) {
  uint64_t v = 7;
  // F56 / F63: strtoull skipped the space and negated the rest.
  for (const char* bad : {" -1", "-1", "+1", " 1", "1 ", "", "0x10", "1e3",
                          "18446744073709551616", "1.0"})
    EXPECT_FALSE(recipe::ParseU64Strict(bad, &v));
  EXPECT_EQ(v, 7u);  // untouched on refusal
  EXPECT_TRUE(recipe::ParseU64Strict("0", &v));
  EXPECT_EQ(v, 0u);
  EXPECT_TRUE(recipe::ParseU64Strict("18446744073709551615", &v));
  EXPECT_EQ(v, ~uint64_t{0});
  double d = 0;
  // strtod alone takes a hex float and skips every isspace() prefix.
  for (const char* bad : {"nan", "inf", "-inf", "infinity", " 1", "+1", "1x",
                          "", "1e999", "0x1p-3", "0x10", "\r1", "\v1", "\f1",
                          "1 "})
    EXPECT_FALSE(recipe::ParseF64Strict(bad, &d));
  // An embedded NUL is not the end of the text.
  EXPECT_FALSE(recipe::ParseF64Strict(std::string_view("1\0" "2", 3), &d));
  EXPECT_TRUE(recipe::ParseF64Strict("-2.5e-1", &d));
  EXPECT_EQ(d, -0.25);
}

TEST(Recipe, EveryFieldIsHeldToItsBoundByName) {
  struct Case {
    const char* field;
    const char* value;
    const char* says;
  };
  const Case cases[] = {
      {"steps", " -1", "--steps (recipe field steps) must be an integer in [1, "},
      {"steps", "0", "--steps (recipe field steps) must be an integer in [1, "},
      {"data_batch", "-4", "--data-batch (recipe field data_batch)"},
      // N18: the rank was unbounded.
      {"lora_rank", "4097",
       "--lora-rank (recipe field lora_rank) must be an integer in [1, 4096], "
       "got '4097'"},
      {"lora_rank", "0", "must be an integer in [1, 4096]"},
      {"lora_alpha", "0", "must be a finite number in (0, 1e+06]"},
      // N26: the temperature is bounded at both ends, in f32.
      {"temperature", "0.0001", "must be a finite number in [0.001, 1000]"},
      {"temperature", "1001", "must be a finite number in [0.001, 1000]"},
      {"temperature", "nan", "--temperature (recipe field temperature)"},
      {"lr", "0", "must be a finite number in (0, 10]"},
      {"lr", "inf", "--lr (recipe field lr)"},
      {"min_lr_factor", "1.5", "must be a finite number in [0, 1]"},
      {"val_frac", "1", "must be a finite number in (0, 1)"},
      {"loss", "hinge", "must be one of xent|mse|kl|xent+kl, got 'hinge'"},
      {"optimizer", "adam", "must be one of adamw|sgd"},
      {"quantize_base", "yes", "must be true or false"},
      {"grad_accum", "4294967296", "--grad-accum (recipe field grad_accum)"},
      {"memory_budget_mib", "0", "--memory-budget-mib"},
      // A nonzero text that underflows f32 is not the 0 a closed bound
      // admits (clip off, no decay): strtof's ERANGE refused it.
      {"clip_norm", "1e-60",
       "--clip-norm (recipe field clip_norm) must be a finite number in "
       "[0, 1e+06], got '1e-60': it underflows the f32 the plan stores"},
      {"weight_decay", "1e-60", "it underflows the f32 the plan stores"},
      {"weight_decay", "1e-40", "it underflows the f32 the plan stores"},
  };
  for (const Case& c : cases) {
    UpdateConfig config;
    Values v = {{c.field, c.value}};
    // The fields these need beside them, so the bound is what refuses.
    if (std::string(c.field) == "temperature") v["loss"] = "kl";
    if (std::string(c.field) == "min_lr_factor") v["lr_schedule"] = "cosine";
    EXPECT_ERROR_CONTAINS(Apply(v, {.teacher = true, .data = true}, &config),
                          c.says);
  }
  UpdateConfig config;
  EXPECT_ERROR_CONTAINS(Apply({{"learning_rate", "1"}}, {}, &config),
                        "unknown recipe field 'learning_rate'");
  // A written zero is the zero the bound admits.
  EXPECT_OK(Apply({{"clip_norm", "0"}, {"weight_decay", "0.0"}}, {}, &config));
  EXPECT_EQ(config.optimizer.clip_norm, 0.0f);
  // Every field of the table has a branch in Apply: none is validated and
  // then dropped. (A value inside each bound reaches the branch; a
  // cross-field refusal after it is another rule's.)
  for (const recipe::Field& f : recipe::kFields) {
    std::string value;
    switch (f.kind) {
      case recipe::Kind::kUInt: value = std::to_string(f.umin); break;
      case recipe::Kind::kFloat:
        value = recipe::Number((f.fmin + f.fmax) / 2);
        break;
      case recipe::Kind::kEnum:
        value = std::string(f.choices).substr(0, std::string(f.choices).find('|'));
        break;
      case recipe::Kind::kBool: value = "true"; break;
      case recipe::Kind::kList: value = "w"; break;
    }
    UpdateConfig each;
    auto applied = Apply({{f.name, value}}, {.teacher = true, .data = true}, &each);
    if (!applied)
      EXPECT_TRUE(applied.error().find("does not apply it") == std::string::npos);
  }
  // The f32 bound itself is admitted ("0.001" is 1e-3f).
  EXPECT_OK(Apply({{"temperature", "0.001"}, {"loss", "kl"}},
                  {.teacher = true}, &config));
  // The f32 stored is the text's own nearest one, not the nearest double's
  // narrowed, which rounds twice: this text sits just above the midpoint of
  // 1.0f and the next float, and its nearest double is that midpoint.
  UpdateConfig near_tie;
  EXPECT_OK(Apply({{"lr", "1.0000000596046448"}}, {}, &near_tie));
  EXPECT_EQ(near_tie.optimizer.lr, std::nextafter(1.0f, 2.0f));
}

TEST(Recipe, FieldsAreHeldToEachOther) {
  UpdateConfig c;
  // N15: the teacher and its knobs without a distillation loss.
  EXPECT_ERROR_CONTAINS(Apply({}, {.teacher = true}, &c),
                        "--teacher applies to a distillation loss only");
  EXPECT_ERROR_CONTAINS(Apply({{"temperature", "2"}}, {}, &c),
                        "--temperature (recipe field temperature) applies to a "
                        "distillation loss only");
  EXPECT_ERROR_CONTAINS(Apply({{"distill_weight", "0.5"}}, {}, &c),
                        "--distill-weight (recipe field distill_weight) applies "
                        "to a distillation loss only");
  c = {};
  EXPECT_ERROR_CONTAINS(
      Apply({{"loss", "kl"}, {"distill_weight", "0.5"}}, {.teacher = true}, &c),
      "weighs the KL term of loss xent+kl");
  c = {};
  EXPECT_OK(Apply({{"loss", "xent+kl"}, {"distill_weight", "0.4"},
                   {"temperature", "2"}},
                  {.teacher = true}, &c));
  EXPECT_EQ(c.distill_weight, 0.4f);
  // N27: targets that name nothing.
  for (const char* empty : {"", ",", ",,"}) {
    c = {};
    EXPECT_ERROR_CONTAINS(Apply({{"targets", empty}}, {}, &c),
                          "--targets (recipe field targets) must name at least "
                          "one weight");
  }
  c = {};
  EXPECT_OK(Apply({{"targets", "wq,,wv"}}, {}, &c));
  EXPECT_EQ(c.lora.target_filters.size(), 2u);
  // The cosine schedule's knobs, and warmup < steps.
  c = {};
  EXPECT_ERROR_CONTAINS(Apply({{"warmup", "3"}}, {}, &c),
                        "--warmup (recipe field warmup) applies to lr_schedule "
                        "cosine only");
  c = {};
  EXPECT_ERROR_CONTAINS(Apply({{"allow_zero_lr", "true"}}, {}, &c),
                        "applies to lr_schedule cosine only");
  c = {};
  EXPECT_OK(Apply({{"allow_zero_lr", "false"}}, {}, &c));
  c = {};
  EXPECT_ERROR_CONTAINS(
      Apply({{"lr_schedule", "cosine"}, {"warmup", "10"}, {"steps", "10"}}, {},
            &c),
      "--warmup (recipe field warmup) (10 steps) must be shorter than steps "
      "(10)");
  c = {};
  EXPECT_ERROR_CONTAINS(
      Apply({{"quantize_base", "true"}, {"bf16_base", "true"}}, {}, &c),
      "mutually exclusive");
  c = {};
  EXPECT_ERROR_CONTAINS(Apply({{"attention", "tiled"},
                               {"attention_cache_budget_mib", "8"}},
                              {}, &c),
                        "applies to attention auto only");
  c = {};
  EXPECT_ERROR_CONTAINS(Apply({{"val_frac", "0.1"}}, {}, &c), "pass --data too");
  c = {};
  EXPECT_ERROR_CONTAINS(Apply({{"split_seed", "3"}}, {.data = true}, &c),
                        "needs val_frac");
  c = {};
  EXPECT_OK(Apply({{"val_frac", "0.25"}, {"split_seed", "3"}}, {.data = true},
                  &c));
  EXPECT_EQ(c.split_seed, 3u);
  // The library path's bounds.
  UpdateConfig lib;
  EXPECT_FALSE(recipe::ConfigViolation(lib).has_value());
  lib.lora.rank = 4097;
  EXPECT_STR_CONTAINS(recipe::ConfigViolation(lib).value_or(""),
                      "lora rank 4097 is outside [1, 4096]");
  lib = {};
  lib.lora.target_filters = {"wq", ""};
  EXPECT_STR_CONTAINS(recipe::ConfigViolation(lib).value_or(""),
                      "target filter is empty");
}

TEST(Recipe, TheFileIsAFlatStrictObjectAndResolvesAsTheFlagsDo) {
  auto parsed = recipe::ParseRecipeFile(
      " {\"loss\": \"xent\", \"lora_rank\": 4, \"targets\": [\"wq\", \"wv\"],\n"
      "  \"lr\": 1e-4, \"lr_schedule\": \"cosine\", \"warmup\": 10,\n"
      "  \"steps\": 300, \"quantize_base\": true}\n");
  ASSERT_OK(parsed);
  EXPECT_EQ(parsed->at("targets"), "wq,wv");
  EXPECT_EQ(parsed->at("lr"), "1e-4");
  EXPECT_EQ(parsed->at("quantize_base"), "true");
  UpdateConfig from_file, from_flags;
  ASSERT_OK(Apply(*parsed, {}, &from_file));
  ASSERT_OK(Apply({{"loss", "xent"}, {"lora_rank", "4"}, {"targets", "wq,wv"},
                   {"lr", "0.0001"}, {"lr_schedule", "cosine"},
                   {"warmup", "10"}, {"steps", "300"},
                   {"quantize_base", "true"}},
                  {}, &from_flags));
  EXPECT_EQ(recipe::ResolvedText(from_file), recipe::ResolvedText(from_flags));
  EXPECT_EQ(recipe::RecipeHash(from_file), recipe::RecipeHash(from_flags));
  EXPECT_STR_CONTAINS(recipe::ResolvedText(from_file), "lora_rank=4\n");
  EXPECT_STR_CONTAINS(recipe::ResolvedText(from_file), "targets=wq,wv\n");
  // Every field is in the text, so any change moves the hash.
  from_flags.optimizer.lr = 2e-4f;
  EXPECT_NE(recipe::RecipeHash(from_file), recipe::RecipeHash(from_flags));
  EXPECT_OK(recipe::ParseRecipeFile("{}"));

  struct Bad {
    const char* text;
    const char* says;
  };
  const Bad bad[] = {
      {"[]", "not a JSON object"},
      {"{\"rank\": 4}", "unknown field 'rank'"},
      {"{\"lora_rank\": 4, \"lora_rank\": 8}", "field 'lora_rank' appears twice"},
      {"{\"lora_rank\": \"4\"}", "field 'lora_rank' takes a number"},
      {"{\"lora_rank\": true}", "field 'lora_rank' takes a number"},
      {"{\"loss\": xent}", "field 'loss' takes a string"},
      {"{\"quantize_base\": 1}", "field 'quantize_base' takes true or false"},
      {"{\"targets\": \"wq\"}", "field 'targets' takes an array of strings"},
      {"{\"targets\": [\"\"]}", "holds an empty name or a comma"},
      {"{\"lr\": {\"v\": 1}}", "field 'lr' takes no nested object"},
      {"{\"lr\": 1} x", "content after the closing '}'"},
      {"{\"lr\": 1", "expected ',' or '}'"},
      {"{\"lr\" 1}", "expected ':' after 'lr'"},
      // A JSON string the reader does not decode is refused as that — a
      // serializer writes \uXXXX for every non-ASCII character by default —
      // not as a value of the wrong type.
      {"{\"targets\": [\"w\\u0031\"]}",
       "field 'targets' uses a string escape this reader does not take"},
      {"{\"loss\": \"x\\rent\"}", "field 'loss' uses a string escape"},
      {"{\"lo\\u0073s\": \"xent\"}", "a field name uses a string escape"},
  };
  for (const Bad& b : bad)
    EXPECT_ERROR_CONTAINS(recipe::ParseRecipeFile(b.text), b.says);
  // A value the file admits by type is still held to its bound by Apply.
  auto wide = recipe::ParseRecipeFile("{\"lora_rank\": 5000}");
  ASSERT_OK(wide);
  UpdateConfig c;
  EXPECT_ERROR_CONTAINS(Apply(*wide, {}, &c), "[1, 4096]");
}

TEST(Recipe, TheRunnersFlagsAreHeldToThePlan) {
  PlanHeader h;
  h.lr_schedule = static_cast<uint32_t>(LrSchedule::kCosineWithWarmup);
  h.warmup_steps = 10;
  recipe::RunFlags ok;
  EXPECT_FALSE(recipe::RunFlagsViolation(h, ok).has_value());
  // N20: a fresh run's --steps inside the compiled warmup.
  recipe::RunFlags f;
  f.steps_given = true;
  f.steps = 10;
  EXPECT_STR_CONTAINS(recipe::RunFlagsViolation(h, f).value_or(""),
                      "--steps 10 must exceed the plan's compiled warmup (10");
  f.steps = 11;
  EXPECT_FALSE(recipe::RunFlagsViolation(h, f).has_value());
  f.steps = 5;
  f.resume = true;  // further steps of the first run's horizon
  EXPECT_FALSE(recipe::RunFlagsViolation(h, f).has_value());
  f.resume = false;
  f.steps = 0;  // the plan's default
  EXPECT_FALSE(recipe::RunFlagsViolation(h, f).has_value());
  PlanHeader constant;
  f.steps = 1;
  EXPECT_FALSE(recipe::RunFlagsViolation(constant, f).has_value());
  // N24: the two flags that were ignored without their partner.
  recipe::RunFlags c;
  c.checkpoint_every_given = true;
  c.checkpoint_every = 5;
  EXPECT_STR_CONTAINS(recipe::RunFlagsViolation(h, c).value_or(""),
                      "--checkpoint-every needs --checkpoint");
  c.has_checkpoint_path = true;
  EXPECT_FALSE(recipe::RunFlagsViolation(h, c).has_value());
  recipe::RunFlags e;
  e.eval_every_given = true;
  e.eval_every = 2;
  EXPECT_STR_CONTAINS(recipe::RunFlagsViolation(h, e).value_or(""),
                      "--eval-every needs a validation split");
  e.has_validation = true;
  EXPECT_FALSE(recipe::RunFlagsViolation(h, e).has_value());
  e.has_validation = false;
  e.eval_every = 0;  // an explicit "off" needs nothing
  EXPECT_FALSE(recipe::RunFlagsViolation(h, e).has_value());
}

}  // namespace
