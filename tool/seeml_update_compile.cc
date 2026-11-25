// =============================================================================
// seeml-update-compile — the SeeAI Model Update Compiler CLI.
//
//   seeml-update-compile
//       --source  model.smf            the on-device model to update
//       --out     out_dir/             emission directory
//       [--recipe recipe.json]         the recipe's fields (source/recipe/
//                                      recipe.h: every flag below that
//                                      configures the fine-tune) as one flat
//                                      JSON object; a flag overrides the
//                                      field it names
//       [--data-batch 32]              compiled batch size
//       [--loss xent|mse|kl|xent+kl]   training objective (default xent)
//       [--teacher open_model.smf]     open-weights teacher (kl / xent+kl)
//       [--distill-weight 0.5]         KL weight for the composite loss
//       [--temperature 2.0]            distillation temperature
//       [--lora-rank 8] [--lora-alpha 16] [--lora-seed 42]
//       [--lora-scaling alpha_over_r|alpha_over_sqrt_r]
//                                      the adapter scale: alpha / r (the
//                                      default) or alpha / sqrt(r)
//       [--targets substr,substr]      restrict adapters to matching weights
//       [--optimizer adamw|sgd] [--lr 2e-3] [--weight-decay 0.01]
//       [--clip-norm 0]                L2 gradient clip (0 = off)
//       [--clip-mode per_tensor|global] each tensor's norm, or one over all (#190)
//       [--lr-schedule cosine|const]   runtime LR schedule (cosine; warmup 5 % of steps)
//       [--warmup N]                   warmup steps (cosine schedule; default 5 % of --steps)
//       [--min-lr-factor 0.1]          cosine floor as a fraction of --lr
//       [--allow-zero-lr]              permit --min-lr-factor 0 (the last
//                                      step then trains at LR 0)
//       [--quantize-base]              int8-quantize frozen weights in rodata
//       [--bf16-base]                  store frozen weights as bfloat16 rodata
//       [--precision f32|certified-bf16] relaxed frozen-weight GEMMs (v18):
//                                      the package then needs a numerics
//                                      certificate (tool/certify_numerics.py)
//                                      (2x smaller, f32 compute; exclusive with
//                                      --quantize-base)
//       [--remat auto|off|on]          recompute the cheap forward activations
//                                      in the backward instead of keeping them
//                                      across the forward (roadmap 2b; same
//                                      bits, a longer train program); auto =
//                                      only when the arena would exceed the
//                                      memory gate's budget
//       [--steps 1000]                 default optimizer-step count baked into the plan
//       [--grad-accum 1]               micro-batches accumulated per optimizer step
//                                      (activations scale with --data-batch, the
//                                      effective batch is data-batch x grad-accum)
//       [--kernel-policy table.json]   the host-keyed kernel-policy table
//                                      tool/autotune.py measured; the entry
//                                      for this host (or --target-host KEY)
//                                      sets the plan's CPU GEMM tiles
//       [--target-host KEY]            look the table up by this host key
//                                      instead of the compile host's
//       [--gemm-tiles K,N]             set the CPU GEMM tiles explicitly
//                                      (throughput only, never bits)
//       [--report report.json]         machine-readable compile report
//       [--adapter-only]               the adapter-only package (F6 #134,
//                                      plan v21): the frozen weights stay
//                                      in the source model file, read there
//                                      at run time; the plan carries no
//                                      rodata (f32 base, exact kernels)
//       [--validation-scores shipped|plan] where a narrow-base plan's eval
//                                      program reads the frozen weights: the
//                                      shipped f32 file (default from a file)
//                                      or the plan's own narrow copies, for a
//                                      device that cannot hold the file
//                                      beside the arena (reported)
//       [--no-embed]                   skip the decimal byte-array TU; let
//                                      tool/pack_update.py embed the plan
//                                      (.incbin) or, above its embed limit,
//                                      leave it beside the binary (#158)
//       [--build]                      run build.sh after emission
//       [--version]                    print the release version
//       [--help | -h]                  print the usage (exit 0), from any
//                                      slot that is not another flag's value:
//                                      `--out -h` is --out without a value
//                                      (exit 2), not a request for help
//
// Every numeric flag is parsed strictly: trailing garbage, overflow, NaN/Inf,
// or an unknown flag is a hard error, never a silent default.
//
// Exit codes: 0 the package was emitted; 1 the work failed (an unreadable or
// malformed model, recipe file or policy table, a compile refusal, a failed
// write or build); 2 a bad command line (an unknown flag, a flag without
// its value, a recipe field outside its bound — whether a flag or the
// recipe file set it — or flags that contradict each other), refused before
// any model is read.
//
// Output: update_plan.seeu + generated TUs + vendored runtime sources +
// build.sh; with --build, the linked self-contained `model_update` binary.
// --no-embed leaves the plan on disk once (the .seeu) and no embedded TU:
// `python3 tool/pack_update.py <out> --build` then embeds it as an .incbin
// assembly stub, which is the route that scales past 100M parameters — or,
// for a plan above the packer's embed limit (1 GiB; the linker and loader
// fail near 1.8 GB on macOS arm64), builds the disk mode, whose binary maps
// update_plan.seeu beside itself at start-up (#158).
// =============================================================================

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/packaging/native_emitter.h"
#include "compiler/backend/architecture/kernel_policy_table.h"
#include "compiler/diagnostics/logger.h"
#include "compiler/driver/statistics.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/ingressor/export_report.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "source/contract/split.h"
#include "source/identity/version.h"
#include "source/platform/durable.h"
#include "source/recipe/recipe.h"
#include "source/recipe/recipe_file.h"

namespace {

using namespace seeml::update;

/// A failure of the work itself — an unreadable or malformed input file, a
/// compile refusal, an unwritable output: exit 1.
[[nodiscard]] int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-update-compile: %s\n", msg.c_str());
  return 1;
}

/// A bad command line — an unknown flag, a flag without its value, a value
/// outside its bound, flags that contradict each other: exit 2, before any
/// model is read.
[[nodiscard]] int Usage(const std::string& msg) {
  std::fprintf(stderr, "seeml-update-compile: %s\n", msg.c_str());
  return 2;
}

void PrintUsage() {
  std::fprintf(
      stderr,
      "usage: seeml-update-compile --source model.smf --out dir/\n"
      "  [--recipe recipe.json]  the fields below as one JSON object;\n"
      "                      a flag overrides the field it names;\n"
      "                      a boolean flag --x has the form --no-x\n"
      "  [--data-batch N] [--loss xent|mse|kl|xent+kl]\n"
      "  [--teacher t.smf] [--distill-weight W] [--temperature T]\n"
      "  [--lora-rank R] [--lora-alpha A] [--lora-seed S]\n"
      "  [--lora-scaling alpha_over_r|alpha_over_sqrt_r]  the adapter scale\n"
      "  [--targets substr,...] [--optimizer adamw|sgd] [--lr LR]\n"
      "  [--weight-decay WD] [--clip-norm C] [--clip-mode per_tensor|global]\n"
      "  [--moments f32|int8]  AdamW's state: f32, or block-wise 8-bit\n"
      "                      (opt-in: it changes the arithmetic)\n"
      "  [--lr-schedule cosine|const] [--warmup N] [--allow-zero-lr]\n"
      "  [--min-lr-factor F] [--quantize-base | --bf16-base]\n"
      "  [--precision f32|certified-bf16]\n"
      "  [--attention auto|cached|tiled] [--attention-cache-budget-mib N]\n"
      "  [--remat auto|off|on] [--memory-budget-mib N]\n"
      "  [--adapter-only] [--validation-scores shipped|plan]\n"
      "  [--data corpus.txt|.jsonl|.sds]  the training data: raw text is\n"
      "                      tokenized with the model's section into\n"
      "                      <out>/corpus.sds; the plan is pinned to it\n"
      "  [--val-frac F] [--split-seed S]  the held-out split, compiled\n"
      "                      from the corpus and named in the plan\n"
      "  [--masked-loss]     the cross-entropy over target positions\n"
      "                      alone (labels of -1 ignored); implied by\n"
      "                      a --data corpus that carries a mask\n"
      "  [--segment-mask auto|off|on]  no attention across a window's\n"
      "                      documents (default auto: when the\n"
      "                      --data corpus holds such windows)\n"
      "  [--steps N]\n"
      "  [--grad-accum G]\n"
      "  [--no-fuse-epilogue] [--no-fuse-elementwise] [--no-fuse-addend]\n"
      "  [--no-rope-table]\n"
      "  [--no-fuse-clip]\n"
      "  [--report out.json] [--dump-sir out.txt]\n"
      "  [--kernel-policy table.json] [--target-host KEY]\n"
      "  [--gemm-tiles K,N]\n"
      "  [--no-embed] [--build] [--version]\n"
      "  [--gpu-peak-gflops G]  the target GPU's peak: the GPU-denominated\n"
      "                      predicted step (and a Metal run's)\n"
      "  [--peak-gflops G]   the target's peak; the backend screen then\n"
      "                      predicts a step's time from the plan's work\n"
      "  [--verbose]         every screen row, the per-pass and per-adapter\n"
      "                      tables, and each pass's note\n");
}

/// The flags outside the recipe's field table that take a value. With the
/// table's non-boolean fields they are every flag whose next argv slot is
/// its value (TakesValue); a flag added to ParseArgs with TakeValue is added
/// here.
constexpr const char* kValueFlags[] = {
    "--source",      "--out",      "--recipe",      "--data",
    "--teacher",     "--dump-sir", "--gemm-tiles",  "--kernel-policy",
    "--target-host", "--report",   "--peak-gflops", "--gpu-peak-gflops"};

[[nodiscard]] bool TakesValue(const char* arg) {
  for (const char* flag : kValueFlags)
    if (std::strcmp(arg, flag) == 0) return true;
  for (const recipe::Field& field : recipe::kFields)
    if (field.kind != recipe::Kind::kBool && std::strcmp(arg, field.flag) == 0)
      return true;
  return false;
}

[[nodiscard]] bool IsHelp(const char* arg) {
  return std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0;
}

/// What the line says about help, read slot by slot the way the runner's
/// ArgsOk reads its own (runtime/host/update_main.cc): a value flag's next
/// slot is its value, so --help or -h is a request only in a slot that is
/// no flag's value. `--out -h` asked for nothing: it left --out without a
/// value (a flag is never a value), `swallowed` names it, and the refusal
/// wins over the help, as it does in the runner. Help standing on its own
/// is help whatever else is wrong with the line, as it always was.
struct HelpRequest {
  bool asked = false;
  // The first value flag with --help or -h in its value slot.
  const char* swallowed = nullptr;
};

[[nodiscard]] HelpRequest FindHelp(int argc, char** argv) {
  HelpRequest help;
  for (int i = 1; i < argc; ++i) {
    if (IsHelp(argv[i])) {
      help.asked = true;
    } else if (TakesValue(argv[i]) && i + 1 < argc) {
      if (IsHelp(argv[i + 1])) {
        if (!help.swallowed) help.swallowed = argv[i];
        ++i;
      } else if (std::strncmp(argv[i + 1], "--", 2) != 0) {
        ++i;  // the value's slot: whatever it holds is not a request
      }
    }
  }
  return help;
}

/// Strict argument cursor: every flag must be known, every value must parse
/// completely. This is the difference between `--lr 1e-3` and `--lr abc`
/// silently training at lr = 0.
class Args {
 public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}

  [[nodiscard]] bool Take(const char* flag) {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        taken_[i] = true;
        return true;
      }
    return false;
  }

  [[nodiscard]] std::optional<std::string> TakeValue(const char* flag) {
    for (int i = 1; i + 1 < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        // A following flag is a missing value, not a value: consuming it
        // ("--targets --quantize-base") would silently disable one option
        // and misconfigure the other — the exact silent default the banner
        // forbids. Remember the culprit so the error names the real
        // mistake ("requires a value"), not "unknown argument".
        if (std::strncmp(argv_[i + 1], "--", 2) == 0) {
          if (!missing_value_) missing_value_ = flag;
          return std::nullopt;
        }
        taken_[i] = taken_[i + 1] = true;
        return std::string(argv_[i + 1]);
      }
    // The flag as the very last argument has no value slot at all.
    if (argc_ > 1 && !taken_[argc_ - 1] &&
        std::strcmp(argv_[argc_ - 1], flag) == 0 && !missing_value_)
      missing_value_ = flag;
    return std::nullopt;
  }

  /// First flag seen with no usable value; set lazily by TakeValue.
  const std::optional<std::string>& MissingValue() const {
    return missing_value_;
  }

  /// Whether `arg` also stands in a slot a Take* call consumed: the
  /// leftover is then a flag given twice, not an unknown one.
  [[nodiscard]] bool TakenElsewhere(const std::string& arg) const {
    for (int i = 1; i < argc_; ++i)
      if (taken_[i] && arg == argv_[i] && arg.starts_with("--")) return true;
    return false;
  }

  /// Any argv slot not consumed by a Take* call is an error.
  [[nodiscard]] std::optional<std::string> FirstUnknown() const {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i]) return std::string(argv_[i]);
    return std::nullopt;
  }

 private:
  int argc_;
  char** argv_;
  bool taken_[256] = {};
  std::optional<std::string> missing_value_;
};

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);  // most names escape nothing
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      // Control characters (a newline in a path, say) are invalid raw JSON.
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x",
                    static_cast<unsigned>(static_cast<unsigned char>(c)));
      out += buf;
    } else {
      out += c;
    }
  }
  return out;
}

/// Single-quotes `s` for POSIX sh, escaping embedded quotes ('\''): the
/// path comes from user-supplied --out, and a bare quote in it would
/// terminate the quoting and execute the remainder as shell code.
std::string ShellQuote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += '\'';
  return out;
}

/// The command line, parsed and held to its bounds (ParseArgs): the recipe's
/// fields are applied to `config`, and the kernel policy is still a request —
/// its table is read only once the command line is known good.
struct CompileArgs {
  std::string source_path, out_dir;
  std::optional<std::string> data_path, teacher_path;
  std::optional<std::string> dump_sir_path, report_path;
  KernelPolicyRequest policy_request;
  bool no_embed = false, verbose = false, want_build = false;
  UpdateConfig config;
};

/// The recipe's values: the --recipe file, then the flags that name the same
/// fields over it. The file is read before any flag is held, so an
/// unreadable or malformed one is exit 1 whatever else the line carries.
[[nodiscard]] std::optional<int> ReadRecipe(Args& args,
                                            recipe::Values& recipe_values) {
  const auto recipe_path = args.TakeValue("--recipe");
  if (recipe_path) {
    std::ifstream f(*recipe_path, std::ios::binary);
    if (!f) return Fail("cannot open recipe '" + *recipe_path + "'");
    std::stringstream text;
    text << f.rdbuf();
    auto parsed = recipe::ParseRecipeFile(text.str());
    if (!parsed) return Fail(*recipe_path + ": " + parsed.error());
    recipe_values = std::move(*parsed);
  }
  for (const recipe::Field& field : recipe::kFields) {
    if (field.kind == recipe::Kind::kBool) {
      // A boolean has two spellings, so a flag can override a recipe file
      // either way: --quantize-base sets the field, --no-quantize-base
      // clears it. Both at once is a contradiction, not an order to guess.
      const std::string off = "--no-" + std::string(field.flag + 2);
      const bool set = args.Take(field.flag);
      const bool clear = args.Take(off.c_str());
      if (set && clear)
        return Usage(std::string(field.flag) + " and " + off +
                     " contradict each other");
      if (set) recipe_values[field.name] = "true";
      if (clear) recipe_values[field.name] = "false";
    } else if (auto v = args.TakeValue(field.flag)) {
      recipe_values[field.name] = *v;
    }
  }
  return std::nullopt;
}

/// --gemm-tiles, --kernel-policy and --target-host: the policy as requested.
[[nodiscard]] std::optional<int> TakeKernelPolicy(
    Args& args, KernelPolicyRequest& policy_request, UpdateConfig& config) {
  if (auto v = args.TakeValue("--gemm-tiles")) {
    auto tiles = ParseGemmTilesFlag(*v);
    if (!tiles) return Usage(tiles.error());
    policy_request.explicit_tiles = *tiles;
  }
  policy_request.table_path = args.TakeValue("--kernel-policy");
  policy_request.target_host = args.TakeValue("--target-host");
  // A host key names an entry of a table: without the table the two flags
  // contradict, which is the command line's fault (exit 2), not a failed
  // lookup (it reached the policy resolver and exited 1).
  if (policy_request.target_host && !policy_request.table_path &&
      !args.MissingValue())
    return Usage(
        "--target-host names a table entry; pass the table with "
        "--kernel-policy");
  if (policy_request.target_host)
    config.target_host = *policy_request.target_host;
  return std::nullopt;
}

/// --peak-gflops and --gpu-peak-gflops: the target description.
[[nodiscard]] std::optional<int> TakePeaks(Args& args, UpdateConfig& config) {
  if (auto v = args.TakeValue("--peak-gflops")) {
    // Parsed as the double it is stored as (and as the runner parses its
    // own --peak-gflops): through an f32, 1000.1 would reach the report
    // and the baked runner constant as 1000.0999755859375. Strictly, as
    // every recipe float is: no whitespace, no hex, nothing non-finite.
    double peak = 0.0;
    if (!recipe::ParseF64Strict(*v, &peak) || !(peak > 0.0))
      return Usage("--peak-gflops must be a positive number, got '" + *v + "'");
    config.peak_gflops = peak;
  }
  // The GPU half of the target description (S6 #152): the backend screen
  // then predicts the step on the GPU too, and a Metal run prints its
  // measured step beside that prediction.
  if (auto v = args.TakeValue("--gpu-peak-gflops")) {
    double peak = 0.0;
    if (!recipe::ParseF64Strict(*v, &peak) || !(peak > 0.0))
      return Usage("--gpu-peak-gflops must be a positive number, got '" + *v +
                   "'");
    config.gpu_peak_gflops = peak;
  }
  return std::nullopt;
}

/// What no flag took, and the flags that cannot stand together: checked
/// last, when every flag has been taken.
[[nodiscard]] std::optional<int> RefuseLeftovers(const Args& args,
                                                 bool no_embed, bool want_build,
                                                 const UpdateConfig& config) {
  if (const auto& m = args.MissingValue())
    return Usage(*m + " requires a value");
  if (auto unknown = args.FirstUnknown())
    return Usage(args.TakenElsewhere(*unknown)
                     ? "'" + *unknown + "' is given more than once"
                     : "unknown argument '" + *unknown + "' (see --help)");
  // Without an embedded TU, build.sh would build the disk mode (#158)
  // regardless of the plan's size; the packer owns that choice (embed
  // below its limit, disk from it) and drives the build itself. Refuse
  // here so a --no-embed package is always packed before it is built.
  if (no_embed && want_build)
    return Usage(
        "--build needs the embedded plan TU that --no-embed skips; "
        "run `python3 tool/pack_update.py <out> --build` instead (it "
        "embeds the plan, or leaves it beside the binary above its "
        "embed limit)");
  // A relaxed plan (v18) is buildable only beside a numerics certificate,
  // and only the Python plane issues one: --build here would link an
  // uncertified relaxed binary (2026-09-24 review, N17). The emitted
  // build.sh refuses the same way, so no path builds one by hand.
  if (want_build && config.precision == Precision::kCertifiedBf16)
    return Usage(
        "--build cannot certify a --precision certified-bf16 plan; "
        "emit it, run `python3 tool/certify_numerics.py certify <out> "
        "--corpus <corpus.sds>`, then `python3 tool/pack_update.py "
        "<out> --build`");
  return std::nullopt;
}

/// The command line: every flag taken in the order that decides which
/// refusal wins. The error is the exit code — 2 for a bad command line, 1
/// for a recipe file that cannot be read, 0 after --help or --version.
[[nodiscard]] std::expected<CompileArgs, int> ParseArgs(int argc, char** argv) {
  if (argc > 255) return std::unexpected(Usage("too many arguments"));
  Args args(argc, argv);

  // Help from any slot that is not another flag's value (/code-review of
  // #213: `--out -h` printed the usage and exited 0 without compiling). In
  // a value's slot it is the flag before it left without a value.
  const HelpRequest help = FindHelp(argc, argv);
  if (help.swallowed)
    return std::unexpected(
        Usage(std::string(help.swallowed) + " requires a value"));
  if (help.asked) {
    PrintUsage();
    return std::unexpected(0);
  }

  // --version in a value flag's slot asked for nothing, as --help there
  // does not: `--out --version` left --out without a value.
  for (int i = 2; i < argc; ++i)
    if (std::strcmp(argv[i], "--version") == 0 && TakesValue(argv[i - 1]))
      return std::unexpected(
          Usage(std::string(argv[i - 1]) + " requires a value"));
  if (args.Take("--version")) {
    std::printf("seeml-update-compile %s\n", kSeemlVersion);
    return std::unexpected(0);
  }

  const auto source_path = args.TakeValue("--source");
  const auto out_dir = args.TakeValue("--out");
  if (!source_path || !out_dir) {
    if (const auto& m = args.MissingValue())
      return std::unexpected(Usage(*m + " requires a value"));
    PrintUsage();
    return std::unexpected(Usage("--source and --out are required"));
  }

  UpdateConfig config;

  // --- The recipe (S4 #150): a recipe file's fields, overridden by the
  // flags that name the same fields, applied by the one function that holds
  // every value to its bound and the fields to each other
  // (source/recipe/recipe.h). No flag is parsed by hand here.
  recipe::Values recipe_values;
  if (auto rc = ReadRecipe(args, recipe_values)) return std::unexpected(*rc);
  // --data (S2 #148): raw text or a pre-tokenized SDS the frontend holds to
  // the model and pins the plan to; the canonical corpus and its manifest
  // are written beside the plan after the compile.
  const auto data_path = args.TakeValue("--data");
  if (data_path) config.data_path = *data_path;
  const auto teacher_path = args.TakeValue("--teacher");
  if (const auto& m = args.MissingValue())
    return std::unexpected(Usage(*m + " requires a value"));
  if (auto ok = recipe::Apply(
          recipe_values,
          {.teacher = teacher_path.has_value(), .data = data_path.has_value()},
          &config);
      !ok)
    return std::unexpected(Usage(ok.error()));
  config.fuse_epilogues = !args.Take("--no-fuse-epilogue");
  // Debug: the final SIR, rendered only when a destination is named.
  const auto dump_sir_path = args.TakeValue("--dump-sir");
  config.dump_sir = dump_sir_path.has_value();
  config.fuse_elementwise = !args.Take("--no-fuse-elementwise");
  config.fuse_gemm_addend = !args.Take("--no-fuse-addend");
  config.rope_table = !args.Take("--no-rope-table");
  config.fuse_clip = !args.Take("--no-fuse-clip");

  // --- Kernel policy: explicit tiles, the tuner's table, or the defaults.
  KernelPolicyRequest policy_request;
  if (auto rc = TakeKernelPolicy(args, policy_request, config))
    return std::unexpected(*rc);

  const auto report_path = args.TakeValue("--report");
  const bool no_embed = args.Take("--no-embed");
  // The statistics screens (S0 #146) are the default output; --verbose
  // adds their verbose rows and detail tables, and the passes' notes.
  const bool verbose = args.Take("--verbose");
  if (auto rc = TakePeaks(args, config)) return std::unexpected(*rc);
  const bool want_build = args.Take("--build");

  if (auto rc = RefuseLeftovers(args, no_embed, want_build, config))
    return std::unexpected(*rc);
  return CompileArgs{.source_path = *source_path,
                     .out_dir = *out_dir,
                     .data_path = data_path,
                     .teacher_path = teacher_path,
                     .dump_sir_path = dump_sir_path,
                     .report_path = report_path,
                     .policy_request = std::move(policy_request),
                     .no_embed = no_embed,
                     .verbose = verbose,
                     .want_build = want_build,
                     .config = std::move(config)};
}

/// The kernel policy — explicit tiles, the tuner's table, or the defaults —
/// written into the config, and the log level the run asked for.
[[nodiscard]] std::expected<KernelPolicyChoice, int> ResolvePolicy(
    CompileArgs& args) {
  const KernelPolicyRequest& policy_request = args.policy_request;
  const bool verbose = args.verbose;
  UpdateConfig& config = args.config;
  // Resolved after the argument check so a malformed table is reported
  // against a well-formed command line. A table without this host is a
  // note (the defaults apply); a table that does not parse is an error.
  auto policy = ResolveKernelPolicy(policy_request);
  if (!policy) return std::unexpected(Fail(policy.error()));
  config.gemm_tile_k = policy->tiles.gemm_tile_k;
  config.gemm_tile_n = policy->tiles.gemm_tile_n;
  seeml::diag::Logger::SetLevel(verbose ? seeml::diag::LogLevel::kInfo
                                        : seeml::diag::LogLevel::kWarn);
  // The backend screen's kernel_policy row states the tiles; where they came
  // from — the host key, the table — is --verbose detail.
  if (verbose) {
    if (policy->source == "default")
      std::fprintf(stderr,
                   "seeml-update-compile: kernel policy default (the runtime's "
                   "GEMM tiles; host \"%s\")\n",
                   policy->host_key.c_str());
    else
      std::fprintf(stderr,
                   "seeml-update-compile: kernel policy from %s: GEMM tiles K "
                   "%u N %u (host \"%s\")\n",
                   policy->source.c_str(), policy->tiles.gemm_tile_k,
                   policy->tiles.gemm_tile_n, policy->host_key.c_str());
  }
  return std::move(*policy);
}

/// The student and, when one was named, the teacher.
struct LoadedModels {
  SmfModel source, teacher;
};

/// Ingest: the model files, and the exporter's report held to the source
/// model's bytes (its record goes into the config).
[[nodiscard]] std::expected<LoadedModels, int> LoadModels(CompileArgs& args) {
  const std::string* source_path = &args.source_path;
  const auto& teacher_path = args.teacher_path;
  UpdateConfig& config = args.config;
  // --- Ingest ---------------------------------------------------------------
  // Student and teacher load concurrently: one file's read overlaps the
  // other's hashing and payload copies.
  std::vector<std::string> model_paths{*source_path};
  if (teacher_path) model_paths.push_back(*teacher_path);
  auto models = LoadSmfMany(model_paths);
  if (!models) return std::unexpected(Fail(models.error()));

  LoadedModels loaded;
  loaded.source = std::move((*models)[0]);
  if (teacher_path) loaded.teacher = std::move((*models)[1]);

  // --- Exporter verification (S4 #150): the report beside the model, held to
  // the bytes just loaded. An imported checkpoint without one is refused.
  {
    auto report = ReadExportReport(*source_path, loaded.source);
    if (!report) return std::unexpected(Fail(report.error()));
    config.export_report = *report;
  }
  return loaded;
}

/// The four screens, and a relaxed plan's obligation.
void PrintScreens(const CompiledUpdate* compiled, bool verbose) {
  // The four screens (S0 #146): frontend, analysis, backend, and the
  // runtime as planned. stdout first: --verbose's notes go there.
  std::fflush(stdout);
  std::fputs(RenderCompileScreens(compiled->screens, verbose).c_str(), stderr);
  // A relaxed plan's obligation, which no screen row carries and the
  // driver's note says only under --verbose: say it on every compile.
  if (const uint64_t relaxed =
          compiled->relaxed_gemms + compiled->relaxed_attention;
      relaxed > 0)
    std::fprintf(stderr,
                 "seeml-update-compile: %" PRIu64
                 " relaxed instruction(s) (--precision certified-bf16): the "
                 "package needs a numerics certificate — `python3 "
                 "tool/certify_numerics.py certify <out> --corpus "
                 "<corpus.sds>`\n",
                 relaxed);
}

/// Where this executable is, as the OS knows it — not as argv[0] spells it.
/// Run through PATH, argv[0] is a bare name and its parent directory is
/// the empty string; through a symlink, it is the link's directory. Either
/// way the vendored runtime was then looked for in the current directory,
/// and the emission failed with a plan already written (review
/// 2026-11-25). Empty when the platform has no answer.
std::filesystem::path ExecutablePath(const char* argv0) {
  std::error_code ec;
#if defined(__APPLE__)
  char buf[4096];
  uint32_t size = sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) == 0) {
    auto real = std::filesystem::canonical(buf, ec);
    if (!ec) return real;
  }
#elif defined(__linux__)
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    buf[n] = '\0';
    auto real = std::filesystem::canonical(buf, ec);
    if (!ec) return real;
  }
#endif
  auto real = std::filesystem::canonical(argv0, ec);
  return ec ? std::filesystem::path() : real;
}

/// The tree the runtime is vendored from: the checkout this binary was
/// built in (<root>/build/seeml-update-compile), else the current
/// directory when it is one. The emitter names the tree it was given if a
/// source is missing from it.
std::string VendorRoot(const char* argv0) {
  const std::filesystem::path root =
      ExecutablePath(argv0).parent_path().parent_path();
  std::error_code ec;
  if (!root.empty() && std::filesystem::exists(root / "source", ec))
    return root.string();
  return ".";
}

/// Emit: the package beside the plan, vendored from this binary's tree.
[[nodiscard]] std::expected<EmitPaths, int> EmitPackage(
    const CompileArgs& args, const CompiledUpdate* compiled,
    const char* argv0) {
  const std::string* out_dir = &args.out_dir;
  const bool no_embed = args.no_embed;
  const UpdateConfig& config = args.config;
  const std::string repo_root = VendorRoot(argv0);
  EmitOptions emit_options;
  // Any kFlagRelaxed instruction (v18 GEMMs, v20 tiled attention) makes
  // the package one that builds only beside a certificate.
  emit_options.relaxed_plan =
      compiled->relaxed_gemms + compiled->relaxed_attention > 0;
  emit_options.embed_plan_tu = !no_embed;
  emit_options.peak_gflops = config.peak_gflops;
  emit_options.gpu_peak_gflops = config.gpu_peak_gflops;
  auto paths =
      EmitNativePackage(compiled->plan, *out_dir, repo_root, emit_options);
  if (!paths) return std::unexpected(Fail(paths.error()));

  std::fprintf(stderr, "seeml-update-compile: emitted %s\n",
               paths->plan_file.c_str());
  return std::move(*paths);
}

/// Staged and renamed into place (as the runner's --report is): a
/// write that fails midway leaves no corpus.sds at all, never a
/// truncated one under the name the plan is pinned to. The two failures
/// are two messages — the bytes did not reach the staged file, or the
/// staged file could not take the name (a directory is in the way, say) —
/// and neither leaves the staged file behind.
[[nodiscard]] std::optional<std::string> WriteStaged(const std::string& path,
                                                     const char* data,
                                                     size_t size) {
  const std::string staged = path + ".tmp";
  std::ofstream f(staged, std::ios::binary | std::ios::trunc);
  f.write(data, static_cast<std::streamsize>(size));
  f.close();
  if (f.fail()) {
    std::remove(staged.c_str());
    return "short write to '" + path + "'";
  }
  if (std::rename(staged.c_str(), path.c_str()) != 0) {
    std::remove(staged.c_str());
    return "cannot move '" + staged + "' to '" + path + "'";
  }
  return std::nullopt;
}

/// The canonical corpus (S2 #148): raw text was tokenized here, so the
/// SDS the plan is pinned to and the manifest of what was dropped are
/// written beside the plan; a given SDS is the corpus as is.
[[nodiscard]] std::optional<int> WriteCorpus(const CompileArgs& args,
                                             const CompiledUpdate* compiled,
                                             std::string& corpus_file) {
  const auto& data_path = args.data_path;
  const std::string* out_dir = &args.out_dir;
  if (data_path) {
    if (compiled->corpus_from_sds) {
      corpus_file = *data_path;
    } else {
      corpus_file = *out_dir + "/corpus.sds";
      const std::string manifest_file = *out_dir + "/corpus_manifest.json";
      if (auto why = WriteStaged(
              corpus_file,
              reinterpret_cast<const char*>(compiled->corpus_sds.data()),
              compiled->corpus_sds.size()))
        return Fail(*why);
      if (auto why =
              WriteStaged(manifest_file, compiled->corpus_manifest.data(),
                          compiled->corpus_manifest.size()))
        return Fail(*why);
      std::fprintf(stderr,
                   "seeml-update-compile: wrote %s (the canonical corpus the "
                   "plan is pinned to) and %s\n",
                   corpus_file.c_str(), manifest_file.c_str());
    }
  }
  return std::nullopt;
}

/// What the package needs next: the embed-limit note, the --no-embed road.
void NoteEmbedding(const CompileArgs& args, const CompiledUpdate* compiled) {
  const std::string* out_dir = &args.out_dir;
  const bool no_embed = args.no_embed;
  // The decimal TU of a plan this large is ~4x its bytes of C++ and, once
  // linked, an image the macOS arm64 linker and loader refuse near 1.8 GB
  // (#158); the packer's disk mode is the form for it. A note, not a
  // refusal: the emission is complete and the package is still a package.
  constexpr size_t kEmbedLimitBytes = size_t{1} << 30;
  if (!no_embed && compiled->plan.size() >= kEmbedLimitBytes)
    std::fprintf(stderr,
                 "seeml-update-compile: note: the plan is %zu bytes, at or "
                 "above the 1 GiB embed limit — an embedded image this large "
                 "may not link or load (macOS arm64 fails near 1.8 GB); "
                 "prefer --no-embed and `python3 tool/pack_update.py %s "
                 "--build`, which leaves such a plan beside the binary\n",
                 compiled->plan.size(), out_dir->c_str());
  if (no_embed)
    std::fprintf(stderr,
                 "seeml-update-compile: no embedded TU (--no-embed); next:"
                 " python3 tool/pack_update.py %s --build\n",
                 out_dir->c_str());
}

/// What --report states: the run's inputs and everything it produced.
struct ReportContext {
  const CompileArgs& args;
  const CompiledUpdate* compiled;
  const EmitPaths* paths;
  const KernelPolicyChoice* policy;
  const std::string& corpus_file;
};

/// The report's scalar keys, through the opening of "adapters".
void PrintReportScalars(std::FILE* f, const ReportContext& context) {
  const UpdateConfig& config = context.args.config;
  const CompiledUpdate* compiled = context.compiled;
  const EmitPaths* paths = context.paths;
  const KernelPolicyChoice* policy = context.policy;
  const std::string& corpus_file = context.corpus_file;
  PlanHeader plan_header;
  std::memcpy(&plan_header, compiled->plan.data(), sizeof(plan_header));
  const HostArchInfo host_arch = DetectHostArch();
  // null when no embedded TU was written (--no-embed): the packer's stub
  // is the package's TU then, and the emitter has no path to report.
  const std::string embedded_tu_json =
      paths->embedded_tu.empty() ? "null"
                                 : "\"" + JsonEscape(paths->embedded_tu) + "\"";
  // The corpus the plan is pinned to (S2 #148): null without --data.
  const std::string corpus_json =
      corpus_file.empty() ? "null" : "\"" + JsonEscape(corpus_file) + "\"";
  // The exact resident footprint the final gate checked (the arena plus
  // the plan blob), which the step-0 estimate is a lower bound on.
  const uint64_t exact_footprint =
      compiled->arena_size + static_cast<uint64_t>(compiled->plan.size());
  // The format is laid out a JSON key to a line.
  // clang-format off
  std::fprintf(f,
               "{\n"
               "  \"schema\": 1,\n"
               "  \"seeml_version\": \"%s\",\n"
               "  \"plan_file\": \"%s\",\n"
               "  \"plan_bytes\": %zu,\n"
               "  \"plan_version\": %u,\n"
               "  \"plan_hash\": \"%016" PRIx64 "\",\n"
               "  \"source_model_hash\": \"%016" PRIx64 "\",\n"
               "  \"recipe_hash\": \"%016" PRIx64 "\",\n"
               "  \"identity\": {\"model_hash\": \"%016" PRIx64
               "\", \"tokenizer_hash\": \"%016" PRIx64
               "\", \"corpus_hash\": \"%016" PRIx64
               "\", \"recipe_hash\": \"%016" PRIx64
               "\", \"val_split_hash\": \"%016" PRIx64
               "\", \"split_seed\": %" PRIu64
               ", \"masked_loss\": %s, \"export_report_hash\": \"%016" PRIx64
               "\"},\n"
               "  \"host_arch\": {\"isa\": \"%s\", \"cpu_model\": \"%s\", "
               "\"physical_cores\": %zu, \"l1d_bytes\": %" PRIu64
               ", \"l2_bytes\": %" PRIu64 ", \"simd_width_f32\": %zu},\n"
               "  \"arena_bytes\": %" PRIu64 ",\n"
               "  \"persistent_bytes\": %" PRIu64 ",\n"
               "  \"io_bytes\": %" PRIu64 ",\n"
               "  \"transient_bytes\": %" PRIu64 ",\n"
               "  \"footprint_estimate\": {\"weights\": %" PRIu64
               ", \"activations\": %" PRIu64 ", \"deltas\": %" PRIu64
               ", \"attention_caches\": %" PRIu64 ", \"total\": %" PRIu64
               ", \"exact\": %" PRIu64 ", \"ratio\": %.4f},\n"
               "  \"remat\": {\"policy\": \"%s\", \"applied\": %s, "
               "\"recomputed_values\": %" PRIu64 ", \"bytes_saved\": %" PRIu64
               "},\n"
               "  \"rodata_bytes\": %" PRIu64 ",\n"
               "  \"train_instructions\": %" PRIu64 ",\n"
               "  \"eval_instructions\": %" PRIu64 ",\n"
               "  \"merge_instructions\": %" PRIu64 ",\n"
               "  \"step_instructions\": %" PRIu64 ",\n"
               "  \"grad_accum_steps\": %u,\n"
               "  \"effective_batch\": %" PRIu64 ",\n"
               "  \"quantized_base\": %s,\n"
               "  \"bf16_base\": %s,\n"
               "  \"adapter_only\": %s,\n"
               "  \"precision\": \"%s\",\n"
               "  \"relaxed_gemms\": %" PRIu64 ",\n"
               "  \"relaxed_attention\": %" PRIu64 ",\n"
               "  \"data\": {\"corpus\": %s, \"corpus_hash\": \"%016" PRIx64
               "\", \"tokenizer_hash\": \"%016" PRIx64
               "\", \"data_flags\": %u, \"val_frac\": %.9g, \"split_seed\": %" PRIu64
               ", \"val_split_hash\": \"%016" PRIx64
               "\", \"held_out_twins\": %" PRIu64 "},\n"
               "  \"validation_scores\": \"%s\",\n"
               "  \"attention\": {\"family\": \"%s\", "
               "\"cached_probs_bytes\": %" PRIu64 "},\n"
               "  \"embedded_tu\": %s,\n"
               "  \"kernel_policy\": {\"source\": \"%s\", \"host_key\": "
               "\"%s\", \"gemm_tile_k\": %u, \"gemm_tile_n\": %u},\n"
               "  \"adapters\": [",
               kSeemlVersion, JsonEscape(paths->plan_file).c_str(),
               compiled->plan.size(), plan_header.version,
               plan_header.plan_hash, plan_header.source_model_hash,
               compiled->recipe_hash,
               plan_header.source_model_hash, plan_header.tokenizer_hash,
               plan_header.corpus_hash, compiled->recipe_hash,
               plan_header.val_split_hash, plan_header.split_seed,
               (plan_header.data_flags & kDataFlagMaskedLoss) ? "true"
                                                              : "false",
               config.export_report.report_hash,
               std::string(host_arch.isa).c_str(),
               JsonEscape(host_arch.cpu_model).c_str(),
               host_arch.physical_cores, host_arch.l1d_bytes,
               host_arch.l2_bytes, host_arch.simd_width_f32,
               compiled->arena_size,
               compiled->persistent_size, compiled->io_size,
               compiled->transient_size, compiled->estimate_weight_bytes,
               compiled->estimate_activation_bytes,
               compiled->estimate_delta_bytes, compiled->estimate_cache_bytes,
               compiled->estimate_bytes, exact_footprint,
               exact_footprint ? static_cast<double>(compiled->estimate_bytes) /
                                     static_cast<double>(exact_footprint)
                               : 0.0,
               compiled->remat_policy == RematKind::kAuto  ? "auto"
               : compiled->remat_policy == RematKind::kOn ? "on"
                                                           : "off",
               compiled->remat_applied ? "true" : "false",
               compiled->remat_recomputed_values,
               compiled->remat_bytes_saved, compiled->rodata_size,
               compiled->train_instruction_count,
               compiled->eval_instruction_count,
               compiled->merge_instruction_count,
               compiled->step_instruction_count, compiled->grad_accum_steps,
               static_cast<uint64_t>(config.batch) * compiled->grad_accum_steps,
               config.quantize_base ? "true" : "false",
               config.bf16_base ? "true" : "false",
               compiled->adapter_only ? "true" : "false",
               config.precision == Precision::kCertifiedBf16
                   ? "certified-bf16" : "f32",
               compiled->relaxed_gemms, compiled->relaxed_attention,
               corpus_json.c_str(), plan_header.corpus_hash,
               plan_header.tokenizer_hash, plan_header.data_flags,
               static_cast<double>(ValFracOf(plan_header.val_frac_bits)),
               plan_header.split_seed, plan_header.val_split_hash,
               compiled->corpus_held_out_twins,
               compiled->scores_shipped ? "shipped" : "plan",
               compiled->attention_tiled ? "tiled" : "cached",
               compiled->probs_cache_bytes,
               embedded_tu_json.c_str(), policy->source.c_str(),
               JsonEscape(policy->host_key).c_str(),
               compiled->gemm_tile_k, compiled->gemm_tile_n);
  // clang-format on
}

/// The report's lists: the adapters, the vendored sources, the passes and
/// the four screens; closes the document.
void PrintReportLists(std::FILE* f, const ReportContext& context) {
  const CompiledUpdate* compiled = context.compiled;
  const EmitPaths* paths = context.paths;
  for (size_t i = 0; i < compiled->adapters.size(); ++i) {
    const auto& a = compiled->adapters[i];
    std::fprintf(f,
                 "%s\n    {\"weight\": \"%s\", \"k\": %" PRId64
                 ", \"m\": %" PRId64 ", \"rank\": %" PRId64
                 ", \"scale\": %g, \"quant_scale\": %g, \"bf16\": %s}",
                 i ? "," : "", JsonEscape(a.weight_name).c_str(), a.k, a.m, a.r,
                 a.scale, a.quant_scale, a.bf16 ? "true" : "false");
  }
  // The package's exact contents (P6): what the packer vendors against.
  std::fprintf(f, "\n  ],\n  \"vendored_sources\": [");
  for (size_t i = 0; i < paths->vendored_sources.size(); ++i)
    std::fprintf(f, "%s\"%s\"", i ? ", " : "",
                 JsonEscape(paths->vendored_sources[i]).c_str());
  std::fprintf(f, "]");
  // Every pass and driver phase, in order, with its wall time: the
  // measurement behind any compile-side performance claim (E5, #84).
  std::fprintf(f, ",\n  \"passes\": [");
  for (size_t i = 0; i < compiled->pass_timings.size(); ++i) {
    const auto& t = compiled->pass_timings[i];
    std::fprintf(f, "%s\n    {\"name\": \"%s\", \"ops\": %zu, \"ms\": %.3f}",
                 i ? "," : "", JsonEscape(t.name).c_str(), t.ops_after, t.ms);
  }
  // The statistics screens (S0 #146): the same rows the console printed,
  // every one of them, by key — the schema seeml-abi publishes. Additive,
  // so the report stays schema 1, as every key added since has been
  // (footprint_estimate, adapter_only, relaxed_attention, ...).
  std::fprintf(f, "\n  ],\n  \"screens\": {");
  const Screen* screens[] = {
      &compiled->screens.frontend, &compiled->screens.analysis,
      &compiled->screens.backend, &compiled->screens.runtime};
  for (size_t i = 0; i < 4; ++i)
    std::fprintf(f, "%s\n    \"%s\": %s", i ? "," : "", screens[i]->spec().name,
                 screens[i]->Json(4).c_str());
  std::fprintf(f, "\n  }\n}\n");
}

/// --report: the machine-readable compile report.
[[nodiscard]] std::optional<int> WriteReport(const ReportContext& context) {
  const std::string* report_path = &*context.args.report_path;
  std::FILE* f = seeml::platform::OpenStreamForWrite(*report_path);
  if (!f) return Fail("cannot write report '" + *report_path + "'");
  PrintReportScalars(f, context);
  PrintReportLists(f, context);
  // ferror catches any failed fprintf above; fclose catches the final
  // flush. A truncated report that exits 0 hands downstream automation
  // invalid JSON with a success status — the silent default this tool's
  // header promises never to have.
  const bool write_failed = std::ferror(f) != 0;
  if (std::fclose(f) != 0 || write_failed)
    return Fail("short write to report '" + *report_path + "'");
  return std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
  auto parsed = ParseArgs(argc, argv);
  if (!parsed) return parsed.error();
  CompileArgs& args = *parsed;
  auto policy = ResolvePolicy(args);
  if (!policy) return policy.error();
  auto models = LoadModels(args);
  if (!models) return models.error();
  SmfModel* teacher = args.teacher_path ? &models->teacher : nullptr;

  // --- Compile ----------------------------------------------------------------
  UpdateCompiler compiler(args.config);
  // The consuming compile: nothing below reads a weight again, so each
  // payload is released as plan assembly packs it (one resident copy of
  // the weights while the blob is built, not two).
  auto compiled = compiler.Compile(std::move(models->source), teacher);
  if (!compiled) return Fail(compiled.error());
  PrintScreens(&*compiled, args.verbose);

  auto paths = EmitPackage(args, &*compiled, argv[0]);
  if (!paths) return paths.error();
  std::string corpus_file;
  if (auto rc = WriteCorpus(args, &*compiled, corpus_file)) return *rc;
  NoteEmbedding(args, &*compiled);

  if (args.dump_sir_path) {
    std::ofstream f(*args.dump_sir_path, std::ios::binary | std::ios::trunc);
    f << compiled->sir_dump;
    f.close();
    if (f.fail()) return Fail("short write to '" + *args.dump_sir_path + "'");
  }

  if (args.report_path)
    if (auto rc =
            WriteReport({args, &*compiled, &*paths, &*policy, corpus_file}))
      return *rc;

  if (args.want_build) {
    const std::string cmd = "sh " + ShellQuote(paths->build_script);
    if (std::system(cmd.c_str()) != 0) return Fail("build.sh failed");
  }
  return 0;
}
