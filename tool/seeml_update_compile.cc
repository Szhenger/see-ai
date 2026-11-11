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
//       [--targets substr,substr]      restrict adapters to matching weights
//       [--optimizer adamw|sgd] [--lr 1e-3] [--weight-decay 0.01]
//       [--clip-norm 0]                per-tensor L2 gradient clip (0 = off)
//       [--lr-schedule const|cosine]   runtime LR schedule
//       [--warmup 0]                   warmup steps (cosine schedule)
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
//
// Every numeric flag is parsed strictly: trailing garbage, overflow, NaN/Inf,
// or an unknown flag is a hard error, never a silent default.
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
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "compiler/backend/architecture/host_arch.h"
#include "compiler/backend/packaging/native_emitter.h"
#include "compiler/backend/architecture/kernel_policy_table.h"
#include "compiler/diagnostics/logger.h"
#include "compiler/driver/statistics.h"
#include "compiler/driver/update_compiler.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "source/contract/split.h"
#include "source/identity/version.h"
#include "source/recipe/recipe.h"
#include "source/recipe/recipe_file.h"

namespace {

using namespace seeml::update;

int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-update-compile: %s\n", msg.c_str());
  return 1;
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: seeml-update-compile --source model.smf --out dir/\n"
               "  [--recipe recipe.json]  the fields below as one JSON object;\n"
               "                      a flag overrides the field it names;\n"
               "                      a boolean flag --x has the form --no-x\n"
               "  [--data-batch N] [--loss xent|mse|kl|xent+kl]\n"
               "  [--teacher t.smf] [--distill-weight W] [--temperature T]\n"
               "  [--lora-rank R] [--lora-alpha A] [--lora-seed S]\n"
               "  [--targets substr,...] [--optimizer adamw|sgd] [--lr LR]\n"
               "  [--weight-decay WD] [--clip-norm C]\n"
               "  [--lr-schedule const|cosine] [--warmup N] [--allow-zero-lr]\n"
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
               "  [--steps N]\n"
               "  [--grad-accum G]\n"
               "  [--no-fuse-epilogue] [--no-fuse-elementwise] [--no-fuse-addend]\n"
               "  [--no-rope-table]\n"
               "  [--no-fuse-clip]\n"
               "  [--report out.json] [--dump-sir out.txt]\n"
               "  [--kernel-policy table.json] [--target-host KEY]\n"
               "  [--gemm-tiles K,N]\n"
               "  [--no-embed] [--build] [--version]\n"
               "  [--peak-gflops G]   the target's peak; the backend screen then\n"
               "                      predicts a step's time from the plan's work\n"
               "  [--verbose]         every screen row, the per-pass and per-adapter\n"
               "                      tables, and each pass's note\n");
}

/// Strict argument cursor: every flag must be known, every value must parse
/// completely. This is the difference between `--lr 1e-3` and `--lr abc`
/// silently training at lr = 0.
class Args {
 public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}

  bool Take(const char* flag) {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        taken_[i] = true;
        return true;
      }
    return false;
  }

  std::optional<std::string> TakeValue(const char* flag) {
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

  /// Any argv slot not consumed by a Take* call is an error.
  std::optional<std::string> FirstUnknown() const {
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

}  // namespace

int main(int argc, char** argv) {
  if (argc > 255) return Fail("too many arguments");
  Args args(argc, argv);

  if (args.Take("--help") || args.Take("-h")) {
    PrintUsage();
    return 0;
  }

  if (args.Take("--version")) {
    std::printf("seeml-update-compile %s\n", kSeemlVersion);
    return 0;
  }

  const auto source_path = args.TakeValue("--source");
  const auto out_dir = args.TakeValue("--out");
  if (!source_path || !out_dir) {
    if (const auto& m = args.MissingValue())
      return Fail(*m + " requires a value");
    PrintUsage();
    return Fail("--source and --out are required");
  }

  UpdateConfig config;

  // --- The recipe (S4 #150): a recipe file's fields, overridden by the
  // flags that name the same fields, applied by the one function that holds
  // every value to its bound and the fields to each other
  // (source/recipe/recipe.h). No flag is parsed by hand here.
  recipe::Values recipe_values;
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
        return Fail(std::string(field.flag) + " and " + off +
                    " contradict each other");
      if (set) recipe_values[field.name] = "true";
      if (clear) recipe_values[field.name] = "false";
    } else if (auto v = args.TakeValue(field.flag)) {
      recipe_values[field.name] = *v;
    }
  }
  // --data (S2 #148): raw text or a pre-tokenized SDS the frontend holds to
  // the model and pins the plan to; the canonical corpus and its manifest
  // are written beside the plan after the compile.
  const auto data_path = args.TakeValue("--data");
  if (data_path) config.data_path = *data_path;
  const auto teacher_path = args.TakeValue("--teacher");
  if (const auto& m = args.MissingValue())
    return Fail(*m + " requires a value");
  if (auto ok = recipe::Apply(
          recipe_values,
          {.teacher = teacher_path.has_value(), .data = data_path.has_value()},
          &config);
      !ok)
    return Fail(ok.error());
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
  if (auto v = args.TakeValue("--gemm-tiles")) {
    auto tiles = ParseGemmTilesFlag(*v);
    if (!tiles) return Fail(tiles.error());
    policy_request.explicit_tiles = *tiles;
  }
  policy_request.table_path = args.TakeValue("--kernel-policy");
  policy_request.target_host = args.TakeValue("--target-host");

  const auto report_path = args.TakeValue("--report");
  const bool no_embed = args.Take("--no-embed");
  // The statistics screens (S0 #146) are the default output; --verbose
  // adds their verbose rows and detail tables, and the passes' notes.
  const bool verbose = args.Take("--verbose");
  if (auto v = args.TakeValue("--peak-gflops")) {
    // Parsed as the double it is stored as (and as the runner parses its
    // own --peak-gflops): through an f32, 1000.1 would reach the report
    // and the baked runner constant as 1000.0999755859375. Strictly, as
    // every recipe float is: no whitespace, no hex, nothing non-finite.
    double peak = 0.0;
    if (!recipe::ParseF64Strict(*v, &peak) || !(peak > 0.0))
      return Fail("--peak-gflops must be a positive number, got '" + *v +
                  "'");
    config.peak_gflops = peak;
  }
  const bool want_build = args.Take("--build");

  if (const auto& m = args.MissingValue())
    return Fail(*m + " requires a value");
  if (auto unknown = args.FirstUnknown())
    return Fail("unknown argument '" + *unknown + "' (see --help)");
  // Without an embedded TU, build.sh would build the disk mode (#158)
  // regardless of the plan's size; the packer owns that choice (embed
  // below its limit, disk from it) and drives the build itself. Refuse
  // here so a --no-embed package is always packed before it is built.
  if (no_embed && want_build)
    return Fail("--build needs the embedded plan TU that --no-embed skips; "
                "run `python3 tool/pack_update.py <out> --build` instead (it "
                "embeds the plan, or leaves it beside the binary above its "
                "embed limit)");
  // A relaxed plan (v18) is buildable only beside a numerics certificate,
  // and only the Python plane issues one: --build here would link an
  // uncertified relaxed binary (2026-09-24 review, N17). The emitted
  // build.sh refuses the same way, so no path builds one by hand.
  if (want_build && config.precision == Precision::kCertifiedBf16)
    return Fail("--build cannot certify a --precision certified-bf16 plan; "
                "emit it, run `python3 tool/certify_numerics.py certify <out> "
                "--corpus <corpus.sds>`, then `python3 tool/pack_update.py "
                "<out> --build`");

  // Resolved after the argument check so a malformed table is reported
  // against a well-formed command line. A table without this host is a
  // note (the defaults apply); a table that does not parse is an error.
  auto policy = ResolveKernelPolicy(policy_request);
  if (!policy) return Fail(policy.error());
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

  // --- Ingest ---------------------------------------------------------------
  // Student and teacher load concurrently: one file's read overlaps the
  // other's hashing and payload copies.
  std::vector<std::string> model_paths{*source_path};
  if (teacher_path) model_paths.push_back(*teacher_path);
  auto models = LoadSmfMany(model_paths);
  if (!models) return Fail(models.error());

  SmfModel source_model = std::move((*models)[0]);
  SmfModel teacher_model;
  SmfModel* teacher = nullptr;
  if (teacher_path) {
    teacher_model = std::move((*models)[1]);
    teacher = &teacher_model;
  }

  // --- Compile ----------------------------------------------------------------
  UpdateCompiler compiler(config);
  // The consuming compile: nothing below reads a weight again, so each
  // payload is released as plan assembly packs it (one resident copy of
  // the weights while the blob is built, not two).
  auto compiled = compiler.Compile(std::move(source_model), teacher);
  if (!compiled) return Fail(compiled.error());

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

  // --- Emit ------------------------------------------------------------------
  const std::string repo_root =
      std::filesystem::path(argv[0]).parent_path().parent_path().string();
  EmitOptions emit_options;
  // Any kFlagRelaxed instruction (v18 GEMMs, v20 tiled attention) makes
  // the package one that builds only beside a certificate.
  emit_options.relaxed_plan =
      compiled->relaxed_gemms + compiled->relaxed_attention > 0;
  emit_options.embed_plan_tu = !no_embed;
  emit_options.peak_gflops = config.peak_gflops;
  auto paths = EmitNativePackage(
      compiled->plan, *out_dir,
      repo_root.empty() || !std::filesystem::exists(repo_root + "/source")
          ? "."
          : repo_root,
      emit_options);
  if (!paths) return Fail(paths.error());

  std::fprintf(stderr, "seeml-update-compile: emitted %s\n",
               paths->plan_file.c_str());
  // The canonical corpus (S2 #148): raw text was tokenized here, so the
  // SDS the plan is pinned to and the manifest of what was dropped are
  // written beside the plan; a given SDS is the corpus as is.
  std::string corpus_file;
  if (data_path) {
    if (compiled->corpus_from_sds) {
      corpus_file = *data_path;
    } else {
      corpus_file = *out_dir + "/corpus.sds";
      const std::string manifest_file = *out_dir + "/corpus_manifest.json";
      // Staged and renamed into place (as the runner's --report is): a
      // write that fails midway leaves no corpus.sds at all, never a
      // truncated one under the name the plan is pinned to.
      auto write_staged = [](const std::string& path, const char* data,
                             size_t size) -> std::optional<std::string> {
        const std::string staged = path + ".tmp";
        std::ofstream f(staged, std::ios::binary | std::ios::trunc);
        f.write(data, static_cast<std::streamsize>(size));
        f.close();
        if (f.fail() || std::rename(staged.c_str(), path.c_str()) != 0) {
          std::remove(staged.c_str());
          return "short write to '" + path + "'";
        }
        return std::nullopt;
      };
      if (auto why = write_staged(
              corpus_file,
              reinterpret_cast<const char*>(compiled->corpus_sds.data()),
              compiled->corpus_sds.size()))
        return Fail(*why);
      if (auto why = write_staged(manifest_file, compiled->corpus_manifest.data(),
                                  compiled->corpus_manifest.size()))
        return Fail(*why);
      std::fprintf(stderr,
                   "seeml-update-compile: wrote %s (the canonical corpus the "
                   "plan is pinned to) and %s\n",
                   corpus_file.c_str(), manifest_file.c_str());
    }
  }
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

  if (dump_sir_path) {
    std::ofstream f(*dump_sir_path, std::ios::binary | std::ios::trunc);
    f << compiled->sir_dump;
    f.close();
    if (f.fail()) return Fail("short write to '" + *dump_sir_path + "'");
  }

  // --- Machine-readable report -------------------------------------------------
  if (report_path) {
    std::FILE* f = std::fopen(report_path->c_str(), "w");
    if (!f) return Fail("cannot write report '" + *report_path + "'");
    PlanHeader plan_header;
    std::memcpy(&plan_header, compiled->plan.data(), sizeof(plan_header));
    const HostArchInfo host_arch = DetectHostArch();
    // null when no embedded TU was written (--no-embed): the packer's stub
    // is the package's TU then, and the emitter has no path to report.
    const std::string embedded_tu_json =
        paths->embedded_tu.empty()
            ? "null"
            : "\"" + JsonEscape(paths->embedded_tu) + "\"";
    // The corpus the plan is pinned to (S2 #148): null without --data.
    const std::string corpus_json =
        corpus_file.empty() ? "null"
                            : "\"" + JsonEscape(corpus_file) + "\"";
    // The exact resident footprint the final gate checked (the arena plus
    // the plan blob), which the step-0 estimate is a lower bound on.
    const uint64_t exact_footprint =
        compiled->arena_size + static_cast<uint64_t>(compiled->plan.size());
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
                 ", \"val_split_hash\": \"%016" PRIx64 "\"},\n"
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
                 recipe::RecipeHash(config),
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
                 compiled->scores_shipped ? "shipped" : "plan",
                 compiled->attention_tiled ? "tiled" : "cached",
                 compiled->probs_cache_bytes,
                 embedded_tu_json.c_str(), policy->source.c_str(),
                 JsonEscape(policy->host_key).c_str(),
                 compiled->gemm_tile_k, compiled->gemm_tile_n);
    for (size_t i = 0; i < compiled->adapters.size(); ++i) {
      const auto& a = compiled->adapters[i];
      std::fprintf(f,
                   "%s\n    {\"weight\": \"%s\", \"k\": %" PRId64
                   ", \"m\": %" PRId64 ", \"rank\": %" PRId64
                   ", \"scale\": %g, \"quant_scale\": %g, \"bf16\": %s}",
                   i ? "," : "", JsonEscape(a.weight_name).c_str(), a.k, a.m,
                   a.r, a.scale, a.quant_scale, a.bf16 ? "true" : "false");
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
                   i ? "," : "", JsonEscape(t.name).c_str(), t.ops_after,
                   t.ms);
    }
    // The statistics screens (S0 #146): the same rows the console printed,
    // every one of them, by key — the schema seeml-abi publishes. Additive,
    // so the report stays schema 1, as every key added since has been
    // (footprint_estimate, adapter_only, relaxed_attention, ...).
    std::fprintf(f, "\n  ],\n  \"screens\": {");
    const Screen* screens[] = {&compiled->screens.frontend,
                               &compiled->screens.analysis,
                               &compiled->screens.backend,
                               &compiled->screens.runtime};
    for (size_t i = 0; i < 4; ++i)
      std::fprintf(f, "%s\n    \"%s\": %s", i ? "," : "",
                   screens[i]->spec().name, screens[i]->Json(4).c_str());
    std::fprintf(f, "\n  }\n}\n");
    // ferror catches any failed fprintf above; fclose catches the final
    // flush. A truncated report that exits 0 hands downstream automation
    // invalid JSON with a success status — the silent default this tool's
    // header promises never to have.
    const bool write_failed = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || write_failed)
      return Fail("short write to report '" + *report_path + "'");
  }

  if (want_build) {
    const std::string cmd = "sh " + ShellQuote(paths->build_script);
    if (std::system(cmd.c_str()) != 0) return Fail("build.sh failed");
  }
  return 0;
}
