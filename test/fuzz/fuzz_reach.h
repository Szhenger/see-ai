#ifndef SEEML_TEST_FUZZ_FUZZ_REACH_H_
#define SEEML_TEST_FUZZ_FUZZ_REACH_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

// =============================================================================
// The fuzz harness's framing and its reach counters (CR3 #208).
//
// A fuzzing run that never gets an input past a seal or a contract covers
// the refusal paths and nothing else — and looks exactly like one that
// does. So the harness counts, at the points that matter, what its inputs
// reached: a plan the loader accepted, a training step that ran, a compile
// that completed, a tokenizer section that loaded. The counters are plain
// statics, always compiled in (one increment each), read by the replay
// suite (test/fuzz/fuzz_replay_test.cc), which runs the checked-in seed
// corpus through the harness's entry point in every build and fails when
// an arm is not reached.
//
// The framing: byte 0 selects the arm (`byte % kArmCount`); the rest is the
// arm's input. An arm that reads several files takes them joined by
// kPartSeparator. test/fuzz/make_seeds.py writes the seed corpus in this
// framing and states the same constants.
// =============================================================================

namespace seeml::fuzz {

enum Arm : uint8_t {
  kArmSmf = 0,          // LoadSmf
  kArmPlan = 1,         // re-seal, load, execute
  kArmSds = 2,          // Dataset::LoadFromFile
  kArmCompile = 3,      // LoadSmf + Compile, no corpus
  kArmPlanRaw = 4,      // load as sealed (no re-seal), execute
  kArmTokenizer = 5,    // [section | kind seq text]: Load, ingest
  kArmRecipe = 6,       // ParseRecipeFile + Apply
  kArmReport = 7,       // [smf | report]: ReadExportReport
  kArmCompileData = 8,  // [smf | recipe | kind corpus]: Compile over data
  kArmCount = 9,
};

/// What joins the files of a multi-file arm: eight bytes no text holds
/// (0xF5 is in no UTF-8 sequence) and a binary part holds only by chance.
inline constexpr std::string_view kPartSeparator = "\xf5seeml\xf5\xf5";

/// The corpus kinds of kArmCompileData (the byte before the corpus,
/// `% 3`), and the text kinds of kArmTokenizer (`% 2`).
inline constexpr uint8_t kDataSds = 0, kDataJsonl = 1, kDataTxt = 2;
inline constexpr uint8_t kTextJsonl = 0, kTextOneDocument = 1;

/// What the inputs so far reached. Monotonic; the replay suite takes the
/// difference across one input.
struct Reach {
  uint64_t inputs = 0;
  // The binary readers.
  uint64_t smf_loaded = 0;
  uint64_t sds_loaded = 0;
  // The plan arms: the loader (seal, both contracts, the validator) took
  // the plan; a plan within the harness's bounds was executed.
  uint64_t plans_accepted = 0;
  uint64_t plans_refused = 0;
  uint64_t instructions_executed = 0;  // of the programs a step ran
  uint64_t train_steps = 0;            // Train returned a report
  uint64_t train_refused = 0;          // Train returned an error
  uint64_t evals = 0;                  // Evaluate returned a loss
  uint64_t merges = 0;                 // RunMerge succeeded
  // Executed instructions by opcode (a step that ran): what holds "one
  // seed per fixture family" to the family's own kernels.
  std::array<uint64_t, 128> opcode_executed{};
  uint64_t relaxed_executed = 0;  // instructions carrying kFlagRelaxed
  // The compile arms.
  uint64_t compiles_started = 0;  // a loaded model within the bounds
  uint64_t compiles_completed = 0;
  uint64_t compiled_plans_accepted = 0;  // the loader took the output
  uint64_t compiled_plans_refused = 0;   // it did not: a compiler defect
  uint64_t corpora_compiled = 0;         // a compile that took a corpus
  // The text-facing parsers.
  uint64_t tokenizers_loaded = 0;
  uint64_t texts_encoded = 0;
  uint64_t jsonl_parsed = 0;        // DocumentsFromJsonl returned documents
  uint64_t ingests = 0;             // Ingest returned a corpus
  uint64_t ingest_windows = 0;      // records those corpora hold
  uint64_t ingest_sds_loaded = 0;   // the corpus reader took Ingest's SDS
  uint64_t ingest_sds_refused = 0;  // it did not: a defect in one of them
  uint64_t recipes_parsed = 0;
  uint64_t recipes_applied = 0;
  uint64_t reports_read = 0;  // a present report, bound to its model
};

/// The harness's counters (test/fuzz/binary_formats.cc defines them).
Reach& TheReach();

/// Makes the harness print each refusal it meets to stderr, as
/// "fuzz: <stage>: <the refusal>" — why a seed stopped where it did. Off
/// in a fuzzing run.
void SetNarration(bool on);

}  // namespace seeml::fuzz

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

#endif  // SEEML_TEST_FUZZ_FUZZ_REACH_H_
