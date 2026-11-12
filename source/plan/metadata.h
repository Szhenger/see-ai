#ifndef SEEML_SOURCE_PLAN_METADATA_H_
#define SEEML_SOURCE_PLAN_METADATA_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "source/plan/instruction.h"
#include "source/plan/schema.h"

// =============================================================================
// The plan's metadata section (plan v23; S4 #150, S5 #151, S6 #152).
//
// The header is 320 bytes, the last size that moves no section, so what the
// three issues add to a plan lives in one trailing section instead: it
// starts at the first 64-aligned byte after the emit table, inside the
// page padding every plan already ends with, and is present exactly when
// the header carries kDataFlagMetadata. No section offset moves; the seal
// covers it like every other byte.
//
//   PlanMetadata   the identity manifest's words the header has no room
//                  for — the resolved recipe's hash and what the exporter
//                  proved about the model (its report's hash, the tokenizer
//                  parity it ran, the logits-parity figure). With the
//                  header's source_model_hash, tokenizer_hash, corpus_hash,
//                  val_split_hash, split_seed and the masked-loss flag,
//                  that is every input of the fine-tune, hashed.
//   LedgerEntry[]  the reduction-order ledger: one line per pass, in the
//                  order they ran — whether its output is bit-identical to
//                  its input, how many instructions carry the change, and
//                  why.
//
// Pure and header-only; the loader, the compiler's own boundary check and
// the dump tool read it through the functions below.
// =============================================================================

namespace seeml::update {

inline constexpr uint32_t kPlanMetadataMagic = 0x4D454553;  // "SEEM"
inline constexpr uint32_t kPlanMetadataFormat = 1;
inline constexpr uint32_t kLedgerMaxEntries = 64;

// PlanMetadata::export_flags.
inline constexpr uint32_t kExportReportPresent = 1u << 0;
inline constexpr uint32_t kExportTokenizerParity = 1u << 1;  // parity was run
inline constexpr uint32_t kExportLogitsParity = 1u << 2;     // parity was run
inline constexpr uint32_t kExportFlagsMask =
    kExportReportPresent | kExportTokenizerParity | kExportLogitsParity;

// LedgerEntry::arithmetic.
inline constexpr uint32_t kLedgerExact = 0;      // bit-identical to its input
inline constexpr uint32_t kLedgerReordered = 1;  // reduction order may change,
                                                 // under a numerics certificate

struct PlanMetadata {
  uint32_t magic = kPlanMetadataMagic;
  uint32_t format = kPlanMetadataFormat;
  uint64_t bytes = 0;               // this struct plus the ledger entries
  uint64_t recipe_hash = 0;         // recipe::RecipeHash of the resolved recipe
  uint64_t export_report_hash = 0;  // ContentHash64 of the exporter's report
  uint32_t export_flags = 0;        // kExport*
  float logits_parity_max_delta = 0.0f;  // max |logit delta| vs the reference
  uint64_t tokenizer_parity_ids = 0;     // ids the parity run compared
  uint32_t ledger_count = 0;
  uint32_t reserved0 = 0;
  uint64_t reserved1 = 0;
};

struct LedgerEntry {
  char pass[32] = {};         // NUL-padded pass name
  uint32_t arithmetic = kLedgerExact;
  uint32_t instructions = 0;  // instructions carrying the change (0 if exact)
  char why[56] = {};          // NUL-padded reason
};

static_assert(sizeof(PlanMetadata) == 64, "PlanMetadata layout is part of the ABI.");
static_assert(sizeof(LedgerEntry) == 96, "LedgerEntry layout is part of the ABI.");

/// Where the section starts: the first 64-aligned byte after the emit table.
inline uint64_t PlanMetadataOffset(const PlanHeader& header) {
  const uint64_t end =
      header.emit_table_offset + header.emit_count * sizeof(EmitEntry);
  return (end + 63) & ~uint64_t{63};
}

inline LedgerEntry MakeLedgerEntry(std::string_view pass, uint32_t arithmetic,
                                   uint32_t instructions,
                                   std::string_view why) {
  LedgerEntry e;
  std::memcpy(e.pass, pass.data(), std::min(pass.size(), sizeof(e.pass) - 1));
  std::memcpy(e.why, why.data(), std::min(why.size(), sizeof(e.why) - 1));
  e.arithmetic = arithmetic;
  e.instructions = instructions;
  return e;
}

/// The section's bytes: the struct, then the ledger.
inline std::vector<uint8_t> EncodePlanMetadata(
    PlanMetadata meta, const std::vector<LedgerEntry>& ledger) {
  meta.ledger_count = static_cast<uint32_t>(ledger.size());
  meta.bytes = sizeof(PlanMetadata) + ledger.size() * sizeof(LedgerEntry);
  std::vector<uint8_t> out(meta.bytes);
  std::memcpy(out.data(), &meta, sizeof(meta));
  if (!ledger.empty())
    std::memcpy(out.data() + sizeof(meta), ledger.data(),
                ledger.size() * sizeof(LedgerEntry));
  return out;
}

/// A plan's decoded metadata; `present` is false for a plan without the
/// flag (every plan before v23).
struct DecodedMetadata {
  bool present = false;
  PlanMetadata meta;
  std::vector<LedgerEntry> ledger;
};

namespace metadata_detail {

inline bool Text(const char* s, size_t n, bool allow_empty) {
  size_t len = 0;
  while (len < n && s[len] != '\0') ++len;
  if (len == n) return false;                 // not NUL-terminated
  if (len == 0 && !allow_empty) return false;
  for (size_t i = 0; i < len; ++i)
    if (static_cast<unsigned char>(s[i]) < 0x20 ||
        static_cast<unsigned char>(s[i]) > 0x7E)
      return false;
  for (size_t i = len; i < n; ++i)
    if (s[i] != '\0') return false;           // padding is zero
  return true;
}

inline uint64_t RelaxedInstructions(const uint8_t* plan, uint64_t offset,
                                    uint64_t count) {
  uint64_t n = 0;
  for (uint64_t i = 0; i < count; ++i) {
    UpdateInstruction ins;
    std::memcpy(&ins, plan + offset + i * sizeof(ins), sizeof(ins));
    if (ins.flags & kFlagRelaxed) ++n;
  }
  return n;
}

}  // namespace metadata_detail

/// The metadata section held to the plan it sits in — the loader's rule.
/// `header` has already passed PlanHeaderViolation for a blob of
/// `plan_size` bytes. The first violated rule's reason, or nothing; `out`
/// (optional) receives the decoded section.
inline std::optional<std::string> PlanMetadataViolation(
    const PlanHeader& header, const uint8_t* plan, uint64_t plan_size,
    DecodedMetadata* out = nullptr) {
  if ((header.data_flags & kDataFlagMetadata) == 0) return std::nullopt;
  const uint64_t off = PlanMetadataOffset(header);
  if (off > plan_size || plan_size - off < sizeof(PlanMetadata))
    return std::string("plan metadata section lies outside the plan");
  PlanMetadata meta;
  std::memcpy(&meta, plan + off, sizeof(meta));
  if (meta.magic != kPlanMetadataMagic)
    return std::string("plan metadata section has a bad magic");
  if (meta.format != kPlanMetadataFormat)
    return std::string("unsupported plan metadata format");
  if (meta.ledger_count > kLedgerMaxEntries)
    return std::string("plan metadata ledger is too long");
  if (meta.bytes != sizeof(PlanMetadata) +
                        uint64_t{meta.ledger_count} * sizeof(LedgerEntry) ||
      meta.bytes > plan_size - off)
    return std::string("plan metadata section size disagrees with its ledger");
  if (meta.reserved0 != 0 || meta.reserved1 != 0)
    return std::string("reserved plan metadata words are not zero");
  if (meta.export_flags & ~kExportFlagsMask)
    return std::string("plan metadata export flags carry unknown bits");
  const bool report = (meta.export_flags & kExportReportPresent) != 0;
  if (!report && (meta.export_flags != 0 || meta.export_report_hash != 0))
    return std::string("plan metadata carries export fields without a report");
  if (report && meta.export_report_hash == 0)
    return std::string("plan metadata names an export report without its hash");
  if (!(meta.export_flags & kExportTokenizerParity) &&
      meta.tokenizer_parity_ids != 0)
    return std::string("plan metadata carries parity ids without the parity flag");
  if (!std::isfinite(meta.logits_parity_max_delta) ||
      meta.logits_parity_max_delta < 0.0f ||
      (!(meta.export_flags & kExportLogitsParity) &&
       meta.logits_parity_max_delta != 0.0f))
    return std::string("plan metadata logits-parity figure is not a finite "
                       "non-negative value under its flag");
  // The ledger against the programs: the instructions it says carry
  // reordered arithmetic are exactly the ones flagged relaxed.
  std::vector<LedgerEntry> ledger(meta.ledger_count);
  if (meta.ledger_count != 0)
    std::memcpy(ledger.data(), plan + off + sizeof(meta),
                ledger.size() * sizeof(LedgerEntry));
  uint64_t reordered = 0;
  for (const LedgerEntry& e : ledger) {
    if (!metadata_detail::Text(e.pass, sizeof(e.pass), false) ||
        !metadata_detail::Text(e.why, sizeof(e.why), true))
      return std::string("plan metadata ledger entry is not NUL-padded text");
    if (e.arithmetic != kLedgerExact && e.arithmetic != kLedgerReordered)
      return std::string("plan metadata ledger entry has an unknown arithmetic");
    if (e.arithmetic == kLedgerExact && e.instructions != 0)
      return std::string("an exact ledger entry counts changed instructions");
    if (e.arithmetic == kLedgerReordered && e.instructions == 0)
      return std::string("a reordered ledger entry changes no instruction");
    reordered += e.instructions;
  }
  const uint64_t relaxed =
      metadata_detail::RelaxedInstructions(plan, header.train_instr_offset,
                                           header.train_instr_count) +
      metadata_detail::RelaxedInstructions(plan, header.step_instr_offset,
                                           header.step_instr_count);
  if (reordered != relaxed)
    return "the ledger accounts for " + std::to_string(reordered) +
           " instruction(s) of reordered arithmetic, the programs carry " +
           std::to_string(relaxed);
  if (out) {
    out->present = true;
    out->meta = meta;
    out->ledger = std::move(ledger);
  }
  return std::nullopt;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_METADATA_H_
