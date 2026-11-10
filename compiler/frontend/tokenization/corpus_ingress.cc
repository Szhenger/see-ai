#include "compiler/frontend/tokenization/corpus_ingress.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <unordered_set>
#include <utility>

#include "compiler/diagnostics/tokenizing/error.h"
#include "runtime/pipeline/dataset.h"
#include "runtime/storage/durable_io.h"
#include "source/identity/hash.h"
#include "source/tokenizer/byte_level_bpe.h"
#include "source/tokenizer/text_ingest.h"

namespace seeml::update::tokenization {

namespace {

using diag::tokenizing::CorpusError;
using seeml::update_rt::Dataset;

/// The file's bytes as text: one sized read (the runtime's reader), the
/// message this stage's.
std::expected<std::string, std::string> ReadText(const std::string& path) {
  auto bytes = seeml::update_rt::ReadFileBytes(path);
  if (!bytes) return CorpusError("cannot read '" + path + "'");
  return std::string(bytes->begin(), bytes->end());
}

std::string Extension(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
    return "";
  std::string ext = path.substr(dot + 1);
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

const char* LabelKindName(uint32_t kind) {
  return kind == 1 ? "class-index" : kind == 2 ? "dense f32" : "no";
}

/// The facts every corpus carries, from the dataset the feeder would serve.
/// `distinct_ids` is taken as given when the caller already counted them
/// (the raw-text stages do); otherwise counted here over a vocabulary-sized
/// table (every id is below the table once HoldToModel has run).
void FillCommon(const Dataset& ds, const sema::Topology& topology,
                uint64_t rows_per_step, uint64_t steps,
                std::optional<uint64_t> distinct_ids, CorpusFacts& f) {
  f.records = ds.num_samples();
  if (ds.input_kind() == 1) {
    const uint64_t seq = ds.input_dim();
    f.record_width = seq + 1;
    f.tokens = f.records * (seq + 1);
    f.total_tokens = f.records * seq;
    f.target_tokens = ds.masked() ? ds.target_positions() : f.total_tokens;
    f.vocabulary = topology.vocabulary;
    if (distinct_ids) {
      f.distinct_ids = *distinct_ids;
    } else if (topology.vocabulary != 0) {
      std::vector<uint8_t> seen(static_cast<size_t>(topology.vocabulary), 0);
      ds.each_token([&](int32_t id) { seen[static_cast<size_t>(id)] = 1; });
      f.distinct_ids = static_cast<uint64_t>(
          std::count(seen.begin(), seen.end(), uint8_t{1}));
    } else {
      std::unordered_set<int32_t> distinct;
      ds.each_token([&](int32_t id) { distinct.insert(id); });
      f.distinct_ids = distinct.size();
    }
    // A step reads rows_per_step token rows: rows_per_step / seq sequences.
    const double sequences_per_step =
        static_cast<double>(rows_per_step) / static_cast<double>(seq);
    f.batches_per_epoch = sequences_per_step > 0
                              ? static_cast<double>(f.records) / sequences_per_step
                              : 0.0;
    f.epochs_covered = f.records
                           ? static_cast<double>(steps) * sequences_per_step /
                                 static_cast<double>(f.records)
                           : 0.0;
  } else {
    f.record_width = ds.input_dim();
    f.batches_per_epoch = rows_per_step
                              ? static_cast<double>(f.records) /
                                    static_cast<double>(rows_per_step)
                              : 0.0;
    f.epochs_covered = f.records
                           ? static_cast<double>(steps) *
                                 static_cast<double>(rows_per_step) /
                                 static_cast<double>(f.records)
                           : 0.0;
  }
  f.label_entropy_nats = ds.LabelEntropyNats();
  f.tokenizer_hash = ds.tokenizer_hash();
  f.masked = ds.masked();
}

/// What the feeder contract will hold the corpus to, held here first, in
/// this stage's words: the tokenizer identity (D9), the geometry, the
/// labels the loss stages, and every id below the bound the device applies
/// — the narrowest of the embedding table and the head. The device's own
/// rule (CorpusGeometryViolation) runs again over the finished header, so
/// nothing can slip between the two statements.
std::expected<void, std::string> HoldToModel(const Dataset& ds,
                                             const SmfModel& model,
                                             const sema::Topology& topology,
                                             PlanLabels labels) {
  const uint64_t model_tok = model.TokenizerHash();
  if (model_tok != 0 && ds.tokenizer_hash() != 0 &&
      ds.tokenizer_hash() != model_tok)
    return CorpusError(
        "the dataset was tokenized by another tokenizer than the model's "
        "section (tokenizer hash mismatch)");
  if (topology.token_input) {
    if (ds.input_kind() != 1)
      return CorpusError("the model takes token ids, but the corpus carries "
                         "feature rows");
    if (ds.input_dim() != model.seq_len)
      return CorpusError("the corpus holds " + std::to_string(ds.input_dim()) +
                         "-token records; the model's seq_len is " +
                         std::to_string(model.seq_len));
    // Token ids index the embedding rows and, as next-token labels, the
    // head's columns: bounded by the narrower, as the device bounds them.
    uint64_t bound = topology.vocabulary;
    if (labels.kind == 1 && labels.width != 0 &&
        (bound == 0 || labels.width < bound))
      bound = labels.width;
    if (auto ok = ds.ValidateClassLabels(bound); !ok)
      return CorpusError(ok.error());
    return {};
  }
  if (ds.input_kind() != 0)
    return CorpusError("the model takes feature rows, but the corpus carries "
                       "token records");
  if (ds.input_dim() != static_cast<uint64_t>(topology.input_width))
    return CorpusError("the corpus holds " + std::to_string(ds.input_dim()) +
                       "-feature rows; the model's input width is " +
                       std::to_string(topology.input_width));
  if (labels.kind != 0) {
    if (ds.label_kind() != labels.kind)
      return CorpusError(std::string("the corpus carries ") +
                         LabelKindName(ds.label_kind()) +
                         " labels, but the loss stages " +
                         LabelKindName(labels.kind) + " labels");
    if (labels.kind == 2 &&
        ds.label_bytes_per_sample() != labels.width * sizeof(float))
      return CorpusError(
          "the corpus's dense labels are " +
          std::to_string(ds.label_bytes_per_sample() / sizeof(float)) +
          " wide; the model's output is " + std::to_string(labels.width));
    if (labels.kind == 1)
      if (auto ok = ds.ValidateClassLabels(labels.width); !ok)
        return CorpusError(ok.error());
  }
  return {};
}

CorpusShape ShapeOf(const Dataset& ds, uint64_t corpus_hash) {
  return CorpusShape{ds.input_kind(),  ds.input_dim(),
                     ds.label_kind(),  ds.label_bytes_per_sample(),
                     ds.tokenizer_hash(), ds.masked(),
                     corpus_hash,      /*split_hash=*/0};
}

}  // namespace

PlanLabels LabelsOf(LossKind loss, uint64_t width) {
  switch (loss) {
    case LossKind::kSoftmaxXEnt:
    case LossKind::kXEntPlusKL:
      return {1, width};
    case LossKind::kMse:
      return {2, width};
    case LossKind::kKLDistill:
      break;
  }
  return {0, 0};
}

std::expected<CorpusIngress, std::string> IngestCorpus(
    const std::string& path, const SmfModel& model,
    const sema::Topology& topology, PlanLabels labels, uint64_t rows_per_step,
    uint64_t steps) {
  const std::string ext = Extension(path);
  CorpusIngress out;

  if (ext == "sds") {
    // The corpus as given: loaded once by the feeder's own reader (which
    // hashes the file as it will on the device), held to the model, pinned
    // by that hash. Nothing is read twice or written back.
    auto ds = Dataset::LoadFromFile(path);
    if (!ds) return CorpusError(ds.error());
    if (auto ok = HoldToModel(*ds, model, topology, labels); !ok)
      return std::unexpected(ok.error());
    out.from_sds = true;
    out.shape = ShapeOf(*ds, ds->content_hash());
    out.facts.kind = "sds";
    FillCommon(*ds, topology, rows_per_step, steps, std::nullopt, out.facts);
    out.facts.corpus_hash = out.shape.corpus_hash;
    return out;
  }

  if (ext != "txt" && ext != "jsonl")
    return CorpusError("unknown data format '" + path +
                       "': --data takes raw text (.txt — one document; .jsonl "
                       "— one {\"text\": ...} per line) or a pre-tokenized "
                       "corpus (.sds)");
  if (!topology.token_input)
    return CorpusError("a feature model takes no raw text: --data needs a "
                       "pre-tokenized .sds for it");
  if (model.tokenizer.empty())
    return CorpusError(
        "the model carries no tokenizer section: raw text needs one (export "
        "the model with its checkpoint's tokenizer.json, or pass a "
        "pre-tokenized .sds)");
  if (model.seq_len == 0) return CorpusError("the model declares no seq_len");

  auto tok = seeml::tokenizer::ByteLevelBpe::Load(model.tokenizer);
  if (!tok) return CorpusError(tok.error());  // "tokenizer section: ..."
  auto text = ReadText(path);
  if (!text) return std::unexpected(text.error());

  std::vector<seeml::tokenizer::Document> documents;
  seeml::tokenizer::IngestOptions options;
  options.seq_len = model.seq_len;
  if (ext == "txt") {
    options.one_document = true;
    documents.push_back({std::move(*text), {}, false});
  } else {
    auto docs = seeml::tokenizer::DocumentsFromJsonl(*text);
    if (!docs) return CorpusError(path + ": " + docs.error());
    documents = std::move(*docs);
    text->clear();
    text->shrink_to_fit();  // the documents hold the text now
  }
  const uint64_t tokenizer_hash = model.TokenizerHash();
  auto ingest = seeml::tokenizer::Ingest(*tok, tokenizer_hash, documents,
                                         options);
  if (!ingest) return CorpusError(ingest.error());

  // The dataset the feeder would serve, from the packed ids (moved, not
  // copied: the SDS bytes are the record of them): the id bound and the
  // label entropy come from it, as they do for a given SDS.
  const auto& m = ingest->manifest;
  auto ds = Dataset::FromTokens(std::move(ingest->ids), m.windows,
                                model.seq_len, std::move(ingest->mask),
                                tokenizer_hash);
  if (!ds) return CorpusError(ds.error());
  if (auto ok = HoldToModel(*ds, model, topology, labels); !ok)
    return std::unexpected(ok.error());

  out.from_sds = false;
  out.sds = std::move(ingest->sds);
  out.manifest_json = seeml::tokenizer::ManifestJson(m);
  out.shape = ShapeOf(*ds, ingest->corpus_hash);
  CorpusFacts& f = out.facts;
  f.kind = ext;
  f.documents_in = m.documents_in;
  f.documents_kept = m.documents_kept;
  f.dropped_empty = m.dropped_empty;
  f.dropped_duplicate = m.dropped_duplicate;
  f.dropped_degenerate = m.dropped_degenerate;
  f.invalid_utf8_bytes = m.sanitized.invalid_utf8_bytes;
  f.control_chars = m.sanitized.control_chars;
  f.bidi_controls = m.sanitized.bidi_controls;
  f.remainder_tokens = m.remainder_tokens;
  f.straddling_windows = m.straddling_windows;
  FillCommon(*ds, topology, rows_per_step, steps, m.distinct_ids, f);
  f.corpus_hash = out.shape.corpus_hash;
  return out;
}

}  // namespace seeml::update::tokenization
