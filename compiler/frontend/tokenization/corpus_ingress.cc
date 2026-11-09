#include "compiler/frontend/tokenization/corpus_ingress.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <unordered_set>

#include "compiler/diagnostics/tokenizing/error.h"
#include "runtime/pipeline/dataset.h"
#include "source/identity/hash.h"
#include "source/tokenizer/byte_level_bpe.h"
#include "source/tokenizer/text_ingest.h"

namespace seeml::update::tokenization {

namespace {

using diag::tokenizing::Error;
using seeml::update_rt::Dataset;

std::expected<std::string, std::string> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Error("cannot open '" + path + "'");
  std::string bytes((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
  if (f.bad()) return Error("cannot read '" + path + "'");
  return bytes;
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

/// The facts every corpus carries, from the dataset the feeder would serve.
void FillCommon(const Dataset& ds, const sema::Topology& topology,
                uint64_t rows_per_step, uint64_t steps, CorpusFacts& f) {
  f.records = ds.num_samples();
  if (ds.input_kind() == 1) {
    const uint64_t seq = ds.input_dim();
    f.record_width = seq + 1;
    f.tokens = f.records * (seq + 1);
    f.total_tokens = f.records * seq;
    f.target_tokens = ds.masked() ? ds.target_positions() : f.total_tokens;
    f.vocabulary = topology.vocabulary;
    {
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

/// What the feeder contract will hold the corpus to, held here first: the
/// tokenizer identity (D9), the geometry, and every id below the table.
std::expected<void, std::string> HoldToModel(const Dataset& ds,
                                             const SmfModel& model,
                                             const sema::Topology& topology) {
  const uint64_t model_tok = model.TokenizerHash();
  if (model_tok != 0 && ds.tokenizer_hash() != 0 &&
      ds.tokenizer_hash() != model_tok)
    return Error(
        "the dataset was tokenized by another tokenizer than the model's "
        "section (tokenizer hash mismatch)");
  if (topology.token_input) {
    if (ds.input_kind() != 1)
      return Error("the model takes token ids, but the corpus carries "
                   "feature rows");
    if (ds.input_dim() != model.seq_len)
      return Error("the corpus holds " + std::to_string(ds.input_dim()) +
                   "-token records; the model's seq_len is " +
                   std::to_string(model.seq_len));
    if (auto ok = ds.ValidateClassLabels(topology.vocabulary); !ok)
      return Error(ok.error());
  } else {
    if (ds.input_kind() != 0)
      return Error("the model takes feature rows, but the corpus carries "
                   "token records");
    if (ds.input_dim() != static_cast<uint64_t>(topology.input_width))
      return Error("the corpus holds " + std::to_string(ds.input_dim()) +
                   "-feature rows; the model's input width is " +
                   std::to_string(topology.input_width));
    if (ds.label_kind() == 1 && topology.output.dims.size() == 2)
      if (auto ok = ds.ValidateClassLabels(
              static_cast<uint64_t>(topology.output.dims[1]));
          !ok)
        return Error(ok.error());
  }
  return {};
}

}  // namespace

std::expected<CorpusIngress, std::string> IngestCorpus(
    const std::string& path, const SmfModel& model,
    const sema::Topology& topology, uint64_t rows_per_step, uint64_t steps) {
  const std::string ext = Extension(path);
  CorpusIngress out;

  if (ext == "sds") {
    auto ds = Dataset::LoadFromFile(path);
    if (!ds) return Error(ds.error());
    auto bytes = ReadFile(path);
    if (!bytes) return std::unexpected(bytes.error());
    if (auto ok = HoldToModel(*ds, model, topology); !ok)
      return std::unexpected(ok.error());
    out.from_sds = true;
    out.sds.assign(bytes->begin(), bytes->end());
    out.corpus_hash = ds->content_hash();
    out.tokenizer_hash = ds->tokenizer_hash();
    out.masked = ds->masked();
    out.facts.kind = "sds";
    FillCommon(*ds, topology, rows_per_step, steps, out.facts);
    out.facts.corpus_hash = out.corpus_hash;
    return out;
  }

  if (ext != "txt" && ext != "jsonl")
    return Error("unknown data format '" + path +
                 "': --data takes raw text (.txt — one document; .jsonl — one "
                 "{\"text\": ...} per line) or a pre-tokenized corpus (.sds)");
  if (!topology.token_input)
    return Error("a feature model takes no raw text: --data needs a "
                 "pre-tokenized .sds for it");
  if (model.tokenizer.empty())
    return Error(
        "the model carries no tokenizer section: raw text needs one (export "
        "the model with its checkpoint's tokenizer.json, or pass a "
        "pre-tokenized .sds)");
  if (model.seq_len == 0) return Error("the model declares no seq_len");

  auto tok = seeml::tokenizer::ByteLevelBpe::Load(model.tokenizer);
  if (!tok) return Error("tokenizer section: " + tok.error());
  auto text = ReadFile(path);
  if (!text) return std::unexpected(text.error());

  std::vector<seeml::tokenizer::Document> documents;
  seeml::tokenizer::IngestOptions options;
  options.seq_len = model.seq_len;
  if (ext == "txt") {
    options.one_document = true;
    documents.push_back({std::move(*text), {}, false});
  } else {
    auto docs = seeml::tokenizer::DocumentsFromJsonl(*text);
    if (!docs) return Error(path + ": " + docs.error());
    documents = std::move(*docs);
  }
  const uint64_t tokenizer_hash = model.TokenizerHash();
  auto ingest = seeml::tokenizer::Ingest(*tok, tokenizer_hash, documents,
                                         options);
  if (!ingest) return Error(ingest.error());

  // The dataset the feeder would serve, from the packed ids: the id bound
  // and the label entropy come from it, as they do for a given SDS.
  auto ds = Dataset::FromTokens(ingest->ids, ingest->manifest.windows,
                                model.seq_len, ingest->mask, tokenizer_hash);
  if (!ds) return Error(ds.error());
  if (auto ok = HoldToModel(*ds, model, topology); !ok)
    return std::unexpected(ok.error());

  out.from_sds = false;
  out.sds = std::move(ingest->sds);
  out.manifest_json = seeml::tokenizer::ManifestJson(ingest->manifest);
  out.corpus_hash = ingest->corpus_hash;
  out.tokenizer_hash = tokenizer_hash;
  out.masked = ingest->masked;
  CorpusFacts& f = out.facts;
  f.kind = ext;
  const auto& m = ingest->manifest;
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
  FillCommon(*ds, topology, rows_per_step, steps, f);
  f.corpus_hash = out.corpus_hash;
  return out;
}

}  // namespace seeml::update::tokenization
