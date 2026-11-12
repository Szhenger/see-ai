// =============================================================================
// SMF container tests: serialization round-trips, layout invariants, and
// rejection of malformed files (the binary-format hardening surface).
// =============================================================================

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

#include <cstdio>

#include "compiler/frontend/ingressor/export_report.h"
#include "compiler/frontend/ingressor/model_reader.h"
#include "compiler/frontend/egressor/model_writer.h"
#include "source/parallel/parallel_for.h"
#include "test/framework/seetest.h"
#include "test/support/builders.h"
#include "test/support/child_process.h"
#include "test/support/scoped_temp_dir.h"

namespace {

using namespace seeml::update;
using seeml::testing::AsBytes;
using seeml::testing::AsPayload;
using seeml::testing::MakeMlp;
using seeml::testing::ScopedTempDir;

std::vector<uint8_t> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

void WriteAll(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
}

TEST(Smf, FindTensor) {
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const SmfTensor* w1 = model.FindTensor("w1");
  ASSERT_NE(w1, nullptr);
  EXPECT_EQ(w1->name, "w1");
  EXPECT_EQ(model.FindTensor("nonexistent"), nullptr);
}

TEST(Smf, SaveLoadRoundTrip) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 1);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  EXPECT_EQ(loaded.input_name, model.input_name);
  EXPECT_EQ(loaded.output_name, model.output_name);
  ASSERT_EQ(loaded.tensors.size(), model.tensors.size());
  ASSERT_EQ(loaded.ops.size(), model.ops.size());

  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const SmfTensor& want = model.tensors[i];
    const SmfTensor& got = loaded.tensors[i];
    EXPECT_EQ(got.name, want.name);
    EXPECT_TRUE(got.dims == want.dims);
    EXPECT_EQ(got.is_const, want.is_const);
    // Constant tensors carry their data through the round trip bit-exactly.
    EXPECT_TRUE(got.data == want.data);
  }
  // The dynamic batch dim of the non-const input survives.
  const SmfTensor* x = loaded.FindTensor("x");
  ASSERT_NE(x, nullptr);
  ASSERT_EQ(x->dims.size(), 2u);
  EXPECT_EQ(x->dims[0], -1);

  for (size_t i = 0; i < model.ops.size(); ++i) {
    EXPECT_EQ(loaded.ops[i].kind, model.ops[i].kind);
    EXPECT_EQ(loaded.ops[i].name, model.ops[i].name);
    EXPECT_TRUE(loaded.ops[i].inputs == model.ops[i].inputs);
    EXPECT_EQ(loaded.ops[i].output, model.ops[i].output);
  }
}

/// Assembles a minimal pre-v3 SMF file (no seq_len, no attr0) with one
/// non-const input tensor and one single-input op of `kind` — the byte
/// layout writers no longer emit, for exercising the per-version gates.
std::vector<uint8_t> LegacySmfBytes(uint32_t version, uint8_t kind) {
  std::vector<uint8_t> b;
  auto u16 = [&](uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto str = [&](const char* s) {
    const size_t n = std::strlen(s);
    u16(static_cast<uint16_t>(n));
    b.insert(b.end(), s, s + n);
  };
  u32(kSmfMagic);
  u32(version);
  u32(1);  // num_tensors
  u32(1);  // num_ops
  str("x");
  str("y");
  str("x");           // tensor name
  b.push_back(1);     // rank
  b.push_back(0);     // flags: non-const
  u64(static_cast<uint64_t>(int64_t{-1}));  // dynamic leading dim
  u64(0);             // data_offset
  u64(0);             // byte_size
  b.push_back(kind);  // op kind
  str("a");
  b.push_back(1);  // num_inputs
  str("x");
  str("y");
  return b;
}

TEST(Smf, KindCeilingIsPerVersion) {
  // A file carrying a kind newer than its own version is corruption, not
  // forward compatibility: v1 ended at Relu (2), v2 at LayerNorm (6).
  ScopedTempDir dir;
  const std::string path = dir.File("legacy.smf");

  WriteAll(path, LegacySmfBytes(1, /*kind=*/3));  // Gelu inside a v1 file
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "unknown op kind");

  WriteAll(path, LegacySmfBytes(2, /*kind=*/7));  // Add inside a v2 file
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "unknown op kind");

  // The ceilings gate exactly at their own vocabulary: the newest v1 and
  // v2 kinds still parse (the reader checks kinds, not arities).
  WriteAll(path, LegacySmfBytes(1, /*kind=*/2));
  EXPECT_OK(LoadSmf(path));
  WriteAll(path, LegacySmfBytes(2, /*kind=*/6));
  EXPECT_OK(LoadSmf(path));
}

TEST(Smf, LoadRejectsDynamicDimBeyondLeading) {
  // The dynamic (-1) marker binds to the compiled batch and is only
  // meaningful as the LEADING dim; anywhere else it would flow into the
  // SIR as a dynamic width, bind zero-byte slots, and seal a plan the
  // runtime always rejects.
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 1);
  model.tensors[0].dims = {4, -1};  // x: dynamic width instead of batch
  const std::string path = dir.File("dynwidth.smf");
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");
}

TEST(Smf, SaveLoadRoundTripsV3TransformerFields) {
  // The v3 additions — model seq_len, per-op attr0, and the transformer op
  // kinds — must survive the byte round trip.
  ScopedTempDir dir;
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  const std::string path = dir.File("decoder.smf");
  ASSERT_OK(SaveSmf(path, model));

  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  EXPECT_EQ(loaded.seq_len, 4u);
  ASSERT_EQ(loaded.ops.size(), model.ops.size());
  for (size_t i = 0; i < model.ops.size(); ++i) {
    EXPECT_EQ(loaded.ops[i].kind, model.ops[i].kind);
    EXPECT_EQ(loaded.ops[i].attr0, model.ops[i].attr0);
  }
}

TEST(Smf, SaveLoadRoundTripsV5RopeBase) {
  // The v5 addition — per-op attr1, the rotary base as f32 bits on kRope —
  // must survive the byte round trip on every op, exactly as written.
  ScopedTempDir dir;
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  const uint32_t llama3_base = std::bit_cast<uint32_t>(500000.0f);
  for (SmfOp& op : model.ops)
    if (op.kind == SmfOpKind::kRope) op.attr1 = llama3_base;
  const std::string path = dir.File("decoder_v5.smf");
  ASSERT_OK(SaveSmf(path, model));

  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  ASSERT_EQ(loaded.ops.size(), model.ops.size());
  size_t rope_ops = 0;
  for (size_t i = 0; i < model.ops.size(); ++i) {
    EXPECT_EQ(loaded.ops[i].attr1, model.ops[i].attr1);
    if (loaded.ops[i].kind == SmfOpKind::kRope) {
      ++rope_ops;
      EXPECT_EQ(loaded.ops[i].attr1, llama3_base);
    } else {
      EXPECT_EQ(loaded.ops[i].attr1, 0u);
    }
  }
  EXPECT_EQ(rope_ops, 2u);
  // The writer stamps the lowest version that carries the model: v6 for a
  // model without a tokenizer section (v7 only with one, SMF v7).
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0, version = 0;
  f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  f.read(reinterpret_cast<char*>(&version), sizeof(version));
  EXPECT_EQ(magic, kSmfMagic);
  EXPECT_EQ(version, 6u);
  EXPECT_EQ(version, 6u);
}

TEST(Smf, SaveLoadRoundTripsV6NormEpsilon) {
  // SMF v6 (P7, #96): attr2 carries a LayerNorm / RmsNorm epsilon's f32
  // bits (0 = 1e-5). A Qwen2-class 1e-6 survives the round trip on every
  // norm op and nowhere else.
  ScopedTempDir dir;
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  const uint32_t qwen = std::bit_cast<uint32_t>(1e-6f);
  size_t norms = 0;
  for (SmfOp& op : model.ops)
    if (op.kind == SmfOpKind::kRmsNorm) {
      op.attr2 = qwen;
      ++norms;
    }
  ASSERT_GT(norms, 0u);
  const std::string path = dir.File("decoder_v6.smf");
  ASSERT_OK(SaveSmf(path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  for (size_t i = 0; i < model.ops.size(); ++i)
    EXPECT_EQ(loaded.ops[i].attr2,
              model.ops[i].kind == SmfOpKind::kRmsNorm ? qwen : 0u);
}

TEST(Smf, LoadRejectsAnAttr2ThatIsNotANormEpsilon) {
  // attr2 is a norm's epsilon and nothing else's; when set it must be a
  // finite positive float.
  ScopedTempDir dir;
  for (const auto& [kind, bits, why] :
       {std::tuple{SmfOpKind::kAttention, std::bit_cast<uint32_t>(1e-6f),
                   "nonzero attr2"},
        std::tuple{SmfOpKind::kRmsNorm, std::bit_cast<uint32_t>(-1e-6f),
                   "finite positive"},
        std::tuple{SmfOpKind::kRmsNorm, std::bit_cast<uint32_t>(INFINITY),
                   "finite positive"}}) {
    SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
    for (SmfOp& op : model.ops)
      if (op.kind == kind) op.attr2 = bits;
    const std::string path = dir.File("bad_attr2.smf");
    ASSERT_OK(SaveSmf(path, model));
    EXPECT_ERROR_CONTAINS(LoadSmf(path), why);
  }
}

TEST(Smf, LoadRejectsAttr1OnKindsThatDoNotDefineIt) {
  // attr1 is reserved (zero) on every kind but kRope; a file setting it
  // elsewhere is refused rather than carrying an undefined meaning.
  ScopedTempDir dir;
  SmfModel model = seeml::testing::MakeTinyDecoder(8, 2, 4, 12, 3, 9);
  for (SmfOp& op : model.ops)
    if (op.kind == SmfOpKind::kAttention) op.attr1 = 1;
  const std::string path = dir.File("attr1_on_attention.smf");
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "nonzero attr1");
}

TEST(Smf, SaveComputesAlignedNonOverlappingOffsets) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 2);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  // SaveSmf rewrites data_offset/byte_size on the model itself.
  uint64_t prev_end = 0;
  for (const SmfTensor& t : model.tensors) {
    if (!t.is_const) continue;
    EXPECT_EQ(t.data_offset % 64, 0u);
    EXPECT_GE(t.data_offset, prev_end);
    EXPECT_EQ(t.byte_size, t.data.size());
    prev_end = t.data_offset + t.byte_size;
  }
  EXPECT_EQ(ReadAll(path).size() % 64, 0u);
}

TEST(Smf, LoadRejectsMissingFile) {
  ScopedTempDir dir;
  EXPECT_ERROR_CONTAINS(LoadSmf(dir.File("nope.smf")), "cannot open");
}

TEST(Smf, LoadRejectsBadMagic) {
  ScopedTempDir dir;
  const std::string path = dir.File("bad.smf");
  WriteAll(path, {'N', 'O', 'P', 'E', 0, 0, 0, 0});
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "bad magic");
}

TEST(Smf, LoadRejectsBadVersion) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 3);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  std::vector<uint8_t> bytes = ReadAll(path);
  bytes[4] = 99;  // u32 version lives at offset 4
  WriteAll(path, bytes);
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "unsupported version");
}

TEST(Smf, LoadRejectsTruncatedMetadata) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 4);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  std::vector<uint8_t> bytes = ReadAll(path);
  bytes.resize(20);  // cuts into the tensor table
  WriteAll(path, bytes);
  EXPECT_ERROR(LoadSmf(path));
}

TEST(Smf, LoadRejectsDataRangeBeyondFile) {
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 5);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));

  // Cut into the data section: some tensor's [offset, offset+size) now
  // exceeds the file.
  std::vector<uint8_t> bytes = ReadAll(path);
  bytes.resize(model.FindTensor("w1")->data_offset + 4);
  WriteAll(path, bytes);
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "exceeds file size");
}

// A v6 file with one non-const input `x` {-1, 2}, const weights at the
// offsets given (each 2x2 f32, 16 bytes), and one MatMul(x, w0) -> y. With
// `write_payloads` the payload bytes are written at the offsets claimed
// (the file grows to hold them), so every range lies inside the file and
// only the placement rules can refuse it; without, the bytes already there
// stand in (for an offset inside the metadata). `metadata_out` receives
// the metadata's length.
std::vector<uint8_t> SmfWithPayloadsAt(const std::vector<uint64_t>& offsets,
                                       bool write_payloads = true,
                                       size_t* metadata_out = nullptr) {
  std::vector<uint8_t> b;
  auto u16 = [&](uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
  };
  auto str = [&](const std::string& s) {
    u16(static_cast<uint16_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  };
  u32(kSmfMagic);
  u32(6);
  u32(static_cast<uint32_t>(1 + offsets.size()));
  u32(1);
  str("x");
  str("y");
  u64(0);  // seq_len
  str("x");
  b.push_back(2);  // rank
  b.push_back(0);  // non-const
  u64(static_cast<uint64_t>(int64_t{-1}));
  u64(2);
  u64(0);
  u64(0);
  for (size_t i = 0; i < offsets.size(); ++i) {
    str("w" + std::to_string(i));
    b.push_back(2);  // rank
    b.push_back(1);  // const
    u64(2);
    u64(2);
    u64(offsets[i]);
    u64(16);
  }
  b.push_back(static_cast<uint8_t>(SmfOpKind::kMatMul));
  str("mm");
  b.push_back(2);
  str("x");
  str("w0");
  str("y");
  u32(0);
  u32(0);
  u32(0);
  if (metadata_out) *metadata_out = b.size();
  if (!write_payloads) return b;
  for (uint64_t off : offsets) {
    if (b.size() < off + 16) b.resize(off + 16, 0);
    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::memcpy(b.data() + off, ones, sizeof(ones));
  }
  return b;
}

TEST(Smf, LoadProvesPayloadPlacement) {
  // S1 #147 (F04): a payload must start on the 64-byte boundary, at or
  // after the metadata, and no two payloads may share bytes. Each rule
  // refused by name; the compliant layout loads.
  ScopedTempDir dir;
  const std::string path = dir.File("placed.smf");
  // The metadata's length depends on the tensor count: measure it for
  // each layout and place the first payload on the boundary past it.
  auto first_after = [](size_t tensors) {
    size_t metadata = 0;
    SmfWithPayloadsAt(std::vector<uint64_t>(tensors, 1u << 20), false,
                      &metadata);
    return static_cast<uint64_t>(((metadata + 63) / 64) * 64);
  };
  const uint64_t one = first_after(1), two = first_after(2);

  WriteAll(path, SmfWithPayloadsAt({two, two + 64}));
  EXPECT_OK(LoadSmf(path));

  WriteAll(path, SmfWithPayloadsAt({one + 4}));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "not 64-byte aligned");

  // Inside the metadata: offset 64 is aligned and inside the file (the
  // metadata of a one-tensor file is longer than 80 bytes), and the bytes
  // there are the tensor table itself.
  WriteAll(path, SmfWithPayloadsAt({64}, /*write_payloads=*/false));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "overlaps the file's metadata");

  // Two tensors claiming the same bytes.
  WriteAll(path, SmfWithPayloadsAt({two, two}));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "overlaps tensor 'w0'");
}

TEST(Smf, TheSweepRefusesNonFiniteWeightsAndReportsRanges) {
  // The ingressor's value rule (S1 #147; CR1 F12 / F34): every constant
  // element finite, named by role, tensor and element; the same pass
  // records each tensor's range, which the frontend screen prints.
  SmfModel model = MakeMlp(4, 8, 2, 5);
  ASSERT_OK_AND_ASSIGN(WeightSweep sweep, SweepWeights(model, "student"));
  size_t consts = 0;
  for (const SmfTensor& t : model.tensors) consts += t.is_const ? 1 : 0;
  EXPECT_EQ(sweep.ranges.size(), consts);
  for (const WeightRange& r : sweep.ranges) EXPECT_LE(r.min, r.max);
  SmfTensor* w = nullptr;
  for (SmfTensor& t : model.tensors)
    if (t.is_const && t.dims.size() == 2) w = &t;
  ASSERT_TRUE(w != nullptr);
  const float inf = INFINITY;
  std::memcpy(w->data.data() + 2 * sizeof(float), &inf, sizeof(inf));
  const auto refused = SweepWeights(model, "teacher");
  ASSERT_FALSE(refused.has_value());
  EXPECT_STR_CONTAINS(refused.error(), "teacher tensor '" + w->name + "'");
  EXPECT_STR_CONTAINS(refused.error(), "element 2");
}

TEST(Smf, TheExportersReportIsHeldToTheModelItDescribes) {
  // S4 #150: the report beside an imported checkpoint binds to the file's
  // hash and the section's; an absent report is a refusal for a model that
  // carries a section, and a report that is there is never ignored.
  seeml::testing::ScopedTempDir dir;
  SmfModel model = seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 16, 41);
  model.tokenizer = {'S', 'M', 'T', 'K', 1, 2, 3, 4};  // opaque to the reader
  const std::string path = dir.File("m.smf");
  ASSERT_OK(SaveSmf(path, model));
  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  ASSERT_TRUE(NeedsExportReport(loaded));
  auto hex = [](uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return std::string(b);
  };
  auto write = [&](const std::string& body) {
    std::ofstream f(path + kExportReportSuffix, std::ios::binary | std::ios::trunc);
    f << body;
  };
  auto report = [&](const std::string& smf, const std::string& tok,
                    const std::string& rest) {
    return "{\"schema\": 1, \"smf_hash\": \"" + smf +
           "\", \"tokenizer_hash\": \"" + tok + "\", " + rest + "}";
  };
  const std::string good_rest =
      "\"tokenizer_parity_checked\": true, \"tokenizer_parity_ids\": 1234, "
      "\"logits_parity_checked\": true, \"logits_parity_max_delta\": 6.4e-05";
  const std::string smf = hex(loaded.content_hash), tok = hex(loaded.TokenizerHash());

  EXPECT_ERROR_CONTAINS(ReadExportReport(path, loaded), "no exporter report");
  write(report(smf, tok, good_rest));
  ASSERT_OK_AND_ASSIGN(ExportReport r, ReadExportReport(path, loaded));
  EXPECT_TRUE(r.present);
  EXPECT_NE(r.report_hash, 0u);
  EXPECT_TRUE(r.tokenizer_parity);
  EXPECT_EQ(r.tokenizer_parity_ids, 1234u);
  EXPECT_TRUE(r.logits_parity);
  EXPECT_EQ(r.logits_parity_max_delta, 6.4e-05f);
  const std::string unproven =
      "\"tokenizer_parity_checked\": false, \"tokenizer_parity_ids\": 0, "
      "\"logits_parity_checked\": false, \"logits_parity_max_delta\": 0.0";
  write(report(smf, tok, unproven));
  ASSERT_OK_AND_ASSIGN(ExportReport u, ReadExportReport(path, loaded));
  EXPECT_TRUE(u.present);
  EXPECT_FALSE(u.tokenizer_parity);
  EXPECT_NE(u.report_hash, r.report_hash);

  struct Bad {
    std::string body;
    const char* says;
  };
  const Bad bad[] = {
      {report(hex(loaded.content_hash ^ 1), tok, good_rest),
       "describes another model file"},
      {report(smf, hex(loaded.TokenizerHash() ^ 1), good_rest),
       "tokenizer_hash is not the model's"},
      {report("12", tok, good_rest), "16 lowercase hex digits"},
      {"{\"schema\": 2, \"smf_hash\": \"" + smf + "\", \"tokenizer_hash\": \"" +
           tok + "\", " + good_rest + "}",
       "unsupported schema 2"},
      {report(smf, tok, good_rest + ", \"extra\": 1"), "unknown key 'extra'"},
      {report(smf, tok, "\"tokenizer_parity_checked\": true"), "is missing"},
      {report(smf, tok,
              "\"tokenizer_parity_checked\": 1, \"tokenizer_parity_ids\": 1, "
              "\"logits_parity_checked\": false, \"logits_parity_max_delta\": 0"),
       "takes true or false"},
      {report(smf, tok,
              "\"tokenizer_parity_checked\": false, \"tokenizer_parity_ids\": 5, "
              "\"logits_parity_checked\": false, \"logits_parity_max_delta\": 0"),
       "the parity was not checked"},
      {report(smf, tok,
              "\"tokenizer_parity_checked\": true, \"tokenizer_parity_ids\": 0, "
              "\"logits_parity_checked\": false, \"logits_parity_max_delta\": 0"),
       "checked over zero ids"},
      {report(smf, tok,
              "\"tokenizer_parity_checked\": false, \"tokenizer_parity_ids\": 0, "
              "\"logits_parity_checked\": true, \"logits_parity_max_delta\": -1"),
       "finite non-negative number"},
      {"[]", "not a JSON object"},
      {report(smf, tok, good_rest) + " x", "content after the closing"},
  };
  for (const Bad& b : bad) {
    write(b.body);
    EXPECT_ERROR_CONTAINS(ReadExportReport(path, loaded), b.says);
  }
  // A model without a section needs none — and one that is there is still
  // held to the file.
  SmfModel plain = seeml::testing::MakeTinyTokenDecoder(16, 8, 2, 4, 16, 41);
  const std::string plain_path = dir.File("plain.smf");
  ASSERT_OK(SaveSmf(plain_path, plain));
  ASSERT_OK_AND_ASSIGN(SmfModel plain_loaded, LoadSmf(plain_path));
  EXPECT_FALSE(NeedsExportReport(plain_loaded));
  ASSERT_OK_AND_ASSIGN(ExportReport none, ReadExportReport(plain_path, plain_loaded));
  EXPECT_FALSE(none.present);
  {
    std::ofstream f(plain_path + kExportReportSuffix);
    f << report(smf, hex(0), unproven);  // another file's report
  }
  EXPECT_ERROR_CONTAINS(ReadExportReport(plain_path, plain_loaded),
                        "describes another model file");
}

TEST(Smf, TheTokenizerSectionRoundTripsAsV7) {
  // SMF v7 (S2 #148): an opaque tokenizer section placed like a payload;
  // its hash is the tokenizer identity. A model without one is still the
  // v6 file it always was.
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 5);
  const std::string plain = dir.File("plain.smf");
  ASSERT_OK(SaveSmf(plain, model));
  std::vector<uint8_t> plain_bytes = ReadAll(plain);
  uint32_t version = 0;
  std::memcpy(&version, plain_bytes.data() + 4, 4);
  EXPECT_EQ(version, 6u);
  EXPECT_EQ(model.TokenizerHash(), 0u);

  for (int i = 0; i < 100; ++i)
    model.tokenizer.push_back(static_cast<uint8_t>(i * 7 + 1));
  const uint64_t hash = model.TokenizerHash();
  EXPECT_TRUE(hash != 0);
  const std::string path = dir.File("tok.smf");
  ASSERT_OK(SaveSmf(path, model));
  std::vector<uint8_t> bytes = ReadAll(path);
  std::memcpy(&version, bytes.data() + 4, 4);
  EXPECT_EQ(version, 7u);
  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  EXPECT_TRUE(loaded.tokenizer == model.tokenizer);
  EXPECT_EQ(loaded.TokenizerHash(), hash);
  EXPECT_EQ(loaded.tensors.size(), model.tensors.size());
  // The section sits 64-aligned past the metadata, before the payloads,
  // and the content hash covers it.
  const size_t after_names = 16 + 2 + model.input_name.size() + 2 +
                             model.output_name.size() + 8;
  uint64_t tok_offset = 0, tok_size = 0;
  std::memcpy(&tok_offset, bytes.data() + after_names, 8);
  std::memcpy(&tok_size, bytes.data() + after_names + 8, 8);
  EXPECT_EQ(tok_size, 100u);
  EXPECT_EQ(tok_offset % 64, 0u);
  for (const SmfTensor& t : loaded.tensors)
    if (t.is_const) EXPECT_GE(t.data_offset, tok_offset + tok_size);
  EXPECT_TRUE(loaded.content_hash != 0);
  // Overlapping the section with a payload is refused like any payload.
  bytes[after_names] = static_cast<uint8_t>(bytes[after_names] + 64);
  WriteAll(path, bytes);
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "overlaps");
}

TEST(Smf, LoadRejectsByteSizeDimsDisagreement) {
  ScopedTempDir dir;
  // dims say 2x2 = 16 bytes, but the blob holds 3 floats. SaveSmf writes
  // byte_size = data.size() = 12, which the loader must reject.
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "x", .dims = {-1, 2}, .is_const = false});
  model.tensors.push_back({.name = "w",
                           .dims = {2, 2},
                           .is_const = true,
                           .data = AsPayload({1.0f, 2.0f, 3.0f})});
  model.ops.push_back({SmfOpKind::kMatMul, "mm", {"x", "w"}, "y"});

  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "byte size disagrees");
}

TEST(Smf, LoadRejectsInvalidDims) {
  ScopedTempDir dir;
  const std::string path = dir.File("model.smf");

  // A zero dim is never valid.
  SmfModel zero_dim;
  zero_dim.input_name = "x";
  zero_dim.output_name = "y";
  zero_dim.tensors.push_back(
      {.name = "w", .dims = {0}, .is_const = true, .data = AsPayload({1.0f})});
  ASSERT_OK(SaveSmf(path, zero_dim));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");

  // A dynamic (-1) dim is only allowed on non-constant tensors.
  SmfModel dynamic_const;
  dynamic_const.input_name = "x";
  dynamic_const.output_name = "y";
  dynamic_const.tensors.push_back(
      {.name = "w", .dims = {-1}, .is_const = true, .data = AsPayload({1.0f})});
  ASSERT_OK(SaveSmf(path, dynamic_const));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");

  // Rank 0 is rejected.
  SmfModel rank0;
  rank0.input_name = "x";
  rank0.output_name = "y";
  rank0.tensors.push_back(
      {.name = "w", .dims = {}, .is_const = true, .data = AsPayload({1.0f})});
  ASSERT_OK(SaveSmf(path, rank0));
  EXPECT_ERROR_CONTAINS(LoadSmf(path), "invalid dims");
}

TEST(Smf, SaveRejectsConstantTensorWithoutData) {
  ScopedTempDir dir;
  SmfModel model;
  model.input_name = "x";
  model.output_name = "y";
  model.tensors.push_back({.name = "w", .dims = {2, 2}, .is_const = true});
  EXPECT_ERROR_CONTAINS(SaveSmf(dir.File("model.smf"), model),
                        "has no data to serialize");
}

TEST(Smf, SaveIsDeterministic) {
  ScopedTempDir dir;
  SmfModel a = MakeMlp(4, 8, 2, 6);
  SmfModel b = MakeMlp(4, 8, 2, 6);
  const std::string pa = dir.File("a.smf");
  const std::string pb = dir.File("b.smf");
  ASSERT_OK(SaveSmf(pa, a));
  ASSERT_OK(SaveSmf(pb, b));
  EXPECT_TRUE(ReadAll(pa) == ReadAll(pb));
}

TEST(Smf, ConcurrentLoadsMatchSequentialLoads) {
  ScopedTempDir dir;
  SmfModel a = MakeMlp(4, 8, 2, 1);
  SmfModel b = MakeMlp(6, 12, 3, 2);
  const std::string pa = dir.File("a.smf");
  const std::string pb = dir.File("b.smf");
  ASSERT_OK(SaveSmf(pa, a));
  ASSERT_OK(SaveSmf(pb, b));

  ASSERT_OK_AND_ASSIGN(SmfModel sa, LoadSmf(pa));
  ASSERT_OK_AND_ASSIGN(SmfModel sb, LoadSmf(pb));

  const std::string paths[] = {pa, pb};
  ASSERT_OK_AND_ASSIGN(auto models, LoadSmfMany(paths));
  ASSERT_EQ(models.size(), 2u);
  EXPECT_EQ(models[0].content_hash, sa.content_hash);
  EXPECT_EQ(models[1].content_hash, sb.content_hash);
  EXPECT_EQ(models[0].tensors.size(), sa.tensors.size());
  EXPECT_EQ(models[1].ops.size(), sb.ops.size());
  ASSERT_NE(models[1].FindTensor("w1"), nullptr);
  EXPECT_TRUE(models[1].FindTensor("w1")->data == sb.FindTensor("w1")->data);
}

TEST(Smf, ConcurrentLoadReportsFirstFailureInInputOrder) {
  ScopedTempDir dir;
  SmfModel a = MakeMlp(4, 8, 2, 1);
  const std::string pa = dir.File("a.smf");
  ASSERT_OK(SaveSmf(pa, a));

  const std::string paths[] = {dir.File("missing.smf"), pa};
  EXPECT_ERROR_CONTAINS(LoadSmfMany(paths), "missing.smf");
}

TEST(ModelIo, ByteBalancedParallelLoadIsExactAtAnyWidth) {
  // One tensor dominating the payload, a tail of small ones, and sizes
  // that are multiples of nothing: the load copies by byte ranges (8 MiB
  // chunks) that start and stop inside tensors (E2, #81), and must
  // reproduce every byte and the content hash at one thread and at eight.
  SmfModel model;
  model.input_name = "x";
  model.tensors.push_back({.name = "x", .dims = {-1, 3}});
  std::string prev = "x";
  const std::vector<std::pair<int64_t, int64_t>> shapes = {
      {3, 1500007}, {1500007, 5}, {5, 7}, {7, 900001}, {900001, 2}};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto [rows, cols] = shapes[i];
    const std::string w = "w" + std::to_string(i);
    model.tensors.push_back(
        {.name = w, .dims = {rows, cols}, .is_const = true,
         .data = AsPayload(seeml::testing::RandnVector(
             static_cast<size_t>(rows * cols), 100 + i))});
    const std::string out = "h" + std::to_string(i);
    model.ops.push_back({.kind = SmfOpKind::kMatMul, .name = "mm" + w,
                         .inputs = {prev, w}, .output = out});
    prev = out;
  }
  model.output_name = prev;
  ScopedTempDir dir;
  const std::string path = dir.File("wide.smf");
  ASSERT_OK(SaveSmf(path, model));
  for (const size_t threads : {size_t{1}, size_t{8}}) {
    seeml::update::SetParallelThreadCount(threads);
    ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
    EXPECT_EQ(loaded.content_hash, model.content_hash);
    ASSERT_EQ(loaded.tensors.size(), model.tensors.size());
    for (size_t i = 0; i < model.tensors.size(); ++i)
      EXPECT_TRUE(loaded.tensors[i].data == model.tensors[i].data);
  }
  seeml::update::SetParallelThreadCount(0);
}

#ifndef _WIN32
TEST(Smf, AWriteThatFailsMidwayLeavesTheModelAndTheDiskUntouched) {
  // CR4 #145 F05, the failure the open-failure test below cannot reach: the
  // disk fills partway through the file. The child's files may not grow
  // past half the model's, and the kernel fails the write that crosses
  // (EFBIG) — no fault seam in SaveSmf. The destination keeps its old
  // bytes, the stage is gone, and the model's offsets, sizes and hash have
  // not moved.
  using seeml::testing::PastTheLimit;
  ScopedTempDir dir;
  const std::string path = dir.File("model.smf");
  SmfModel old_model = MakeMlp(4, 8, 2, 2);
  ASSERT_OK(SaveSmf(path, old_model));
  const std::vector<uint8_t> old_bytes = ReadAll(path);
  SmfModel probe = MakeMlp(64, 256, 16, 3);
  ASSERT_OK(SaveSmf(dir.File("probe.smf"), probe));
  const uint64_t full = ReadAll(dir.File("probe.smf")).size();
  SmfModel model = MakeMlp(64, 256, 16, 3);
  for (SmfTensor& t : model.tensors)
    if (t.is_const) {
      t.data_offset = 7777;
      t.byte_size = 1;
    }
  model.content_hash = 42;
  const int status = seeml::testing::RunUnderFileSizeLimit(
      full / 2, PastTheLimit::kFail, [&] {
        auto r = SaveSmf(path, model);
        if (r) return 1;
        if (r.error().find("short write") == std::string::npos) return 2;
        if (ReadAll(path) != old_bytes) return 3;
        if (std::ifstream(path + ".tmp").good()) return 4;
        for (const SmfTensor& t : model.tensors)
          if (t.is_const && (t.data_offset != 7777 || t.byte_size != 1))
            return 5;
        if (model.content_hash != 42) return 6;
        return 0;
      });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(Smf, AWriterKilledMidSaveLeavesTheOldFile) {
  // CR4 #145, kill mid-save: killed partway through the file (SIGXFSZ's
  // default action at the child's file-size limit), the writer leaves the
  // destination as it was — the rename never ran — and a stage short of
  // the whole file, the proof it died mid-write. The old model still loads,
  // and the next save replaces the destination and clears the stage.
  using seeml::testing::PastTheLimit;
  ScopedTempDir dir;
  const std::string path = dir.File("model.smf");
  SmfModel old_model = MakeMlp(4, 8, 2, 2);
  ASSERT_OK(SaveSmf(path, old_model));
  const std::vector<uint8_t> old_bytes = ReadAll(path);
  SmfModel model = MakeMlp(64, 256, 16, 3);
  SmfModel probe = model;
  ASSERT_OK(SaveSmf(dir.File("probe.smf"), probe));
  const uint64_t full = ReadAll(dir.File("probe.smf")).size();
  const int status = seeml::testing::RunUnderFileSizeLimit(
      full / 2, PastTheLimit::kKill, [&] {
        (void)SaveSmf(path, model);
        return 0;  // reached only if the writer survived
      });
  ASSERT_TRUE(WIFSIGNALED(status));
  EXPECT_EQ(WTERMSIG(status), SIGXFSZ);
  EXPECT_TRUE(ReadAll(path) == old_bytes);
  const std::vector<uint8_t> stage = ReadAll(path + ".tmp");
  ASSERT_TRUE(std::ifstream(path + ".tmp").good());
  EXPECT_LE(uint64_t{stage.size()}, full / 2);
  ASSERT_OK_AND_ASSIGN(SmfModel loaded, LoadSmf(path));
  EXPECT_EQ(loaded.tensors.size(), old_model.tensors.size());
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_FALSE(std::ifstream(path + ".tmp").good());
  EXPECT_EQ(uint64_t{ReadAll(path).size()}, full);
}
#endif  // _WIN32

TEST(Smf, AFailedSaveLeavesTheModelAndTheDiskUntouched) {
  // CR4 #145 F05: SaveSmf computed the new offsets into the model before a
  // write that could fail, so a failed save left new offsets paired with
  // the old hash. The offsets, sizes and hash change only once the file is
  // in place, and the stage is gone either way.
  ScopedTempDir dir;
  SmfModel model = MakeMlp(4, 8, 2, 2);
  for (SmfTensor& t : model.tensors)
    if (t.is_const) {
      t.data_offset = 7777;
      t.byte_size = 1;
    }
  model.content_hash = 42;
  EXPECT_ERROR_CONTAINS(SaveSmf(dir.File("no-such-dir/model.smf"), model),
                        "cannot write");
  for (const SmfTensor& t : model.tensors)
    if (t.is_const) {
      EXPECT_EQ(t.data_offset, 7777u);
      EXPECT_EQ(t.byte_size, 1u);
    }
  EXPECT_EQ(model.content_hash, 42u);
  const std::string path = dir.File("model.smf");
  ASSERT_OK(SaveSmf(path, model));
  EXPECT_FALSE(std::ifstream(path + ".tmp").good());
  EXPECT_NE(model.content_hash, 42u);
  for (const SmfTensor& t : model.tensors)
    if (t.is_const) EXPECT_EQ(t.byte_size, t.data.size());
}

}  // namespace
