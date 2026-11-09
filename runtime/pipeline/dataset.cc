#include "runtime/pipeline/dataset.h"

#include "runtime/diagnostics/feeding/error.h"
#include "source/identity/hash.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <optional>

namespace seeml::update_rt {

namespace {

// Overflow-checked u64 multiply for validating file-supplied sizes.
bool MulU64(uint64_t a, uint64_t b, uint64_t* out) {
  if (b != 0 && a > UINT64_MAX / b) return false;
  *out = a * b;
  return true;
}

}  // namespace

uint64_t Dataset::target_positions() const {
  if (input_kind_ != 1) return 0;
  if (!masked_) return num_samples_ * input_dim_;
  uint64_t n = 0;
  for (const uint8_t m : mask_) n += m;
  return n;
}

uint64_t Dataset::label_bytes_per_sample() const {
  // Token records derive one class label per POSITION, so a record's label
  // payload is input_dim (= seq) i32s — the shifted token view.
  if (input_kind_ == 1) return input_dim_ * sizeof(int32_t);
  switch (label_kind_) {
    case 1: return sizeof(int32_t);
    case 2: return label_dim_ * sizeof(float);
    default: return 0;
  }
}

std::expected<Dataset, std::string> Dataset::FromTokens(
    std::vector<int32_t> tokens, uint64_t num_records, uint64_t seq,
    std::vector<uint8_t> mask, uint64_t tokenizer_hash) {
  Dataset d;
  d.input_kind_ = 1;
  d.label_kind_ = 1;  // derived next-token class ids
  d.num_samples_ = num_records;
  d.input_dim_ = seq;
  d.tokens_ = std::move(tokens);
  d.tokenizer_hash_ = tokenizer_hash;
  if (num_records == 0) return diag::feeding::Error("zero samples");
  if (seq == 0) return diag::feeding::Error("zero input dim");
  uint64_t want = 0;
  if (seq >= UINT64_MAX || !MulU64(num_records, seq + 1, &want) ||
      d.tokens_.size() != want)
    return diag::feeding::Error("token buffer size mismatch");
  for (uint64_t i = 0; i < want; ++i)
    if (d.tokens_[i] < 0)
      return diag::feeding::Error("negative token id at position " +
                                  std::to_string(i));
  if (!mask.empty()) {
    uint64_t positions = 0;
    if (!MulU64(num_records, seq, &positions) || mask.size() != positions)
      return diag::feeding::Error("mask buffer size mismatch");
    if (auto ok = CheckMask(mask, num_records, seq); !ok)
      return std::unexpected(ok.error());
    d.mask_ = std::move(mask);
    d.masked_ = true;
  }
  return d;
}

std::expected<Dataset, std::string> Dataset::FromMemory(
    std::vector<float> inputs, std::vector<uint8_t> labels,
    uint64_t num_samples, uint64_t input_dim, uint32_t label_kind,
    uint64_t label_dim) {
  Dataset d;
  d.num_samples_ = num_samples;
  d.input_dim_ = input_dim;
  d.label_kind_ = label_kind;
  d.label_dim_ = label_dim;
  d.inputs_ = std::move(inputs);
  d.labels_ = std::move(labels);
  if (num_samples == 0) return diag::feeding::Error("zero samples");
  // Same admission checks as LoadFromFile: label_bytes_per_sample()
  // multiplies label_dim unchecked, so an unvalidated dim would wrap the
  // expected-size math and admit a mis-sized label buffer.
  if (input_dim == 0) return diag::feeding::Error("zero input dim");
  if (label_kind > 2) return diag::feeding::Error("unknown label kind");
  if (label_kind == 2 &&
      (label_dim == 0 || label_dim > UINT64_MAX / sizeof(float)))
    return diag::feeding::Error("label dim out of range");
  uint64_t want_inputs = 0, want_labels = 0;
  if (!MulU64(num_samples, input_dim, &want_inputs) ||
      d.inputs_.size() != want_inputs)
    return diag::feeding::Error("input buffer size mismatch");
  if (!MulU64(num_samples, d.label_bytes_per_sample(), &want_labels) ||
      d.labels_.size() != want_labels)
    return diag::feeding::Error("label buffer size mismatch");
  return d;
}

std::expected<void, std::string> Dataset::CheckMask(
    const std::vector<uint8_t>& mask, uint64_t num_records, uint64_t seq) {
  // Every mask byte is 0 or 1, and every record keeps at least one target:
  // a record with none would divide the masked loss by zero, so the
  // ingest drops such windows (S3 #149) and the loader refuses them.
  for (uint64_t r = 0; r < num_records; ++r) {
    bool any = false;
    for (uint64_t j = 0; j < seq; ++j) {
      const uint8_t m = mask[r * seq + j];
      if (m > 1)
        return diag::feeding::Error("a loss mask byte is not 0 or 1 (record " +
                                    std::to_string(r) + ")");
      any = any || m == 1;
    }
    if (!any)
      return diag::feeding::Error("record " + std::to_string(r) +
                                  " carries no target position");
  }
  return {};
}

namespace {

/// ContentHash64 of the whole file: the corpus identity a v22 plan pins
/// (kDataFlagCorpusPinned). A second read of the file, once, at load.
std::expected<uint64_t, std::string> HashFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return diag::feeding::Error("cannot open '" + path + "'");
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0) return diag::feeding::Error("cannot stat '" + path + "'");
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  if (!bytes.empty() &&
      !f.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size())))
    return diag::feeding::Error("cannot read '" + path + "'");
  return seeml::update::ContentHash64(bytes.data(), bytes.size());
}

}  // namespace

std::expected<Dataset, std::string> Dataset::LoadFromFile(
    const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return diag::feeding::Error("cannot open '" + path + "'");

  // The file size bounds every allocation below: a corrupt header cannot ask
  // for more sample data than the file actually holds.
  f.seekg(0, std::ios::end);
  const std::streamoff end_pos = f.tellg();
  if (end_pos < 0) return diag::feeding::Error("cannot stat '" + path + "'");
  const uint64_t file_size = static_cast<uint64_t>(end_pos);
  f.seekg(0);

  auto read = [&](void* dst, size_t n) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
  };

  uint32_t magic = 0, version = 0, label_kind = 0, input_kind = 0;
  uint64_t num_samples = 0, input_dim = 0, label_dim = 0;
  if (!read(&magic, 4) || magic != kSdsMagic)
    return diag::feeding::Error("bad magic in '" + path + "'");
  if (!read(&version, 4) || version < kSdsMinVersion || version > kSdsVersion)
    return diag::feeding::Error("unsupported version");
  if (!read(&num_samples, 8) || !read(&input_dim, 8) || !read(&label_kind, 4) ||
      !read(&input_kind, 4) || !read(&label_dim, 8))
    return diag::feeding::Error("truncated header");
  // v3 (S2 #148 / S3 #149): the tokenizer identity and the flags; the two
  // reserved words must be zero so a later meaning can never be misread.
  const uint64_t header_bytes = SdsHeaderBytes(version);
  uint64_t tokenizer_hash = 0, reserved1 = 0;
  uint32_t flags = 0, reserved0 = 0;
  if (version >= 3) {
    if (!read(&tokenizer_hash, 8) || !read(&flags, 4) || !read(&reserved0, 4) ||
        !read(&reserved1, 8))
      return diag::feeding::Error("truncated header");
    if (reserved0 != 0 || reserved1 != 0)
      return diag::feeding::Error("reserved SDS header words are not zero");
    if (flags & ~kSdsFlagsMask) return diag::feeding::Error("unknown SDS flags");
  }

  Dataset d;
  d.tokenizer_hash_ = tokenizer_hash;
  d.masked_ = (flags & kSdsFlagMasked) != 0;
  d.num_samples_ = num_samples;
  d.input_dim_ = input_dim;
  d.label_kind_ = label_kind;
  d.label_dim_ = label_dim;
  d.input_kind_ = input_kind;
  if (num_samples == 0 || input_dim == 0)
    return diag::feeding::Error("empty dataset");
  if (label_kind > kSdsLabelKindMax)
    return diag::feeding::Error("unknown label kind");
  if (label_kind == 2 &&
      (label_dim == 0 || label_dim > UINT64_MAX / sizeof(float)))
    return diag::feeding::Error("label dim out of range");
  // The input_kind word was header padding in v1 (always zero); a nonzero
  // value there is corruption, and token corpora require the derived
  // next-token label discipline.
  if (input_kind > 1 || (version < 2 && input_kind != 0))
    return diag::feeding::Error("unknown input kind");
  if (d.masked_ && input_kind != 1)
    return diag::feeding::Error("a loss mask needs token records");
  // The sample section fills the file exactly (CR2's F19): trailing bytes
  // are not a corpus the writer meant, so they are refused, not ignored.
  auto exact_fill = [&](uint64_t payload) -> std::optional<std::string> {
    if (payload > file_size - header_bytes)
      return "sample section exceeds file size";
    if (payload != file_size - header_bytes)
      return "dataset payload does not fill the file: " +
             std::to_string(file_size - header_bytes - payload) +
             " trailing byte(s)";
    return std::nullopt;
  };
  if (input_kind == 1) {
    if (label_kind != 1 || label_dim != 0)
      return diag::feeding::Error(
          "token corpora derive next-token class labels (label_kind 1, "
          "label_dim 0)");
    // Records are (seq + 1) contiguous i32 ids with no stored label, and
    // under the mask flag a u8 mask[seq] padded to 4 bytes after them.
    const uint64_t mask_bytes = d.masked_ ? SdsMaskBytes(input_dim) : 0;
    uint64_t record_ids = 0, payload = 0;
    if (input_dim >= UINT64_MAX ||
        !MulU64(input_dim + 1, sizeof(int32_t), &record_ids) ||
        record_ids > UINT64_MAX - mask_bytes ||
        !MulU64(num_samples, record_ids + mask_bytes, &payload) ||
        payload > file_size - header_bytes)
      return diag::feeding::Error("sample section exceeds file size");
    if (auto why = exact_fill(payload)) return diag::feeding::Error(*why);
    if (payload > SIZE_MAX)
      return diag::feeding::Error("dataset too large for this host");
    const uint64_t record_bytes = record_ids + mask_bytes;
    d.tokens_.resize(num_samples * (input_dim + 1));
    if (!d.masked_) {
      if (!read(d.tokens_.data(), static_cast<size_t>(payload)))
        return diag::feeding::Error("truncated inputs");
    } else {
      d.mask_.resize(num_samples * input_dim);
      std::vector<uint8_t> rec(static_cast<size_t>(record_bytes));
      for (uint64_t r = 0; r < num_samples; ++r) {
        if (!read(rec.data(), rec.size()))
          return diag::feeding::Error("truncated inputs");
        std::memcpy(d.tokens_.data() + r * (input_dim + 1), rec.data(),
                    static_cast<size_t>(record_ids));
        std::memcpy(d.mask_.data() + r * input_dim, rec.data() + record_ids,
                    static_cast<size_t>(input_dim));
      }
      if (auto ok = CheckMask(d.mask_, num_samples, input_dim); !ok)
        return std::unexpected(ok.error());
    }
    for (const int32_t t : d.tokens_)
      if (t < 0) return diag::feeding::Error("negative token id");
    if (auto h = HashFile(path); !h) return std::unexpected(h.error());
    else d.content_hash_ = *h;
    return d;
  }

  // Overflow-safe sizing, cross-checked against the actual file size before
  // any allocation.
  const uint64_t lbytes = d.label_bytes_per_sample();
  uint64_t input_bytes = 0, payload = 0;
  if (!MulU64(input_dim, sizeof(float), &input_bytes) ||
      input_bytes > UINT64_MAX - lbytes ||
      !MulU64(num_samples, input_bytes + lbytes, &payload) ||
      payload > file_size - header_bytes)
    return diag::feeding::Error("sample section exceeds file size");
  if (auto why = exact_fill(payload)) return diag::feeding::Error(*why);
  // On 32-bit hosts size_t is narrower than the u64 the checks above ran
  // in: a corpus that cannot fit the address space must be refused here,
  // not silently truncated by the resize below.
  if (payload > SIZE_MAX)
    return diag::feeding::Error("dataset too large for this host");

  d.inputs_.resize(num_samples * input_dim);
  d.labels_.resize(num_samples * lbytes);

  if (lbytes == 0) {
    // No labels: the sample section is one contiguous f32 block — read it
    // straight into place, a single bulk transfer instead of one syscall
    // per sample.
    if (!read(d.inputs_.data(), num_samples * input_bytes))
      return diag::feeding::Error("truncated inputs");
    return d;
  }

  // Interleaved records: stream fixed chunks of whole records through a
  // bounded buffer and deinterleave in memory. Bulk reads amortize the
  // stream overhead; the buffer bound keeps peak memory flat no matter how
  // large the corpus is.
  const uint64_t record = input_bytes + lbytes;
  constexpr uint64_t kChunkBudget = 256 * 1024;
  const uint64_t per_chunk =
      record >= kChunkBudget ? 1 : kChunkBudget / record;
  std::vector<uint8_t> chunk(per_chunk * record);
  for (uint64_t i = 0; i < num_samples;) {
    const uint64_t n = std::min(per_chunk, num_samples - i);
    if (!read(chunk.data(), n * record))
      return diag::feeding::Error("truncated inputs");
    for (uint64_t s = 0; s < n; ++s) {
      const uint8_t* rec = chunk.data() + s * record;
      std::memcpy(d.inputs_.data() + (i + s) * input_dim, rec, input_bytes);
      std::memcpy(d.labels_.data() + (i + s) * lbytes, rec + input_bytes,
                  lbytes);
    }
    i += n;
  }
  return d;
}

std::expected<void, std::string> Dataset::ValidateClassLabels(
    uint64_t num_classes) const {
  if (label_kind_ != 1 || num_classes == 0) return {};
  // For a token corpus every token is both an embedding index and (via
  // the shifted view) a class label, so the whole stream is bounded — the
  // caller passes the narrowest of the vocab and softmax widths.
  const int32_t* ids =
      input_kind_ == 1 ? tokens_.data()
                       : reinterpret_cast<const int32_t*>(labels_.data());
  const uint64_t count = input_kind_ == 1 ? tokens_.size() : num_samples_;

  // One scan ever: the extent of the ids answers every later question.
  if (!labels_scanned_) {
    int64_t lo = 0, hi = -1;
    for (uint64_t i = 0; i < count; ++i) {
      const int64_t v = ids[i];
      if (i == 0 || v < lo) lo = v;
      if (i == 0 || v > hi) hi = v;
    }
    proven_min_label_ = lo;
    proven_max_label_ = hi;
    labels_scanned_ = true;
  }
  if (count == 0 ||
      (proven_min_label_ >= 0 &&
       static_cast<uint64_t>(proven_max_label_) < num_classes))
    return {};

  // Refused: find the first offender for the diagnostic (the error path
  // may rescan; the success path never does).
  for (uint64_t i = 0; i < count; ++i)
    if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= num_classes)
      return diag::feeding::Error(
          std::string(input_kind_ == 1 ? "token id " : "class label ") +
          std::to_string(ids[i]) +
          (input_kind_ == 1 ? " at position " : " at sample ") +
          std::to_string(i) + " outside [0, " + std::to_string(num_classes) +
          ")");
  return {};
}

std::expected<void, std::string> Dataset::SaveToFile(
    const std::string& path) const {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return diag::feeding::Error("cannot write '" + path + "'");
  auto write = [&](const void* src, size_t n) {
    f.write(reinterpret_cast<const char*>(src),
            static_cast<std::streamsize>(n));
  };
  // The lowest version that carries the content: a plain corpus stays
  // v1 / v2 byte for byte; a tokenizer identity or a mask makes it v3.
  const uint32_t version = (tokenizer_hash_ != 0 || masked_) ? 3
                           : input_kind_ == 1                 ? 2
                                                              : 1;
  write(&kSdsMagic, 4);
  write(&version, 4);
  write(&num_samples_, 8);
  write(&input_dim_, 8);
  write(&label_kind_, 4);
  write(&input_kind_, 4);  // header padding in v1 (always 0 there)
  write(&label_dim_, 8);
  if (version >= 3) {
    const uint32_t flags = masked_ ? kSdsFlagMasked : 0;
    const uint32_t reserved0 = 0;
    const uint64_t reserved1 = 0;
    write(&tokenizer_hash_, 8);
    write(&flags, 4);
    write(&reserved0, 4);
    write(&reserved1, 8);
  }
  if (input_kind_ == 1) {
    if (!masked_) {
      // Token records are one contiguous i32 block; labels are derived,
      // never stored.
      write(tokens_.data(), tokens_.size() * sizeof(int32_t));
    } else {
      // Each record: its ids, then its mask padded to 4 bytes.
      const uint64_t stride = input_dim_ + 1;
      std::vector<uint8_t> pad(static_cast<size_t>(
          SdsMaskBytes(input_dim_) - input_dim_));
      for (uint64_t r = 0; r < num_samples_; ++r) {
        write(tokens_.data() + r * stride, stride * sizeof(int32_t));
        write(mask_.data() + r * input_dim_, input_dim_);
        if (!pad.empty()) write(pad.data(), pad.size());
      }
    }
    f.close();
    if (f.fail())
      return diag::feeding::Error("short write to '" + path + "'");
    return {};
  }
  const uint64_t lbytes = label_bytes_per_sample();
  if (lbytes == 0) {
    // Unlabeled: the sample section is exactly the input block.
    write(inputs_.data(), num_samples_ * input_dim_ * sizeof(float));
  } else {
    // Interleave whole records through a bounded staging chunk and write in
    // bulk — the mirror of LoadFromFile's chunked reader.
    const uint64_t input_bytes = input_dim_ * sizeof(float);
    const uint64_t record = input_bytes + lbytes;
    constexpr uint64_t kChunkBudget = 256 * 1024;
    const uint64_t per_chunk =
        record >= kChunkBudget ? 1 : kChunkBudget / record;
    std::vector<uint8_t> chunk(per_chunk * record);
    for (uint64_t i = 0; i < num_samples_;) {
      const uint64_t n = std::min(per_chunk, num_samples_ - i);
      for (uint64_t s = 0; s < n; ++s) {
        uint8_t* rec = chunk.data() + s * record;
        std::memcpy(rec, inputs_.data() + (i + s) * input_dim_, input_bytes);
        std::memcpy(rec + input_bytes, labels_.data() + (i + s) * lbytes,
                    lbytes);
      }
      write(chunk.data(), n * record);
      i += n;
    }
  }
  // Close before the state check: the tail of the stream buffer is flushed
  // at close, and a failure there (disk full) after an early state check
  // would report a truncated .sds file as saved.
  f.close();
  if (f.fail()) return diag::feeding::Error("short write to '" + path + "'");
  return {};
}

void Dataset::FillBatch(uint64_t batch, float* input_slot,
                        uint8_t* label_slot) {
  if (input_kind_ == 1) {
    // `batch` counts token ROWS; the feeder contract proved it a whole
    // number of records. Record r contributes tokens[0..S) as the input
    // rows and the shifted view tokens[1..S] as the derived labels. The
    // cursor/order/epoch machinery is untouched — it simply walks records.
    const uint64_t S = input_dim_;
    const uint64_t records = batch / S;
    auto* in = reinterpret_cast<int32_t*>(input_slot);
    auto* lab = reinterpret_cast<int32_t*>(label_slot);
    for (uint64_t r = 0; r < records; ++r) {
      const uint64_t i = order_.empty() ? cursor_ : order_[cursor_];
      const int32_t* rec = tokens_.data() + i * (S + 1);
      std::memcpy(in + r * S, rec, S * sizeof(int32_t));
      if (lab && !masked_) {
        std::memcpy(lab + r * S, rec + 1, S * sizeof(int32_t));
      } else if (lab) {
        // A masked-out position stages kSdsIgnoreLabel: the masked
        // cross-entropy counts only the positions the mask names.
        const uint8_t* m = mask_.data() + i * S;
        for (uint64_t j = 0; j < S; ++j)
          lab[r * S + j] = m[j] ? rec[j + 1] : kSdsIgnoreLabel;
      }
      cursor_ = cursor_ + 1;
      if (cursor_ == num_samples_) {
        cursor_ = 0;
        if (!order_.empty()) {
          Reshuffle();  // fresh permutation every epoch
          ++epoch_;
        }
      }
    }
    return;
  }
  const uint64_t lbytes = label_bytes_per_sample();

  if (order_.empty()) {
    // Sequential serving: samples are contiguous in memory, so copy whole
    // runs (up to the wraparound point) instead of one sample at a time.
    uint64_t b = 0;
    while (b < batch) {
      const uint64_t run = std::min(batch - b, num_samples_ - cursor_);
      std::memcpy(input_slot + b * input_dim_,
                  inputs_.data() + cursor_ * input_dim_,
                  run * input_dim_ * sizeof(float));
      if (label_slot && lbytes)
        std::memcpy(label_slot + b * lbytes, labels_.data() + cursor_ * lbytes,
                    run * lbytes);
      b += run;
      cursor_ += run;
      if (cursor_ == num_samples_) cursor_ = 0;
    }
    return;
  }

  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t i = order_[cursor_];
    std::memcpy(input_slot + b * input_dim_, inputs_.data() + i * input_dim_,
                input_dim_ * sizeof(float));
    if (label_slot && lbytes)
      std::memcpy(label_slot + b * lbytes, labels_.data() + i * lbytes, lbytes);
    cursor_ = cursor_ + 1;
    if (cursor_ == num_samples_) {
      cursor_ = 0;
      Reshuffle();  // fresh permutation every epoch
      ++epoch_;
    }
  }
}

namespace {

// splitmix64: tiny, deterministic, and high-quality enough for shuffling.
// Avoids dragging <random> (and its per-platform distribution differences)
// into the device runtime — the permutation must be reproducible everywhere.
uint64_t SplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

void Dataset::EnableShuffle(uint64_t seed) {
  // Fold the seed through one splitmix step so seed 0 is a valid choice
  // (shuffle_state_ == 0 means "shuffling off").
  shuffle_state_ = seed;
  shuffle_state_ = SplitMix64(&shuffle_state_) | 1ULL;
  shuffle_origin_ = shuffle_state_;
  epoch_ = 0;
  order_.resize(num_samples_);
  Reshuffle();
  cursor_ = 0;
}

void Dataset::RestoreServingPos(ServingPos pos) {
  cursor_ = pos.cursor;
  if (order_.empty() || epoch_ == pos.epoch) return;
  // The epoch moved past the snapshot: replay the permutation sequence from
  // the shuffle origin. Each epoch's permutation is a pure function of
  // (origin, epoch index), so the replay reproduces the exact order the
  // snapshot indexed — O(epoch * n), paid once at feeder teardown.
  shuffle_state_ = shuffle_origin_;
  epoch_ = 0;
  Reshuffle();
  while (epoch_ < pos.epoch) {
    Reshuffle();
    ++epoch_;
  }
}

std::expected<void, std::string> Dataset::SkipServed(uint64_t rows) {
  uint64_t records = rows;
  if (input_kind_ == 1) {
    // One record serves input_dim_ rows. The compiler fixes batch as a
    // whole number of sequences, so a remainder here means the caller's
    // step arithmetic and this corpus disagree — refuse loudly rather
    // than silently desynchronize the replayed cursor.
    if (rows % input_dim_ != 0)
      return diag::feeding::Error(
          "served rows are not a whole number of token records — cannot "
          "replay the serving position");
    records = rows / input_dim_;
  }
  RestoreServingPos({records % num_samples_, records / num_samples_});
  return {};
}

void Dataset::Reshuffle() {
  for (uint64_t i = 0; i < num_samples_; ++i) order_[i] = i;
  // Fisher–Yates with an unbiased-enough bound (num_samples << 2^64).
  for (uint64_t i = num_samples_ - 1; i > 0; --i) {
    const uint64_t j = SplitMix64(&shuffle_state_) % (i + 1);
    std::swap(order_[i], order_[j]);
  }
}

std::expected<Dataset, std::string> Dataset::SplitValidation(double fraction) {
  if (!(fraction > 0.0) || fraction >= 1.0)
    return diag::feeding::Error("validation fraction must be in (0, 1)");
  if (num_samples_ < 2)
    return diag::feeding::Error("too few samples to split");
  if (!order_.empty())
    return diag::feeding::Error("split before enabling shuffle");

  uint64_t val_n = static_cast<uint64_t>(
      static_cast<double>(num_samples_) * fraction);
  if (val_n == 0) val_n = 1;
  if (val_n >= num_samples_) val_n = num_samples_ - 1;
  const uint64_t train_n = num_samples_ - val_n;
  const uint64_t lbytes = label_bytes_per_sample();

  Dataset val;
  val.num_samples_ = val_n;
  val.input_dim_ = input_dim_;
  val.label_kind_ = label_kind_;
  val.label_dim_ = label_dim_;
  val.input_kind_ = input_kind_;
  val.tokenizer_hash_ = tokenizer_hash_;
  val.masked_ = masked_;
  if (input_kind_ == 1) {
    // Token corpora split at RECORD (sequence) granularity: the tail
    // records move wholesale, so no sequence is ever cut in half.
    const uint64_t rec_ids = input_dim_ + 1;
    val.tokens_.assign(
        tokens_.begin() + static_cast<ptrdiff_t>(train_n * rec_ids),
        tokens_.end());
    tokens_.resize(train_n * rec_ids);
    if (masked_) {
      val.mask_.assign(
          mask_.begin() + static_cast<ptrdiff_t>(train_n * input_dim_),
          mask_.end());
      mask_.resize(train_n * input_dim_);
    }
    num_samples_ = train_n;
    cursor_ = 0;
    labels_scanned_ = false;  // the extent was the whole corpus's
    return val;
  }
  val.inputs_.assign(inputs_.begin() + static_cast<ptrdiff_t>(train_n * input_dim_),
                     inputs_.end());
  val.labels_.assign(labels_.begin() + static_cast<ptrdiff_t>(train_n * lbytes),
                     labels_.end());

  inputs_.resize(train_n * input_dim_);
  labels_.resize(train_n * lbytes);
  num_samples_ = train_n;
  cursor_ = 0;
  labels_scanned_ = false;  // the extent was the whole corpus's
  return val;
}

std::optional<double> Dataset::LabelEntropyNats() const {
  if (label_kind_ != 1) return std::nullopt;
  // Every class label the corpus serves, in record order. Token record r
  // is tokens_[r·(seq+1), ...]: the inputs are its first seq ids, the
  // labels positions 1..seq (the shifted view).
  auto each_label = [&](auto&& visit) {
    if (input_kind_ == 1) {
      // A masked corpus's labels are the target positions alone.
      const uint64_t stride = input_dim_ + 1;
      for (uint64_t r = 0; r < num_samples_; ++r)
        for (uint64_t j = 1; j <= input_dim_; ++j)
          if (!masked_ || mask_[r * input_dim_ + (j - 1)])
            visit(tokens_[r * stride + j]);
    } else {
      for (uint64_t i = 0; i < num_samples_; ++i) {
        int32_t label;
        std::memcpy(&label, labels_.data() + i * sizeof(label),
                    sizeof(label));
        visit(label);
      }
    }
  };
  uint64_t n = 0;
  int32_t lo = INT32_MAX, hi = INT32_MIN;
  each_label([&](int32_t v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    ++n;
  });
  if (n == 0) return std::nullopt;
  // Counted in a dense table over [lo, hi] — class and token ids are a
  // vocabulary-sized range, which the feeder contract bounds before
  // training — and in a hash map only for a range too wide to tabulate.
  // Summed over the sorted counts: one order on every host, so the same
  // bits wherever the screen is printed.
  std::vector<uint64_t> sorted;
  const auto span =
      static_cast<uint64_t>(static_cast<int64_t>(hi) - lo) + 1;
  if (span <= (uint64_t{1} << 20)) {
    std::vector<uint64_t> counts(span, 0);
    each_label([&](int32_t v) {
      ++counts[static_cast<uint64_t>(static_cast<int64_t>(v) - lo)];
    });
    for (const uint64_t c : counts)
      if (c != 0) sorted.push_back(c);
  } else {
    std::unordered_map<int32_t, uint64_t> counts;
    each_label([&](int32_t v) { ++counts[v]; });
    sorted.reserve(counts.size());
    for (const auto& [label, count] : counts) sorted.push_back(count);
  }
  std::sort(sorted.begin(), sorted.end());
  double entropy = 0.0;
  const double total = static_cast<double>(n);
  for (const uint64_t c : sorted) {
    const double p = static_cast<double>(c) / total;
    entropy -= p * std::log(p);
  }
  return entropy;
}

}  // namespace seeml::update_rt
