#include "compiler/frontend/egressor/model_writer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#endif

#include "compiler/diagnostics/tokenizing/error.h"
#include "source/identity/hash.h"

namespace seeml::update {

namespace tokenizing = seeml::diag::tokenizing;

namespace {

constexpr uint64_t kDataAlignment = 64;

uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// --- Little-endian primitive writer into an in-memory buffer -----------------

struct Writer {
  std::vector<uint8_t> buf;

  template <typename T>
  void Write(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
  }

  void WriteStr(const std::string& s) {
    Write<uint16_t>(static_cast<uint16_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
  }
};

}  // namespace

std::expected<void, std::string> SaveSmf(const std::string& path,
                                         SmfModel& model) {
  // Every length/count below is narrowed into a fixed-width prefix while the
  // full payload is written; a wrapped prefix would be silent on-disk
  // corruption, so reject anything that does not fit its field first.
  auto name_ok = [](const std::string& s) { return s.size() <= UINT16_MAX; };
  if (model.tensors.size() > UINT32_MAX)
    return tokenizing::Error("too many tensors to serialize");
  if (model.ops.size() > UINT32_MAX)
    return tokenizing::Error("too many ops to serialize");
  if (!name_ok(model.input_name) || !name_ok(model.output_name))
    return tokenizing::Error("model input/output name is too long");
  for (const auto& t : model.tensors) {
    if (!name_ok(t.name))
      return tokenizing::TensorError(t.name, "name is too long to serialize");
    if (t.dims.size() > UINT8_MAX)
      return tokenizing::TensorError(t.name, "has too many dims to serialize");
  }
  for (const auto& op : model.ops) {
    if (!name_ok(op.name) || !name_ok(op.output))
      return tokenizing::Error("op '" + op.name +
                               "' has a name too long to serialize");
    if (op.inputs.size() > UINT8_MAX)
      return tokenizing::Error("op '" + op.name +
                               "' has too many inputs to serialize");
    for (const auto& in : op.inputs)
      if (!name_ok(in))
        return tokenizing::Error("op '" + op.name +
                                 "' has an input name too long to serialize");
  }

  // Pass 1: serialize the header/metadata with zeroed data offsets to learn
  // the metadata section size, then lay out the 64-aligned data section.
  // The data layout is computed into `layout` and written from it; the
  // model's own offsets, sizes and hash change only once the file is in
  // place (CR4 #145 F05: a failed save used to leave new offsets paired
  // with the old hash).
  std::vector<std::pair<uint64_t, uint64_t>> layout(model.tensors.size());
  auto serialize_meta = [&](Writer& w) {
    w.Write<uint32_t>(kSmfMagic);
    w.Write<uint32_t>(kSmfVersion);
    w.Write<uint32_t>(static_cast<uint32_t>(model.tensors.size()));
    w.Write<uint32_t>(static_cast<uint32_t>(model.ops.size()));
    w.WriteStr(model.input_name);
    w.WriteStr(model.output_name);
    w.Write<uint64_t>(model.seq_len);
    for (size_t i = 0; i < model.tensors.size(); ++i) {
      const auto& t = model.tensors[i];
      w.WriteStr(t.name);
      w.Write<uint8_t>(static_cast<uint8_t>(t.dims.size()));
      w.Write<uint8_t>(t.is_const ? 1 : 0);
      for (int64_t d : t.dims) w.Write<int64_t>(d);
      w.Write<uint64_t>(layout[i].first);
      w.Write<uint64_t>(layout[i].second);
    }
    for (const auto& op : model.ops) {
      w.Write<uint8_t>(static_cast<uint8_t>(op.kind));
      w.WriteStr(op.name);
      w.Write<uint8_t>(static_cast<uint8_t>(op.inputs.size()));
      for (const auto& in : op.inputs) w.WriteStr(in);
      w.WriteStr(op.output);
      w.Write<uint32_t>(op.attr0);
      w.Write<uint32_t>(op.attr1);
      w.Write<uint32_t>(op.attr2);
    }
  };

  Writer probe;
  serialize_meta(probe);

  uint64_t cursor = AlignUp(probe.buf.size(), kDataAlignment);
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const auto& t = model.tensors[i];
    if (!t.is_const) continue;
    if (t.data.empty())
      return tokenizing::Error("constant tensor '" + t.name +
                               "' has no data to serialize");
    layout[i] = {cursor, t.data.size()};
    cursor = AlignUp(cursor + t.data.size(), kDataAlignment);
  }

  // Pass 2: re-serialize with the final offsets, append the data section.
  // The probe told us the exact final size — reserve it up front so neither
  // the metadata append nor the data-section resize ever reallocates.
  Writer w;
  w.buf.reserve(cursor);
  serialize_meta(w);
  // Append, don't resize-then-overwrite: only the alignment gaps are zero
  // bytes; zero-filling the whole data section first wrote every payload
  // byte twice (E2, #81). The reservation above means no append reallocates.
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const auto& t = model.tensors[i];
    if (!t.is_const) continue;
    w.buf.resize(static_cast<size_t>(layout[i].first), 0);  // the gap only
    w.buf.insert(w.buf.end(), t.data.begin(), t.data.end());
  }
  w.buf.resize(static_cast<size_t>(cursor), 0);

  // Staged beside the destination, every write and the close checked, the
  // data synced, then renamed into place; any failure removes the stage
  // and leaves the destination as it was (CR4 #145 F05).
  const std::string tmp = path + ".tmp";
  // Created 0644 (the durable writer's mode; CodeQL's world-writable-file
  // rule on fopen's 0666), then wrapped for the buffered writes below.
#ifndef _WIN32
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  std::FILE* f = fd < 0 ? nullptr : ::fdopen(fd, "wb");
  if (!f && fd >= 0) ::close(fd);
#else
  const int fd = ::_open(tmp.c_str(),
                         _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY,
                         _S_IREAD | _S_IWRITE);
  std::FILE* f = fd < 0 ? nullptr : ::_fdopen(fd, "wb");
  if (!f && fd >= 0) ::_close(fd);
#endif
  if (!f) return tokenizing::FileError("cannot write", tmp);
  auto fail = [&](const char* what) {
    std::remove(tmp.c_str());
    return tokenizing::FileError(what, tmp);
  };
  const bool written =
      std::fwrite(w.buf.data(), 1, w.buf.size(), f) == w.buf.size() &&
      std::fflush(f) == 0;
#ifndef _WIN32
  const bool synced = written && ::fsync(::fileno(f)) == 0;
#else
  const bool synced = written && ::_commit(::_fileno(f)) == 0;
#endif
  const bool closed = std::fclose(f) == 0;
  if (!synced || !closed) return fail("short write to");
  if (std::rename(tmp.c_str(), path.c_str()) != 0)
    return fail("cannot move the staged file over");
  // The file is in place: now the model corresponds to the saved bytes —
  // its offsets and sizes, and its identity, so a plan compiled from this
  // in-memory model patches this exact file.
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    if (!model.tensors[i].is_const) continue;
    model.tensors[i].data_offset = layout[i].first;
    model.tensors[i].byte_size = layout[i].second;
  }
  model.content_hash = ContentHash64(w.buf.data(), w.buf.size());
  return {};
}

}  // namespace seeml::update
