#ifndef SEEML_SOURCE_LANGUAGE_SMF_PREAMBLE_H_
#define SEEML_SOURCE_LANGUAGE_SMF_PREAMBLE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>

#include "source/language/model_format.h"

// =============================================================================
// The SMF preamble, read on the device (#202 item 2): the version, the
// model's seq_len and where the v7 tokenizer section lies. model_update
// needs no more of the model file to ingest raw text with the model's own
// tokenizer — the compiler's full reader (compiler/frontend/ingressor/
// model_reader.cc, LoadSmf) stays in the compiler. The layout is LoadSmf's:
//   u32 magic, u32 version, u32 num_tensors, u32 num_ops,
//   str input_name, str output_name        (str = u16 length, then bytes)
//   u64 seq_len                            (v3+)
//   u64 tokenizer_offset, u64 tokenizer_size (v7+)
// Header-only, so the package vendors it as it is.
// =============================================================================

namespace seeml::update {

struct SmfPreamble {
  uint32_t version = 0;
  uint64_t seq_len = 0;           // 0 before v3
  uint64_t tokenizer_offset = 0;  // 0 / 0: no tokenizer section
  uint64_t tokenizer_size = 0;
};

/// The longest preamble a valid file can have: the fixed words and two
/// maximal names.
inline constexpr size_t kSmfPreambleMaxBytes = 4 * 4 + 2 * (2 + 65535) + 8 + 16;

/// Reads the preamble from the file's first `n` bytes (at least
/// min(file_size, kSmfPreambleMaxBytes)); the tokenizer section must lie
/// inside a file of `file_size` bytes.
inline std::expected<SmfPreamble, std::string> ReadSmfPreamble(
    const uint8_t* bytes, size_t n, uint64_t file_size) {
  size_t pos = 0;
  bool ok = true;
  auto take = [&](void* dst, size_t len) {
    if (!ok || len > n - pos) {
      ok = false;
      return;
    }
    std::memcpy(dst, bytes + pos, len);
    pos += len;
  };
  auto u32 = [&] { uint32_t v = 0; take(&v, 4); return v; };
  auto u64 = [&] { uint64_t v = 0; take(&v, 8); return v; };
  auto skip_str = [&] {
    uint16_t len = 0;
    take(&len, 2);
    if (ok && len > n - pos) ok = false;
    if (ok) pos += len;
  };
  if (u32() != kSmfMagic || !ok) return std::unexpected("not an SMF model (bad magic)");
  SmfPreamble p;
  p.version = u32();
  if (!ok || p.version < kSmfMinVersion || p.version > kSmfVersion)
    return std::unexpected("unsupported SMF version " + std::to_string(p.version));
  (void)u32();  // num_tensors
  (void)u32();  // num_ops
  skip_str();   // input_name
  skip_str();   // output_name
  if (p.version >= 3) p.seq_len = u64();
  if (p.version >= 7) {
    p.tokenizer_offset = u64();
    p.tokenizer_size = u64();
  }
  if (!ok) return std::unexpected("truncated SMF preamble");
  if (p.tokenizer_offset > file_size ||
      p.tokenizer_size > file_size - p.tokenizer_offset)
    return std::unexpected("the tokenizer section exceeds the file");
  return p;
}

}  // namespace seeml::update

#endif  // SEEML_SOURCE_LANGUAGE_SMF_PREAMBLE_H_
