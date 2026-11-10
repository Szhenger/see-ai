// seeml-tokenize — the core's tokenizer, run from the command line (S2 #148).
//
//   seeml-tokenize model.smf --info
//       the tokenizer section's summary: kind, pre-tokenizer, vocabulary,
//       merges, special tokens, eos / bos, whether a chat template is
//       carried, and the section's hash (the identity corpora bind to).
//   seeml-tokenize model.smf text.txt [--pieces | --lines]
//       the ids of the file's bytes, one per line (the whole file is one
//       document: no EOS appended, nothing dropped but bytes without a
//       token); --pieces prints the pre-tokenizer's pieces instead;
//       --lines encodes every line of the file as a document of its own
//       and prints one line of space-separated ids per input line.
//
// What this prints is what `seeml-update-compile --data text.txt` encodes
// and what the device's pipeline re-runs on its own data; the exporter's
// parity harness (tool/export_model.py --tokenizer-parity) holds it to the
// reference tokenizer on a golden corpus. Exit 2 on usage, 1 on refusal.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/frontend/ingressor/model_reader.h"
#include "source/tokenizer/byte_level_bpe.h"

namespace {

int Usage() {
  std::fprintf(stderr,
               "usage: seeml-tokenize model.smf --info\n"
               "       seeml-tokenize model.smf text.txt [--pieces | --lines]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return Usage();
  const std::string model_path = argv[1];
  bool info = false, pieces = false, lines = false;
  std::string text_path;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--info") == 0)
      info = true;
    else if (std::strcmp(argv[i], "--pieces") == 0)
      pieces = true;
    else if (std::strcmp(argv[i], "--lines") == 0)
      lines = true;
    else if (argv[i][0] == '-' && argv[i][1] != '\0')
      return Usage();
    else if (text_path.empty())
      text_path = argv[i];
    else
      return Usage();
  }
  if (!info && text_path.empty()) return Usage();

  auto model = seeml::update::LoadSmf(model_path);
  if (!model) {
    std::fprintf(stderr, "seeml-tokenize: %s\n", model.error().c_str());
    return 1;
  }
  if (model->tokenizer.empty()) {
    std::fprintf(stderr,
                 "seeml-tokenize: '%s' carries no tokenizer section (SMF v7): "
                 "export it with tool/export_model.py --hf from a checkpoint "
                 "with a tokenizer.json\n",
                 model_path.c_str());
    return 1;
  }
  auto tok = seeml::tokenizer::ByteLevelBpe::Load(model->tokenizer);
  if (!tok) {
    std::fprintf(stderr, "seeml-tokenize: %s\n", tok.error().c_str());
    return 1;
  }
  if (info) {
    const auto& in = tok->info();
    std::printf("kind            byte-level BPE\n");
    std::printf("pre-tokenizer   %s\n",
                in.pre_tokenizer == seeml::tokenizer::PreTokenizer::kGpt2
                    ? "GPT-2 pattern"
                    : "Digits(individual), then the GPT-2 pattern");
    std::printf("vocabulary      %u\n", in.vocab_size);
    std::printf("merges          %u\n", in.num_merges);
    std::printf("special tokens  %u\n", in.num_special);
    std::printf("eos / bos       %d / %d\n", in.eos_id, in.bos_id);
    std::printf("chat template   %s\n",
                in.chat_template.empty()
                    ? "none"
                    : (std::to_string(in.chat_template.size()) + " bytes (stored, not interpreted)").c_str());
    std::printf("section bytes   %zu\n", model->tokenizer.size());
    std::printf("tokenizer hash  %016llx\n",
                static_cast<unsigned long long>(model->TokenizerHash()));
    return 0;
  }
  std::ifstream f(text_path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "seeml-tokenize: cannot open '%s'\n", text_path.c_str());
    return 1;
  }
  const std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  if (pieces) {
    for (const auto& [b, e] : tok->PreTokenize(text)) {
      std::fwrite(text.data() + b, 1, e - b, stdout);
      std::fputc('\n', stdout);
    }
    return 0;
  }
  std::string out;
  if (lines) {
    // Every line of the file as a document of its own — the pieces of
    // text.split("\n"), the last possibly empty — one line of
    // space-separated ids per input line (an empty line for no ids), so a
    // parity run over a corpus of N lines loads the model once, not N
    // times.
    std::string_view rest = text;
    while (true) {
      const size_t nl = rest.find('\n');
      const std::string_view line =
          nl == std::string_view::npos ? rest : rest.substr(0, nl);
      std::vector<int32_t> ids;
      if (auto ok = tok->Encode(line, ids); !ok) {
        std::fprintf(stderr, "seeml-tokenize: %s\n", ok.error().c_str());
        return 1;
      }
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ' ';
        out += std::to_string(ids[i]);
      }
      out += '\n';
      if (nl == std::string_view::npos) break;
      rest = rest.substr(nl + 1);
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
  }
  std::vector<int32_t> ids;
  if (auto ok = tok->Encode(text, ids); !ok) {
    std::fprintf(stderr, "seeml-tokenize: %s\n", ok.error().c_str());
    return 1;
  }
  out.reserve(ids.size() * 6);
  for (int32_t id : ids) {
    out += std::to_string(id);
    out += '\n';
  }
  std::fwrite(out.data(), 1, out.size(), stdout);
  return 0;
}
