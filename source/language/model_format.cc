#include "source/language/model_format.h"

#include "source/identity/hash.h"

namespace seeml::update {

const SmfTensor* SmfModel::FindTensor(std::string_view name) const {
  for (const auto& t : tensors)
    if (t.name == name) return &t;
  return nullptr;
}

uint64_t SmfModel::TokenizerHash() const {
  if (tokenizer.empty()) return 0;
  return ContentHash64(tokenizer.data(), tokenizer.size());
}

}  // namespace seeml::update
