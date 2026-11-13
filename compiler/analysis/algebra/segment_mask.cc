#include "compiler/analysis/algebra/segment_mask.h"

#include <vector>

namespace seeml::update {

namespace sir = seeml::sir;

std::expected<size_t, std::string> SegmentMasker::Run(sir::Block& block) {
  std::vector<sir::Operation*> attention;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_high.attention") attention.push_back(op);
  });
  if (attention.empty()) return size_t{0};
  if (tokens_ == nullptr || tokens_->dtype() != sir::DataType::I32)
    return std::unexpected(std::string(
        "segment mask: the model reads no token ids to find documents in"));
  const int64_t seq = attention.front()->getAttrAs<int64_t>("seq").value_or(0);
  for (sir::Operation* op : attention) {
    if (op->hasAttribute("tiled"))
      return std::unexpected(std::string(
          "segment mask: the tiled attention family is not masked yet "
          "(#191 step 2); compile with --attention cached"));
    if (op->numOperands() != 3)
      return std::unexpected(std::string(
          "segment mask: an attention op already carries a fourth operand"));
    if (op->getAttrAs<int64_t>("seq").value_or(0) != seq)
      return std::unexpected(std::string(
          "segment mask: attention ops disagree on the sequence length"));
  }
  // Declared at the end, moved above the first attention op while it has no
  // operand (the move's own rule), then handed its one operand: the block's
  // token argument, which precedes every op, so the order stays SSA-sound.
  sir::Operation* starts_op = block.appendOp("sc_low.segment_starts");
  starts_op->setAttribute("seq", seq);
  starts_op->setAttribute("eos", static_cast<int64_t>(eos_));
  sir::Value* starts = starts_op->addResult(
      "segment_starts", sir::DataType::I32, tokens_->shape());
  block.moveOpBefore(starts_op, attention.front());
  starts_op->addOperand(tokens_);
  for (sir::Operation* op : attention) op->addOperand(starts);
  return attention.size();
}

}  // namespace seeml::update
