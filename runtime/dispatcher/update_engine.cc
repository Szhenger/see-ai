#include "runtime/dispatcher/update_engine.h"

#include <algorithm>
#include <cerrno>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "runtime/storage/checkpoint.h"
#include "runtime/storage/checkpoint_format.h"
#include "runtime/storage/durable_io.h"
#include "runtime/diagnostics/executing/error.h"
#include "runtime/loader/contract.h"
#include "runtime/loader/plan_loader.h"
#include "runtime/executor/backend.h"
#include "runtime/pipeline/batch_pipeline.h"
#include "runtime/verifier/plan_validator.h"
#include "source/identity/hash.h"
#include "source/platform/file_map.h"
#include "source/platform/memory.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update_rt {

namespace up = seeml::update;

static_assert(LossWindows::kMax == kCkptWindowMax,
              "the train-loss windows are the checkpoint's ring");

namespace {

/// The header's kernel policy: the v11 GEMM tiles, zero meaning the
/// runtime's compiled-in default for that dimension.
kernels::KernelPolicy KernelPolicyOf(const up::PlanHeader& header) {
  kernels::KernelPolicy policy;
  if (header.gemm_tile_k) policy.gemm_tiles.k = header.gemm_tile_k;
  if (header.gemm_tile_n) policy.gemm_tiles.n = header.gemm_tile_n;
  return policy;
}

}  // namespace

UpdateEngine::UpdateEngine() : backend_(CreateCpuBackend()) {}

kernels::GemmTiles UpdateEngine::gemm_tiles() const {
  return KernelPolicyOf(header_).gemm_tiles;
}

std::expected<void, std::string> UpdateEngine::BindSourceModel(
    const std::string& source_model_path) {
  if (!arena_) return diag::executing::Error("no plan loaded");
  if (source_required_ == 0) return VerifySourceModel(source_model_path);
  // Map the file read-only: the eval program reads the f32 weights in place
  // (the page cache serves them; nothing is copied), and the hash below is
  // taken over the very bytes the kernels will read.
  auto mapped =
      platform::MapFile(source_model_path, platform::MapAccess::kReadOnly,
                        /*require_regular=*/false);
  if (!mapped)
    return diag::executing::Error(
        std::string(mapped.error().stage == platform::MapFailure::kOpen
                        ? "cannot open"
                    : mapped.error().stage == platform::MapFailure::kSize
                        ? "cannot size"
                        : "cannot map") +
        " the source model '" + source_model_path + "'");
  platform::FileMapping candidate = std::move(*mapped);
  const size_t bytes = candidate.bytes;
  if (up::ContentHash64(candidate.data, bytes) != header_.source_model_hash)
    return diag::executing::Error(
        "source model does not match the plan's source_model_hash — "
        "refusing to score against '" +
        source_model_path + "'");
  if (bytes < source_required_)
    return diag::executing::Error(
        "source model '" + source_model_path + "' is shorter than the " +
        std::to_string(source_required_) + " bytes its programs read from it");
  if (auto r = backend_->BindSource(candidate.data, bytes); !r)
    return diag::executing::Error("backend cannot bind the source model: " +
                                  r.error());
  source_ = std::move(candidate);  // releases the previous mapping
  return {};
}

UpdateEngine::~UpdateEngine() {
  // The backend may hold a zero-copy view of the arena (a GPU buffer
  // wrapping it) and of the source mapping: release the views before the
  // memory.
  backend_.reset();
  source_.Release();
  plan_map_.Release();
  std::free(arena_);
}

std::expected<void, std::string> UpdateEngine::SelectBackend(
    BackendKind requested) {
  auto sel = CreateBackend(requested);
  if (!sel)
    return diag::executing::Error("backend '" +
                                  std::string(BackendKindName(requested)) +
                                  "': " + sel.error());
  // A plan already loaded must stay executable on the new backend, or the
  // switch is refused and the current backend keeps its binding.
  if (arena_) {
    if (auto r = sel->backend->Bind(arena_, arena_bytes_, rodata_,
                                    header_.rodata_size,
                                    plan_size_ - header_.rodata_offset);
        !r)
      return diag::executing::Error(
          "backend '" + std::string(sel->backend->name()) +
          "' cannot bind the loaded plan: " + r.error());
    sel->backend->Configure(KernelPolicyOf(header_));
    if (source_.data)
      if (auto r = sel->backend->BindSource(source_.data, source_.bytes); !r)
        return diag::executing::Error(
            "backend '" + std::string(sel->backend->name()) +
            "' cannot bind the source model: " + r.error());
  }
  backend_ = std::move(sel->backend);
  backend_kind_ = sel->resolved;
  backend_note_ = std::move(sel->note);
  return {};
}

const char* UpdateEngine::backend_name() const { return backend_->name(); }
std::string UpdateEngine::backend_device() const { return backend_->device(); }
uint64_t UpdateEngine::relaxed_executions() const {
  return backend_->relaxed_executions();
}

uint64_t UpdateEngine::fused_dispatches() const {
  return backend_->fused_dispatches();
}

const float* UpdateEngine::ReadPtr(uint64_t ref) const {
  const uint64_t offset = up::RefOffset(ref);
  if (up::IsRodataRef(ref))
    return reinterpret_cast<const float*>(rodata_ + offset);
  return reinterpret_cast<const float*>(arena_ + offset);
}

float* UpdateEngine::WritePtr(uint64_t ref) {
  // Lowering never emits a rodata destination; the frozen weights are
  // physically unwritable from the instruction stream by construction.
  return reinterpret_cast<float*>(arena_ + up::RefOffset(ref));
}

std::expected<void, std::string> UpdateEngine::LoadFromMemory(
    const uint8_t* plan, size_t size, RelaxedPlanPermit permit) {
  if (auto r = Initialize(plan, size, permit); !r) return r;
  // The committed plan is borrowed; a previously file-owned or mapped plan
  // is no longer referenced by plan_/rodata_ and can be released.
  owned_plan_.clear();
  owned_plan_.shrink_to_fit();
  plan_map_.Release();
  return {};
}

std::expected<void, std::string> UpdateEngine::LoadFromFile(
    const std::string& path, RelaxedPlanPermit permit) {
  auto bytes = ReadFileBytes(path);
  if (!bytes) return std::unexpected(bytes.error());
  // Initialize against the candidate buffer first: replacing owned_plan_ up
  // front would free the buffer a still-loaded previous plan's rodata_
  // points into even when the candidate is rejected. The move below keeps
  // the heap buffer (and the just-committed plan_/rodata_) stable.
  if (auto r = Initialize(bytes->data(), bytes->size(), permit); !r) return r;
  owned_plan_ = std::move(*bytes);
  plan_map_.Release();
  return {};
}

std::expected<void, std::string> UpdateEngine::LoadFromFileMapped(
    const std::string& path, RelaxedPlanPermit permit) {
  // MAP_PRIVATE with PROT_WRITE: copy-on-write, so the file is never
  // modified through the mapping and a stray write costs one private page
  // instead of the package — the same protection the .incbin embedding's
  // writable data section gives (the runtime re-seals the plan before
  // merge and commit either way). The pages are page-aligned and writable,
  // which is what the Metal backend asks before wrapping the frozen
  // weights zero-copy; the page cache backs them, so a multi-GB plan costs
  // no copy and no second resident image.
  auto mapped = platform::MapFile(path, platform::MapAccess::kCopyOnWrite,
                                  /*require_regular=*/true);
  if (!mapped) {
    const platform::MapFailure f = mapped.error();
    if (f.stage == platform::MapFailure::kSize)
      return diag::executing::Error("cannot size the update plan '" + path +
                                    "' (not a non-empty regular file)");
    return diag::executing::Error(
        std::string(f.stage == platform::MapFailure::kOpen ? "cannot open"
                                                           : "cannot map") +
        " the update plan '" + path + "' (" + std::strerror(f.err) + ")");
  }
  platform::FileMapping candidate = std::move(*mapped);
  // Validate the candidate first: a rejected file must leave the previous
  // plan (borrowed, owned or mapped) exactly as it was. The candidate's
  // destructor unmaps it on the error path; the error names the contract
  // that refused the bytes, and the caller names the file.
  if (auto r = Initialize(candidate.data, candidate.bytes, permit); !r)
    return r;
  owned_plan_.clear();
  owned_plan_.shrink_to_fit();
  plan_map_ = std::move(candidate);  // releases the previous mapping
  return {};
}

std::expected<void, std::string> UpdateEngine::Initialize(
    const uint8_t* plan, size_t plan_size, RelaxedPlanPermit permit) {
  // Everything below validates into locals; engine members are only
  // assigned in the commit block at the end, after every contract has
  // passed. A failed re-Load must leave the previous plan fully usable —
  // not a half-overwritten mixture of two plans that the `if (!arena_)`
  // guards would happily execute.
  // The loader (S7 #153): every container and executor rule, the decode
  // and the scans, then the one arena — all into locals; members are only
  // assigned in the commit block at the end.
  auto decoded = DecodePlan(plan, plan_size, permit);
  if (!decoded) return std::unexpected(decoded.error());
  DecodedPlan& d = *decoded;
  const up::PlanHeader& header = d.header;
  auto allocated = AllocateArena(header, plan);
  if (!allocated) return std::unexpected(allocated.error());
  uint8_t* arena = allocated->data;
  const size_t arena_bytes = allocated->bytes;

  // Executor boundary, second half: the backend must accept the address
  // spaces (a GPU backend wraps them as device-visible buffers here). A
  // refusal releases the candidate arena and re-binds the previous plan, so
  // a rejected re-Load still leaves the old plan fully executable.
  if (auto r =
          backend_->Bind(arena, arena_bytes, plan + header.rodata_offset,
                         header.rodata_size, plan_size - header.rodata_offset);
      !r) {
    std::free(arena);
    // The contract (backend.h) keeps the previous binding intact on a
    // refused Bind; re-binding is belt and braces. If even that fails the
    // engine must not claim a loaded plan over an unbound backend.
    if (arena_) {
      if (auto again =
              backend_->Bind(arena_, arena_bytes_, rodata_, header_.rodata_size,
                             plan_size_ - header_.rodata_offset);
          !again) {
        std::free(arena_);
        arena_ = nullptr;
        plan_ = nullptr;
        return diag::executing::Error(
            "backend '" + std::string(backend_->name()) +
            "' cannot bind the plan (" + r.error() +
            ") and lost the previous one (" + again.error() + ")");
      }
    }
    return diag::executing::Error("backend '" + std::string(backend_->name()) +
                                  "' cannot bind the plan: " + r.error());
  }

  // The header's kernel policy (the GEMM tiles, v11) reaches the backend
  // once the binding is accepted; the contract above proved it valid.
  backend_->Configure(KernelPolicyOf(header));

  // Commit — nothing below can fail.
  std::free(arena_);
  arena_ = arena;
  arena_bytes_ = arena_bytes;
  header_ = header;
  metadata_ = std::move(d.metadata);
  train_program_ = std::move(d.train);
  merge_program_ = std::move(d.merge);
  eval_program_ = std::move(d.eval);
  step_program_ = std::move(d.step);
  grad_accum_ = std::max<uint64_t>(1, header.grad_accum_steps);
  // The step-latency split's phase boundaries (the profiler's, #202).
  profiler_.Bind(train_program_);
  emit_table_ = std::move(d.emit);
  num_classes_ = d.num_classes;
  vocab_bound_ = d.vocab_bound;
  eval_probs_ref_ = d.eval_probs_ref;
  eval_softmax_rows_ = d.eval_softmax_rows;
  eval_softmax_cols_ = d.eval_softmax_cols;
  eval_tail_ = d.eval_tail;
  // A new plan needs its own source binding: the old file (if any) was
  // verified against another plan's hash.
  if (source_.data) (void)backend_->BindSource(nullptr, 0);
  source_.Release();
  source_required_ = d.source_required;
  train_reads_source_ = d.train_reads_source;
  merge_reads_source_ = d.merge_reads_source;
  plan_ = plan;
  plan_size_ = plan_size;
  rodata_ = plan + header.rodata_offset;
  step_ = 0;
  horizon_ = 0;
  merged_ = false;
  binding_ = {};
  gate_.Reset();
  windows_ = LossWindows{};
  resume_evaluations_ = 0;
  return {};
}

std::expected<void, std::string> UpdateEngine::Execute(
    const std::vector<up::UpdateInstruction>& program) {
  return ExecuteRange(program, 0, program.size());
}

/// The optimizer program under gradient accumulation; the profiler charges
/// it to the optimizer phase of the step-timing split (#202).
std::expected<void, std::string> UpdateEngine::ExecuteStepProgram() {
  return profiler_.StepProgram([&] { return Execute(step_program_); });
}

/// The training stream. The profiler (#202) splits it at the phase
/// boundaries it scanned at bind and clocks each phase in -DSEEML_STEP_TIMING
/// builds (the benchmark harness); every other build runs it as one range,
/// so the shipped runtime never reads a clock inside the step.
std::expected<void, std::string> UpdateEngine::ExecuteTrainProgram() {
  return profiler_.TrainProgram(train_program_.size(),
                                [&](size_t begin, size_t end) {
    return ExecuteRange(train_program_, begin, end);
  });
}

std::expected<void, std::string> UpdateEngine::ExecuteRange(
    const std::vector<up::UpdateInstruction>& program, size_t begin,
    size_t end) {
  // The per-step scalars the optimizer instructions read: the scheduled
  // learning rate is a pure function of (header_, step_), evaluated once
  // per range — every instruction of one execution sees the same step.
  const StepParams params{.lr = EffectiveLr(),
                          .beta1 = header_.beta1,
                          .beta2 = header_.beta2,
                          .eps = header_.eps,
                          .weight_decay = header_.weight_decay,
                          .step = step_};
  // The backend sees the range before its first instruction — and the
  // whole program around it, so what it plans across instructions is
  // decided from everything that follows (the Metal backend's low-rank
  // fold), not from a phase of the step-timing split: where the source
  // model is off the device, a range that reads it runs on the CPU whole
  // (BeginRange), and EndRange follows the Flush below.
  backend_->BeginRange(std::span<const up::UpdateInstruction>(program), begin,
                       end);
  for (size_t idx = begin; idx < end; ++idx)
    if (auto r = backend_->Execute(program[idx], params); !r) {
      backend_->EndRange();
      return diag::executing::Error(
          "backend '" + std::string(backend_->name()) +
          "' failed at instruction " + std::to_string(idx) + ": " + r.error());
    }
  // Every range ends coherent: the engine reads the loss slot, the eval
  // probabilities, the merge deltas and the persistent segment straight off
  // the arena after Execute returns, and the step-timing split (below)
  // must charge deferred GPU work to the phase that issued it.
  auto flushed = backend_->Flush();
  backend_->EndRange();
  if (!flushed)
    return diag::executing::Error("backend '" + std::string(backend_->name()) +
                                  "' failed to complete: " + flushed.error());
  return {};
}

float UpdateEngine::LossValue() const { return *ReadPtr(header_.loss_ref); }

float UpdateEngine::EffectiveLr() const {
  const float base = header_.lr;
  if (static_cast<up::LrSchedule>(header_.lr_schedule) !=
      up::LrSchedule::kCosineWithWarmup)
    return base;
  // Linear warmup over warmup_steps, cosine decay to lr*min_lr_factor across
  // the RUN's horizon (horizon(): the steps this update actually trains,
  // E8 #91 — not the plan's compile-time budget); clamped at the floor
  // beyond it.
  if (header_.warmup_steps > 0 && step_ <= header_.warmup_steps)
    return base * static_cast<float>(step_) /
           static_cast<float>(header_.warmup_steps);
  const uint64_t run = this->horizon();
  const uint64_t horizon =
      run > header_.warmup_steps ? run - header_.warmup_steps : 0;
  const float floor = base * header_.min_lr_factor;
  if (horizon == 0 || step_ - header_.warmup_steps >= horizon) return floor;
  const float t = static_cast<float>(step_ - header_.warmup_steps) /
                  static_cast<float>(horizon);
  constexpr float kPi = 3.14159265358979323846f;
  return floor + (base - floor) * 0.5f * (1.0f + std::cos(kPi * t));
}

void UpdateEngine::ExecuteTrainOnce() {
  if (step_ == 0) step_ = 1;  // AdamW bias correction is 1-indexed
  // Same discipline as TrainImpl and LoadCheckpoint: the step moves the
  // adapter parameters, so deltas materialized by an earlier RunMerge are
  // stale and commit must re-merge first.
  merged_ = false;
  profiler_.CountStep();
  if (auto r = ExecuteTrainProgram(); !r) {
    // A verification hook, not a product path: the CPU backend never fails,
    // and a GPU failure here is a bug the test must not paper over.
    std::fprintf(stderr, "ExecuteTrainOnce: %s\n", r.error().c_str());
    std::abort();
  }
}

std::expected<void, std::string> UpdateEngine::ValidateDataset(
    Dataset& data) const {
  // Feeder boundary: the corpus must match the compiled plan's geometry.
  return VerifyFeederContract(header_, data, num_classes_, vocab_bound_);
}

std::expected<float, std::string> UpdateEngine::Evaluate(Dataset& data) {
  auto m = EvaluateMetrics(data);
  if (!m) return std::unexpected(m.error());
  return m->loss;
}

// Every evaluation is clocked (outside the step: the shipped runtime's
// step reads no clock) so the report can say what the gate cost (F9).
std::expected<EvalMetrics, std::string> UpdateEngine::EvaluateMetrics(
    Dataset& data) {
  const auto t0 = std::chrono::steady_clock::now();
  auto m = EvaluateMetricsImpl(data);
  eval_seconds_ +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return m;
}

std::expected<EvalMetrics, std::string> UpdateEngine::EvaluateMetricsImpl(
    Dataset& data) {
  if (!arena_) return diag::executing::Error("no plan loaded");
  if (eval_program_.empty())
    return diag::executing::Error("plan carries no eval program");
  if (source_required_ > 0 && !source_.data)
    return diag::executing::Error(
        "this plan scores the shipped model: its eval program reads the "
        "student's f32 weights from the source model file — bind it with "
        "BindSourceModel before evaluating");
  if (auto r = ValidateDataset(data); !r) return std::unexpected(r.error());

  // Every evaluation of a given set must score the same sample sequence:
  // the validation gate compares a pre- and post-training Evaluate, and a
  // cursor left mid-set by the previous pass would make the two passes
  // weight different samples. Rewind alone is not enough for a shuffled
  // set — a pass whose final partial batch wraps advances the permutation
  // epoch, so the next pass would iterate a different order and
  // double-count a different wrapped tail. Snapshot-and-restore pins the
  // epoch too: the restore replays the permutation deterministically, so
  // every pass over this set scores the identical multiset.
  const Dataset::ServingPos entry_pos = data.SaveServingPos();
  data.Rewind();

  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  // Accuracy needs class labels staged in the label slot and a softmax in
  // the eval program whose row count covers the batch.
  const bool track_accuracy =
      header_.label_kind == 1 && label_slot != nullptr &&
      eval_probs_ref_ != up::kNullRef && eval_softmax_cols_ > 0 &&
      eval_softmax_rows_ >= header_.batch;

  // One pass over the set in compiled-batch chunks (final partial batch
  // wraps — the fixed-shape contract admits no ragged batch). The feeder
  // stages batch b+1 while batch b evaluates, exactly as TrainImpl
  // pipelines; the batch sequence is identical to the serial one, and the
  // eval program never mutates the dataset. Scoped so the feeder joins
  // before the caller sees the dataset single-threaded again.
  // A token corpus's sample unit is one RECORD of seq_len rows; a feature
  // corpus's is one row. All serving arithmetic below counts samples.
  const uint64_t rows_per_sample =
      header_.input_kind == 1 ? header_.seq_len : 1;
  const uint64_t samples_per_step = header_.batch / rows_per_sample;
  const uint64_t batches = std::max<uint64_t>(
      1, (data.num_samples() + samples_per_step - 1) / samples_per_step);
  double total = 0.0;
  uint64_t correct = 0, counted = 0;
  // The final batch wraps when the set is not a whole number of batches:
  // its last rows are duplicates of samples already scored. Its loss is
  // then taken over the real rows alone (G13, #97) — exactly, from the
  // row-separable loss's own operands, or by real-row weight for a
  // composite loss. A set of whole batches keeps the plain mean of the
  // batch losses, bit for bit — except under the masked loss (S3 #149),
  // whose batch means are each over that batch's own target count: a mean
  // of those would weight a batch of ten targets like one of a hundred and
  // change with how the records fall into batches, so a masked plan's set
  // is always scored over its target rows, as PyTorch's ignore_index mean
  // over the set is.
  const bool ragged = data.num_samples() % samples_per_step != 0;
  const bool per_row = ragged || eval_tail_.kind == EvalTail::kXentMasked;
  double real_row_loss = 0.0;  // sum of per-row losses over real rows
  uint64_t real_rows = 0;
  {
    BatchPipeline feeder(data, header_.batch, header_.input_floats,
                         header_.label_kind == 0 ? 0 : header_.label_bytes);
    for (uint64_t b = 0; b < batches; ++b) {
      feeder.NextBatch(input_slot, label_slot);
      if (auto r = Execute(eval_program_); !r) {
        data.RestoreServingPos(entry_pos);
        return std::unexpected(r.error());
      }
      total += LossValue();
      if (per_row) {
        const uint64_t served_samples = b * samples_per_step;
        const uint64_t real_here =
            std::min<uint64_t>(samples_per_step,
                               data.num_samples() - served_samples) *
            rows_per_sample;
        uint64_t counted_here = 0;
        real_row_loss += RealRowLoss(real_here, label_slot, &counted_here);
        real_rows += counted_here;
      }
      if (track_accuracy) {
        // Rows past the dataset's tail in the final batch are wrapped
        // duplicates — the loss's fixed-shape mean cannot exclude them, but
        // accuracy can and does: each real sample is judged exactly once.
        // NextBatch copied the staged batch into the slots before Execute,
        // so label_slot still holds THIS batch while the feeder stages the
        // next one internally.
        const uint64_t served = b * samples_per_step;
        const uint64_t real_samples = std::min<uint64_t>(
            samples_per_step,
            data.num_samples() > served ? data.num_samples() - served : 0);
        // Every ROW of a real sample is judged (token records carry
        // seq_len positions each); wrapped duplicate samples are excluded.
        const uint64_t real = real_samples * rows_per_sample;
        const float* probs = ReadPtr(eval_probs_ref_);
        const auto* labels = reinterpret_cast<const int32_t*>(label_slot);
        // The argmax runs over the standard chunk geometry (E3, #82): it
        // was a serial scan of rows x classes — at a 49k vocabulary the
        // dominant eval cost — behind a GEMM that had used every core.
        // Per-chunk integer tallies summed afterwards: exact at any width.
        const uint64_t cols = eval_softmax_cols_;
        // A masked plan (S3 #149) judges the target rows alone: a row
        // whose label is -1 is neither a hit nor counted. A plain plan
        // never stages -1, so every real row counts, as before.
        uint64_t tallies[up::kMaxParallelChunks] = {};
        uint64_t judged[up::kMaxParallelChunks] = {};
        up::ParallelFor(real, kernels::RowGrain(cols, kernels::kGrainCheap),
                        [&](size_t r0, size_t r1, size_t chunk) {
          uint64_t hits = 0, rows = 0;
          for (size_t r = r0; r < r1; ++r) {
            if (labels[r] < 0) continue;
            ++rows;
            const float* row = probs + r * cols;
            uint64_t argmax = 0;
            for (uint64_t c = 1; c < cols; ++c)
              if (row[c] > row[argmax]) argmax = c;
            if (static_cast<uint64_t>(labels[r]) == argmax) ++hits;
          }
          tallies[chunk] = hits;
          judged[chunk] = rows;
        });
        for (const uint64_t hits : tallies) correct += hits;
        for (const uint64_t rows : judged) counted += rows;
      }
    }
  }
  data.RestoreServingPos(entry_pos);
  EvalMetrics m;
  m.loss = static_cast<float>(
      per_row
          ? (real_rows ? real_row_loss / static_cast<double>(real_rows) : 0.0)
          : total / static_cast<double>(batches));
  if (track_accuracy && counted > 0) {
    m.accuracy = static_cast<float>(static_cast<double>(correct) /
                                    static_cast<double>(counted));
    m.has_accuracy = true;
  }
  return m;
}

std::expected<TrainReport, std::string> UpdateEngine::Train(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  auto report = TrainImpl(data, steps, options);
  // The diagnostics contract at the engine's main boundary: every error
  // must name a registered unit, or the failing process cannot be
  // delimited (mirrors the driver's Compile boundary).
  if (!report && !WellFormedDiagnostic(report.error()))
    return diag::executing::Error(
        "unattributed diagnostic escaped a subsystem: " + report.error());
  return report;
}

std::expected<bool, std::string> UpdateEngine::ResumeIfAsked(
    Dataset& data, const TrainOptions& options, bool explicit_steps) {
  // --resume is strict (CR4 #221). No file at the path is "nothing to
  // resume" and a fresh run, said so. A file that IS there and cannot be
  // resumed — unreadable, corrupt, another plan's, another run's — fails the
  // call: falling through to a fresh run would overwrite it at the first
  // periodic checkpoint, and would hand the caller a full-budget run it did
  // not ask for.
  bool from_checkpoint = false;
  if (options.resume && !options.checkpoint_path.empty()) {
    if (!CheckpointExists(options.checkpoint_path)) {
      std::fprintf(stderr,
                   "seeml-update: no checkpoint at '%s': nothing to resume, "
                   "starting a fresh run\n",
                   options.checkpoint_path.c_str());
    } else {
      // A refused resume leaves the engine — the segment, the step, the
      // horizon, the binding, the gate, the windows — and the feeder as
      // they were: the refusal is the call's only effect.
      const std::vector<uint8_t> held_segment(arena_,
                                              arena_ + header_.persistent_size);
      const uint64_t held_step = step_, held_horizon = horizon_;
      const RunBinding held_binding = binding_;
      const GateState held_gate = gate_;
      const LossWindows held_windows = windows_;
      const bool held_merged = merged_;
      const Dataset::ServingPos held_pos = data.SaveServingPos();
      if (auto r = ResumeFromCheckpoint(data, options, explicit_steps); !r) {
        std::memcpy(arena_, held_segment.data(), held_segment.size());
        step_ = held_step;
        horizon_ = held_horizon;
        binding_ = held_binding;
        gate_ = held_gate;
        windows_ = held_windows;
        merged_ = held_merged;
        data.RestoreServingPos(held_pos);
        return std::unexpected(r.error());
      }
      from_checkpoint = true;
    }
  }
  return from_checkpoint;
}

std::expected<uint64_t, std::string> UpdateEngine::DecideStepsAndHorizon(
    uint64_t steps, bool explicit_steps, const TrainOptions& options) {
  // The step count and the schedule horizon, decided AFTER the checkpoint
  // is in (E8, #91): until then a resume could not know how much of the
  // run was left, so it asked for a full default budget — all of it past
  // the old fixed horizon.
  const bool resumed = step_ > 0;
  if (!explicit_steps) {
    if (resumed && horizon_ > 0)
      steps = horizon_ > step_ ? horizon_ - step_ : 0;  // the remainder
    else
      steps = header_.default_steps;
    if (steps == 0 && !resumed)
      return diag::executing::Error(
          "no steps requested and the plan has no default");
  }
  if (options.horizon_steps != 0) {
    horizon_ = options.horizon_steps;
  } else if (explicit_steps || horizon_ == 0) {
    uint64_t end = 0;
    if (__builtin_add_overflow(step_, steps, &end))
      return diag::executing::Error("the requested step count overflows");
    horizon_ = end;
  }  // else: a resume of the remainder keeps the checkpoint's horizon
  if (horizon_ < step_ + steps && options.horizon_steps != 0)
    return diag::executing::Error("the requested horizon (" +
                                  std::to_string(horizon_) +
                                  " steps) ends before the run does (step " +
                                  std::to_string(step_ + steps) + ")");
  // The warmup rule (S4 #150, review N20), held where the run's true start
  // is known: a horizon inside the compiled warmup leaves the cosine phase
  // no steps, so the whole run sits on the ramp. The runner refuses a fresh
  // run's --steps early as a usage error; this covers what only the engine
  // can see — a resume whose step + steps still ends inside the ramp, and a
  // run that asked to resume but started fresh because its checkpoint did
  // not load.
  if (header_.lr_schedule ==
          static_cast<uint32_t>(up::LrSchedule::kCosineWithWarmup) &&
      header_.warmup_steps != 0 && horizon_ <= header_.warmup_steps)
    return diag::executing::Error(
        "the run's horizon (" + std::to_string(horizon_) +
        " steps) does not exceed the plan's compiled warmup (" +
        std::to_string(header_.warmup_steps) +
        " steps): the run would end inside the ramp");
  return steps;
}

std::expected<void, std::string> UpdateEngine::OpenValidation(
    const TrainOptions& options, bool track_best, TrainReport* report) {
  const bool resumed = step_ > 0;
  report->has_validation = true;
  if (resumed && gate_.has_val_initial) {
    // The gate's "before" is the SOURCE model's score — not the resumed
    // adapter's, which would be the baseline of an evaluation made here
    // and would reject an update that merely held steady over the resumed
    // segment. A resume from a checkpoint has just re-measured this number
    // on the source state and found the stored bits (CR4 #221); a second
    // Train call in one process carries the one its first call measured.
    report->val_initial_loss = gate_.val_initial_loss;
    report->has_val_accuracy = gate_.val_initial_has_accuracy;
    report->val_initial_accuracy = gate_.val_initial_accuracy;
  } else {
    auto v = EvaluateMetrics(*options.validation);
    if (!v) return std::unexpected(v.error());
    report->val_initial_loss = v->loss;
    report->has_val_accuracy = v->has_accuracy;
    report->val_initial_accuracy = v->accuracy;
    if (resumed)
      std::fprintf(stderr,
                   "seeml-update: note — the run so far measured no "
                   "source-model score; the gate's baseline is the "
                   "current state's loss %.6f\n",
                   v->loss);
    gate_.has_val_initial = true;
    gate_.val_initial_loss = v->loss;
    gate_.val_initial_has_accuracy = v->has_accuracy;
    gate_.val_initial_accuracy = v->accuracy;
  }
  if (track_best) {
    report->best_tracked = true;
    // The bar to beat is the score of the state the run starts from:
    // the source model's on a fresh run, the checkpoint's best on a
    // resume. Until something beats it, the best state IS this one —
    // labelled with ITS OWN score. A resume whose checkpoint carries the
    // source model's score but no best (the first run did not track)
    // starts from the resumed segment, not the source: scoring it as
    // the source would let any later state between the two displace it.
    if (!gate_.has_best) {
      float start_loss = report->val_initial_loss;
      float start_accuracy = report->val_initial_accuracy;
      if (resumed) {
        auto v = EvaluateMetrics(*options.validation);
        if (!v) return std::unexpected(v.error());
        start_loss = v->loss;
        start_accuracy = v->accuracy;
      }
      gate_.Snapshot(arena_, header_.persistent_size, step_, start_loss,
                     start_accuracy);
    }
  }
  return {};
}

std::expected<void, std::string> UpdateEngine::RunSteps(
    Dataset& data, uint64_t steps, const TrainOptions& options, bool track_best,
    bool from_checkpoint, TrainReport* report) {
  const uint64_t eval_every = report->eval_every;
  float* input_slot = WritePtr(header_.input_ref);
  uint8_t* label_slot =
      header_.label_kind == 0
          ? nullptr
          : reinterpret_cast<uint8_t*>(WritePtr(header_.label_ref));

  // The train-loss gate's windows (runtime/gating/gate.h). The last window
  // is over the steps actually run, kept as a ring: a run stopped early
  // (patience, should_stop) ends at its own last steps, not at the requested
  // run's — which it never reached. They are the RUN's (CR4 #221): a call
  // that resumed a checkpoint carries on with the windows the checkpoint
  // holds — the first run's size, its first window, its ring — so the gate
  // weighs the whole run, and a finished run resumed weighs what it weighed.
  // Any other call starts them, sized by its own step count, as before.
  if (!(from_checkpoint && windows_.live())) windows_.Start(steps);
  if (options.record_loss_curve) report->loss_curve.reserve(steps);

  // N12 (2026-11-16 review, D4): the checkpoint of the evaluation that
  // spent the patience is written before the loop below breaks, so a resume
  // from it starts with the patience already spent. It stops where the
  // uninterrupted run stopped — no step, straight to the bracketing
  // evaluation and the best-state restore that run made — rather than
  // training on to its next evaluation, where it could adopt a best state
  // the uninterrupted run never saw.
  if (track_best && options.patience > 0 &&
      gate_.stale_evals >= options.patience) {
    report->stopped_by_patience = true;
    steps = 0;
  }
  const uint64_t start = step_;
  uint64_t executed = 0;
  double step_seconds = 0.0;  // the steps alone (S0 #146)
  // Training is about to move the adapter parameters, so any previously
  // materialized merge deltas are stale: commit must be preceded by a fresh
  // RunMerge. Cleared before the first step, not after the loop — the error
  // exits inside the loop (non-finite loss, checkpoint failure) leave
  // mutated parameters behind too.
  merged_ = false;
  {
    // The feeder thread stages batch s+1 (shuffle gather + epoch reshuffles)
    // while step s computes; the batch sequence is exactly the serial one, so
    // pipelining never changes what is trained on. Scoped to the loop: the
    // destructor joins the feeder on every exit path — including the error
    // returns below — and the surrounding Evaluate() calls see the dataset
    // single-threaded again.
    BatchPipeline feeder(data, header_.batch, header_.input_floats,
                         header_.label_kind == 0 ? 0 : header_.label_bytes);
    for (uint64_t s = start; s < start + steps; ++s) {
      if (options.should_stop && options.should_stop()) {
        report->stopped_early = true;
        // #202 item 1: a stop (the host's SIGINT/SIGTERM) leaves a
        // checkpoint of the state it stopped at, so a resume finishes the
        // run the uninterrupted one ran. Written HERE, at the step boundary
        // the poll sits on and before the bracketing evaluation below could
        // snapshot a best at a step the uninterrupted run never evaluated or
        // restore one into the arena — either would put a different run in
        // the checkpoint (N12's failure, again). Skipped when this step's
        // periodic checkpoint already holds this state, or when no step ran.
        // The report says whether the file holds the state the run stopped
        // at (CR4 #221): it does once a step ran, and before the first step
        // only when this call resumed from that very file — a fresh run
        // stopped before its first step has written nothing.
        if (!options.checkpoint_path.empty()) {
          if (executed > 0 && !(options.checkpoint_every &&
                                step_ % options.checkpoint_every == 0)) {
            DurableWriteStats written;
            if (auto r = SaveCheckpoint(options.checkpoint_path, &written); !r)
              return std::unexpected(r.error());
            ++report->checkpoints;
            report->checkpoint_bytes += written.bytes;
            report->checkpoint_fsyncs += written.fsyncs;
          }
          report->stop_checkpointed = executed > 0 || from_checkpoint;
        }
        break;
      }
      step_ = s + 1;  // 1-indexed timestep for AdamW bias correction
      // Gradient accumulation: G grad executions (one micro-batch each,
      // folded into the persistent accumulators), then the step program
      // once. The reported loss is the mean over the G micro-batch losses
      // — the loss of the effective batch. With G == 1 the grad program IS
      // the whole step and the loop body is the classic one.
      // An error return from inside this loop leaves the accumulators
      // partially folded; a failed Train() ends the engine's usable state
      // (checkpoints are written only after the step program), so no
      // resume ever starts from them.
      const auto step_t0 = std::chrono::steady_clock::now();
      double loss_sum = 0.0;
      for (uint64_t g = 0; g < grad_accum_; ++g) {
        feeder.NextBatch(input_slot, label_slot);
        if (auto r = ExecuteTrainProgram(); !r)
          return std::unexpected(r.error());
        loss_sum += LossValue();
      }
      if (!step_program_.empty()) {
        if (auto r = ExecuteStepProgram(); !r)
          return std::unexpected(r.error());
      }
      step_seconds += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - step_t0)
                          .count();
      ++executed;
      // one optimizer step: G grad executions + the step
      profiler_.CountStep();

      const float loss =
          static_cast<float>(loss_sum / static_cast<double>(grad_accum_));
      // A non-finite loss means the parameters (and any AdamW moments) are
      // already poisoned; continuing can only burn energy. Fail the update —
      // the source model on disk is untouched by construction.
      if (!std::isfinite(loss))
        return diag::executing::Error("loss became non-finite at step " +
                                      std::to_string(step_) +
                                      " — aborting the update");
      if (options.record_loss_curve) report->loss_curve.push_back(loss);
      windows_.Observe(loss);
      if (options.log_every && (s - start) % options.log_every == 0)
        std::fprintf(stderr, "seeml-update: step %llu  loss %.6f\n",
                     static_cast<unsigned long long>(step_), loss);
      // Periodic evaluation (E9): every eval_every steps, except the last
      // one, which the bracketing evaluation below scores. Before the
      // checkpoint, so a checkpoint written this step carries the verdict.
      // The eval program writes transients only, and the train program
      // writes every transient it reads, so this changes no training bit.
      bool out_of_patience = false;
      // Keyed on the GLOBAL step: a resume re-enters mid-run, and keying
      // on this call's count would evaluate at different steps than the
      // uninterrupted run — and so, possibly, keep a different best.
      if (track_best && step_ % eval_every == 0 && s + 1 < start + steps) {
        auto v = EvaluateMetrics(*options.validation);
        if (!v) return std::unexpected(v.error());
        ++report->evaluations;
        out_of_patience = gate_.Observe(arena_, header_.persistent_size, step_,
                                        v->loss, v->accuracy, options.patience);
        if (options.log_every)
          std::fprintf(stderr,
                       "seeml-update: step %llu  validation loss %.6f "
                       "(best %.6f at step %llu)\n",
                       static_cast<unsigned long long>(step_), v->loss,
                       gate_.best_loss,
                       static_cast<unsigned long long>(gate_.best_step));
      }
      if (options.checkpoint_every && !options.checkpoint_path.empty() &&
          step_ % options.checkpoint_every == 0) {
        DurableWriteStats written;
        if (auto r = SaveCheckpoint(options.checkpoint_path, &written); !r)
          return std::unexpected(r.error());
        ++report->checkpoints;
        report->checkpoint_bytes += written.bytes;
        report->checkpoint_fsyncs += written.fsyncs;
      }
      if (out_of_patience) {
        report->stopped_by_patience = true;
        break;
      }
    }
  }
  report->steps = executed;
  report->step_seconds = step_seconds;
  return {};
}

std::expected<void, std::string> UpdateEngine::CloseValidation(
    const TrainOptions& options, bool track_best, TrainReport* report) {
  auto v = EvaluateMetrics(*options.validation);
  if (!v) return std::unexpected(v.error());
  report->val_last_loss = v->loss;
  report->val_last_accuracy = v->accuracy;
  if (track_best) {
    if (v->loss < gate_.best_loss)
      gate_.Snapshot(arena_, header_.persistent_size, step_, v->loss,
                     v->accuracy);
    // Leave the best evaluated state in the arena: it is what RunMerge
    // and CommitToModel will ship. A checkpoint written after this
    // carries it as both payloads.
    if (gate_.RestoreBest(arena_, step_)) {
      if (options.log_every)
        std::fprintf(stderr,
                     "seeml-update: restored the best state (step %llu, "
                     "validation loss %.6f); the last state (step %llu) "
                     "scored %.6f\n",
                     static_cast<unsigned long long>(gate_.best_step),
                     gate_.best_loss, static_cast<unsigned long long>(step_),
                     v->loss);
    }
    report->best_step = gate_.best_step;
    report->val_final_loss = gate_.best_loss;
    report->val_final_accuracy = gate_.best_accuracy;
  } else {
    report->val_final_loss = v->loss;
    report->val_final_accuracy = v->accuracy;
  }
  return {};
}

std::expected<TrainReport, std::string> UpdateEngine::TrainImpl(
    Dataset& data, uint64_t steps, const TrainOptions& options) {
  if (!arena_) return diag::executing::Error("no plan loaded");
  // An adapter-only plan (v21) reads its frozen weights from the source
  // model file in the train program too: unbound, the backends would read
  // through no mapping (the CPU) or the arena standing in for the source
  // buffer (Metal). Refused the way Evaluate refuses, before anything runs.
  if (train_reads_source_ && !source_.data)
    return diag::executing::Error(
        "this plan reads its frozen weights from the source model file in "
        "the train program (an adapter-only plan) — bind it with "
        "BindSourceModel before training");
  const bool explicit_steps = steps != 0;
  // Every evaluation of this call is clocked, a resume's re-measurements
  // included.
  eval_seconds_ = 0.0;
  resume_evaluations_ = 0;
  if (auto r = ValidateDataset(data); !r) return std::unexpected(r.error());
  if (options.validation)
    if (auto r = ValidateDataset(*options.validation); !r)
      return std::unexpected(r.error());

  const auto from_checkpoint = ResumeIfAsked(data, options, explicit_steps);
  if (!from_checkpoint) return std::unexpected(from_checkpoint.error());
  const auto decided = DecideStepsAndHorizon(steps, explicit_steps, options);
  if (!decided) return std::unexpected(decided.error());
  steps = *decided;

  // The run binding every checkpoint of this run will carry.
  binding_ = BindingOf(data, options.validation);

  // Periodic evaluation (E9): "auto" resolves against the RUN — its
  // horizon, already decided above — never against this call's step count,
  // so a resumed run evaluates on the cadence the uninterrupted one did.
  const uint64_t eval_every =
      !options.validation ? 0
      : options.eval_every == TrainOptions::kEvalEveryAuto
          ? std::max<uint64_t>(1, horizon_ / 10)
          : options.eval_every;
  const bool track_best = options.validation && eval_every > 0;

  TrainReport report;
  report.eval_every = eval_every;
  report.patience = track_best ? options.patience : 0;
  report.resumed_step = *from_checkpoint ? step_ : 0;
  report.resume_evaluations = resume_evaluations_;
  if (options.validation)
    if (auto r = OpenValidation(options, track_best, &report); !r)
      return std::unexpected(r.error());
  if (auto r =
          RunSteps(data, steps, options, track_best, *from_checkpoint, &report);
      !r)
    return std::unexpected(r.error());
  report.initial_avg_loss = windows_.FirstMean();
  report.final_avg_loss = windows_.LastMean();

  // A stopped run is not weighed (#202 item 1): the host commits nothing and
  // the resume re-enters from the checkpoint written at the stop, so the
  // bracketing evaluation would cost a validation pass for a verdict no one
  // reads — and its best-state restore would leave the arena holding a state
  // other than the one checkpointed.
  if (options.validation && !report.stopped_early)
    if (auto r = CloseValidation(options, track_best, &report); !r)
      return std::unexpected(r.error());
  report.eval_seconds = eval_seconds_;
  return report;
}

double UpdateEngine::RealRowLoss(uint64_t real, const uint8_t* label_slot,
                                 uint64_t* counted) const {
  // The sum of the per-row losses of the batch just evaluated over its
  // first `real` rows — each row's loss evaluated with the loss kernel's
  // own expression, so a full batch sums to rows x the kernel's mean up to
  // the order of a double sum. `counted` is the rows the sum is over: the
  // real rows, or for a masked loss (S3 #149) the real TARGET rows.
  *counted = real;
  switch (eval_tail_.kind) {
    case EvalTail::kXent:
    case EvalTail::kXentMasked: {
      const float* probs = ReadPtr(eval_tail_.a);
      const auto* labels = reinterpret_cast<const int32_t*>(label_slot);
      double sum = 0.0;
      uint64_t rows = 0;
      for (uint64_t r = 0; r < real; ++r) {
        if (labels[r] < 0) continue;
        ++rows;
        const float p =
            probs[r * eval_tail_.width + static_cast<uint64_t>(labels[r])];
        const double safe = std::isnan(p)
                                ? static_cast<double>(p)
                                : static_cast<double>(std::fmax(p, 1e-12f));
        sum -= std::log(safe);
      }
      *counted = rows;
      return sum;
    }
    case EvalTail::kMse: {
      // MseFwd is the mean over every element; a row's share is the sum of
      // its squared differences over the elements per row, scaled so the
      // real rows' mean is the mean over their elements.
      const float* pred = ReadPtr(eval_tail_.a);
      const float* target = ReadPtr(eval_tail_.b);
      const uint64_t w = eval_tail_.width;
      double sum = 0.0;
      for (uint64_t i = 0; i < real * w; ++i) {
        const double d = static_cast<double>(pred[i]) - target[i];
        sum += d * d;
      }
      return sum / static_cast<double>(w);
    }
    case EvalTail::kWeighted:
      break;
  }
  // A composite loss keeps no per-row values: the batch's mean stands for
  // each of its real rows (the duplicates' influence on that mean remains;
  // documented in docs/runtime.md).
  return static_cast<double>(LossValue()) * static_cast<double>(real);
}

std::expected<void, std::string> UpdateEngine::RunMerge() {
  if (!arena_) return diag::executing::Error("no plan loaded");
  if (merge_reads_source_ && !source_.data)
    return diag::executing::Error(
        "this plan's merge program reads the source model file — bind it "
        "with BindSourceModel before merging");
  // Re-seal the plan before its frozen weights are merged and committed.
  // The blob is read-only by contract but lives in ordinary memory — in a
  // packed Apple package on writable pages, so the GPU can share them —
  // and a stray write since load must fail here, not ship as corrupted
  // deltas. One parallel hash of the plan, once per update.
  if (up::PlanSelfHash(plan_, plan_size_,
                       offsetof(up::PlanHeader, plan_hash)) !=
      header_.plan_hash)
    return diag::executing::Error(
        "plan bytes changed since load — refusing to merge");
  if (auto r = Execute(merge_program_); !r) return r;
  // Finiteness gate on the materialized deltas. The training loop's loss
  // guard reads the loss written BEFORE each step's backward + optimizer,
  // so a gradient that overflows on the final executed step can poison the
  // adapters with no later loss read — and the clamped xent/KL losses can
  // never go infinite themselves. With a validation split the regression
  // gate rejects the NaN; without one, this scan is the last line between
  // a poisoned delta and the committed model file.
  for (const up::EmitEntry& e : emit_table_) {
    const auto* delta = reinterpret_cast<const float*>(arena_ + e.arena_offset);
    const size_t count = e.byte_size / sizeof(float);
    for (size_t i = 0; i < count; ++i)
      if (!std::isfinite(delta[i]))
        return diag::executing::Error(
            "merged delta contains a non-finite value — the adapters are "
            "numerically poisoned; refusing to stage a commit");
  }
  merged_ = true;
  return {};
}

std::expected<void, std::string> UpdateEngine::VerifySourceModel(
    const std::string& source_model_path) const {
  if (header_.source_model_hash == 0) return {};
  auto hash = HashFileContent(source_model_path);
  if (!hash) return std::unexpected(hash.error());
  if (*hash != header_.source_model_hash)
    return diag::executing::Error(
        "source model does not match the plan's "
        "source_model_hash — refusing to run against '" +
        source_model_path + "'");
  return {};
}

std::expected<void, std::string> UpdateEngine::WriteCommittedModelReport(
    const std::string& out_path) const {
  if (header_.tokenizer_hash == 0) return {};
  auto hash = HashFileContent(out_path);
  if (!hash) return std::unexpected(hash.error());
  const bool parity = metadata_.present && (metadata_.meta.export_flags &
                                            update::kExportTokenizerParity);
  char text[512];
  const int n =
      std::snprintf(text, sizeof(text),
                    "{\n \"schema\": 1,\n \"smf_hash\": \"%016llx\",\n"
                    " \"tokenizer_hash\": \"%016llx\",\n"
                    " \"tokenizer_parity_checked\": %s,\n"
                    " \"tokenizer_parity_ids\": %llu,\n"
                    " \"logits_parity_checked\": false,\n"
                    " \"logits_parity_max_delta\": 0.0\n}\n",
                    static_cast<unsigned long long>(*hash),
                    static_cast<unsigned long long>(header_.tokenizer_hash),
                    parity ? "true" : "false",
                    static_cast<unsigned long long>(
                        parity ? metadata_.meta.tokenizer_parity_ids : 0));
  return WriteFileDurable(out_path + ".export.json",
                          reinterpret_cast<const uint8_t*>(text),
                          static_cast<size_t>(n));
}

std::expected<void, std::string> UpdateEngine::CommitToModel(
    const std::string& source_model_path, const std::string& out_path) const {
  if (!merged_) return diag::executing::Error("RunMerge() must precede commit");

  // Serialize committers: without the lock, two updates targeting the same
  // output race at the atomic rename and the loser's work silently
  // disappears. The lock dies with the process, so a crash cannot wedge
  // future commits.
  auto lock = CommitLock::Acquire(out_path);
  if (!lock) return std::unexpected(lock.error());

  // Copy-on-write sidecar first, then verify the identity of the *copy* —
  // the exact bytes about to be patched — so no window exists between the
  // check and the patch. The whole path streams in bounded chunks: commit
  // memory is O(chunk), never O(model).
  auto edit = DurableFileEdit::Begin(source_model_path, out_path);
  if (!edit) return std::unexpected(edit.error());

  // Identity check before any byte moves: the emit table's offsets are only
  // meaningful inside the exact file the plan was compiled from. A same-sized
  // different model would otherwise be silently corrupted.
  if (header_.source_model_hash != 0) {
    auto hash = HashFileContent(edit->sidecar_path());
    if (!hash) return std::unexpected(hash.error());
    if (*hash != header_.source_model_hash)
      return diag::executing::Error(
          "source model does not match the plan's "
          "source_model_hash — refusing to patch '" +
          source_model_path + "'");
  }

  // Apply every adapter delta to its f32 weight range: W' = W + Δ. The
  // file's pristine weights are the base, so quantized plans commit no
  // quantization error. Read-modify-write in bounded chunks; the buffer is
  // float-typed, so alignment holds regardless of the file offset.
  constexpr uint64_t kPatchChunkFloats = 1u << 16;  // 256 KiB per chunk
  std::vector<float> buf;
  for (const up::EmitEntry& e : emit_table_) {
    if (!RangeOk(e.smf_data_offset, e.byte_size, edit->size()) ||
        e.byte_size % sizeof(float) != 0)
      return diag::executing::Error(
          "emit entry exceeds the source model file — plan and "
          "model are out of sync");
    const auto* delta = reinterpret_cast<const float*>(arena_ + e.arena_offset);
    const uint64_t count = e.byte_size / sizeof(float);
    for (uint64_t i = 0; i < count; i += kPatchChunkFloats) {
      const uint64_t n = std::min<uint64_t>(kPatchChunkFloats, count - i);
      buf.resize(n);
      const uint64_t at = e.smf_data_offset + i * sizeof(float);
      if (auto r = edit->ReadAt(at, reinterpret_cast<uint8_t*>(buf.data()),
                                n * sizeof(float));
          !r)
        return std::unexpected(r.error());
      for (uint64_t j = 0; j < n; ++j) buf[j] += delta[i + j];
      if (auto r =
              edit->WriteAt(at, reinterpret_cast<const uint8_t*>(buf.data()),
                            n * sizeof(float));
          !r)
        return std::unexpected(r.error());
    }
  }

  // Transactional and durable: fsync'd sidecar, atomic rename, dir fsync.
  return edit->Commit();
}

uint32_t UpdateEngine::CheckpointBackend() const {
  return backend_kind_ == BackendKind::kMetal ? kCkptBackendMetal
                                              : kCkptBackendCpu;
}

UpdateEngine::RunBinding UpdateEngine::BindingOf(
    const Dataset& data, const Dataset* validation) const {
  RunBinding b;
  b.bound = true;
  b.shuffle_origin = data.shuffle_origin();
  b.train_samples = data.num_samples();
  b.val_samples = validation ? validation->num_samples() : 0;
  b.corpus_hash = data.content_hash();
  b.split_hash = data.split_hash();
  b.backend = CheckpointBackend();
  return b;
}

// The strict resume (CR4 #221). The checkpoint is an input: the storage
// role has verified the file as a file (version, header hash, field bounds,
// length, plan, payload hashes); here it is held to the RUN — and the two
// numbers the gate will weigh are measured again instead of read.
std::expected<void, std::string> UpdateEngine::ResumeFromCheckpoint(
    Dataset& data, const TrainOptions& options, bool explicit_steps) {
  const std::string& path = options.checkpoint_path;
  auto refuse = [&](const std::string& why) {
    return diag::executing::Error("resume refused: '" + path + "' " + why);
  };
  if (auto r = LoadCheckpoint(path); !r)
    return refuse("cannot be resumed (" + r.error() + "); it is left as it is");
  auto hex = [](uint64_t v) {
    char buf[20];
    std::snprintf(buf, sizeof buf, "%016llx",
                  static_cast<unsigned long long>(v));
    return std::string(buf);
  };
  // A run writes no checkpoint before its first step, and every one it
  // writes carries its binding and its horizon.
  if (step_ == 0)
    return refuse(
        "holds step 0 (field 'step'): a run writes no checkpoint "
        "before its first step");
  if (!binding_.bound)
    return refuse(
        "carries no run binding (field 'flags'): it was not "
        "written by a training run, and a resume is held to the "
        "run's corpus, split, seed and backend");
  if (horizon_ == 0)
    return refuse("carries no horizon (field 'horizon_steps')");
  const RunBinding now = BindingOf(data, options.validation);
  // The backend first: the stored losses are one backend's bits, and the
  // re-measurement below compares bit for bit — which only the backend
  // that wrote them can reproduce. Refused by name rather than compared at
  // a tolerance: a tolerance is a band a wrong number fits in, and a run
  // that changes backend midway is neither backend's uninterrupted run.
  if (binding_.backend != now.backend)
    return refuse(std::string("was written on the ") +
                  (binding_.backend == kCkptBackendMetal ? "metal"
                   : binding_.backend == kCkptBackendCpu ? "cpu"
                                                         : "unstated") +
                  " backend and this run is on " + backend_name() +
                  " (field 'backend'): the gate's losses are re-measured "
                  "bit for bit, which holds on one backend only — resume "
                  "on the backend the run started on");
  // The run binding (v5): the checkpoint's segment was trained on ONE
  // shuffle stream over ONE train/validation split. A different seed or
  // split would silently move the boundary — training on the first run's
  // validation rows — so it is refused, not adjusted.
  if (binding_.shuffle_origin != now.shuffle_origin ||
      binding_.train_samples != now.train_samples ||
      binding_.val_samples != now.val_samples)
    return diag::executing::Error(
        "resume refused: the checkpoint was written on a different "
        "shuffle seed or train/validation split (" +
        std::to_string(binding_.train_samples) + " train / " +
        std::to_string(binding_.val_samples) +
        " validation samples); resume with the first run's --seed "
        "and --val-frac");
  // The corpus and the split themselves (v6): the same record counts are
  // not the same records. Both hashes are compared as they are — 0 is "the
  // dataset names none" (an in-memory corpus; a positional split), and a
  // named one never resumes an unnamed one.
  if (binding_.corpus_hash != now.corpus_hash)
    return refuse("was written over another corpus (content hash " +
                  hex(binding_.corpus_hash) + "; this run's is " +
                  hex(now.corpus_hash) +
                  ", field 'corpus_hash'): resume with the first run's "
                  "--data");
  if (binding_.split_hash != now.split_hash)
    return refuse("was written over another validation split (split hash " +
                  hex(binding_.split_hash) + "; this run's is " +
                  hex(now.split_hash) + ", field 'split_hash')");
  // The fields only the run bounds. A stated horizon on a resume of the
  // remainder must be the run's own; an explicit step count is a request
  // for more steps on a schedule stretched to cover them (E8), which
  // re-decides the horizon below.
  if (options.horizon_steps != 0 && !explicit_steps &&
      options.horizon_steps != horizon_)
    return refuse("has a horizon of " + std::to_string(horizon_) +
                  " steps and this run states " +
                  std::to_string(options.horizon_steps) +
                  " (field 'horizon_steps')");
  // Equality is the checkpoint of the evaluation that spent the patience
  // (N12); more than the run's patience is a count no run of it reaches.
  if (options.patience > 0 && gate_.stale_evals > options.patience)
    return refuse("counts " + std::to_string(gate_.stale_evals) +
                  " evaluations without a new best, past this run's "
                  "patience of " +
                  std::to_string(options.patience) + " (field 'stale_evals')");
  if (options.validation && !gate_.has_val_initial)
    return refuse(
        "carries no source-model validation score (field "
        "'val_initial_loss') though its run had a validation set");

  // The gate's numbers, measured again — before the feeder moves, so every
  // refusal comes before anything the run would build on.
  if (options.validation)
    if (auto r = RemeasureGate(*options.validation, path); !r)
      return std::unexpected(r.error());

  // Replay the feeder to where the interrupted run stood: step_ batches
  // of header_.batch rows each. Without this, a resumed process serves
  // the permutation from the top, and its batch order — hence its
  // final bits — diverges from the same run left uninterrupted. The
  // dataset owns the rows-to-cursor exchange rate (token corpora
  // advance one record per seq_len rows) and replays per-epoch
  // permutations from the shuffle seed, so the position is exact even
  // across epoch boundaries.
  // Under accumulation each optimizer step consumed G micro-batches.
  uint64_t served = 0;
  if (!MulOk(step_, header_.batch, &served) ||
      !MulOk(served, grad_accum_, &served))
    return diag::executing::Error(
        "resumed step count overflows the "
        "feeder position");
  if (auto r = data.SkipServed(served); !r) return std::unexpected(r.error());
  return {};
}

// The gate's numbers are not taken from the checkpoint on trust (CR4 #221).
// The starting validation loss is the SOURCE model's: the persistent
// segment at the plan's initial image (LoRA's B is zero there, so the
// adapted function is the source's) is exactly the state a fresh run's
// first evaluation scores, through this same EvaluateMetrics — so on the
// backend that wrote the checkpoint the bits are the stored ones, or the
// file is not this run's. The best state's loss is measured from the best
// payload the same way. The restored segment is set aside for the passes
// and put back, byte for byte, on every path out.
std::expected<void, std::string> UpdateEngine::RemeasureGate(
    Dataset& validation, const std::string& path) {
  const size_t bytes = static_cast<size_t>(header_.persistent_size);
  const std::vector<uint8_t> held(arena_, arena_ + bytes);
  auto restore = [&] { std::memcpy(arena_, held.data(), bytes); };

  // The plan's initial image of the segment, as AllocateArena lays it down.
  const size_t init =
      std::min(bytes, static_cast<size_t>(header_.persist_init_size));
  std::memset(arena_, 0, bytes);
  std::memcpy(arena_, plan_ + header_.persist_init_offset, init);
  auto source = EvaluateMetrics(validation);
  ++resume_evaluations_;
  if (!source) {
    restore();
    return std::unexpected(source.error());
  }
  EvalMetrics best;
  if (gate_.has_best) {
    std::memcpy(arena_, gate_.best_state.data(), bytes);
    auto scored = EvaluateMetrics(validation);
    ++resume_evaluations_;
    if (!scored) {
      restore();
      return std::unexpected(scored.error());
    }
    best = *scored;
  }
  restore();

  auto bits = [](float v) { return std::bit_cast<uint32_t>(v); };
  auto mismatch = [&](const char* field, const char* what, float stored,
                      float measured) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "states %s of %.9g (bits %08x) but it measures %.9g (bits "
                  "%08x) on this run's held-out records [backend %s] (field "
                  "'%s'): the gate's numbers are re-measured, not trusted",
                  what, static_cast<double>(stored), bits(stored),
                  static_cast<double>(measured), bits(measured), backend_name(),
                  field);
    return diag::executing::Error("resume refused: '" + path + "' " + buf);
  };
  if (bits(source->loss) != bits(gate_.val_initial_loss))
    return mismatch("val_initial_loss", "a source-model validation loss",
                    gate_.val_initial_loss, source->loss);
  // The accuracy the --require-accuracy gate compares against is read from
  // the same record, so it is held the same way.
  if (source->has_accuracy != gate_.val_initial_has_accuracy ||
      bits(source->accuracy) != bits(gate_.val_initial_accuracy))
    return mismatch("val_initial_accuracy",
                    "a source-model validation accuracy",
                    gate_.val_initial_accuracy, source->accuracy);
  if (gate_.has_best) {
    if (bits(best.loss) != bits(gate_.best_loss))
      return mismatch("best_loss", "a best-state validation loss",
                      gate_.best_loss, best.loss);
    if (bits(best.accuracy) != bits(gate_.best_accuracy))
      return mismatch("best_accuracy", "a best-state validation accuracy",
                      gate_.best_accuracy, best.accuracy);
  }
  return {};
}

std::expected<void, std::string> UpdateEngine::SaveCheckpoint(
    const std::string& path, DurableWriteStats* stats) const {
  // The run horizon rides along (v4): a resume trains the remainder to it;
  // (v5) the run binding, the source model's score and the best state; and
  // (v6) the corpus, the split, the backend and the train-loss windows.
  CheckpointRecord record;
  record.step = step_;
  record.horizon_steps = horizon_;
  // Say whether the binding is live (CR4 #145 F71): the file carries the
  // flag, and a load no longer reads zero fields as a binding.
  record.has_binding = binding_.bound;
  record.shuffle_origin = binding_.shuffle_origin;
  record.train_samples = binding_.train_samples;
  record.val_samples = binding_.val_samples;
  record.corpus_hash = binding_.corpus_hash;
  record.split_hash = binding_.split_hash;
  // The backend is the engine's whether or not a run bound anything: the
  // stored losses are its bits.
  record.backend = CheckpointBackend();
  record.has_windows = windows_.live();
  record.window = windows_.window;
  record.window_count = windows_.count;
  record.first_window_sum = windows_.first_sum;
  std::copy(std::begin(windows_.last), std::end(windows_.last),
            std::begin(record.last_losses));
  record.has_val_initial = gate_.has_val_initial;
  record.has_accuracy = gate_.val_initial_has_accuracy;
  record.val_initial_loss = gate_.val_initial_loss;
  record.val_initial_accuracy = gate_.val_initial_accuracy;
  record.has_best = gate_.has_best;
  record.best_step = gate_.best_step;
  record.best_loss = gate_.best_loss;
  record.best_accuracy = gate_.best_accuracy;
  record.stale_evals = gate_.stale_evals;
  // The best payload is written only while it differs from the segment
  // being saved: at the moment of a new best the two are the same bytes,
  // and a resume rebuilds the copy from the main payload.
  if (gate_.has_best && gate_.best_step != step_)
    record.best_payload = gate_.best_state;
  return SaveCheckpointRecord(path, header_.plan_hash, record, arena_,
                              header_.persistent_size, stats);
}

std::expected<void, std::string> UpdateEngine::LoadCheckpoint(
    const std::string& path) {
  auto record = LoadCheckpointRecord(path, header_.plan_hash,
                                     header_.persistent_size, arena_);
  if (!record) return std::unexpected(record.error());
  step_ = record->step;
  horizon_ = record->horizon_steps;  // 0: saved outside a run (plan default)
  binding_.bound = record->has_binding;
  binding_.shuffle_origin = record->shuffle_origin;
  binding_.train_samples = record->train_samples;
  binding_.val_samples = record->val_samples;
  binding_.corpus_hash = record->corpus_hash;
  binding_.split_hash = record->split_hash;
  binding_.backend = record->backend;
  windows_ = LossWindows{};
  if (record->has_windows) {
    windows_.window = record->window;
    windows_.count = record->window_count;
    windows_.first_sum = record->first_window_sum;
    std::copy(std::begin(record->last_losses), std::end(record->last_losses),
              std::begin(windows_.last));
  }
  gate_.has_val_initial = record->has_val_initial;
  gate_.val_initial_has_accuracy = record->has_accuracy;
  gate_.val_initial_loss = record->val_initial_loss;
  gate_.val_initial_accuracy = record->val_initial_accuracy;
  gate_.has_best = record->has_best;
  gate_.best_step = record->best_step;
  gate_.best_loss = record->best_loss;
  gate_.best_accuracy = record->best_accuracy;
  gate_.stale_evals = record->stale_evals;
  // The best payload, or — when the checkpoint was written at a new best —
  // the main payload, which is the same bytes.
  if (!gate_.has_best)
    gate_.best_state.clear();
  else if (!record->best_payload.empty())
    gate_.best_state = std::move(record->best_payload);
  else
    gate_.best_state.assign(arena_, arena_ + header_.persistent_size);
  // The restored persistent segment replaces the adapter state any earlier
  // RunMerge materialized deltas from; committing those would patch deltas
  // that no longer match the parameters.
  merged_ = false;
  return {};
}

}  // namespace seeml::update_rt
