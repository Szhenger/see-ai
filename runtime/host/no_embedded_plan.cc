// The runner's two plan symbols with no plan behind them: the disk mode
// (#158), in which model_update reads update_plan.seeu beside the
// executable or the file --plan names. An emitted package's build.sh writes
// these three lines itself when it carries no embedded plan; this file is
// the same unit for the tree's own build of the runner
// (build/seeml-model-update), which exists so that runtime/host/
// update_main.cc — the device's main — is compiled by the tree's warnings,
// its sanitizer job and its static analysis like every other source
// (review 2026-11-25: it was in neither build list).
#include <cstddef>

extern const unsigned char kSeemlUpdatePlan[];
extern const size_t kSeemlUpdatePlanSize;
const unsigned char kSeemlUpdatePlan[1] = {0};
const size_t kSeemlUpdatePlanSize = 0;
