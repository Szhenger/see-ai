// =============================================================================
// The compiled held-out split (S3 #149): a pure function of (records, seed,
// fraction) the compiler and the device agree on. Determinism, the count
// rule, sortedness and distinctness, the hash's sensitivity, and a literal
// fixture the Python mirror (tool/frontier_exec.py) pins to the same values.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "source/contract/split.h"
#include "test/framework/seetest.h"

namespace {

using namespace seeml::update;

TEST(Split, TheCountRuleIsThePositionalSplits) {
  EXPECT_EQ(HeldOutCount(0, 0.5f), 0u);
  EXPECT_EQ(HeldOutCount(1, 0.5f), 0u);
  EXPECT_EQ(HeldOutCount(2, 0.1f), 1u);    // at least one
  EXPECT_EQ(HeldOutCount(2, 0.99f), 1u);   // at most n - 1
  EXPECT_EQ(HeldOutCount(10, 0.3f), 3u);
  EXPECT_EQ(HeldOutCount(501, 0.1f), 50u);
}

TEST(Split, IndicesAreSortedDistinctInRangeAndDeterministic) {
  for (const uint64_t n : {2u, 3u, 10u, 501u, 4096u}) {
    for (const uint64_t seed : {0u, 1u, 12345u}) {
      const auto a = HeldOutIndices(n, seed, 0.1f);
      const auto b = HeldOutIndices(n, seed, 0.1f);
      EXPECT_EQ(a, b);
      EXPECT_EQ(a.size(), HeldOutCount(n, 0.1f));
      std::set<uint64_t> distinct(a.begin(), a.end());
      EXPECT_EQ(distinct.size(), a.size());
      for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_LT(a[i], n);
        if (i > 0) EXPECT_LT(a[i - 1], a[i]);
      }
    }
  }
  // Another seed, another set (for any corpus big enough to tell).
  EXPECT_NE(HeldOutIndices(501, 0, 0.1f), HeldOutIndices(501, 1, 0.1f));
  // The seed is not the shuffle's stream: the same seed in the split's
  // domain draws a different first word.
  uint64_t shuffle = 7, split = 7 ^ kSplitDomain;
  EXPECT_NE(SplitMix64Next(&shuffle), SplitMix64Next(&split));
  EXPECT_TRUE(HeldOutIndices(1, 3, 0.5f).empty());
}

TEST(Split, TheHashNamesTheSetAndTheLiteralFixtureHoldsThePythonMirror) {
  const auto a = HeldOutIndices(501, 0, 0.1f);
  const auto b = HeldOutIndices(501, 1, 0.1f);
  EXPECT_NE(SplitHash(a), SplitHash(b));
  EXPECT_EQ(SplitHash(a), SplitHash(HeldOutIndices(501, 0, 0.1f)));
  // The fixture: n = 10, seed = 1, 30 % held out. The Python mirror pins
  // these same three indices and this hash (frontier_exec_test).
  const std::vector<uint64_t> fixture = HeldOutIndices(10, 1, 0.3f);
  ASSERT_EQ(fixture.size(), 3u);
  std::printf("split fixture n=10 seed=1 frac=0.3: %llu %llu %llu hash %016llx\n",
              (unsigned long long)fixture[0], (unsigned long long)fixture[1],
              (unsigned long long)fixture[2],
              (unsigned long long)SplitHash(fixture));
  const std::vector<uint64_t> expected = {1, 5, 7};
  EXPECT_EQ(fixture, expected);
  EXPECT_EQ(SplitHash(fixture), 0xe731eed086e2a95dULL);
  EXPECT_EQ(ValFracOf(ValFracBits(0.3f)), 0.3f);
}

}  // namespace
