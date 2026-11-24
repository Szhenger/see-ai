"""No function over 200 lines in compiler/, runtime/ or tool/ (S8 #154,
acceptance 3), by the scanner of driver_shape_test.py: a function's length
runs from the first line of its head to its closing '}'.

The functions still over the limit are named in EXEMPT with the reason each
is; the list only shrinks. An exempt function that is no longer over the
limit fails the test until its line is removed, so the list cannot go stale
and a new long function cannot hide behind it. Standard library only.
"""

import glob
import os
import unittest

from driver_shape_test import REPO, function_lengths

LIMIT = 200
TREES = ("compiler", "runtime", "tool")
SUFFIXES = (".cc", ".h", ".mm")

# (file, a unique part of the function's head): why it is over the limit.
EXEMPT = {}


def long_functions():
    """[(lines, 'file:line', head, file)] over LIMIT, longest first."""
    rows = []
    for tree in TREES:
        pattern = os.path.join(REPO, tree, "**", "*")
        for path in sorted(glob.glob(pattern, recursive=True)):
            if not path.endswith(SUFFIXES):
                continue
            rel = os.path.relpath(path, REPO).replace(os.sep, "/")
            with open(path, encoding="utf-8") as f:
                for n, first, head in function_lengths(f.read()):
                    if n > LIMIT:
                        rows.append((n, f"{rel}:{first}", head, rel))
    return sorted(rows, reverse=True)


def exemption(row):
    for (path, name) in EXEMPT:
        if row[3] == path and name in row[2]:
            return (path, name)
    return None


class FunctionLengthTest(unittest.TestCase):
    def test_no_function_exceeds_the_limit(self):
        over = [row[:3] for row in long_functions() if not exemption(row)]
        self.assertEqual(over, [], f"functions over {LIMIT} lines")

    def test_every_exemption_is_still_needed(self):
        used = {exemption(row) for row in long_functions()}
        stale = sorted(set(EXEMPT) - used)
        self.assertEqual(stale, [], "under the limit now: remove from EXEMPT")

    def test_the_scan_reads_the_three_trees(self):
        for tree in TREES:
            files = [p for p in glob.glob(os.path.join(REPO, tree, "**", "*"),
                                          recursive=True)
                     if p.endswith(SUFFIXES)]
            self.assertGreater(len(files), 5, tree)


if __name__ == "__main__":
    unittest.main()
