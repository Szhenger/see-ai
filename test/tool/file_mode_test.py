"""No file is created with fopen (CWE-732): fopen(path, "w") asks for mode
0666 and leaves the result to the umask. A file is created through the
platform seam — CreateForWrite or OpenStreamForWrite in
source/platform/durable.h — which states its mode. Standard library only.
"""

import glob
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREES = ("compiler", "runtime", "source", "tool", "test")
SUFFIXES = (".cc", ".h", ".mm")
# A creating mode: "w", "wb", "w+", "a", "ab", ... ("r+b" opens what exists).
CREATES = re.compile(r'\bfopen\s*\([^;]*?,\s*"[wa][^"]*"\s*\)', re.S)
# The seam's own Win32 branch, where the CRT has no mode to state.
ALLOWED = {"source/platform/durable.h": 1}


def creating_fopens():
    found = {}
    for tree in TREES:
        pattern = os.path.join(REPO, tree, "**", "*")
        for path in sorted(glob.glob(pattern, recursive=True)):
            if not path.endswith(SUFFIXES):
                continue
            with open(path, encoding="utf-8") as f:
                n = len(CREATES.findall(f.read()))
            if n:
                found[os.path.relpath(path, REPO).replace(os.sep, "/")] = n
    return found


class FileModeTest(unittest.TestCase):
    def test_no_file_is_created_with_fopen(self):
        self.assertEqual(creating_fopens(), ALLOWED)

    def test_the_pattern_sees_a_creating_fopen(self):
        self.assertTrue(CREATES.search('f = std::fopen(p.c_str(),\n  "wb");'))
        self.assertFalse(CREATES.search('f = std::fopen(p.c_str(), "r+b");'))
        self.assertFalse(CREATES.search('f = std::fopen(p.c_str(), "rb");'))


if __name__ == "__main__":
    unittest.main()
