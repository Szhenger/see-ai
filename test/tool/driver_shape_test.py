"""The driver contains orchestration only (S6 #152 acceptance 2): no function
in compiler/driver/ is longer than 150 lines.

Function bodies are found by brace matching over the source with comments
and string literals blanked: a '{' opens a function body when the text since
the previous ';', '{' or '}' has a parameter list and is not a type, a
namespace, an initializer, a control statement or a lambda, and no enclosing
brace is already a function. A function's length runs from the first line
of its head to its closing '}'; lambdas and local blocks count toward the
function that holds them. Preprocessor lines are blanked with the comments:
an `#include <x>` or an `#if defined(X)` before a file's first brace is not
the head of anything, and read as one it made the whole file a single
"function" that hid every function inside it. Standard library only.

test/tool/function_length_test.py holds the rest of the tree to its own
limit with this scanner.
"""

import glob
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIMIT = 150

_NOT_A_FUNCTION = re.compile(
    r"^(?:template\s*<.*>\s*)?(?:struct|class|namespace|enum|union)\b"
    r"|\b(?:if|for|while|switch|catch)\s*\(.*\)$"
    r"|^(?:else|do|try)$"
    r"|\[[^\]]*\]\s*(?:\([^)]*\))?\s*(?:mutable\s*)?(?:->\s*[\w:<>,\s*&]+)?$"
    r"|(?:^|[=,(])$")


def _blank(src):
    """`src` with comments and the insides of string / char literals
    replaced by spaces, newlines kept, so braces in them do not count."""
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif src.startswith('R"(', i):
            j = src.find(')"', i)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif src[i] == "#" and not src[src.rfind("\n", 0, i) + 1:i].strip():
            # A directive, through its backslash continuations.
            j = i
            while True:
                k = src.find("\n", j)
                k = n if k < 0 else k
                if k == n or not src[i:k].rstrip().endswith("\\"):
                    break
                j = k + 1
            out.append(re.sub(r"[^\n]", " ", src[i:k]))
            i = k
        elif src[i] in "\"'":
            q, j = src[i], i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(q + re.sub(r"[^\n]", " ", src[i + 1:j]) + q)
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def function_lengths(text):
    """[(lines, first line, head)] for every outermost function body."""
    s = _blank(text)
    found, stack, head_at, line = [], [], 0, 1
    for i, c in enumerate(s):
        if c == "{":
            raw = s[head_at:i]
            head = " ".join(raw.split())
            is_fn = ("(" in head and ")" in head
                     and not _NOT_A_FUNCTION.search(head)
                     and not any(fn for fn, _, _ in stack))
            lead = len(raw) - len(raw.lstrip())
            first = line - raw[lead:].count("\n")
            stack.append((is_fn, first, head[:100]))
            head_at = i + 1
        elif c == "}":
            if stack:
                is_fn, first, head = stack.pop()
                if is_fn:
                    found.append((line - first + 1, first, head))
            head_at = i + 1
        elif c == ";":
            head_at = i + 1
        if c == "\n":
            line += 1
    return found


class DriverShapeTest(unittest.TestCase):
    def test_the_scanner_finds_a_long_function_and_skips_the_rest(self):
        body = "\n".join(f"  x += {i};" for i in range(200))
        text = (
            "namespace n {\nstruct S {\n  int f(int a) {\n    return a;\n  }\n};\n"
            "int g(int x) {\n  if (x) {\n    auto k = [&](int y) {\n"
            "      return y;\n    };\n  }\n" + body + "\n  return x;\n}\n"
            "const char* s = \"{ not a brace }\";  // nor this {\n}\n")
        rows = sorted(function_lengths(text), reverse=True)
        self.assertEqual([r[2].split("(")[0] for r in rows], ["int g", "int f"])
        self.assertEqual(rows[0][0], 208)  # int g's head through its '}'
        self.assertEqual(rows[1][0], 3)

    def test_a_directive_before_the_first_brace_is_not_a_function_head(self):
        body = "\n".join(f"  x += {i};" for i in range(10))
        text = ("#include <vector>\n#if defined(__APPLE__)\n#define F(a) \\\n"
                "  { a }\n#endif\nnamespace {\nint g(int x) {\n" + body +
                "\n  return x;\n}\n}  // namespace\n")
        rows = function_lengths(text)
        self.assertEqual([(r[0], r[1], r[2]) for r in rows],
                         [(13, 7, "int g(int x)")])

    def test_no_driver_function_exceeds_the_limit(self):
        files = sorted(glob.glob(os.path.join(REPO, "compiler", "driver", "*.cc")))
        self.assertTrue(files)
        longest = []
        for path in files:
            with open(path, encoding="utf-8") as f:
                for n, first, head in function_lengths(f.read()):
                    longest.append((n, f"{os.path.relpath(path, REPO)}:{first}", head))
        longest.sort(reverse=True)
        over = [row for row in longest if row[0] > LIMIT]
        self.assertEqual(over, [], f"driver functions over {LIMIT} lines")
        self.assertGreater(longest[0][0], 50)  # the scan found real bodies


if __name__ == "__main__":
    unittest.main()
