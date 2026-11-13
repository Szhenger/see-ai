"""The driver contains orchestration only (S6 #152 acceptance 2): no function
in compiler/driver/ is longer than 150 lines.

Function bodies are found by brace matching over the source with comments
and string literals blanked: a '{' opens a function body when the text since
the previous ';', '{' or '}' has a parameter list and is not a type, a
namespace, an initializer, a control statement or a lambda, and no enclosing
brace is already a function. A function's length runs from the first line
of its head to its closing '}'; lambdas and local blocks count toward the
function that holds them. Standard library only.
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
