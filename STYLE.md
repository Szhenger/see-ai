# SeeAI House Style

What the tree already does, written down (S8, #154). Where the tree is not
yet uniform, the rule is stated and the gap is named. Layout is not written
down here: it is what `clang-format` 19 makes of the committed `.clang-format`
files (section 1, "Formatting").

## 1. Two dialects, by subsystem

The dialect question is decided: **document per subsystem, do not rename.**

| | Google style | SIR dialect (MLIR-flavoured) |
|---|---|---|
| Where | everything else: `seeml::update` (compiler), `seeml::update_rt` (runtime), `source/`, `tool/`, `test/` | `seeml::sir` only: `compiler/frontend/representation/` and `compiler/frontend/operator/` (13 files) |
| Functions and methods | `PascalCase` — `LoadSmf`, `ValidateInstruction` | `camelCase` — `appendOp`, `numOperands`, `replaceAllUsesWith`, `OpBuilder::gemm` |
| Indent | 2 spaces, access specifiers at 1 | 4 spaces, access specifiers at 1 |
| Programmer-contract violations | never `assert` (0 in `runtime/`, `source/`, `tool/`, and in the compiler outside SIR) | `assert(cond && "fn: what")` |

Shared by both: `PascalCase` types, `snake_case` locals, parameters and
struct fields, `trailing_underscore_` private members, `kPascalCase`
constants and enumerators (`enum class`, sized: `enum class LossKind :
uint8_t`), `snake_case` file names, `SEEML_<PATH>_H_` include guards,
includes by repository-rooted path (`"compiler/driver/update_compiler.h"`),
`///` on declarations and a `// ====` banner that says what the file is for,
80 columns. A conversion is a named cast (`static_cast`, `reinterpret_cast`):
both build drivers compile with `-Wold-style-cast` under `-Werror`. Neither
compiles the generated runner, `runtime/host/update_main.cc` — a package's
own `build.sh` does, with `-Wall -Wextra` alone — so no build holds that
file to this rule or to a dropped `[[nodiscard]]` result (section 2); it is
held by review.

**Formatting: the formatter is the rule.** The root `.clang-format` is the
Google dialect at 80 columns; `compiler/frontend/representation/.clang-format`
and `compiler/frontend/operator/.clang-format` inherit it and set the SIR
dialect's indent. One command formats the tree:

    git ls-files -z '*.cc' '*.h' '*.mm' | xargs -0 clang-format-19 -i

and the check is the same file list with `--dry-run -Werror` in place of
`-i`, which exits non-zero on any file the formatter would change:

    git ls-files -z '*.cc' '*.h' '*.mm' | xargs -0 clang-format-19 --dry-run -Werror

- **The version is 19**, the LLVM release the Linux toolchain is pinned to.
  Other releases lay some constructs out differently (Apple's `clang-format`
  is not 19): a tree formatted by one fails the check. Where no
  `clang-format-19` is packaged, `pip install 'clang-format==19.1.*'` into a
  virtual environment provides it as `clang-format`.
- **The formatter lays code out and changes nothing else.** It does not sort
  or regroup `#include`s (their order is semantic here), does not reorder
  `using` declarations, does not re-wrap a comment and does not split a
  string literal. So the 80 columns are the author's to keep in a comment and
  in a literal, and a trailing comment that would run past 80 goes on its own
  line above the statement — otherwise the formatter breaks the *code* to
  shorten the line.
- **`.clang-format-ignore`** names what the formatter skips even when asked:
  the generated `source/tokenizer/unicode_tables.h`, `test/fixtures/`, and
  the ledger's probe under `docs/`. A C++ file that is not code to maintain
  is added there, by path.
- **`// clang-format off` … `// clang-format on` fences a hand-laid table**,
  and only that: rows the formatter would explode to one item a line or
  cannot lay out. Six today — the `OpCode` table with its operand comments
  (`source/plan/instruction.h`), the Metal pipeline ids and their kernel
  names (`runtime/executor/metal_backend.mm`), the two X-macro field lists
  of `tool/seeml_abi.cc`, the header field list of `tool/seeml_seeu_dump.cc`
  and the report's format string in `tool/seeml_update_compile.cc`. The
  Metal shader source (`runtime/executor/metal_kernels.h`) needs no fence: it
  is raw string literals, which the formatter does not enter.
- `.git-blame-ignore-revs` names the commit that reformatted the tree:
  `git config blame.ignoreRevsFile .git-blame-ignore-revs` makes `git blame`
  look through it.

A file belongs to the dialect of the namespace it *defines* in. The passes,
sema, the parser and the lowering live in `seeml::update` and are Google
style; they call SIR's camelCase methods through `namespace sir =
seeml::sir;` — that is use, not a third dialect. Do not add PascalCase
methods to `seeml::sir` or camelCase ones anywhere else.

## 2. Errors: `std::expected`, and where `assert` stops

- **Anything an input can cause is a `std::expected<T, std::string>`.** A
  model, a corpus, a plan, a checkpoint, a recipe, a command line: every
  byte is assumed malformed or hostile, and a bad one is a returned error,
  on the build host and again on the device. The runtime validates a plan
  in full at load and then dispatches without checks; it contains no
  `assert` and must not gain one.
- **`assert` is for a caller breaking an API contract inside the SIR core**
  — a null operand, removing an op that still has users, an anchor from
  another block. Nothing a file can contain may reach one: the structural
  check that inputs *can* fail is `Block::verify()`, which returns
  `std::expected` and is rerun by the pass manager after every pass.
- **No exceptions for errors.** The tree does not throw to report a
  failure (the worker pool carries a throwing body's exception back to the
  submitter; one `mmap` failure surfaces as `std::bad_alloc`).
- **`[[nodiscard]]` on every function that returns `std::expected`**, and
  on diagnostic builders. The headers of `compiler/` and `runtime/` hold to
  this but for five declarations; `source/` and `tool/` hold to it too, and
  there it also covers the `…Violation` checks, the strict parsers, the
  bounds helpers and the durable-I/O calls whose `bool` is a write's fate.
  A result dropped on purpose (a best-effort close) says so with `(void)`.
- Propagate with an early return:
  `if (auto r = Step(); !r) return std::unexpected(r.error());`

## 3. The diagnostic-unit contract

Every diagnostic is **one line, `"<unit>: <message>"`** — the unit names
the component that spoke (`PassManager`, `PlanValidator`, `HostArch`).

- Unit names are `inline constexpr std::string_view k…` constants in a
  *process module*, not literals repeated at call sites. The compiler
  has six (`compiler/diagnostics/`: `tokenizing`, `parsing`, `passing`,
  `updating`, `architecting`, `generating`), the runtime four
  (`runtime/diagnostics/`: `feeding`, `validating`, `executing`,
  `persisting`). Each is header-only: the unit registry plus builders for
  the message shapes that process repeats.
- Builders return `std::unexpected<std::string>` (via `diag::Fail(unit,
  message)`), so `return updating::Something(...);` fits any `expected`.
- Inside the compiler and the runtime's engine an error is propagated
  **verbatim** — not re-prefixed on the way up. The attributions added are
  the pass manager's (corruption found after a pass names that pass) and a
  command line's (`model_update` and the tools prefix the flag or file).
- A message states the rule and the value that broke it, and names the
  flag, field or tensor (`--lora-rank (recipe field lora_rank) must be an
  integer in [1, 4096], got '5000'`). Tests match these strings.
- The compiler's `Logger` is the only stateful piece (`Note` = INFO,
  `Fallback` = WARN). The runtime has no logger. A tool prints
  `"<tool-name>: <message>"` to stderr and results to stdout.

## 4. Exit codes

The six build-host CLIs (`seeml-update-compile`, `seeml-seeu-dump`,
`seeml-plan-probe`, `seeml-tokenize`, `seeml-abi`, `seeml-bench`):

| Code | Meaning |
|---|---|
| `0` | the work was done |
| `1` | the work failed: an input that cannot be read or is not what it claims to be, a refusal, a failed write or build |
| `2` | a bad command line — unknown flag, stray argument, missing value, a value that does not parse or is out of bound, contradicting flags — refused before any model or plan is read |

The generated `model_update` runner extends the table: `0` committed, `1`
runtime error, `2` bad arguments, `3` rejected by the regression gate
(device untouched), `4` committed but `--report` / `--loss-log` not written,
`5` stopped by a signal at a step boundary (resumable). SeeTest suites: `0`
no test failed, `1` a test failed, `2` usage or an empty selection.

In a tool, the two refusals are two functions — `Usage(msg)` returns 2,
`Fail(msg)` returns 1 — and `main` returns their result.
`test/tool/cli_exit_codes_test.py` pins the table for all six. Known gap:
`seeml-bench` has one `Fail` that returns 2 for a failed compile or write
too; its command-line errors are right, its run-time failures are not yet 1.

## 5. Ownership and views

- **`std::unique_ptr` owns; a raw pointer or reference never does.** A
  `Block` owns its `Operation`s and `Value`s; `Value*` / `Operation*`
  handed around the passes are borrowed and live as long as the block.
  There is no `std::shared_ptr` in the tree and no `delete`; the two
  `new`s are deliberately leaked function-local statics.
- **A view is a `std::span`** (or `std::string_view`): accessors return
  `std::span<const T>`, not a container reference or a copy. About thirty
  `(const uint8_t* data, size_t size)` parameter pairs remain; new
  signatures take a span.
- File-supplied offsets and counts go through the overflow-checked helpers
  of `source/contract/bounds.h` (`MulOk`, `RangeOk`) — the one way bounds
  math is written, on both planes.
- The runtime's working memory is one arena, sized by the compiler.
- **No file is created with `fopen`.** `fopen(path, "w")` asks for mode
  0666 and leaves the rest to the umask; the platform seam
  (`source/platform/durable.h`: `CreateForWrite`, or `OpenStreamForWrite`
  for a stdio stream) states the mode — 0644, narrowed by the umask —
  and `test/tool/file_mode_test.py` holds the tree to having no creating
  `fopen`. Not yet uniform: eight `std::ofstream` sites (the emitter's
  package files, `SaveSds`, the in-place edit's sidecar, the staged
  corpus and `--dump-sir`) create files the way `fopen` did; new code
  uses the seam.
- `source/` is what both halves share and neither owns: the compiler and
  the runtime include it and never include each other. The on-disk structs
  there are the ABI; `tool/seeml/abi.json` is generated from them and CI
  fails when it is stale.
- `using namespace` never appears in a product header.

## 6. Tests

- A suite lives where its subject lives (`test/<tree>/<subsystem>/
  <name>_test.cc` → `build/seeml_<name>_test`) and includes only the two
  façades, `test/framework/seetest.h` and `test/support/builders.h`. A new
  source or suite is added to **both** `build/build.sh` and
  `CMakeLists.txt`.
- `TEST(Suite, NameThatStatesTheProperty)`. `EXPECT_*` records and
  continues, `ASSERT_*` records and returns (so it belongs in the `TEST`
  body). For `std::expected`: `ASSERT_OK`, `ASSERT_OK_AND_ASSIGN(auto x,
  F())`, `EXPECT_ERROR_CONTAINS(F(), "the rule's words")`.
- **A test that cannot run says `SKIP("why")`; it never returns green for
  zero work.** A missing Metal device, a platform without the API: the
  test is printed and counted as `[  SKIPPED ]` with its reason, apart from
  `[  PASSED  ]`, and does not fail the run. `SKIP` is for a capability the
  host lacks — not for a test that is broken or slow.
- Fixtures are seeded and deterministic; file I/O goes through
  `ScopedTempDir`. A golden changes only by an explicit regenerate step,
  in the commit that explains why.
- Most suites open with `using namespace seeml::update;`. Per-symbol
  `using` declarations are the direction (#154); new suites use them.
- Python tests (`test/tool/*_test.py`) are `unittest`, standard library
  first. One that needs a built binary skips without it, with a reason
  the skip guard of CI's end-to-end job matches (`no built …`) — there the
  binary was built, so the skip is a failure.

A comment says why the code is so and names the issue or finding behind it
(`CR4 #145 F61`); a commit subject starts with the issue (`#202 item 3a: …`)
and states what is now true.
