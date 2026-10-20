"""Tests for tool/pack_update.py, the package assembler.

Standard library only, like the tool. The suite fabricates package
directories in the shape seeml-update-compile emits (a plan file with the
SEEU magic, a build.sh) so no compiler build is needed; the one test that
assembles the stub for real is skipped when no C++ driver is on PATH.

    python3 -m unittest discover -s test/tool -p '*_test.py'
"""

import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tool"))

import pack_update as pu  # noqa: E402

# The pre-P1 emitter's build.sh, abridged to the lines the packer touches.
LEGACY_BUILD_SH = (
    "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nCXX=\"${CXX:-c++}\"\n"
    "FLAGS=\"-std=c++23 -O2 -Wall -Wextra -pthread -I. $TILE_FLAGS\"\n"
    "$CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o\n"
    "$CXX $FLAGS -c update_main.cc -o update_main.o\n"
    "$CXX $FLAGS update_main.o update_plan_embedded.o -o model_update\n")

# A post-P1, pre-#158 emitter's build.sh branches on the stub itself and
# stops the build when no plan TU is present.
PRE_DISK_BUILD_SH = (
    "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nCXX=\"${CXX:-c++}\"\n"
    "if [ -f update_plan_embedded.S ]; then\n"
    "  $CXX -c update_plan_embedded.S -o update_plan_embedded.o\n"
    "elif [ -f update_plan_embedded.cc ]; then\n"
    "  $CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o\n"
    "else\n  exit 1\nfi\n")

# A current emitter's build.sh (#158) builds the disk mode with neither TU:
# its else-branch compiles the two-line empty-plan TU.
MODERN_BUILD_SH = (
    "#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\nCXX=\"${CXX:-c++}\"\n"
    "if [ -f update_plan_embedded.S ]; then\n"
    "  $CXX -c update_plan_embedded.S -o update_plan_embedded.o\n"
    "elif [ -f update_plan_embedded.cc ]; then\n"
    "  $CXX $FLAGS -c update_plan_embedded.cc -o update_plan_embedded.o\n"
    "else\n"
    "  printf '%s\\n' 'x' > update_plan_none.cc\n"
    "  $CXX $FLAGS -c update_plan_none.cc -o update_plan_embedded.o\n"
    "fi\n")


CXX = (os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
       or shutil.which("g++"))

BUILD = os.environ.get("SEEML_BUILD_DIR", os.path.join(REPO, "build"))
COMPILER = os.path.join(BUILD, "seeml-update-compile")
GOLDEN = os.path.join(REPO, "test", "fixtures", "golden")


def repo_file(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


class PackageDir:
    """A fabricated package directory."""

    def __init__(self, build_sh=MODERN_BUILD_SH, plan=b"SEEU" + bytes(range(256)) * 3):
        self.dir = tempfile.mkdtemp(prefix="pack_update_test-")
        self.plan = plan
        with open(os.path.join(self.dir, pu.PLAN_FILE), "wb") as f:
            f.write(plan)
        with open(os.path.join(self.dir, pu.BUILD_SCRIPT), "w") as f:
            f.write(build_sh)
        os.chmod(os.path.join(self.dir, pu.BUILD_SCRIPT), 0o755)
        with open(os.path.join(self.dir, pu.DECIMAL_TU), "w") as f:
            f.write("const unsigned char kSeemlUpdatePlan[] = {0};\n")

    def path(self, name):
        return os.path.join(self.dir, name)

    def read(self, name, mode="r"):
        with open(self.path(name), mode) as f:
            return f.read()

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def run_main(argv):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = pu.main(argv)
    return rc, err.getvalue()


class SeamTest(unittest.TestCase):
    """The packer restates three facts the C++ side owns. Until P6's ABI
    manifest carries them, these tests are the machine check that the
    copies agree (they read the repository sources, so they run only from
    a checkout)."""

    def test_build_script_lines_match_the_emitters_template(self):
        emitter = repo_file("compiler/backend/packaging/native_emitter.cc")
        # C++ string literals: the trailing newline is spelled \n.
        for line in (pu.STUB_COMPILE_LINE, pu.DECIMAL_COMPILE_LINE,
                     pu.DISK_COMPILE_LINE):
            self.assertIn(line[:-1] + "\\n", emitter, line)
        self.assertIn("if [ -f update_plan_embedded.S ]; then", emitter)
        # The disk mode's TU (#158) defines exactly the symbols the stub
        # and the decimal TU do, as an empty plan — the driver keys the
        # mode off the zero count.
        self.assertIn("const unsigned char kSeemlUpdatePlan[1] = {0};", emitter)
        self.assertIn("const size_t kSeemlUpdatePlanSize = 0;", emitter)
        self.assertIn("> update_plan_none.cc", emitter)
        self.assertIn("kSeemlUpdatePlanSize > 0", emitter)

    def test_stub_symbols_match_the_generated_drivers_externs(self):
        emitter = repo_file("compiler/backend/packaging/native_emitter.cc")
        self.assertIn("extern const unsigned char kSeemlUpdatePlan[];", emitter)
        self.assertIn("extern const size_t kSeemlUpdatePlanSize;", emitter)
        stub = pu.render_stub(16384)
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlan)\n", stub)
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlanSize)\n", stub)

    def test_the_driver_reads_the_plan_file_the_packer_leaves(self):
        # The disk mode's contract (#158): the packer leaves PLAN_FILE
        # beside the binary, and the generated driver maps exactly that
        # name from its own directory (or --plan FILE).
        emitter = repo_file("compiler/backend/packaging/native_emitter.cc")
        self.assertIn(f'kPlanFileName = "{pu.PLAN_FILE}"', emitter)
        self.assertIn('ExecutableDir(argv[0]) + "/" + kPlanFileName', emitter)
        self.assertIn('"--plan"', emitter)
        self.assertIn("LoadFromFileMapped(plan_path)", emitter)

    def test_plan_magic_matches_the_plan_schema(self):
        schema = repo_file("source/plan/schema.h")
        m = re.search(r"kSeeuMagic\s*=\s*0x([0-9A-Fa-f]+)", schema)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1), 16),
                         int.from_bytes(pu.PLAN_MAGIC, "little"))


class StubTest(unittest.TestCase):
    def test_declares_the_drivers_symbols_page_aligned(self):
        stub = pu.render_stub(16384)
        self.assertIn('.incbin "update_plan.seeu"', stub)
        self.assertIn(".balign 16384", stub)
        self.assertIn("SEEML_SYM(kSeemlUpdatePlan):", stub)
        self.assertIn("SEEML_SYM(kSeemlUpdatePlanSize):", stub)
        # The size is a label difference, never a literal that can go stale.
        self.assertRegex(stub, r"\.quad SEEML_SYM\(kSeemlUpdatePlan_end\) - "
                               r"SEEML_SYM\(kSeemlUpdatePlan\)\n")
        self.assertNotRegex(stub, r"\.(quad|long)\s+\d")
        self.assertIn("__USER_LABEL_PREFIX__", stub)
        # Both symbols are exported; the end label is not.
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlan)\n", stub)
        self.assertIn(".globl SEEML_SYM(kSeemlUpdatePlanSize)\n", stub)
        self.assertNotIn(".globl SEEML_SYM(kSeemlUpdatePlan_end)", stub)
        # Only C comments: the preprocessor strips them on every target,
        # while the assembler's own comment character differs per ISA.
        self.assertNotIn("//", stub)
        self.assertNotIn("#!", stub)

    @unittest.skipIf(CXX is None, "no C++ driver on PATH")
    def test_assembles_links_and_matches_the_plan_bytes(self):
        """The contract, end to end on this host's toolchain: the stub
        assembles under the C++ driver, links against the exact extern
        declarations update_main.cc uses, the symbol is page-aligned, and
        the bytes equal the file's — for an odd-sized plan, so the size
        symbol cannot be satisfied by padding."""
        pkg = PackageDir(plan=b"SEEU" + os.urandom(100003 - 4))
        self.addCleanup(pkg.cleanup)
        with open(pkg.path(pu.STUB_FILE), "w") as f:
            f.write(pu.render_stub(16384))
        probe = r"""
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern const unsigned char kSeemlUpdatePlan[];
extern const size_t kSeemlUpdatePlanSize;
int main() {
  std::FILE* f = std::fopen("update_plan.seeu", "rb");
  if (!f) return 2;
  std::vector<unsigned char> want(kSeemlUpdatePlanSize + 1);
  const size_t got = std::fread(want.data(), 1, want.size(), f);
  std::fclose(f);
  const bool same = got == kSeemlUpdatePlanSize &&
      std::memcmp(want.data(), kSeemlUpdatePlan, got) == 0;
  std::printf("%zu %zu %d\n", kSeemlUpdatePlanSize,
              (size_t)(reinterpret_cast<uintptr_t>(kSeemlUpdatePlan) % 16384),
              same ? 1 : 0);
  return same ? 0 : 1;
}
"""
        with open(pkg.path("probe.cc"), "w") as f:
            f.write(probe)
        # Exactly the build.sh recipe: the stub takes no C++ flags.
        subprocess.run([CXX, "-c", pu.STUB_FILE, "-o", "stub.o"],
                       cwd=pkg.dir, check=True)
        subprocess.run([CXX, "-std=c++17", "probe.cc", "stub.o", "-o", "probe"],
                       cwd=pkg.dir, check=True)
        out = subprocess.run(["./probe"], cwd=pkg.dir, check=True,
                             capture_output=True, text=True).stdout.split()
        self.assertEqual(out, [str(len(pkg.plan)), "0", "1"])


class PlanCheckTest(unittest.TestCase):
    def test_accepts_a_plan_and_returns_its_size(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assertEqual(pu.check_plan(pkg.path(pu.PLAN_FILE)), len(pkg.plan))

    def test_rejects_missing_empty_and_foreign_files(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        with self.assertRaisesRegex(pu.PackError, "cannot read plan"):
            pu.check_plan(pkg.path("absent.seeu"))
        open(pkg.path("empty.seeu"), "wb").close()
        with self.assertRaisesRegex(pu.PackError, "is empty"):
            pu.check_plan(pkg.path("empty.seeu"))
        with open(pkg.path("model.smf"), "wb") as f:
            f.write(b"SMF1" + bytes(64))
        with self.assertRaisesRegex(pu.PackError, "plan magic"):
            pu.check_plan(pkg.path("model.smf"))


class BuildScriptTest(unittest.TestCase):
    def test_rewrites_only_the_legacy_compile_line(self):
        out = pu.adapt_build_script(LEGACY_BUILD_SH)
        self.assertIsNotNone(out)
        self.assertNotIn("-c update_plan_embedded.cc", out)
        self.assertIn(pu.STUB_COMPILE_LINE, out)
        # Everything else — including the link line naming the .o — stays.
        self.assertIn("update_main.o update_plan_embedded.o -o model_update", out)
        self.assertEqual(out.count("\n"), LEGACY_BUILD_SH.count("\n"))

    def test_leaves_a_stub_aware_script_alone(self):
        self.assertIsNone(pu.adapt_build_script(MODERN_BUILD_SH))

    def test_refuses_a_script_it_does_not_recognize(self):
        with self.assertRaisesRegex(pu.PackError, "not an unmodified seeml-update-compile"):
            pu.adapt_build_script("#!/bin/sh\nmake\n")
        # Two compile lines is not the emitter's script either.
        with self.assertRaisesRegex(pu.PackError, "refusing"):
            pu.adapt_build_script(LEGACY_BUILD_SH + pu.DECIMAL_COMPILE_LINE)
        # A user-edited decimal line (extra flags) is refused, not guessed.
        edited = LEGACY_BUILD_SH.replace("-c update_plan_embedded.cc",
                                         "-march=native -c update_plan_embedded.cc")
        with self.assertRaisesRegex(pu.PackError, "unmodified"):
            pu.adapt_build_script(edited)

    def test_a_comment_naming_the_stub_does_not_count_as_support(self):
        commented = "# built from update_plan_embedded.S one day\n" + LEGACY_BUILD_SH
        out = pu.adapt_build_script(commented)
        self.assertIsNotNone(out)
        self.assertIn(pu.STUB_COMPILE_LINE, out)

    def test_disk_mode_needs_the_disk_branch(self):
        pu.check_disk_capable(MODERN_BUILD_SH)
        for script in (PRE_DISK_BUILD_SH, LEGACY_BUILD_SH,
                       "# update_plan_none.cc mentioned, never compiled\n"):
            with self.assertRaisesRegex(pu.PackError,
                                        "cannot build without an embedded"):
                pu.check_disk_capable(script)


class ModeChoiceTest(unittest.TestCase):
    """Policy, not a flag the user has to know (#158): below the limit the
    plan is embedded, from it the plan stays on disk; a forced mode wins
    at any size."""

    def test_auto_switches_at_the_limit(self):
        self.assertEqual(pu.choose_mode("auto", 0), "embed")
        self.assertEqual(pu.choose_mode("auto", pu.EMBED_LIMIT - 1), "embed")
        self.assertEqual(pu.choose_mode("auto", pu.EMBED_LIMIT), "disk")
        self.assertEqual(pu.choose_mode("auto", 3 * pu.EMBED_LIMIT), "disk")

    def test_forced_modes_ignore_the_size(self):
        self.assertEqual(pu.choose_mode("embed", 3 * pu.EMBED_LIMIT), "embed")
        self.assertEqual(pu.choose_mode("disk", 1), "disk")

    def test_the_limit_sits_under_the_measured_failures(self):
        # 1.15 GB linked and ran; 1.85 GB segfaulted the linker; 2.72 GB
        # could not be loaded (the issue's table, macOS arm64).
        self.assertEqual(pu.EMBED_LIMIT, 1 << 30)
        self.assertLess(pu.EMBED_LIMIT, 1_850_000_000)


class AtomicWriteTest(unittest.TestCase):
    def test_replaces_the_target_and_leaves_no_temp_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        target = os.path.join(d, "out.txt")
        with open(target, "w") as f:
            f.write("old")
        pu.write_atomic(target, b"new", executable=True)
        with open(target) as f:
            self.assertEqual(f.read(), "new")
        self.assertTrue(os.stat(target).st_mode & stat.S_IXUSR)
        self.assertEqual(os.listdir(d), ["out.txt"])

    def test_files_get_the_ordinary_create_mode_not_mkstemps_0600(self):
        """A package is copied to and built on other machines and by other
        users: the stub must be world-readable under a normal umask."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        pu.write_atomic(os.path.join(d, "a.S"), b"x")
        pu.write_atomic(os.path.join(d, "b.sh"), b"x", executable=True)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(d, "a.S")).st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(d, "b.sh")).st_mode), 0o755)


class PackTest(unittest.TestCase):
    def test_embeds_and_removes_the_decimal_tu(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.exists(pkg.path(pu.STUB_FILE)))
        self.assertFalse(os.path.exists(pkg.path(pu.DECIMAL_TU)))
        self.assertIn("removed update_plan_embedded.cc", err)
        # A stub-aware build.sh is untouched, byte for byte.
        self.assertEqual(pkg.read(pu.BUILD_SCRIPT), MODERN_BUILD_SH)
        # The plan itself is never rewritten.
        self.assertEqual(pkg.read(pu.PLAN_FILE, "rb"), pkg.plan)

    def test_adapts_a_legacy_package(self):
        pkg = PackageDir(build_sh=LEGACY_BUILD_SH)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 0, err)
        self.assertIn("rewrote build.sh", err)
        script = pkg.read(pu.BUILD_SCRIPT)
        self.assertIn(pu.STUB_COMPILE_LINE, script)
        self.assertTrue(os.stat(pkg.path(pu.BUILD_SCRIPT)).st_mode & stat.S_IXUSR)

    def test_is_idempotent(self):
        pkg = PackageDir(build_sh=LEGACY_BUILD_SH)
        self.addCleanup(pkg.cleanup)
        self.assertEqual(run_main([pkg.dir])[0], 0)
        first = pkg.read(pu.STUB_FILE), pkg.read(pu.BUILD_SCRIPT)
        self.assertEqual(run_main([pkg.dir])[0], 0)
        self.assertEqual((pkg.read(pu.STUB_FILE), pkg.read(pu.BUILD_SCRIPT)), first)

    def test_page_align_reaches_the_stub(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assertEqual(run_main([pkg.dir, "--page-align", "4096"])[0], 0)
        self.assertIn(".balign 4096", pkg.read(pu.STUB_FILE))

    def test_no_embed_leaves_the_plan_on_disk(self):
        """The disk mode, forced (#158): no stub is written, every plan TU
        a previous pack or emission left is removed, build.sh is untouched
        (its else-branch builds the mode), the plan is untouched, and the
        report says which mode and why."""
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        with open(pkg.path(pu.STUB_FILE), "w") as f:
            f.write("stale stub from an earlier pack\n")
        rc, err = run_main([pkg.dir, "--no-embed", "--report",
                            pkg.path("pack.json")])
        self.assertEqual(rc, 0, err)
        self.assertFalse(os.path.exists(pkg.path(pu.STUB_FILE)))
        self.assertFalse(os.path.exists(pkg.path(pu.DECIMAL_TU)))
        self.assertIn("stays on disk (--no-embed)", err)
        self.assertIn("maps update_plan.seeu beside itself", err)
        self.assertIn("removed update_plan_embedded.S", err)
        self.assertIn("removed update_plan_embedded.cc", err)
        self.assertNotIn("embedded update_plan.seeu", err)
        self.assertEqual(pkg.read(pu.BUILD_SCRIPT), MODERN_BUILD_SH)
        self.assertEqual(pkg.read(pu.PLAN_FILE, "rb"), pkg.plan)
        report = json.loads(pkg.read("pack.json"))
        self.assertEqual(report["schema"], 2)
        self.assertEqual(report["mode"], "disk")
        self.assertTrue(report["mode_forced"])
        self.assertIsNone(report["embedded_tu"])
        self.assertTrue(report["stub_removed"])
        self.assertTrue(report["decimal_tu_removed"])
        self.assertEqual(report["embed_limit"], pu.EMBED_LIMIT)

    def test_a_plan_at_the_limit_stays_on_disk_by_default(self):
        """The policy end to end, on a sparse file of exactly EMBED_LIMIT
        bytes (the packer sizes the plan; it reads its header only)."""
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        with open(pkg.path(pu.PLAN_FILE), "r+b") as f:
            f.seek(pu.EMBED_LIMIT - 1)
            f.write(b"\0")
        self.assertEqual(os.path.getsize(pkg.path(pu.PLAN_FILE)),
                         pu.EMBED_LIMIT)
        rc, err = run_main([pkg.dir, "--report", pkg.path("pack.json")])
        self.assertEqual(rc, 0, err)
        self.assertFalse(os.path.exists(pkg.path(pu.STUB_FILE)))
        self.assertIn(f"at or above the {pu.EMBED_LIMIT}-byte embed limit",
                      err)
        report = json.loads(pkg.read("pack.json"))
        self.assertEqual(report["mode"], "disk")
        self.assertFalse(report["mode_forced"])
        # --embed overrides the policy, with the limit named.
        rc, err = run_main([pkg.dir, "--embed"])
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.exists(pkg.path(pu.STUB_FILE)))
        self.assertIn("warning: --embed at", err)
        self.assertIn("may fail to link or load", err)
        # One byte under the limit, the default embeds silently.
        os.truncate(pkg.path(pu.PLAN_FILE), pu.EMBED_LIMIT - 1)
        rc, err = run_main([pkg.dir, "--report", pkg.path("pack.json")])
        self.assertEqual(rc, 0, err)
        self.assertNotIn("warning", err)
        self.assertEqual(json.loads(pkg.read("pack.json"))["mode"], "embed")

    def test_disk_mode_refuses_a_pre_disk_build_script(self):
        pkg = PackageDir(build_sh=PRE_DISK_BUILD_SH)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--no-embed"])
        self.assertEqual(rc, 1)
        self.assertIn("cannot build without an embedded plan", err)
        self.assertIn("--embed", err)
        # Nothing was written or removed: the decimal TU is still there.
        self.assertTrue(os.path.exists(pkg.path(pu.DECIMAL_TU)))
        self.assertFalse(os.path.exists(pkg.path(pu.STUB_FILE)))
        # The same package embeds fine.
        self.assertEqual(run_main([pkg.dir, "--embed"])[0], 0)
        self.assertTrue(os.path.exists(pkg.path(pu.STUB_FILE)))

    def test_missing_plan_or_script_is_exit_1(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        os.unlink(pkg.path(pu.PLAN_FILE))
        rc, err = run_main([pkg.dir])
        self.assertEqual(rc, 1)
        self.assertIn("cannot read plan", err)
        pkg2 = PackageDir()
        self.addCleanup(pkg2.cleanup)
        os.unlink(pkg2.path(pu.BUILD_SCRIPT))
        rc, err = run_main([pkg2.dir])
        self.assertEqual(rc, 1)
        self.assertIn("build.sh", err)
        # The failure happened before anything was written.
        self.assertFalse(os.path.exists(pkg2.path(pu.STUB_FILE)))

    def test_build_runs_the_packages_script_with_cxx(self):
        script = ("#!/bin/sh\nset -e\ncd \"$(dirname \"$0\")\"\n"
                  "printf '%s' \"$CXX\" > cxx.txt\n"
                  "CXX=:\n" + pu.STUB_COMPILE_LINE +
                  "printf '#!/bin/sh\\n' > model_update\nchmod +x model_update\n")
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build", "--cxx", "my-cross-c++",
                            "--report", pkg.path("pack.json")])
        self.assertEqual(rc, 0, err)
        self.assertEqual(pkg.read("cxx.txt"), "my-cross-c++")
        self.assertIn("built", err)
        report = json.loads(pkg.read("pack.json"))
        self.assertEqual(report["schema"], 2)
        self.assertTrue(report["built"])
        self.assertEqual(report["mode"], "embed")
        self.assertFalse(report["mode_forced"])
        self.assertEqual(report["embedded_tu"], pkg.path(pu.STUB_FILE))
        self.assertEqual(report["plan_bytes"], len(pkg.plan))
        self.assertIsInstance(report["build_seconds"], float)

    def test_failing_build_is_exit_1(self):
        script = "#!/bin/sh\nCXX=:\n" + pu.STUB_COMPILE_LINE + "exit 7\n"
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build"])
        self.assertEqual(rc, 1)
        self.assertIn("exit 7", err)

    def test_build_that_leaves_no_binary_is_exit_1(self):
        script = "#!/bin/sh\nCXX=:\n" + pu.STUB_COMPILE_LINE
        pkg = PackageDir(build_sh=script)
        self.addCleanup(pkg.cleanup)
        rc, err = run_main([pkg.dir, "--build"])
        self.assertEqual(rc, 1)
        self.assertIn("no executable", err)


class CliTest(unittest.TestCase):
    """Strict like every SeeAI tool: exit 2, never a default."""

    def assert_exit_2(self, argv):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                pu.parse_args(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_bad_arguments(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        self.assert_exit_2([])
        self.assert_exit_2([pkg.dir, "--bogus"])
        self.assert_exit_2([pkg.dir, "--bui"])           # no abbreviations
        self.assert_exit_2([pkg.dir, "--page-align", "1000"])
        self.assert_exit_2([pkg.dir, "--page-align", "32"])
        self.assert_exit_2([pkg.dir, "--page-align", "131072"])
        self.assert_exit_2([pkg.dir, "--page-align", "big"])
        self.assert_exit_2([pkg.dir, "--cxx"])
        self.assert_exit_2([pkg.dir, "--cxx", "--build"])
        self.assert_exit_2([pkg.dir, "--keep-decimal-tu"])   # no such flag
        self.assert_exit_2([pkg.dir, "--embed", "--no-embed"])  # one mode
        self.assert_exit_2([pkg.dir, "--disk"])              # no such flag
        self.assert_exit_2([pkg.path("absent")])
        self.assert_exit_2([pkg.path(pu.PLAN_FILE)])      # a file, not a dir

    def test_defaults(self):
        pkg = PackageDir()
        self.addCleanup(pkg.cleanup)
        opts = pu.parse_args([pkg.dir])
        self.assertEqual(opts.page_align, 16384)
        self.assertFalse(opts.build)
        self.assertIsNone(opts.cxx)
        self.assertIsNone(opts.report)
        self.assertEqual(opts.mode, "auto")
        self.assertEqual(pu.parse_args([pkg.dir, "--embed"]).mode, "embed")
        self.assertEqual(pu.parse_args([pkg.dir, "--no-embed"]).mode, "disk")


@unittest.skipUnless(os.path.exists(COMPILER) and CXX is not None,
                     f"no built seeml-update-compile under {BUILD}")
class DiskModeEndToEndTest(unittest.TestCase):
    """The disk mode for real (#158): compile the golden MLP with
    --no-embed, pack it with --no-embed, build it, and run the binary. The
    plan is found beside the executable (not in the working directory), a
    missing or swapped file fails closed with exit 1 and a message naming
    the mode, --plan overrides the location, and a plan that passes every
    check trains and commits."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="pack_update_e2e-")
        cls.pkg = os.path.join(cls.tmp, "pkg")
        cls.model = os.path.join(GOLDEN, "mlp.smf")
        cls.corpus = os.path.join(GOLDEN, "class.sds")
        done = subprocess.run(
            [COMPILER, "--source", cls.model, "--out", cls.pkg,
             "--data-batch", "4", "--steps", "4", "--no-embed"],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise AssertionError(done.stderr)
        rc, err = run_main([cls.pkg, "--no-embed", "--build",
                            "--report", os.path.join(cls.tmp, "pack.json")])
        if rc != 0:
            raise AssertionError(err)
        cls.pack_log = err
        cls.binary = os.path.join(cls.pkg, pu.BINARY)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_update(self, *extra, cwd=None, binary=None):
        out = os.path.join(self.tmp, "updated.smf")
        if os.path.exists(out):
            os.unlink(out)
        done = subprocess.run(
            [binary or self.binary, "--model", self.model, "--data",
             self.corpus, "--out", out, "--steps", "2", "--val-frac", "0",
             "--eval-every", "0", "--force", "--backend", "cpu", *extra],
            cwd=cwd or self.tmp, capture_output=True, text=True)
        return done.returncode, done.stderr, out

    def test_the_package_built_without_a_plan_tu(self):
        # (build.sh's own "built: ... (no embedded plan ...)" line streams
        # through on stdout; the packer's log is its stderr.)
        self.assertIn("stays on disk (--no-embed)", self.pack_log)
        self.assertIn("built", self.pack_log)
        self.assertFalse(os.path.exists(os.path.join(self.pkg, pu.STUB_FILE)))
        self.assertFalse(os.path.exists(os.path.join(self.pkg, pu.DECIMAL_TU)))
        self.assertFalse(os.path.exists(
            os.path.join(self.pkg, "update_plan_none.cc")))
        self.assertTrue(os.access(self.binary, os.X_OK))
        with open(os.path.join(self.tmp, "pack.json")) as f:
            report = json.load(f)
        self.assertEqual(report["mode"], "disk")
        self.assertTrue(report["built"])

    def test_the_plan_beside_the_binary_trains_and_commits(self):
        report = os.path.join(self.tmp, "run.json")
        rc, err, out = self.run_update("--report", report)
        self.assertEqual(rc, 0, err)
        plan = os.path.join(self.pkg, pu.PLAN_FILE)
        self.assertIn(f"plan from '{os.path.realpath(plan)}'", err)
        self.assertIn("mapped", err)
        self.assertIn("committed", err)
        self.assertTrue(os.path.getsize(out) > 0)
        with open(report) as f:
            run = json.load(f)
        self.assertEqual(run["plan"]["source"], "file")
        self.assertEqual(run["plan"]["path"], os.path.realpath(plan))
        self.assertEqual(run["plan"]["bytes"], os.path.getsize(plan))
        self.assertGreater(run["lifecycle"]["load_s"], 0.0)
        self.assertTrue(run["committed"])

    def test_the_plan_is_found_beside_the_executable_not_the_cwd(self):
        # The driver resolves its own location through the OS, symlinks
        # included (a macOS temp dir is one), so the message names the
        # real path.
        elsewhere = os.path.realpath(os.path.join(self.tmp, "elsewhere"))
        os.makedirs(elsewhere, exist_ok=True)
        moved = os.path.join(elsewhere, pu.BINARY)
        shutil.copy2(self.binary, moved)
        # The binary alone, run from the package directory: no plan beside
        # the executable, so it fails closed — exit 1, and the message
        # names the mode and both remedies.
        rc, err, _ = self.run_update(cwd=self.pkg, binary=moved)
        self.assertEqual(rc, 1, err)
        self.assertIn("cannot open the update plan", err)
        self.assertIn(os.path.join(elsewhere, pu.PLAN_FILE), err)
        self.assertIn("this package carries no embedded plan", err)
        self.assertIn("--plan", err)
        # --plan names the file wherever it is.
        plan = os.path.join(self.pkg, pu.PLAN_FILE)
        rc, err, _ = self.run_update("--plan", plan, binary=moved)
        self.assertEqual(rc, 0, err)
        self.assertIn(f"plan from '{plan}'", err)
        # And beside the moved binary, it is found again.
        shutil.copy2(plan, os.path.join(elsewhere, pu.PLAN_FILE))
        rc, err, _ = self.run_update(binary=moved)
        self.assertEqual(rc, 0, err)

    def test_a_swapped_or_corrupt_plan_fails_closed(self):
        plan = os.path.join(self.pkg, pu.PLAN_FILE)
        # Not a plan at all.
        rc, err, _ = self.run_update("--plan", self.model)
        self.assertEqual(rc, 1, err)
        self.assertIn("bad plan magic", err)
        # One flipped byte: the seal refuses it.
        with open(plan, "rb") as f:
            data = bytearray(f.read())
        data[len(data) // 2] ^= 0x5A
        corrupt = os.path.join(self.tmp, "corrupt.seeu")
        with open(corrupt, "wb") as f:
            f.write(data)
        rc, err, _ = self.run_update("--plan", corrupt)
        self.assertEqual(rc, 1, err)
        self.assertIn("plan hash mismatch", err)
        # Too short to hold a header.
        short = os.path.join(self.tmp, "short.seeu")
        with open(short, "wb") as f:
            f.write(data[:16])
        rc, err, _ = self.run_update("--plan", short)
        self.assertEqual(rc, 1, err)
        self.assertIn("smaller than a plan header", err)
        # A valid plan compiled from another model: the source-model hash
        # check refuses the pairing, exactly as it does for an embedded
        # plan.
        other_pkg = os.path.join(self.tmp, "other")
        done = subprocess.run(
            [COMPILER, "--source", os.path.join(GOLDEN, "eps6.smf"),
             "--out", other_pkg, "--data-batch", "4", "--loss", "mse",
             "--no-embed"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        rc, err, _ = self.run_update(
            "--plan", os.path.join(other_pkg, pu.PLAN_FILE))
        self.assertEqual(rc, 1, err)
        self.assertIn("source_model_hash", err)
        # A missing --plan value is a usage error, not a load error.
        done = subprocess.run([self.binary, "--plan"], capture_output=True,
                              text=True)
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
