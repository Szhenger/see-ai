#ifndef SEEML_TEST_FRAMEWORK_REGISTRY_H_
#define SEEML_TEST_FRAMEWORK_REGISTRY_H_

#include <string>

// =============================================================================
// registry/ discipline of SeeTest: test registration and the runner. Tests
// self-register via static Registrar objects (constructed by the TEST macro
// in assert.h); the runner executes them in registration order and owns the
// per-test state that ReportFailure and ReportSkip record into.
// =============================================================================

namespace seeml::testing {

using TestFn = void (*)();

/// Registers a test. Invoked by TEST() through static Registrar objects; the
/// runner executes tests in registration order (deterministic within a TU).
void RegisterTest(const char* suite, const char* name, TestFn fn);

/// Records a failure against the currently running test and prints it.
/// Aborts if no test is running (assertion used outside the harness).
void ReportFailure(const char* file, int line, const std::string& message);

/// Marks the currently running test skipped, with the reason it cannot run
/// here (a capability the host lacks), and prints it. Invoked by SKIP(),
/// which also returns from the test. A skipped test is not a passed one: it
/// is reported and counted apart. A failure recorded before the skip still
/// fails the test. Aborts if no test is running.
void ReportSkip(const char* file, int line, const std::string& reason);

/// Runs every registered test selected by argv. Returns the process exit
/// code: 0 iff no selected test failed (a skipped test does not fail the
/// run).
int RunAllTests(int argc, char** argv);

namespace internal {

struct Registrar {
  Registrar(const char* suite, const char* name, TestFn fn) {
    RegisterTest(suite, name, fn);
  }
};

}  // namespace internal
}  // namespace seeml::testing

#endif  // SEEML_TEST_FRAMEWORK_REGISTRY_H_
