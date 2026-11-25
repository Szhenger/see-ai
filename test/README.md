# The SeeAI Test Tree

## How do you trust a compiler?

Think about what SeeAI claims: that it can differentiate your model correctly, plan every byte of memory without collisions, survive power cuts, and produce identical bits at any thread count. Extraordinary claims, and each one is *checkable* — which is what this tree is for. It's organized in the same fashion as the `compiler/` and `runtime/` subsystems: the harness and fixtures are partitioned per discipline behind façade headers, and the suites mirror the subsystem partition of the code they verify — **a suite lives where its subject lives**, so if you can find the code, you can find its tests, and vice versa.

```
test/
  framework/            SeeTest, the in-tree harness (façade: seetest.h)
    registry.{h,cc}     test registration + the runner (--list, --filter)
    matcher.h           value formatting, safe comparisons, failure messages
    assert.h            the TEST / EXPECT_* / ASSERT_* / SKIP macro surface
    seetest_main.cc     the runner main linked into every suite
  support/              shared fixtures (façade: builders.h)
    models.cc           deterministic SMF (SeeAI Model Format) model builders + BaseConfig
    corpora.cc          synthetic datasets (classification/regression/unlabeled)
    probes.cc           engine-arena introspection + the test-run environment
    scoped_temp_dir     filesystem sandbox for I/O suites
  fixtures/             the checked-in inputs
    golden/             four SMF / SDS files (make_golden.py writes them), read by both planes; screens/ holds the screens' golden text
    tokenizer/          adversarial.txt, the Python plane's tokenizer parity corpus
  source/               one folder per source-language subsystem
    parallel/           parallel_for
    identity/           hash
    statistics/         screen
    tokenizer/          bpe  ingest
    contract/           split
    recipe/             recipe
  compiler/             one folder per compiler subsystem
    frontend/           model_io  resource_analyzer  sir  operator  parser  golden_fixture
    analysis/           update_passes  pass_manager  statistics
    backend/            architecture  lowering  native_emitter
    driver/             update_compiler  driver  screens
    diagnostics/        diagnostics
  runtime/              one folder per runtime subsystem
    pipeline/           dataset  batch_pipeline
    executor/           kernels  metal_gemm  metal_backend
    verifier/           verifier
    storage/            storage
    dispatcher/         engine  update_engine
  system/               the cross-half end-to-end update
    update_system_test
  fuzz/                 the fuzz harness, its seed corpus, and the suite that replays it
    binary_formats.cc   the harness: nine arms behind one entry point (libFuzzer under -DSEEML_FUZZ=ON)
    fuzz_reach.h        the input framing, and the counters of what the inputs reached
    plan_mutate.h       the plan-aware edit: one operand field of one instruction (the libFuzzer build's custom mutator)
    fuzz_replay_test    every seed through the harness in every build, what it must reach asserted
    make_seeds.py       writes seeds/ (the standard library and the built seeml-update-compile)
    seeds/              one valid input per arm, one sealed plan per fixture family; regressions/ keeps inputs that once hung or crashed
  tool/                 the Python plane (unittest, standard library only)
    autotune_test.py          tool/autotune.py: the sweep, the scoring, the decision rule and the table merge, against a fake seeml-bench
    bench_compare_test.py     tool/bench_compare.py, the Tier A regression gate, on synthetic reports and two recorded incidents
    bench_runtime_test.py     seeml-bench rows carry the runtime screen model_update prints (runs the built seeml-bench)
    certify_numerics_test.py  tool/certify_numerics.py: what a certificate binds to, the relaxed model's bound, certificates over compiled plans
    cli_exit_codes_test.py    the exit-code contract of the six built C++ tools (0 done, 1 failed, 2 bad command line)
    driver_shape_test.py      the driver contains orchestration only: no function in compiler/driver/ is longer than 150 lines
    energy_test.py            tool/seeml/energy.py: the energy meter never raises; the per-step regression is the step time's
    export_model_test.py      tool/export_model.py: the streaming writers against the original per-row and whole-blob algorithms, byte for byte
    file_mode_test.py         no file is created with fopen: the platform seam states the mode (CWE-732)
    formats_test.py           the byte seam between the planes: formats.py against abi.json, seeml-abi, the golden fixtures, the two plan decoders
    frontier_exec_test.py     tool/frontier_exec.py, the reference executor: on its own (finite differences), then differentially against the C++ runtime
    frontier_run_test.py      tool/frontier_run.py, the frontier harness: the adapted-set rule, the parity checks, the split rule, the report
    function_length_test.py   no function over 200 lines in compiler/, runtime/ or tool/, by driver_shape_test.py's scanner
    fuzz_seeds_test.py        test/fuzz/seeds/ is what test/fuzz/make_seeds.py writes today: byte for byte but for the plans, which the replay suite holds to being accepted
    masked_loss_test.py       the masked cross-entropy kernels of the oracle against PyTorch's cross_entropy(ignore_index=-1) and its gradient
    pack_update_test.py       tool/pack_update.py, the package assembler, on fabricated package directories
    torch_step_test.py        the compiled train step next to PyTorch: a decoder and an MLP written in torch (float64), compiled with LoRA, a few steps through seeml-plan-probe, the adapters against torch.autograd and torch.optim (tier 2: torch)
    recipe_test.py            the recipe at the command line: a recipe file compiles to the plan its flags compile to; the refusals name the field
    runner_stop_test.py       cooperative cancellation and --threads through a built model_update: SIGTERM stops at a step boundary, exit 5, resumable
    runner_text_test.py       raw text on the device: a run on .txt / .jsonl commits the bytes a run on the compiler's canonical SDS commits
    seetest_skip_test.py      SeeTest's SKIP(), seen from outside the process: printed and counted as SKIPPED, never a pass or a failure
    tokenizer_test.py         the SMF tokenizer section and the core's encoder: the builder's refusals, parity with the reference tokenizer
```

## Why an in-tree harness?

Fair question — GoogleTest exists. But recall the product's core constraint: the runtime is *zero-dependency*, and the project prizes knowing exactly what every byte of its code does. SeeTest is a few hundred lines implementing precisely what the suites need — a `TEST` macro that self-registers (each `TEST` block constructs a static registrar object before `main` runs, the classic trick), `EXPECT_*` (record the failure, keep going) versus `ASSERT_*` (this test can't meaningfully continue), a matcher layer that formats both sides of a failed comparison so failures read like sentences, `SKIP("why")` for a test the host cannot run (no Metal device, say) — printed and counted as `[  SKIPPED ]` with its reason, apart from the passes, so a hardware-gated test never reports green for zero work — and a runner with `--list` and `--filter`. Nothing more, because nothing more is needed — and when a test fails on some exotic target, the entire harness is right there to read.

The `support/` fixtures earn their keep the same way: **deterministic** model builders and synthetic corpora (seeded, so a failure reproduces exactly), `probes.cc` for peeking inside an engine's arena from a test, and `scoped_temp_dir`, which sandboxes every I/O suite into a temp directory that cleans itself up — tests that touch the filesystem must not be able to touch *your* filesystem.

## What the suites actually prove

Worth knowing the flavors, because they answer different questions:

- **Unit suites** pin one module's behavior — `sir` proves `Block::verify()` catches each corruption it claims to; `verifier` proves each opcode's bounds math, including the overflow cases; `hash` pins the digests so an accidental algorithm change can't slip by.
- **Numeric suites** check kernels against independent references — and the autodiff machinery is verified by **finite-difference gradient checking**: nudge a parameter by ε, watch the loss, and compare `(L(θ+ε) − L(θ−ε)) / 2ε` against the gradient the compiled backward pass computes. Calculus, cross-examined by arithmetic. (This is why the compiler can emit a plan *without* the optimizer step — a pure forward+backward program exists precisely to be gradient-checked.)
- **Property/regression suites** pin the invariants the docs promise: the pipelined batch sequence is *exactly* the serial one; every instruction the compiler emits passes the runtime's validator; corrupted checkpoints are rejected; parallel results match serial bit-for-bit.
- **The system suite** (`system/update_system_test`) runs the whole story — export-shaped model in, train, gate, merge, commit — proving the two halves agree about every format and contract between them.
- **Fuzzing** (`fuzz/binary_formats.cc`, libFuzzer) feeds every parser of outside bytes adversarial garbage for hours on end — the SMF, SDS and plan readers, the tokenizer section, the JSONL and text ingest, the recipe file, the exporter's report — and *runs* what they accept: an accepted plan is executed, an accepted model compiled and its plan executed. Run under sanitizers, it's the sharpest tool we have against parser bugs. But a fuzzer that never gets an input past a seal looks exactly like one that does, so the harness **counts what its inputs reach** (`fuzz_reach.h`), and `fuzz/fuzz_replay_test` — an ordinary suite, in every build and under every sanitizer job — runs the checked-in seed corpus through the harness's own entry point and fails, naming the seed, when an arm is not reached: no accepted plan, no training step, no completed compile, no ingested window. It also replays `seeds/regressions/`, refuses a battery of malformed inputs, and runs a fixed pass of structured mutations over every seed. A plan is megabytes of weights around a few hundred instructions, so a byte-level mutator almost never writes an operand — the fields the validator's bounds rules read. `fuzz/plan_mutate.h` does exactly that edit: the libFuzzer build makes it for three in four mutations of a re-sealed plan unit (`LLVMFuzzerCustomMutator`), and the replay suite makes a few thousand of them over the plan seeds in every build, where a bounds rule that stopped refusing ends the suite in a crash.

## Running things

Every suite is one executable (`seeml_<basename>`), built by both `build/build.sh` and CMake/CTest; run one directly with e.g.

```bash
./build/seeml_pass_manager_test --filter=DeadCodeElimination
```

Suites include only the façades (`test/framework/seetest.h`, `test/support/builders.h`); the split units behind them can be reorganized without touching any suite — the same façade discipline as the code under test. (`fuzz_replay_test` also includes `test/fuzz/fuzz_reach.h` and links the harness: the harness is its subject.)

The seed corpus is generated, never edited:

```bash
python3 test/fuzz/make_seeds.py            # after a build: rewrite test/fuzz/seeds/
python3 test/fuzz/make_seeds.py --check    # what test/tool/fuzz_seeds_test.py runs
```

A plan-version bump — or any change that makes the loader refuse yesterday's plans — turns `fuzz_replay_test` red, naming each stale seed and the first command above; regenerate in the commit that moved the format. Everything but the plans is byte-compared against the generator on every host; a plan is the compiler's output on the host that generated it, so it is held to being *accepted and executed* instead. `seeds/regressions/` holds inputs that once hung or crashed the harness, named by libFuzzer's prefix for the kind (`slow-unit-`, `crash-`, `oom-`, `timeout-`). One that is a plan is *derived* in `make_seeds.py` (`build_regressions`: a seed with a header field rewritten), so a plan-version bump regenerates it with everything else; add a reproducer there, with its fix.

The suite's mutation pass is tunable from the environment — `SEEML_FUZZ_MUTATIONS` (inputs per seed, default 128), `SEEML_FUZZ_MUTATION_SEED`, and `SEEML_FUZZ_KEEP=1`, which writes each input before it runs to `seeml_fuzz_keep.unit` in the system's temp directory, so a crash or a hang leaves its reproducer on disk (a switch, not a path: the suite opens no file the environment names):

```bash
SEEML_FUZZ_MUTATIONS=5000 SEEML_FUZZ_KEEP=1 ./build/seeml_fuzz_replay_test --filter=MutatedSeeds
```

The Python-plane tools have their own suite under `test/tool/`, run with `python3 -m unittest discover -s test/tool -p '*_test.py'` and needing nothing beyond the standard library (plus a C++ driver on `PATH` for the one test that assembles the packer's `.incbin` stub for real).
