# SM_120 Test Harness

The `tests/` directory contains the starting scaffolding for unit and integration suites that will drive SM_120 correctness work later.

## Layout
- `unit/` builds `sm120_unit_tests`, a placeholder executable that accepts `--mode`, `--list`, and `--verbose` switches.
- `integration/` currently builds `sm120_integration_tests`, which simply reports that no kernels run yet.
- `run_tests.sh` wraps configuration, build, and `ctest` invocation while writing logs to `artifacts/tests/`.

## Running the harness
```
./tests/run_tests.sh [build-dir] [output-dir]
```
- `build-dir` defaults to `build/tests` under the repo root.
- `output-dir` defaults to `artifacts/tests` and captures `ctest` output for later review.
- The script ensures `sm120_test_harness` is built and runs every registered test.
