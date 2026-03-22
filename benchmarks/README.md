# SM_120 Benchmark Harness

The benchmark harness currently consists of a single placeholder executable and an orchestration script.

## Components
- `benchmark_main.cpp` builds `sm120_benchmark_main`, which supports `--profile`, `--iterations`, `--list`, and `--verbose` switches but does not execute real kernels yet.
- `run_benchmarks.sh` configures and builds the benchmark target, then runs the binary while piping output into `artifacts/benchmarks/benchmarks.log` for later inspection.
- The CMake target `sm120_benchmark_harness` groups the binary so the script can build the entire harness with one target.

## Running the harness
```
./benchmarks/run_benchmarks.sh [build-dir] [output-dir] [--extra bench args]
```
- `build-dir` defaults to `build/benchmarks` under the repo root.
- `output-dir` defaults to `artifacts/benchmarks`.
- `--extra bench args` are forward to `sm120_benchmark_main`, so you can set `--profile debug` or `--iterations 3` as needed.

## M2-C reproducible benchmark

Use `benchmarks/run_m2_benchmark.sh` to capture the reproducible M2-C benchmark artifacts that power our `artifacts/m2` directory. The script runs a full configure/build/run cycle and writes the logs described in `artifacts/m2/README.md`.

```
./benchmarks/run_m2_benchmark.sh [options] [-- extra bench args]
```

- Default build dir: `build/m2-benchmarks`.
- Default output dir: `artifacts/m2`, where the script places `m2-configure.log`, `m2-build.log`, `m2-benchmark.log`, `m2-run-metadata.txt`, and `m2-summary.md`.
- Default harness args: `--profile m2-c --iterations 3`.
- Additional options:
  - `--build-dir DIR`, `--output-dir DIR` override the directories.
  - `--profile NAME`, `--iterations N` override the harness options.
  - `--cmake-arg ARG` appends extra knobs to `cmake`.
  - Use `--` to forward remaining arguments straight to `sm120_benchmark_main` (e.g., `--verbose`).
