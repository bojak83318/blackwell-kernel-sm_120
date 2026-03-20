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
