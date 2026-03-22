# M2 Benchmark Artifacts

Store reproducible M2-C benchmark evidence in this directory after running `benchmarks/run_m2_benchmark.sh`.

Contents:

- `m2-configure.log`: output from the `cmake` configure step with the options that were used.
- `m2-build.log`: build-phase output for the `sm120_benchmark_main` target.
- `m2-benchmark.log`: the benchmark binary's stdout plus the profile/iteration header the script emits.
- `m2-run-metadata.txt`: git commit, status, system info, and full command strings so the run can be replayed.
- `m2-summary.md`: quick-reference summary of the profile, iteration count, and artifact names.

The script overwrites these files on each run to keep the directory focused on the latest reproducible evidence.
