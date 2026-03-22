# M1-C Build, Run, and Disassembly Workflow

This project exposes a dedicated helper for the *M1-C* milestone: `scripts/run_m1c.sh`. The helper mirrors the expected SM_120 build steps, exercises the placeholder test/benchmark harnesses, and collects the artifacts required for later analysis.

## Running the helper

```sh
scripts/run_m1c.sh
```

On a clean checkout the script will:

1. Configure a release build under `build/m1` with `SM120_TARGET_ARCH=sm_120`, tests enabled, and benchmark targets enabled.
2. Build the project using all available CPUs (`nproc` by default) and write the console output to `artifacts/m1/logs/configure.log` and `artifacts/m1/logs/build.log`.
3. Run `ctest` (parallelized) with output in `artifacts/m1/logs/tests.log`.
4. Launch the benchmark harness (`sm120_benchmark_main --profile context_throughput --iterations 2 --verbose`) and log to `artifacts/m1/logs/benchmarks.log`.
5. Capture any PTX files emitted by the kernel-level custom targets (`gdn_kernel_sm120.ptx`, `gdn_tiled_sm120.ptx`, `tcgen05_probe_sm120.ptx`) into `artifacts/m1/ptx/`.
6. Disassemble the first `gdn_kernel.cu.o` object that CMake produces using `cuobjdump`, `nvdisasm`, or `objdump`, with the transcript stored at `artifacts/m1/logs/disassembly.log`.
7. Print a completion message pointing to `artifacts/m1` so you can inspect the results.

### Customizing behavior

The script exposes the following flags (see `--help` for the same text):

- `--skip-build` – only useful if the build is already configured/built; avoids rerunning `cmake`.
- `--skip-tests` / `--skip-bench` – skip the test or benchmark runs when fast iteration is needed.
- `--skip-disasm` – skip the disassembly step when the tooling (e.g., `cuobjdump`/`nvdisasm`) is unavailable.

Environment variables you can adjust:

- `BUILD_DIR` and `ARTIFACT_ROOT` to point the script at non-default paths.
- `PARALLEL_JOBS` to override the number of `cmake --build`/`ctest` worker threads.
- `CMAKE_BUILD_TYPE`, `SM120_TARGET_ARCH`, or other `CMAKE_ARGS` can be modified by editing the script or passing extra `CMAKE_ARGS` directly in the helper (the defaults already match the SM_120 goals).

## Artifact layout

| Path | Description |
|---|---|
| `artifacts/m1/logs/configure.log` | CMake configure output with dependencies and options. |
| `artifacts/m1/logs/build.log` | `cmake --build` output for the release target. |
| `artifacts/m1/logs/tests.log` | `ctest` log for all unit/integration harnesses. |
| `artifacts/m1/logs/benchmarks.log` | Benchmark harness stdout (placeholder numbers). |
| `artifacts/m1/logs/disassembly.log` | The disassembly transcript from `cuobjdump`/`nvdisasm`/`objdump`. |
| `artifacts/m1/ptx/*.ptx` | PTX files emitted by the SM_120 CUDA targets to aid manual inspection. |

The script also guarantees `artifacts/m1/.gitkeep` exists so Git can track the directory even before your first run.

## When to use this helper

- As a quick verification that the SM_120 build still configures/builts on the target machine.
- When capturing an initial disassembly for `gdn_kernel.cu` so upstream reviewers can inspect the generated SASS.
- Before running longer vLLM/TensorRT-LLM workloads so you can reference the `artifacts/m1` logs and PTX as a baseline.

If you need to re-run portions of the workflow, combine the `--skip-*` flags (for example `scripts/run_m1c.sh --skip-build --skip-tests`).
