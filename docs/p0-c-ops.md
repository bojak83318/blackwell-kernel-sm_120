# SM_120 Phase 0-C Operational Notes

## Environment assumptions
- Local work runs inside `/home/rocm/workspace/blackwell-kernel-worktrees/sm_120-dev` with an existing CMake/Ninja build stack (confirm via `cmake --version` and `ninja --version`).
- The repo already includes `docs/environment.md`, so this slice adds the actionable execution steps that reference the shared dependency matrix.
- CUDA tooling is expected locally for non-kernel tasks, but the remote validation node (Ubuntu, NVIDIA GeForce RTX 5090) still does **not** expose `nvcc` globally, so any CUDA/CUTLASS/host build must run inside a container that supplies the CUDA toolchain. The host does expose `cmake`, `docker`, and `kubectl` globally.
- Remote node services: Docker, `kubectl` (for the local k3s control plane), and hardware (RTX 5090) are present. All remote validation runs assume GPU access through Docker+NVIDIA Container Toolkit and Kubernetes operations through `kubectl` bound to the local `k3s` cluster.
- Network access is limited to the agent tunnel; keep remote commands precise and idempotent to avoid repeated data transfer.

## Local command prerequisites
1. **Source checkout & configuration**
   - Fetch/update submodules if needed, then run `cmake -S . -B build -DSM120_BUILD_TESTS=ON` from the repo root. The top-level `CMakeLists.txt` and `cmake/SM120Dependencies.cmake` encode the version matrix, so the configure step must complete before running targets.
2. **Build/test entry points**
   - After configuration, invoke `cmake --build build --target <target>` (e.g., `sm120-bootstrap`, `sm120-tests`) or run `ninja -C build` directly. Keep `build/bin` on the PATH for subsequent commands, or call targets through `cmake --build` each time.
3. **Python dependency validation**
   - Confirm `python -m pip show torch` (should report `2.9.1+cu128`) and `python -m pip show vllm` (≥ 0.12)—see `PRD_GDN_Kernel.md` for the pinned versions—before executing Python scripts under `python/` or `scripts/`. Set `PYTHON_EXECUTABLE` in CMake if the interpreter differs from the default on the system.
4. **Documentation / evidence captures**
   - Use `docs/environment.md` as the source of truth for dependency versions. Link any new tooling requirements back to that file when adding CMake config.

## Remote node workflow: disk-pressure preflight and Docker-based validation
1. **Current remote state**
   - Host: `ubuntu@192.168.0.233` (`kimi-k3s-node`). GPU: NVIDIA GeForce RTX 5090. Existing tooling: Docker (with NVIDIA runtime support) and `kubectl` pointed at a preconfigured k3s cluster. Reported disk use: root filesystem ~91% used with ~46 GiB free, inode usage ~3%.
2. **K3s disk-pressure preflight**
   - SSH into the node and run `kubectl describe node kimi-k3s-node` to ensure `DiskPressure=False` (and also confirm `MemoryPressure=False` and `PIDPressure=False`). If the condition changes, halt and report the precise status.
   - Double-check storage headroom via `df -h /` and `df -i /` before container work. If free space drops below ~40 GiB or `DiskPressure` flips to `True`, pause and wait for remediation.
3. **Docker-based validation prerequisites**
   - The remote environment lacks `cmake`/`nvcc`, so run CUDA-sensitive builds inside a Docker image that bundles the required toolchain. Example flow:
     ```sh
     docker pull nvcr.io/nvidia/pytorch:24.11-py3
     docker run --rm -it --gpus all \
       -v /home/rocm/workspace/blackwell-kernel-worktrees:/workspace \
       -w /workspace/blackwell-kernel-worktrees/sm_120-dev \
       nvcr.io/nvidia/pytorch:24.11-py3 \
       bash -lc 'cmake -S . -B build -DSM120_BUILD_TESTS=ON && cmake --build build'
     ```
   - Ensure the chosen container supplies `nvcc` or `cuda-compiler` and the NVIDIA driver stack matching the RTX 5090. The host already provides `cmake`; keep `ninja` in the container if the image does not ship it.
   - Inside the container, run `nvidia-smi` to confirm the GPU is exposed before invoking CUTLASS/CUDA tasks.
4. **Remote verification steps**
   - Use `ssh` to copy any generated artifacts back to the main workspace if further analysis is required, but prefer to store logs under `artifacts/remote/` within the repo tree.
   - When running `kubectl` workflows (e.g., applying a job or checking cluster logs), keep the commands idempotent and note any generated manifests in `docs/` for future reference.

## Milestone harness targets & artifacts

- **M1 test harness** – The correctness milestone focuses on the `sm120_test_harness` target declared in `tests/CMakeLists.txt`. The convenience script `./tests/run_tests.sh` configures the tree with `-DSM120_BUILD_TESTS=ON`, builds the aggregate target, runs `ctest`, and writes `ctest.log` into `artifacts/m1/tests`. Keep M1 evidence (unit + integration pass reports, diff comparisons, etc.) inside that directory; its `README.md` explains how to append new artifacts without confusing later phases.
- **M2 benchmark harness** – The performance milestone centers on the `sm120_benchmark_harness` target defined in `benchmarks/CMakeLists.txt`. Running `./benchmarks/run_benchmarks.sh` (itself invoking `cmake --build ... --target sm120_benchmark_harness`) produces `benchmarks.log` under `artifacts/m2/benchmarks` and uses the same path as the `SM120_BENCHMARK_EVIDENCE_DIR` variable for future kernel plumbing. Record throughput traces, `nvprof` output, or compiled kernel metadata in that directory so M2 reviewers can find them without digging through build caches.
- **Verifying harness commands** – Treat these scripts as the canonical way to drive each harness. After tweaks to the tooling, rerun `./tests/run_tests.sh ./build/tests ./artifacts/m1/tests` and `./benchmarks/run_benchmarks.sh ./build/benchmarks ./artifacts/m2/benchmarks` to confirm the targets still configure, build, and log correctly. Capture the exact command line in `docs/` or under `artifacts/m1/tests/commands.log` (and the M2 equivalent) to document what “working harness” looked like when this phase was closed.

## Command & environment coordination notes
- Always capture the exact commands executed (including arguments) in `docs/` or `artifacts/` so the next phase can reproduce the preflight steps.
- Align any new environment variables (e.g., `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`) with both local and dockerized runs; mention them in this doc before introducing them elsewhere.
- If a container run fails due to missing GPU access or disk pressure, record the failure and the host status (DiskPressure, available disk) before retrying.
