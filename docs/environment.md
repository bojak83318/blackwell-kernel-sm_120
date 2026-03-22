# SM_120 Environment Notes

## Overall dependency matrix
The minimal dependency matrix is defined in `cmake/SM120Dependencies.cmake`. The current baseline values are:
- CUDA 12.8 (toolkit + driver that exposes `sm_120` compute capability — see the [NVIDIA CUDA 12.8 release overview](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/))
- CUTLASS 4.2.1 (pinned CUTLASS release for the kernels and the Python DSL preview that aligns with the PRD — see the [CUTLASS 4.2.1 overview](https://docs.nvidia.com/cutlass/4.2.1/media/docs/pythonDSL/overview.html))
- PyTorch 2.9.1+cu128 (runtime version targeted by the extension; the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) documents the CUDA 12.8 `cu128` wheel for 2.9.1)
- vLLM 0.12+ (the early integration hook for the GDN path; the [vLLM v0.12.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.12.0) enumerate the base features and CUDA alignment)
- TensorRT 9.2 (the later TRT-LLM plugin target / build-time reference)

The helper `sm120_print_dependency_matrix` prints this list during configuration so any change is easy to audit. Keep this file in sync with actual dependency upgrades; the version strings in `cmake/SM120Dependencies.cmake` drive both the root CMake build and downstream packaging. This matrix mirrors the baseline defined in `PRD_GDN_Kernel.md` and is backed by the cited release notes.

## Toolchain assumptions
- Build and test commands rely on the stock `cmake`/`ninja`/`make` trio from the host OS. Confirm `cmake --version` before recursively configuring to avoid silent platform drift.
- Python(command) (the same interpreter that runs the vLLM smoke tests) should match the interpreter used by CMake (`PYTHON_EXECUTABLE`), so run `python -m pip install -r ...` if needed for dependencies like `torch` or `vllm`.
- Ensure the local CUDA driver stack exposes compute capability `sm_120`. The RTX 5090 in the remote node already claims that, but validate `nvidia-smi` locally if you ever move to another GPU.
- Directory hierarchy follows the planned layout from `implementation.md`: top-level `include/`, `src/`, `python/`, `tests/`, `benchmarks/`, `docker/`, `scripts/`, `docs/`, and `artifacts/`. Keeping that layout consistent makes later phases deterministic.

## Arch and target assumptions
- The custom kernels target `sm_120`. Expect to compile with `-arch=sm_120` and to guard PTX for exactly that compute capability.
- During early phases the code should compile as a standalone CUTLASS kernel, not yet tied to PyTorch or vLLM, so keep those integration steps behind the documented gates.
- Long-context validation happens through the vLLM path, while TRT-LLM work waits until the vLLM path is stable; reference this doc when deciding whether a change belongs to Phase 2 or Phase 4.

## Quick verification checklist
1. Run `cmake -S . -B build -DSM120_BUILD_TESTS=ON` and ensure the dependency matrix logs the versions above.
2. Confirm `python -m pip show torch` (look for `2.9.1+cu128`) and `python -m pip show vllm` (≥ 0.12) before running integration scripts.
3. If you add a new dependency, update `cmake/SM120Dependencies.cmake` and rerun the configuration step to keep the matrix accurate.
