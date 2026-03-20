# SM_120 Environment Notes

## Overall dependency matrix
The minimal dependency matrix is defined in `cmake/SM120Dependencies.cmake`. The current baseline values are:
- CUDA 12.0 (toolkit + driver that exposes `sm_120` compute capability)
- CUTLASS 3.0 (pinned CUTLASS release for the kernels)
- PyTorch 2.2 (runtime version targeted by the extension)
- vLLM 1.0 (the early integration hook for the GDN path)
- TensorRT 9.2 (the later TRT-LLM plugin target / build-time reference)

The helper `sm120_print_dependency_matrix` prints this list during configuration so any change is easy to audit. Keep this file in sync with actual dependency upgrades; the version strings in `cmake/SM120Dependencies.cmake` drive both the root CMake build and downstream packaging.

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
2. Confirm `python -m pip show torch vllm` (or install the pinned versions) before running integration scripts.
3. If you add a new dependency, update `cmake/SM120Dependencies.cmake` and rerun the configuration step to keep the matrix accurate.
