# M4-C (vLLM) Run Commands

This document records the commands that prove the SM_120 GDN kernel can be exercised through the vLLM execution path once the PyTorch extension and registration helpers exist.

## Prerequisites
- Follow `docs/environment.md` to install CUDA 12.8, CUTLASS 4.2.1, PyTorch 2.9.1+cu128, and vLLM 0.12+.
- Ensure `torch.ops.sm120` gets created by loading the extension via the helper from `docs/vllm_registration.md` before running vLLM.
- Use the same Python interpreter that CMake will register (`PYTHON_EXECUTABLE`).

## Local workflow
1. **Configure / rebuild the M4 tree**
   ```sh
   cmake -S . -B build/m4 \
         -DSM120_BUILD_TESTS=ON \
         -DSM120_ENABLE_VLLM_CHECKS=ON \
         -DSM120_BUILD_BENCHMARKS=OFF \
         -DSM120_ENABLE_TENSORRT_LLM=OFF
   cmake --build build/m4 --target sm120_kernel
   ```
   Keep the build directory dedicated to the M4 slice so logs and artifacts stay organized.

2. **Run the vLLM readiness check**
   ```sh
   python test/test_vllm_integration.py | tee artifacts/m4/logs/vllm_check.log
   ```
   This smoke test verifies the dependency versions, CUDA availability, and the existence of `torch.ops.sm120.gdn`. Capture the output alongside the other `artifacts/m4` logs.

3. **Document the registration command**
   - Because `vllm` requires the custom op to be registered before launching the server, log the registration call that loads `libsm120_pytorch.so` and patches `vllm` in `artifacts/m4/logs/register.log`.

4. **Optional: start a minimal vLLM session**
   When the integration hook is in place, a representative command looks like:
   ```sh
   python -m vllm.entrypoints.run \
         --model qwen-3.5-7b-demo \
         --max-output-length 8 \
         --tensor-parallel-size 1
   ```
   Keep the actual invocation aligned with the checkpoint/weights you are validating. Redirect the stdout/stderr to `artifacts/m4/logs/vllm_run.log` whenever you capture evidence.

## Remote (Docker) verification
Use the NVIDIA container image that already bundles CUDA/PyTorch if the local machine lacks the full toolchain.

```sh
docker run --rm --gpus all \
  -v /home/rocm/workspace/blackwell-kernel-worktrees:/workspace \
  -w /workspace/submini/m4-c \
  nvcr.io/nvidia/pytorch:24.11-py3 \
  bash -lc '
    cmake -S . -B build/m4 \
          -DSM120_BUILD_TESTS=ON \
          -DSM120_ENABLE_VLLM_CHECKS=ON \
          -DSM120_BUILD_BENCHMARKS=OFF \
          -DSM120_ENABLE_TENSORRT_LLM=OFF && \
    cmake --build build/m4 --target sm120_kernel && \
    python test/test_vllm_integration.py
  '
```

Keep the logs produced inside the container (for example, redirect from the `python` command) and copy them back into `artifacts/m4/logs` for future reference.

## After the run
1. Verify that `artifacts/m4/logs/*` contains:
   - `vllm_check.log` (test output)
   - `register.log` (registration command)
   - `vllm_run.log` (optional vLLM inference)
2. If the smoke test fails, revisit `docs/vllm_registration.md` to ensure the extension is loaded before the run.
3. Keep this document updated whenever you tweak the commands, especially the Docker run line and the path to the shared object.
