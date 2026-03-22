# vLLM SM_120 Registration Recipe

Use this recipe to connect the SM_120 GDN kernel into vLLM once the PyTorch extension and custom op are available.

## 1. Build the SM_120 extension

The first build step is already covered by the shared `cmake` configuration. The goal is to produce a library that exposes `torch.ops.sm120`. Typical commands:

```sh
cmake -S . -B build/m4 \
      -DSM120_BUILD_TESTS=ON \
      -DSM120_ENABLE_VLLM_CHECKS=ON \
      -DSM120_BUILD_BENCHMARKS=OFF
cmake --build build/m4 --target sm120_ops
```

The PyTorch binding target is `sm120_ops` and emits `libsm120_ops.so` (for example, `build/m4/src/ops/libsm120_ops.so`). Keep the resulting `.so` path handy for the registration script.

## 2. Load the custom op before vLLM starts

Before any vLLM worker instantiates the Qwen 3.5 model, load the shared object so `torch.ops.sm120` exists. A helper script might look like this:

```py
import torch
from pathlib import Path

LIB_PATH = Path("build/m4/src/ops/libsm120_ops.so")

if not LIB_PATH.exists():
    raise FileNotFoundError(f"Missing SM_120 extension: {LIB_PATH}")

torch.ops.load_library(str(LIB_PATH))
assert hasattr(torch.ops, "sm120"), "sm120 namespace should exist after loading the extension"
```

Run the helper once in the same environment that will run vLLM.

## 3. Patch vLLM's attention pipeline to call the custom op

With the custom op exposed, wire it into vLLM by replacing the default GDN implementation that the model would normally call. The exact hook depends on the vLLM version, but the general pattern is:

1. Import the module that defines the GDN kernel or attention layer.
2. Replace or wrap the GDN helper with a thin wrapper that calls `torch.ops.sm120.gdn`.
3. Ensure the wrapper signature matches the tensors vLLM passes (query/key/value shapes, dtype, stride).

For example (pseudocode):

```py
from vllm.ops import gdn as vllm_gdn
import torch

def custom_gdn(query, key, value, **kwargs):
    return torch.ops.sm120.gdn(query, key, value, **kwargs)

vllm_gdn._fast_path = custom_gdn
```

Document the exact symbol that needs patching once the integration file structure is finalized; treat the above snippet as guidance for when the hook exists.

In this repository, use these entrypoints:
- `scripts/serve_qwen35_hf_baseline.sh` for native vLLM baseline.
- `scripts/serve_qwen35_hf_sm120.sh` to load `ops.vllm_sm120_plugin` (via `VLLM_PLUGINS`) and register `torch.ops.sm120.gdn` before server startup.

## 4. Keep registration deterministic

- Always load the shared object before `vllm.entrypoints` spins up worker processes so the custom op is available in every process.
- If you spawn multiple worker processes (for example via `vllm.cli:serve`), keep the registration helper in the entrypoint or use `multiprocessing.set_start_method("spawn")` to avoid missing symbols.
- Capture the registration log (the commands that load the shared object and patch vLLM) in `artifacts/m4/logs/register.log` for future reference.

## 5. Verify the registration path

Run `python test/test_vllm_integration.py` after you load the library. The script checks that:

- PyTorch and vLLM meet the minimum versions from `cmake/SM120Dependencies.cmake`.
- CUDA is available (warns if not).
- The `torch.ops.sm120.gdn` symbol exists.

This script is the lightweight smoke test for the M4-C slice; keep the log next to the other `artifacts/m4` outputs.
