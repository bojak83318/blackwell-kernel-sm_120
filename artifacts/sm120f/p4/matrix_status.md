# SM120f Target Matrix Status

Evidence captured on 2026-03-29.

## Scope

Acceptance gate: `The full target matrix passes on RTX 5090, RTX 5080, and RTX 5070 Ti with CUDA 13.0+`.

## What is executable now

- RTX 5090 host `kimi-k3s-node` is reachable over SSH and reports `GPU 0: NVIDIA GeForce RTX 5090`.
- The node is healthy for validation work:
  - `MemoryPressure=False`
  - `DiskPressure=False`
  - `PIDPressure=False`
  - `Ready=True`
- The remote root filesystem has headroom for validation runs: `/` is `474G` total with `120G` free.
- The remote host has the basic operator tools needed for validation orchestration:
  - `Python 3.12.3`
  - `cmake 3.28.3`
  - `Docker 28.2.2`
  - `kubectl v1.33.6+k3s1`
- Existing artifact evidence already proves the runnable SM120 path:
  - `artifacts/sm120f/p1/test_output.log` records `compute_120f` on CUDA `>= 13.0` and fallback to `compute_120a` below that floor.
  - `artifacts/sm120f/p3/verification.txt` records TMA symbol and ACQBULK instruction presence.
  - `artifacts/sm120f/p4/verification.txt` records zero divergence validation.

## Explicit blockers

### RTX 5080

- No accessible RTX 5080 host is available in this workspace.
- No `nvidia-smi` evidence, benchmark log, or CUDA 13+ execution artifact exists for this GPU in `artifacts/sm120f/`.
- Result: this leg cannot be executed or marked complete until hardware access is provided.

### RTX 5070 Ti

- No accessible RTX 5070 Ti host is available in this workspace.
- No `nvidia-smi` evidence, benchmark log, or CUDA 13+ execution artifact exists for this GPU in `artifacts/sm120f/`.
- Result: this leg cannot be executed or marked complete until hardware access is provided.

### CUDA 13+ build path on the remote node

- The remote 5090 node does not expose `nvcc` globally.
- Result: any fresh CUDA compile or benchmark on that node still needs a container image that supplies the CUDA toolchain.

## Command evidence

- `ssh ubuntu@192.168.0.233 'hostname && nvidia-smi -L && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && df -h /'`
- `ssh ubuntu@192.168.0.233 'kubectl describe node kimi-k3s-node'`
- `ssh ubuntu@192.168.0.233 'python3 --version; cmake --version; docker --version; kubectl version --client --output=yaml'`
