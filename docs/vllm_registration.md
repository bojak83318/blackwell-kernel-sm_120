# GDN SM_120 Op — vLLM Registration Guide

## Overview

`libgdn_sm120_op.so` exposes the fused GDN state-update kernel as a PyTorch
custom op: `torch.ops.gdn_sm120.state_update`. It can be loaded into any
PyTorch or vLLM process without modifying framework source.

## Build

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

cmake -S . -B build/m4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSM120_BUILD_TESTS=OFF \
  -DSM120_BUILD_OPS=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

cmake --build build/m4 --target gdn_sm120_op --parallel $(nproc)
# output: build/m4/src/ops/libgdn_sm120_op.so
```

## Load in Python

```python
from src.ops.gdn_op import load_gdn_op, state_update
load_gdn_op()  # registers torch.ops.gdn_sm120.state_update

S_out_data, S_out_scales = state_update(
    S_data, S_scales, k, v,
    alpha=0.95, beta=0.05, d=2048)
```

## Load via torch.ops directly

```python
import torch
torch.ops.load_library("build/m4/src/ops/libgdn_sm120_op.so")
S_out_data, S_out_scales = torch.ops.gdn_sm120.state_update(
    S_data, S_scales, k, v, 0.95, 0.05, 2048)
```

## vLLM serve integration (stretch goal — requires vLLM 0.12+)

```bash
vllm serve <model> \
  --custom-op-lib /home/ubuntu/blackwell-kernel/sm_120/build/m4/src/ops/libgdn_sm120_op.so
```

To attach the op to Qwen3.5 GDN attention layers, patch the model's
attention forward pass to call `torch.ops.gdn_sm120.state_update` for
linear attention layers (those with `layer_type == "gdn"`), replacing
the default BF16 fallback.

## Op signature

```
torch.ops.gdn_sm120.state_update(
    S_data:   Tensor[uint8,    (d*d)//2 ],  # packed E2M1 NVFP4
    S_scales: Tensor[uint8,    (d*d)//16],  # UE4M3 block scales
    k:        Tensor[bfloat16, d        ],  # key projection
    v:        Tensor[bfloat16, d        ],  # value projection
    alpha:    float,                        # decay gate
    beta:     float,                        # update scale
    d:        int,                          # state dimension
) -> (Tensor[uint8, (d*d)//2], Tensor[uint8, (d*d)//16])
```

## Correctness guarantee

Output satisfies FR-6: max element-wise relative error vs BF16 reference
< 5% for d=512, 1024, 2048. Verified by M3 correctness gate (M3C_EXIT=0).

## Hardware requirement

RTX 5090 (SM_120) or any Blackwell GPU with `compute_120f` support.
Requires CUDA 12.9+ and driver 590.x+.
