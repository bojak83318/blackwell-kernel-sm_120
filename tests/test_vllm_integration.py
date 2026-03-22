"""
test_vllm_integration.py — M4 core gate

Tests:
  1. libgdn_sm120_op.so loads via torch.ops.load_library (no vLLM needed)
  2. torch.ops.gdn_sm120.state_update is callable
  3. Output shapes are correct
  4. Output values match M3 BF16 reference within 5% relative error (FR-6)
  5. [optional] vllm serve accepts --custom-op-lib (skipped if vLLM absent)

Exit: pytest returns 0 if all non-optional tests pass.
"""

import os
import pathlib
import sys

import pytest
import torch
import numpy as np

# ── locate the .so ────────────────────────────────────────────────────────────

def find_so() -> pathlib.Path:
    repo = pathlib.Path(__file__).parent.parent
    candidates = [
        repo / "build" / "m4" / "src" / "ops" / "libgdn_sm120_op.so",
        repo / "build" / "src" / "ops" / "libgdn_sm120_op.so",
    ]
    # also accept env override
    env = os.environ.get("GDN_OP_LIB")
    if env:
        candidates.insert(0, pathlib.Path(env))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "libgdn_sm120_op.so not found.\n"
        "Build with: cmake --build build/m4 --target gdn_sm120_op\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


# ── BF16 reference (pure PyTorch, no CUDA kernel) ─────────────────────────────

def gdn_reference(
    S_data: torch.Tensor,
    S_scales: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    beta: float,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU/GPU BF16 reference for one GDN step using PyTorch ops."""
    # dequantise S_prev from NVFP4
    S_fp32 = nvfp4_dequant(S_data, S_scales, d)
    # outer product
    k_fp32 = k.float()
    v_fp32 = v.float()
    outer = torch.outer(k_fp32, v_fp32)
    # GDN update
    S_out_fp32 = alpha * S_fp32 + beta * outer
    # requantise
    return nvfp4_quant(S_out_fp32, d)


def nvfp4_dequant(data: torch.Tensor, scales: torch.Tensor, d: int) -> torch.Tensor:
    """Dequantise packed E2M1 + UE4M3 scales to FP32."""
    n = d * d
    data_cpu = data.cpu().numpy().astype(np.uint8)
    scales_cpu = scales.cpu().numpy().astype(np.uint8)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        byte = data_cpu[i // 2]
        nibble = int(byte & 0x0F) if i % 2 == 0 else int((byte >> 4) & 0x0F)
        raw = _e2m1_to_f32(nibble)
        scale = _ue4m3_to_f32(int(scales_cpu[i // 16]))
        out[i] = raw * scale
    return torch.from_numpy(out).view(d, d).to(data.device)


def nvfp4_quant(S: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise FP32 [d,d] tensor to packed E2M1 + UE4M3."""
    n = d * d
    flat = S.float().cpu().numpy().flatten()
    data = np.zeros(n // 2, dtype=np.uint8)
    scales = np.zeros(n // 16, dtype=np.uint8)
    E2M1_MAGS = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    for b in range(n // 16):
        block = flat[b*16:(b+1)*16]
        amax = np.max(np.abs(block))
        local_scale = amax / 6.0 if amax > 0 else 1.0
        enc = _f32_to_ue4m3(float(local_scale))
        scales[b] = enc
        dec = _ue4m3_to_f32(int(enc))
        inv = 1.0 / dec if dec > 0 else 0.0
        for i in range(16):
            v = float(block[i]) * inv
            s = 1 if v < 0 else 0
            av = abs(v)
            best = min(range(8), key=lambda j: abs(av - E2M1_MAGS[j]))
            nibble = best | (s << 3)
            idx = b * 16 + i
            if idx % 2 == 0:
                data[idx // 2] = (data[idx // 2] & 0xF0) | nibble
            else:
                data[idx // 2] = (data[idx // 2] & 0x0F) | (nibble << 4)
    dev = S.device
    return (torch.from_numpy(data).to(dev),
            torch.from_numpy(scales).to(dev))


def _e2m1_to_f32(nibble: int) -> float:
    s = (nibble >> 3) & 1
    e = (nibble >> 1) & 3
    m = nibble & 1
    if e == 0:
        mag = m * 0.5
    else:
        mag = (1.0 + m * 0.5) * float(1 << (e - 1))
    return -mag if s else mag


def _ue4m3_to_f32(b: int) -> float:
    if b == 0:
        return 0.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    return (1.0 + m / 8.0) * float(1 << e)


def _f32_to_ue4m3(v: float) -> int:
    if v <= 0.0:
        return 0
    best_enc, best_err = 0, float('inf')
    for enc in range(1, 256):
        decoded = _ue4m3_to_f32(enc)
        err = abs(decoded - v)
        if err < best_err:
            best_err, best_enc = err, enc
    return best_enc


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def so_path():
    return find_so()


@pytest.fixture(scope="session")
def op_loaded(so_path):
    torch.ops.load_library(str(so_path))
    return True


# ── tests ─────────────────────────────────────────────────────────────────────

def test_so_exists():
    """SO file must exist at a known build path."""
    p = find_so()
    assert p.exists(), f"libgdn_sm120_op.so not found at {p}"


def test_op_loads(op_loaded):
    """torch.ops.load_library must succeed without error."""
    assert op_loaded


def test_op_registered(op_loaded):
    """torch.ops.gdn_sm120.state_update must be callable after load."""
    assert hasattr(torch.ops, "gdn_sm120"), \
        "torch.ops.gdn_sm120 namespace not found after load"
    assert hasattr(torch.ops.gdn_sm120, "state_update"), \
        "torch.ops.gdn_sm120.state_update not found"
    assert callable(torch.ops.gdn_sm120.state_update)


@pytest.mark.parametrize("d", [512, 1024, 2048])
def test_output_shapes(op_loaded, d):
    """Output tensors must have correct NVFP4 shapes."""
    n = d * d
    device = torch.device("cuda:0")
    S_data   = torch.zeros(n // 2,  dtype=torch.uint8,    device=device)
    S_scales = torch.ones (n // 16, dtype=torch.uint8,    device=device)
    k = torch.randn(d, dtype=torch.bfloat16, device=device)
    v = torch.randn(d, dtype=torch.bfloat16, device=device)

    out_data, out_scales = torch.ops.gdn_sm120.state_update(
        S_data, S_scales, k, v, 0.9, 0.1, d)

    assert out_data.shape   == (n // 2,),  f"data shape wrong: {out_data.shape}"
    assert out_scales.shape == (n // 16,), f"scales shape wrong: {out_scales.shape}"
    assert out_data.dtype   == torch.uint8
    assert out_scales.dtype == torch.uint8


@pytest.mark.parametrize("d,alpha,beta", [
    (512,  0.9,  0.1),
    (1024, 0.8,  0.2),
    (2048, 0.95, 0.05),
])
def test_correctness_vs_reference(op_loaded, d, alpha, beta):
    """Fused kernel output must match PyTorch BF16 reference within 5% (FR-6)."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    n = d * d

    # build synthetic NVFP4 state
    S_fp32 = torch.empty(d, d, device=device).uniform_(-4.0, 4.0)
    S_data, S_scales = nvfp4_quant(S_fp32, d)
    k = torch.randn(d, dtype=torch.bfloat16, device=device)
    v = torch.randn(d, dtype=torch.bfloat16, device=device)

    # fused kernel
    out_data, out_scales = torch.ops.gdn_sm120.state_update(
        S_data, S_scales, k, v, alpha, beta, d)
    fused_fp32 = nvfp4_dequant(out_data, out_scales, d)

    # reference
    ref_data, ref_scales = gdn_reference(
        S_data, S_scales, k, v, alpha, beta, d)
    ref_fp32 = nvfp4_dequant(ref_data, ref_scales, d)

    # compare
    denom = ref_fp32.abs().clamp(min=1e-6)
    rel_err = ((fused_fp32 - ref_fp32).abs() / denom).max().item()
    assert rel_err <= 0.05, \
        f"d={d}: max relative error {rel_err:.4f} exceeds 5% tolerance (FR-6)"


@pytest.mark.skipif(
    not pytest.importorskip("vllm", reason="vllm not installed"),
    reason="vllm not installed"
)
def test_vllm_serve_accepts_custom_op_lib(so_path):
    """vllm serve must accept --custom-op-lib without error (stretch goal)."""
    import subprocess
    result = subprocess.run(
        ["vllm", "serve", "--help"],
        capture_output=True, text=True
    )
    assert "--custom-op-lib" in result.stdout or "--custom-op-lib" in result.stderr, \
        "vllm serve does not support --custom-op-lib flag"
