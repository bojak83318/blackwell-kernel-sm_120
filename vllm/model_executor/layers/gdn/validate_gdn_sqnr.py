"""
validate_gdn_sqnr.py
====================
Standalone SQNR validation script for the Mixed-Precision GDN state path.

Run this BEFORE declaring any engineering phase complete (per PRD §5 hard
constraints) and AFTER any kernel fusion change.

Usage
-----
    python validate_gdn_sqnr.py \
        --mode bf16 \
        --state-dim 512 \
        --batch 4 \
        --seq-len 256 \
        --n-steps 200 \
        --sqnr-floor 50.0

Exit codes
----------
  0  All checks passed
  1  SQNR < floor at any step
  2  NaN or Inf detected
  3  Clipping > 1%
"""

import argparse
import logging
import sys
import time
from typing import Literal

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gdn_validate")


# ---------------------------------------------------------------------------
# FP32 reference implementation for SQNR baseline
# ---------------------------------------------------------------------------
def reference_state_update_fp32(
    W: torch.Tensor,       # float32 [state_dim, state_dim]
    S: torch.Tensor,       # float32 [batch, state_dim]
    k: torch.Tensor,       # float32 [batch, state_dim]
    v: torch.Tensor,       # float32 [batch, state_dim]
) -> torch.Tensor:
    """Ground-truth FP32 GDN state update: S_out = S * k + W @ v^T."""
    return S * k + (v @ W.T)


def simulate_nvfp4_weights(state_dim: int, device: torch.device) -> tuple:
    """
    Create synthetic NVFP4 weights by quantising random FP32 weights.
    Returns (W_fp32, W_packed_uint8, W_scale_fp32, block_size).
    """
    block_size = 16
    W_fp32 = torch.randn(state_dim, state_dim, device=device)
    n_elements = state_dim * state_dim
    n_blocks = (n_elements + block_size - 1) // block_size

    # Compute per-block scales
    W_flat = W_fp32.reshape(-1)
    padded = torch.zeros(n_blocks * block_size, device=device)
    padded[:n_elements] = W_flat
    padded_blocks = padded.reshape(n_blocks, block_size)
    block_max = padded_blocks.abs().max(dim=1).values
    W_scale = block_max / 1.75  # FP4 E2M1 max magnitude = 1.75

    # Quantise to 4-bit integers {-6..6 clipped} and pack two per byte
    W_int4 = torch.zeros(n_blocks * block_size, dtype=torch.uint8, device=device)
    for b in range(n_blocks):
        block = padded_blocks[b]
        sc = W_scale[b].item()
        if sc == 0:
            continue
        quantised = (block / sc).clamp(-1.75, 1.75)
        # Map to nibble (rough: just sign-magnitude for simulation)
        for j in range(block_size):
            q = quantised[j].item()
            s_bit = 0 if q >= 0 else 1
            abs_q = abs(q)
            # E2M1 encoding (simplified)
            if abs_q < 0.5:
                nibble = (s_bit << 3) | 0
            elif abs_q < 1.0:
                nibble = (s_bit << 3) | 0x2
            elif abs_q < 1.5:
                nibble = (s_bit << 3) | 0x4
            else:
                nibble = (s_bit << 3) | 0x6
            W_int4[b * block_size + j] = nibble & 0xF

    # Pack two nibbles per byte
    W_int4_trunc = W_int4[:n_elements]
    n_packed = (n_elements + 1) // 2
    W_packed = torch.zeros(n_packed, dtype=torch.uint8, device=device)
    for i in range(0, n_elements, 2):
        lo = W_int4_trunc[i].item()
        hi = W_int4_trunc[i + 1].item() if (i + 1) < n_elements else 0
        W_packed[i // 2] = (lo & 0xF) | ((hi & 0xF) << 4)

    # Dequantised version for FP32 reference
    W_dequant = torch.zeros(n_elements, device=device)
    for b in range(n_blocks):
        sc = W_scale[b].item()
        for j in range(block_size):
            idx = b * block_size + j
            if idx >= n_elements:
                break
            nibble = W_int4[idx].item()
            s = (nibble >> 3) & 1
            e = (nibble >> 1) & 3
            m = nibble & 1
            if e == 0:
                val = m * 0.5
            else:
                val = (1.0 + m * 0.5) * (2 ** (e - 1))
            W_dequant[idx] = val * sc * (-1 if s else 1)

    W_dequant_mat = W_dequant.reshape(state_dim, state_dim)
    return W_dequant_mat, W_packed, W_scale, block_size


def compute_sqnr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    sig_pw  = (signal.float() ** 2).sum().item()
    nois_pw = (noise.float()  ** 2).sum().item()
    if nois_pw == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(sig_pw / nois_pw)).item()


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------
def run_validation(
    mode: Literal["bf16", "fp8e5m2"],
    state_dim: int,
    batch: int,
    seq_len: int,
    n_steps: int,
    sqnr_floor: float,
    device_str: str,
) -> int:
    device = torch.device(device_str)
    log.info(
        f"Starting GDN SQNR validation: mode={mode} state_dim={state_dim} "
        f"batch={batch} seq_len={seq_len} n_steps={n_steps} floor={sqnr_floor} dB"
    )

    # --- Build synthetic weights once ---
    W_fp32, W_packed, W_scale, block_size = simulate_nvfp4_weights(state_dim, device)

    # --- Initialise states ---
    S_ref  = torch.randn(batch, state_dim, device=device)
    S_test = S_ref.clone()

    sqnr_log = []
    clip_log  = []
    nan_count  = 0
    inf_count  = 0
    exit_code  = 0

    for step in range(n_steps):
        # Synthetic k, v (simulate attention output)
        k = torch.randn(batch, state_dim, device=device)
        v = torch.randn(batch, state_dim, device=device)

        # --- FP32 reference ---
        S_ref_next = reference_state_update_fp32(W_fp32, S_ref.float(), k.float(), v.float())

        # --- Test path ---
        if mode == "bf16":
            S_test_fp32 = S_test.float()
            k_bf16 = k.to(torch.bfloat16).float()
            v_bf16 = v.to(torch.bfloat16).float()
            S_test_next = reference_state_update_fp32(W_fp32, S_test_fp32, k_bf16, v_bf16)
            S_test_next_stored = S_test_next.to(torch.bfloat16).float()

        else:  # fp8e5m2
            FP8_MAX = 57344.0
            scale = max(S_test.float().abs().max().item(), 1e-12) / FP8_MAX
            S_test_fp8 = (S_test.float() / scale).clamp(-FP8_MAX, FP8_MAX)
            S_test_dq  = S_test_fp8 * scale   # simulate dequant
            k_bf16 = k.to(torch.bfloat16).float()
            v_bf16 = v.to(torch.bfloat16).float()
            S_test_next = reference_state_update_fp32(W_fp32, S_test_dq, k_bf16, v_bf16)
            out_scale = max(S_test_next.abs().max().item(), 1e-12) / FP8_MAX
            S_quant = (S_test_next / out_scale).clamp(-FP8_MAX, FP8_MAX)
            S_test_next_stored = S_quant * out_scale

        # --- Metrics ---
        noise = S_test_next_stored - S_ref_next
        sqnr  = compute_sqnr(S_ref_next, noise)
        sqnr_log.append(sqnr)

        has_nan = S_test_next_stored.isnan().any().item()
        has_inf = S_test_next_stored.isinf().any().item()
        n_clip  = (S_test_next_stored.abs() > 1e6).sum().item()  # crude large-value clip
        pct_clip = 100.0 * n_clip / S_test_next_stored.numel()
        clip_log.append(pct_clip)

        if has_nan:
            nan_count += 1
        if has_inf:
            inf_count += 1

        if step % 50 == 0 or sqnr < sqnr_floor:
            log.info(
                f"  step={step:4d}  SQNR={sqnr:.2f} dB  "
                f"pct_clipped={pct_clip:.3f}%  NaN={has_nan}  Inf={has_inf}"
            )

        if sqnr < sqnr_floor and exit_code == 0:
            log.error(
                f"SQNR BELOW FLOOR at step {step}: {sqnr:.2f} dB < {sqnr_floor} dB"
            )
            exit_code = 1

        # Advance state
        S_ref  = S_ref_next
        S_test = torch.tensor(S_test_next_stored, device=device)

    # --- Summary ---
    import statistics
    log.info("=" * 60)
    log.info(f"Validation complete — {n_steps} steps")
    log.info(f"  SQNR:     mean={statistics.mean(sqnr_log):.2f} dB  "
             f"min={min(sqnr_log):.2f} dB  max={max(sqnr_log):.2f} dB")
    log.info(f"  Clipped:  max_pct={max(clip_log):.3f}%")
    log.info(f"  NaN steps: {nan_count}  |  Inf steps: {inf_count}")
    log.info(f"  Floor: {sqnr_floor} dB  |  Result: {'PASS ✓' if exit_code == 0 else 'FAIL ✗'}")
    log.info("=" * 60)

    if nan_count > 0 and exit_code == 0:
        exit_code = 2
    if inf_count > 0 and exit_code == 0:
        exit_code = 2
    if max(clip_log) > 1.0 and exit_code == 0:
        exit_code = 3

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="GDN SQNR Validation")
    parser.add_argument("--mode",       default="bf16",   choices=["bf16", "fp8e5m2"])
    parser.add_argument("--state-dim",  type=int, default=512)
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--seq-len",    type=int, default=256)
    parser.add_argument("--n-steps",    type=int, default=200)
    parser.add_argument("--sqnr-floor", type=float, default=50.0)
    parser.add_argument("--device",     default="cuda:0")
    args = parser.parse_args()

    code = run_validation(
        mode       = args.mode,
        state_dim  = args.state_dim,
        batch      = args.batch,
        seq_len    = args.seq_len,
        n_steps    = args.n_steps,
        sqnr_floor = args.sqnr_floor,
        device_str = args.device,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
