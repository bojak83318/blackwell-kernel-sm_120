"""
gdn_mixed_precision.py
======================
Python-side integration for the Mixed-Precision GDN state update shim.

Drop this file into:
    vllm/model_executor/layers/gdn/

And import GdnMixedPrecisionManager from your vLLM worker init.

Responsibilities
----------------
1. Load the compiled CUDA extension (gdn_state_update_ext).
2. Expose a toggle: BF16 (Mode A) vs FP8-E5M2 (Mode B) for state storage.
3. Expose a Triton bypass hook so the *output projection logits* can
   still be handled by Triton while only the state storage path is swapped.
4. Log telemetry (max_abs_err, rel_err, pct_clipped, SQNR) every N steps.

Usage in worker
---------------
    from vllm.model_executor.layers.gdn.gdn_mixed_precision import (
        GdnMixedPrecisionManager,
    )

    gdn_mgr = GdnMixedPrecisionManager(
        state_dim=512,
        batch=1,
        state_mode="bf16",          # "bf16" | "fp8e5m2"
        triton_output_proj=True,    # let Triton handle output projection
        telemetry_every=50,         # log every 50 forward steps
        sqnr_floor_db=50.0,         # auto-alert if SQNR drops below this
    )

    # Inside the model forward pass:
    S_out = gdn_mgr.step(S_prev, k, v, W_packed, W_scale)
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telemetry dataclass (mirrors GdnStateTelemetry in the C++ header)
# ---------------------------------------------------------------------------
@dataclass
class GdnTelemetryRecord:
    step: int
    mode: str
    max_abs_err: float
    mean_rel_err: float
    pct_clipped: float
    sqnr_db: float
    n_nan: int
    n_inf: int
    total_elements: int
    wall_ms: float

    def is_healthy(self, sqnr_floor: float = 50.0) -> bool:
        return (
            self.sqnr_db >= sqnr_floor
            and self.n_nan == 0
            and self.n_inf == 0
            and self.pct_clipped < 1.0
        )

    def __str__(self) -> str:
        return (
            f"[GDN Telemetry step={self.step} mode={self.mode}] "
            f"SQNR={self.sqnr_db:.2f} dB  "
            f"max_abs_err={self.max_abs_err:.4e}  "
            f"rel_err={self.mean_rel_err:.4e}  "
            f"pct_clipped={self.pct_clipped:.3f}%  "
            f"NaN={self.n_nan}  Inf={self.n_inf}  "
            f"wall={self.wall_ms:.3f} ms"
        )


# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------
_EXT = None

def _load_extension() -> Optional[object]:
    """
    Attempt to load the compiled CUDA extension.
    Falls back to a pure-PyTorch reference path if the .so is not found.
    This allows the Python integration layer to be imported and tested
    independently of the CUDA build.
    """
    global _EXT
    if _EXT is not None:
        return _EXT

    # Try the installed extension first (built via setup.py / cmake)
    try:
        import gdn_state_update_ext as ext  # type: ignore
        _EXT = ext
        logger.info("gdn_state_update_ext CUDA extension loaded.")
        return _EXT
    except ImportError:
        pass

    # Fallback: JIT-compile if source is adjacent
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(src_dir, "gdn_state_update.cu")
    if os.path.isfile(cu_path):
        try:
            from torch.utils.cpp_extension import load  # type: ignore
            _EXT = load(
                name="gdn_state_update_ext",
                sources=[cu_path],
                extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math"],
                verbose=True,
            )
            logger.info("gdn_state_update_ext JIT-compiled from source.")
            return _EXT
        except Exception as exc:
            logger.warning(f"JIT compile failed: {exc}. Using PyTorch fallback.")

    logger.warning(
        "gdn_state_update_ext not found. Using pure-PyTorch reference path. "
        "Performance will be significantly lower."
    )
    return None


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (fallback + validation baseline)
# ---------------------------------------------------------------------------

def _reference_state_update_bf16(
    W_packed_flat: torch.Tensor,   # uint8 packed, shape [out_dim * in_dim // 2]
    W_scale: torch.Tensor,         # float32, shape [num_blocks]
    block_size: int,
    S_prev: torch.Tensor,          # bfloat16 [batch, state_dim]
    k: torch.Tensor,               # bfloat16 [batch, state_dim]
    v: torch.Tensor,               # bfloat16 [batch, state_dim]
) -> tuple[torch.Tensor, dict]:
    """
    Pure-PyTorch GDN state update with NVFP4 weight simulation.
    Used as the validation reference and CUDA-unavailable fallback.
    Returns (S_out_bf16, telemetry_dict).
    """
    # Dequantise NVFP4 weights in FP32
    batch, state_dim = S_prev.shape
    out_dim = state_dim

    packed = W_packed_flat.to(torch.int32)
    lo_nibble = packed & 0x0F
    hi_nibble = (packed >> 4) & 0x0F

    def nibble_to_float(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = (n >> 0) & 0x1
        # Subnormals (e==0): val = m * 0.5; normals: val = (1 + m*0.5) * 2^(e-1)
        val_sub  = m.float() * 0.5
        exp_val  = (torch.ones_like(m, dtype=torch.float32)
                    + m.float() * 0.5) * (2.0 ** (e.float() - 1))
        val      = torch.where(e == 0, val_sub, exp_val)
        sign     = torch.where(s == 0, torch.ones_like(val), -torch.ones_like(val))
        return val * sign

    # Interleave lo/hi to recover full weight vector
    w_lo = nibble_to_float(lo_nibble)
    w_hi = nibble_to_float(hi_nibble)
    # Shape: [out_dim, in_dim]
    w_interleaved = torch.stack([w_lo, w_hi], dim=1).reshape(-1)
    num_w = out_dim * state_dim
    W_dequant = w_interleaved[:num_w]

    # Apply per-block scales
    scales_expanded = W_scale.repeat_interleave(block_size)[:num_w]
    W_dequant = (W_dequant * scales_expanded).reshape(out_dim, state_dim)

    # GDN state update in FP32 accumulators
    S_fp32 = S_prev.float()
    k_fp32 = k.float()
    v_fp32 = v.float()

    Wv = (v_fp32 @ W_dequant.T)          # [batch, out_dim]
    S_out_fp32 = S_fp32 * k_fp32 + Wv

    # BF16 round-trip telemetry
    S_out_bf16 = S_out_fp32.to(torch.bfloat16)
    S_rt_fp32  = S_out_bf16.float()

    abs_err   = (S_rt_fp32 - S_out_fp32).abs()
    rel_err   = abs_err / (S_out_fp32.abs() + 1e-8)
    signal_pw = (S_out_fp32 ** 2).sum().item()
    noise_pw  = (abs_err ** 2).sum().item()

    sqnr = 10.0 * torch.log10(
        torch.tensor(signal_pw / max(noise_pw, 1e-30))
    ).item()

    BF16_SAT  = 3.4e38
    n_clipped = (S_out_fp32.abs() >= BF16_SAT).sum().item()
    pct_clip  = 100.0 * n_clipped / S_out_fp32.numel()

    telem = {
        "max_abs_err":   abs_err.max().item(),
        "mean_rel_err":  rel_err.mean().item(),
        "pct_clipped":   pct_clip,
        "sqnr_db":       sqnr,
        "n_nan":         S_out_bf16.isnan().sum().item(),
        "n_inf":         S_out_bf16.isinf().sum().item(),
        "total_elements": S_out_fp32.numel(),
    }
    return S_out_bf16, telem


# ---------------------------------------------------------------------------
# Triton bypass hook
# ---------------------------------------------------------------------------
# When triton_output_proj=True, the output projection logits are computed
# via a Triton kernel instead of the CUDA extension, allowing us to isolate
# *only* the state storage path for comparison.

_TRITON_AVAILABLE = False
try:
    import triton                          # type: ignore
    import triton.language as tl           # type: ignore
    _TRITON_AVAILABLE = True
except ImportError:
    pass


def _maybe_triton_output_proj(
    hidden: torch.Tensor,           # [batch, hidden_dim]
    W_out: torch.Tensor,            # [vocab_size, hidden_dim]  or [out_dim, hidden_dim]
    use_triton: bool,
) -> torch.Tensor:
    """
    Compute output projection logits.
    If use_triton=True and Triton is available, dispatches to triton.ops.matmul.
    Otherwise falls through to torch.matmul (which may itself use cuBLAS/CUTLASS).
    """
    if use_triton and _TRITON_AVAILABLE:
        # triton.ops.matmul is a drop-in for torch.mm with custom tiling
        try:
            return triton.ops.matmul(hidden, W_out.T)  # type: ignore
        except Exception:
            pass  # silent fallback

    return F.linear(hidden, W_out)


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------

class GdnMixedPrecisionManager:
    """
    Manages the mixed-precision GDN state update in a vLLM worker.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the GDN recurrent state vector.
    batch : int
        Maximum batch size (used to pre-allocate device buffers).
    state_mode : "bf16" | "fp8e5m2"
        Mode A (BF16) or Mode B (FP8-E5M2) for state storage.
    triton_output_proj : bool
        If True, output projection logits route through Triton; the state
        update path is isolated so mode swaps don't touch the logit path.
    telemetry_every : int
        Emit a telemetry log line every N forward steps.
    sqnr_floor_db : float
        Raise GdnSqnrAlertError if SQNR drops below this value.
    device : torch.device
        Target device (default: cuda:0).
    """

    def __init__(
        self,
        state_dim: int,
        batch: int,
        state_mode: Literal["bf16", "fp8e5m2"] = "bf16",
        triton_output_proj: bool = True,
        telemetry_every: int = 50,
        sqnr_floor_db: float = 50.0,
        device: Optional[torch.device] = None,
    ):
        self.state_dim         = state_dim
        self.batch             = batch
        self.state_mode        = state_mode
        self.triton_output_proj = triton_output_proj
        self.telemetry_every   = telemetry_every
        self.sqnr_floor_db     = sqnr_floor_db
        self.device            = device or torch.device("cuda:0")
        self._step             = 0
        self._ext              = _load_extension()
        self._history: list[GdnTelemetryRecord] = []

        # Allocate persistent device-side telemetry buffer (tiny, 32 bytes)
        self._d_telemetry = torch.zeros(8, dtype=torch.float32, device=self.device)

        # FP8 scale factor (scalar, updated each step for Mode B)
        self._fp8_state_scale = torch.ones(1, dtype=torch.float32, device=self.device)

        logger.info(
            f"GdnMixedPrecisionManager init: "
            f"state_dim={state_dim} batch={batch} mode={state_mode} "
            f"triton_proj={triton_output_proj} "
            f"ext={'loaded' if self._ext else 'fallback'}"
        )

    # ------------------------------------------------------------------
    # Public: swap state precision mode at runtime
    # ------------------------------------------------------------------
    def set_state_mode(self, mode: Literal["bf16", "fp8e5m2"]) -> None:
        """
        Toggle between BF16 (Mode A) and FP8-E5M2 (Mode B) state storage.
        Safe to call between forward passes; state tensors are recast on
        the next step() call.
        """
        if mode not in ("bf16", "fp8e5m2"):
            raise ValueError(f"state_mode must be 'bf16' or 'fp8e5m2', got {mode!r}")
        old = self.state_mode
        self.state_mode = mode
        logger.info(f"GDN state mode switched: {old} → {mode}")

    # ------------------------------------------------------------------
    # Public: main forward step
    # ------------------------------------------------------------------
    def step(
        self,
        S_prev: torch.Tensor,        # current state [batch, state_dim]
        k: torch.Tensor,             # key projection [batch, state_dim]
        v: torch.Tensor,             # value projection [batch, state_dim]
        W_packed: torch.Tensor,      # uint8 NVFP4 weights [out_dim, state_dim//2]
        W_scale: torch.Tensor,       # float32 per-block scales
        block_size: int = 16,
    ) -> torch.Tensor:
        """
        Execute one GDN state update step.
        Returns S_out in the configured state dtype (BF16 or FP8-E5M2).
        If telemetry_every > 0, logs metrics every N calls.
        """
        t0 = time.perf_counter()
        self._step += 1

        capture_telem = (
            self.telemetry_every > 0
            and self._step % self.telemetry_every == 0
        )

        if self._ext is not None:
            S_out, raw_telem = self._dispatch_cuda(
                S_prev, k, v, W_packed, W_scale, block_size, capture_telem
            )
        else:
            S_out, raw_telem = self._dispatch_reference(
                S_prev, k, v, W_packed, W_scale, block_size
            )

        wall_ms = (time.perf_counter() - t0) * 1e3

        if capture_telem and raw_telem is not None:
            record = GdnTelemetryRecord(
                step            = self._step,
                mode            = self.state_mode,
                max_abs_err     = raw_telem["max_abs_err"],
                mean_rel_err    = raw_telem["mean_rel_err"],
                pct_clipped     = raw_telem["pct_clipped"],
                sqnr_db         = raw_telem["sqnr_db"],
                n_nan           = int(raw_telem["n_nan"]),
                n_inf           = int(raw_telem["n_inf"]),
                total_elements  = int(raw_telem["total_elements"]),
                wall_ms         = wall_ms,
            )
            self._history.append(record)
            logger.info(str(record))

            if not record.is_healthy(self.sqnr_floor_db):
                logger.error(
                    f"GDN SQNR ALERT at step {self._step}: "
                    f"SQNR={record.sqnr_db:.2f} dB < floor {self.sqnr_floor_db} dB  "
                    f"NaN={record.n_nan}  Inf={record.n_inf}  "
                    f"clipped={record.pct_clipped:.2f}%"
                )
                raise GdnSqnrAlertError(record)

        return S_out

    # ------------------------------------------------------------------
    # Public: output projection dispatch (Triton bypass hook)
    # ------------------------------------------------------------------
    def output_projection(
        self,
        hidden: torch.Tensor,
        W_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute output projection logits.
        Routes through Triton when triton_output_proj=True, otherwise
        falls through to torch (cuBLAS/CUTLASS).  This hook deliberately
        decouples the output logit path from the state storage path so
        that mode swaps on the state do NOT affect the logit computation.
        """
        return _maybe_triton_output_proj(hidden, W_out, self.triton_output_proj)

    # ------------------------------------------------------------------
    # Public: telemetry summary
    # ------------------------------------------------------------------
    def telemetry_summary(self, last_n: int = 10) -> dict:
        """Return aggregate stats over the last N telemetry records."""
        records = self._history[-last_n:]
        if not records:
            return {}
        import statistics
        sqnrs = [r.sqnr_db for r in records]
        return {
            "n_records":     len(records),
            "sqnr_mean":     statistics.mean(sqnrs),
            "sqnr_min":      min(sqnrs),
            "sqnr_max":      max(sqnrs),
            "pct_clipped_max": max(r.pct_clipped for r in records),
            "n_nan_total":   sum(r.n_nan for r in records),
            "n_inf_total":   sum(r.n_inf for r in records),
            "wall_ms_mean":  statistics.mean(r.wall_ms for r in records),
        }

    # ------------------------------------------------------------------
    # Internal: CUDA extension dispatch
    # ------------------------------------------------------------------
    def _dispatch_cuda(
        self,
        S_prev, k, v, W_packed, W_scale, block_size, capture_telem
    ):
        """
        Dispatch to the compiled CUDA extension.
        The extension is expected to export:
            gdn_state_update_bf16(W_packed, W_scale, block_size,
                                  S_prev, k, v, d_telemetry)
              -> S_out (bf16 tensor)

            gdn_state_update_fp8e5m2(W_packed, W_scale, block_size,
                                     S_prev_fp8, s_prev_scale,
                                     k, v, d_telemetry)
              -> (S_out_fp8 tensor, new_scale scalar)
        """
        telem_buf = self._d_telemetry if capture_telem else None

        if self.state_mode == "bf16":
            # Ensure correct dtypes
            S_prev_bf16 = S_prev.to(torch.bfloat16)
            k_bf16      = k.to(torch.bfloat16)
            v_bf16      = v.to(torch.bfloat16)

            S_out = self._ext.gdn_state_update_bf16(
                W_packed.contiguous(),
                W_scale.contiguous(),
                block_size,
                S_prev_bf16.contiguous(),
                k_bf16.contiguous(),
                v_bf16.contiguous(),
                telem_buf,
            )
            raw_telem = self._read_telemetry(telem_buf) if capture_telem else None
            return S_out, raw_telem

        else:  # fp8e5m2
            # Cast state to FP8 storage if not already
            if S_prev.dtype != torch.float8_e5m2:
                scale = self._fp8_state_scale.item()
                S_prev_fp8 = (S_prev.float() / max(scale, 1e-12)).to(torch.float8_e5m2)
            else:
                S_prev_fp8 = S_prev
                scale = self._fp8_state_scale.item()

            k_bf16 = k.to(torch.bfloat16)
            v_bf16 = v.to(torch.bfloat16)

            S_out_fp8, new_scale = self._ext.gdn_state_update_fp8e5m2(
                W_packed.contiguous(),
                W_scale.contiguous(),
                block_size,
                S_prev_fp8.contiguous(),
                scale,
                k_bf16.contiguous(),
                v_bf16.contiguous(),
                telem_buf,
            )
            # EMA scale update (α=0.1 smoothing to avoid scale instability)
            self._fp8_state_scale.mul_(0.9).add_(new_scale * 0.1)
            raw_telem = self._read_telemetry(telem_buf) if capture_telem else None
            return S_out_fp8, raw_telem

    # ------------------------------------------------------------------
    # Internal: reference fallback dispatch
    # ------------------------------------------------------------------
    def _dispatch_reference(self, S_prev, k, v, W_packed, W_scale, block_size):
        """Always runs BF16 reference path for correctness validation."""
        S_prev_bf16 = S_prev.to(torch.bfloat16)
        S_out_bf16, telem = _reference_state_update_bf16(
            W_packed.view(-1), W_scale, block_size,
            S_prev_bf16, k.to(torch.bfloat16), v.to(torch.bfloat16),
        )
        if self.state_mode == "fp8e5m2":
            scale = max(S_out_bf16.float().abs().max().item(), 1e-12) / 57344.0
            self._fp8_state_scale.fill_(scale)
            S_out = (S_out_bf16.float() / scale).to(torch.float8_e5m2)
        else:
            S_out = S_out_bf16
        return S_out, telem

    # ------------------------------------------------------------------
    # Internal: telemetry buffer readback
    # ------------------------------------------------------------------
    def _read_telemetry(self, telem_buf: Optional[torch.Tensor]) -> Optional[dict]:
        if telem_buf is None:
            return None
        t = telem_buf.cpu()
        return {
            "max_abs_err":    t[0].item(),
            "mean_rel_err":   t[1].item(),
            "pct_clipped":    t[2].item(),
            "sqnr_db":        t[3].item(),
            "n_nan":          int(t[4].item()),
            "n_inf":          int(t[5].item()),
            "total_elements": int(t[6].item()),
        }


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class GdnSqnrAlertError(RuntimeError):
    def __init__(self, record: GdnTelemetryRecord):
        self.record = record
        super().__init__(
            f"GDN SQNR {record.sqnr_db:.2f} dB dropped below floor. "
            f"Step={record.step}  mode={record.mode}  "
            f"NaN={record.n_nan}  clipped={record.pct_clipped:.2f}%"
        )
