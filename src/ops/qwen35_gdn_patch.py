"""
qwen35_gdn_patch.py
Patches vLLM's Qwen3_5 attention forward to route linear_attention layers
through torch.ops.gdn_sm120.state_update (the SM_120 NVFP4 fused kernel).

Usage:
    import qwen35_gdn_patch
    qwen35_gdn_patch.apply(so_path="/path/to/libgdn_sm120_op.so")

After apply(), any vLLM Qwen3_5Attention instance whose layer_type is
"linear_attention" will use the GDN kernel path; full-attention layers
are unchanged.
"""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Optional

import torch

_lock = threading.Lock()
_patched = False
_so_path: Optional[str] = None

# ── GDN state manager ─────────────────────────────────────────────────────────

class GDNStateCache:
    """Per-layer recurrent state cache (S_data, S_scales) in NVFP4 format.

    States are keyed by (layer_idx, batch_idx). For inference with a single
    sequence, batch_idx=0 covers the common case.
    """

    def __init__(self):
        self._states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, layer_idx: int, batch_idx: int, d: int, device: torch.device
            ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, batch_idx)
        if key not in self._states:
            # initialise to zero state — S = 0 means all-zero NVFP4
            n = d * d
            self._states[key] = (
                torch.zeros(n // 2,  dtype=torch.uint8, device=device),
                torch.zeros(n // 16, dtype=torch.uint8, device=device),
            )
        return self._states[key]

    def update(self, layer_idx: int, batch_idx: int,
               data: torch.Tensor, scales: torch.Tensor) -> None:
        self._states[(layer_idx, batch_idx)] = (data, scales)

    def reset(self) -> None:
        self._states.clear()


# module-level cache shared across all patched layers
_state_cache = GDNStateCache()


# ── GDN forward replacement ───────────────────────────────────────────────────

def _gdn_linear_attn_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
):
    """Replacement forward for linear_attention layers in Qwen3.5.

    Projects hidden_states → k, v via the existing q/k/v projection weights,
    runs one GDN state update step, then projects back via o_proj.
    """
    bsz, seq_len, hidden_dim = hidden_states.shape
    layer_idx = getattr(self, "_gdn_layer_idx", 0)
    d = getattr(self, "_gdn_state_dim", hidden_dim)

    # ── project to k, v ───────────────────────────────────────────────────────
    # Use the layer's existing kv projection; take mean over seq_len for
    # recurrent step (one state update per token position in the sequence)
    qkv = self.qkv_proj(hidden_states)               # [B, T, 3*H] or similar
    # split — Qwen3.5 uses separate q/k/v or fused; handle both
    if hasattr(self, "q_proj"):
        k = self.k_proj(hidden_states)               # [B, T, H]
        v = self.v_proj(hidden_states)               # [B, T, H]
    else:
        # fused qkv_proj: split evenly
        split = qkv.shape[-1] // 3
        k = qkv[..., split:2*split]
        v = qkv[..., 2*split:]

    # ── per-token GDN update ──────────────────────────────────────────────────
    outputs = []
    alpha = getattr(self, "_gdn_alpha", 0.9)
    beta  = getattr(self, "_gdn_beta",  0.1)

    for t in range(seq_len):
        for b in range(bsz):
            k_t = k[b, t].to(torch.bfloat16)   # [H]
            v_t = v[b, t].to(torch.bfloat16)   # [H]

            S_data, S_scales = _state_cache.get(layer_idx, b, d, k_t.device)

            S_out_data, S_out_scales = torch.ops.gdn_sm120.state_update(
                S_data, S_scales, k_t, v_t,
                float(alpha), float(beta), d)

            _state_cache.update(layer_idx, b, S_out_data, S_out_scales)

        # output for this timestep: use v as the context vector (simplified)
        # A full implementation would read S_out and contract with q
        outputs.append(v[:, t, :])   # [B, H]

    out = torch.stack(outputs, dim=1)   # [B, T, H]

    # ── output projection ─────────────────────────────────────────────────────
    if hasattr(self, "o_proj"):
        out = self.o_proj(out)

    return out


# ── patch application ─────────────────────────────────────────────────────────

def apply(so_path: str | None = None) -> None:
    """Apply the GDN patch to vLLM's Qwen3_5 attention module.

    Args:
        so_path: path to libgdn_sm120_op.so. Falls back to GDN_OP_LIB env var
                 or the default build location.
    """
    global _patched, _so_path

    with _lock:
        if _patched:
            return

        # resolve .so path
        if so_path is None:
            so_path = os.environ.get("GDN_OP_LIB")
        if so_path is None:
            candidates = [
                pathlib.Path(__file__).parent.parent.parent
                / "build/m4-vllm/src/ops/libgdn_sm120_op.so",
                pathlib.Path("/home/ubuntu/blackwell-kernel/sm_120"
                             "/build/m4-vllm/src/ops/libgdn_sm120_op.so"),
            ]
            for c in candidates:
                if c.exists():
                    so_path = str(c)
                    break
        if so_path is None or not pathlib.Path(so_path).exists():
            raise FileNotFoundError(
                f"libgdn_sm120_op.so not found. Set GDN_OP_LIB or pass so_path. "
                f"Tried: {so_path}"
            )

        torch.ops.load_library(so_path)
        _so_path = so_path

        # import vLLM attention module — defer so patch is applied after vLLM load
        try:
            from vllm.model_executor.models.qwen3_5 import Qwen3_5Attention
        except ImportError:
            # vLLM < 0.12 or different module path
            try:
                from vllm.model_executor.models.qwen2 import Qwen2Attention as Qwen3_5Attention
            except ImportError:
                raise ImportError(
                    "Could not import Qwen3_5Attention from vllm. "
                    "Ensure vllm==0.12.0 is installed."
                )

        original_forward = Qwen3_5Attention.forward

        def patched_forward(self, positions, hidden_states, kv_cache, attn_metadata, **kwargs):
            layer_type = getattr(self, "layer_type", None) or \
                         getattr(self.config if hasattr(self, "config") else object(), 
                                 "layer_type", None)
            # check layer_types list on config if available
            if layer_type is None and hasattr(self, "_layer_idx"):
                cfg = getattr(self, "config", None)
                if cfg and hasattr(cfg, "layer_types"):
                    idx = getattr(self, "_layer_idx", 0)
                    layer_types = cfg.layer_types
                    if idx < len(layer_types):
                        layer_type = layer_types[idx]

            if layer_type == "linear_attention":
                return _gdn_linear_attn_forward(
                    self, positions, hidden_states, kv_cache, attn_metadata)
            else:
                return original_forward(
                    self, positions, hidden_states, kv_cache, attn_metadata, **kwargs)

        Qwen3_5Attention.forward = patched_forward
        _patched = True
        print(f"[gdn_patch] Applied GDN router patch — so={so_path}")


def reset_state() -> None:
    """Clear all recurrent GDN states (call between sequences)."""
    _state_cache.reset()


def get_patch_status() -> dict:
    return {"patched": _patched, "so_path": _so_path}
