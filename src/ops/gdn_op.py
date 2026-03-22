"""
gdn_op.py — Load the GDN SM_120 custom op and expose it as
            torch.ops.gdn_sm120.state_update

Usage:
    from src.ops.gdn_op import load_gdn_op
    load_gdn_op()   # registers torch.ops.gdn_sm120.state_update
    S_out_data, S_out_scales = torch.ops.gdn_sm120.state_update(
        S_data, S_scales, k, v, alpha, beta, d)
"""

import pathlib
import torch

_loaded = False

def load_gdn_op(lib_path: str | None = None) -> None:
    """Load libgdn_sm120_op.so and register torch.ops.gdn_sm120.state_update.

    Args:
        lib_path: explicit path to the .so file. If None, searches the
                  standard build output locations relative to this file.
    """
    global _loaded
    if _loaded:
        return

    if lib_path is not None:
        so_path = pathlib.Path(lib_path)
    else:
        # search standard locations
        repo_root = pathlib.Path(__file__).parent.parent.parent
        candidates = [
            repo_root / "build" / "m4" / "src" / "ops" / "libgdn_sm120_op.so",
            repo_root / "build" / "src" / "ops" / "libgdn_sm120_op.so",
            pathlib.Path("/tmp/libgdn_sm120_op.so"),
        ]
        so_path = None
        for c in candidates:
            if c.exists():
                so_path = c
                break
        if so_path is None:
            raise FileNotFoundError(
                "libgdn_sm120_op.so not found. Build with:\n"
                "  cmake --build build/m4 --target gdn_sm120_op\n"
                f"Searched: {[str(c) for c in candidates]}"
            )

    torch.ops.load_library(str(so_path))
    _loaded = True


def state_update(
    S_data: torch.Tensor,
    S_scales: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    beta: float,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one GDN state update step.

    Args:
        S_data:   uint8 CUDA tensor, shape [(d*d)//2]  — packed E2M1 NVFP4
        S_scales: uint8 CUDA tensor, shape [(d*d)//16] — UE4M3 block scales
        k:        bfloat16 CUDA tensor, shape [d]
        v:        bfloat16 CUDA tensor, shape [d]
        alpha:    decay gate scalar
        beta:     update scale scalar
        d:        state dimension

    Returns:
        (S_out_data, S_out_scales) — same shapes as inputs
    """
    load_gdn_op()
    return torch.ops.gdn_sm120.state_update(S_data, S_scales, k, v, alpha, beta, d)
