"""M4-B loader for ``torch.ops.gdn_sm120.state_update``.

This module locates and loads the SM_120 GDN PyTorch extension, then
re-exports the registered ``state_update`` operator so higher layers can use
it without repeating the loading logic. The loader honors the
``SM120_GDN_LIBRARY`` environment variable and otherwise probes a few
likely directories under the repository root (build/, src/pytorch/, etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import torch

__all__ = ["register_gdn_ops", "ensure_gdn_ops", "state_update"]

_LIBRARY_ENV = "SM120_GDN_LIBRARY"
_NAMESPACE = "gdn_sm120"
_OP_NAME = "state_update"
_CANDIDATE_LIB_BASES = (
    "libgdn_sm120",
    "gdn_sm120",
    "sm120_pytorch",
    "libsm120_pytorch",
)

_loaded = False
_loaded_library: Optional[Path] = None


def _library_suffix() -> str:
    platform = sys.platform
    if platform.startswith("linux"):
        return ".so"
    if platform == "darwin":
        return ".dylib"
    if platform == "win32":
        return ".dll"
    return ".so"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_directories(root: Path) -> Iterable[Path]:
    directories = [
        root,
        root / "build",
        root / "build" / "lib",
        root / "build" / "pytorch",
        root / "build" / "src",
        root / "build" / "src" / "pytorch",
        root / "src" / "pytorch",
        root / "python",
        root / "python" / "build",
        root / "artifacts",
        root / "artifacts" / "lib",
    ]
    seen: set[Path] = set()
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def _candidate_paths(root: Path, suffix: str) -> Iterable[Path]:
    env_override = os.environ.get(_LIBRARY_ENV)
    if env_override:
        yield Path(env_override)
    for base in _CANDIDATE_LIB_BASES:
        for directory in _candidate_directories(root):
            yield directory / f"{base}{suffix}"
    for base in _CANDIDATE_LIB_BASES:
        yield root / f"{base}{suffix}"


def _resolve_library(explicit_path: Optional[Union[str, Path]] = None) -> Path:
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"SM_120 GDN library '{candidate}' was provided but does not exist."
        )

    root = _project_root()
    suffix = _library_suffix()
    attempted: list[str] = []
    for candidate in _candidate_paths(root, suffix):
        attempted.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the SM_120 GDN PyTorch extension. "
        f"Set {_LIBRARY_ENV} or pass the exact path explicitly. "
        f"Tried: {', '.join(attempted)}"
    )


def register_gdn_ops(library_path: Optional[Union[str, Path]] = None) -> Path:
    """Load the GDN custom op and return the resolved library path."""
    global _loaded, _loaded_library
    if _loaded:
        if library_path is not None and Path(library_path) != _loaded_library:
            raise RuntimeError(
                "GDN ops already loaded from a different library. "
                f"Current: {_loaded_library}, requested: {library_path}."
            )
        if _loaded_library is None:
            raise RuntimeError("GDN ops were marked as loaded but no path is recorded.")
        return _loaded_library

    resolved = _resolve_library(library_path)
    try:
        torch.ops.load_library(str(resolved))
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            f"Failed to load the SM_120 GDN library '{resolved}': {exc}"
        ) from exc

    namespace = getattr(torch.ops, _NAMESPACE, None)
    if namespace is None:
        raise RuntimeError(
            f"Expected namespace 'torch.ops.{_NAMESPACE}' after loading '{resolved}', "
            "but it was not found."
        )
    if not hasattr(namespace, _OP_NAME):
        raise RuntimeError(
            f"Library '{resolved}' did not register 'torch.ops.{_NAMESPACE}.{_OP_NAME}'."
        )

    _loaded = True
    _loaded_library = resolved
    return resolved


def ensure_gdn_ops(library_path: Optional[Union[str, Path]] = None) -> Path:
    """Ensure the GDN ops have been loaded (loads the default library if needed)."""
    return register_gdn_ops(library_path)


def _state_update_callable() -> Any:
    ensure_gdn_ops()
    namespace = getattr(torch.ops, _NAMESPACE)
    return getattr(namespace, _OP_NAME)


def state_update(*args: Any, **kwargs: Any) -> Any:
    """Call the registered ``torch.ops.gdn_sm120.state_update`` operator."""
    return _state_update_callable()(*args, **kwargs)
