#!/usr/bin/env python3
"""Basic readiness check for the SM_120 vLLM integration path."""

from __future__ import annotations

import logging
import re

# Minimum versions are copied from the cmake/SM120Dependencies.cmake matrix.
MIN_TORCH_VERSION = (2, 9, 1)
MIN_VLLM_VERSION = (0, 12, 0)


def version_tuple(version: str) -> tuple[int, ...]:
    digits = re.findall(r"\d+", version)
    if not digits:
        raise ValueError(f"unable to parse version string: {version!r}")
    return tuple(int(token) for token in digits)


def validate_version(name: str, actual: str, minimum: tuple[int, ...]) -> None:
    actual_tuple = version_tuple(actual)
    if actual_tuple < minimum:
        raise RuntimeError(
            f"{name} {actual} is older than the required "
            f"{'.'.join(str(part) for part in minimum)}"
        )
    logging.debug("%s version %s >= %s", name, actual, minimum)


def get_sm120_ops() -> tuple[bool, bool]:
    import torch

    op_namespace = getattr(torch.ops, "sm120", None)
    if op_namespace is None:
        return False, False
    has_gdn = hasattr(op_namespace, "gdn")
    return True, has_gdn


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        import torch
    except ImportError as exc:
        logging.error("PyTorch import failed: %s", exc)
        return 1

    try:
        import vllm
    except ImportError as exc:
        logging.error("vLLM import failed: %s", exc)
        return 1

    try:
        validate_version("PyTorch", torch.__version__, MIN_TORCH_VERSION)
    except Exception as exc:  # noqa: BLE001
        logging.error(exc)
        return 1

    try:
        vllm_version = getattr(vllm, "__version__", "")
        if not vllm_version:
            raise RuntimeError("vLLM does not expose __version__")
        validate_version("vLLM", vllm_version, MIN_VLLM_VERSION)
    except Exception as exc:  # noqa: BLE001
        logging.error(exc)
        return 1

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        logging.info("CUDA device available: %s (SM %s)", device_name, capability)
    else:
        logging.warning("CUDA is not available; vLLM integration will need a GPU to run this kernel.")

    namespace_present, gdn_present = get_sm120_ops()
    if namespace_present:
        if gdn_present:
            logging.info("torch.ops.sm120.gdn is registered and ready to be wired into vLLM.")
        else:
            logging.warning(
                "torch.ops.sm120 is present but the gdn op is missing (expecting sm120.gdn)."
            )
    else:
        logging.warning(
            "torch.ops.sm120 namespace is missing. Build the PyTorch extension and load it via sm120_register.py before running vLLM workloads."
        )

    logging.info("Environment checks passed. Run docs/vllm_registration.md for the registration recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
