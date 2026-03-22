"""vLLM plugin that loads the SM_120 custom op library on import."""

from __future__ import annotations

import logging

from .gdn_op import register_gdn_ops

LOGGER = logging.getLogger("vllm_sm120_plugin")


def _load() -> None:
    path = register_gdn_ops()
    LOGGER.info("Loaded SM_120 custom op library: %s", path)


_load()

