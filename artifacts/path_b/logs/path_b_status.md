# Path B Status — Qwen3.5 BF16 + GDN op

## Status: BLOCKED (upstream)

## Root cause
Fine-tuned checkpoint was saved with transformers 5.3.0.dev0.
All current stable inference runtimes (vLLM 0.16-0.18, transformers serve)
require transformers <5.0. The qwen3_5_text architecture is not registered
in any 4.x release.

## What was completed
- M1-M4 kernel milestones: PASS
- libgdn_sm120_op.so: built and validated
- torch.ops.gdn_sm120.state_update: registered and tested
- vllm-gdn venv: created, isolated from trtllm venv
- Dockerfile: written (docker/qwen35-gdn/)
- Router patch: written (src/ops/qwen35_gdn_patch.py)

## Unblocking condition
Either:
1. vLLM releases support for transformers 5.x class names (tracking issue #35998)
2. nvidia/Qwen3.5-35B-A3B-NVFP4 checkpoint published (Path C)
3. Manual auto_map injection into config.json (risky for fine-tuned weights)

## Production baseline remains
Qwen3-30B-A3B NVFP4 @ 134.84 T/s via TRT-LLM 1.2.0 — fully operational
