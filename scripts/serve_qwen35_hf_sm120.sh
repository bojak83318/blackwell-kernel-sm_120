#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/ubuntu/vllm-gdn/.venv/bin/activate

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export VLLM_PLUGINS="ops.vllm_sm120_plugin"
export SM120_GDN_LIBRARY="${SM120_GDN_LIBRARY:-${REPO_ROOT}/build/m4/src/ops/libsm120_ops.so}"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8001 \
  --language-model-only \
  --generation-config vllm

