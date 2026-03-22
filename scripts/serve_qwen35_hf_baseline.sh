#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/vllm-gdn/.venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8001 \
  --language-model-only \
  --generation-config vllm

