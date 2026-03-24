#!/usr/bin/env bash
set -euo pipefail

# Remote Node Environment
VENV_PATH="/home/ubuntu/labs/sglang_nvfp4_gdn/.venv"
VLLM_SOURCE="/home/ubuntu/vllm-gdn/vllm-sm120"
MODEL_PATH="/home/ubuntu/labs/sglang_nvfp4_gdn/models/qwen35_nvfp4"

source "${VENV_PATH}/bin/activate"

export PYTHONPATH="${VLLM_SOURCE}:${PYTHONPATH:-}"
export VLLM_PLUGINS="ops.vllm_sm120_plugin"

# P1 launch — --enforce-eager REMOVED
# Increasing gpu-memory-utilization to 0.85 to provide headroom for KV cache
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --port 18080 \
  --served-model-name qwen35-nvfp4
