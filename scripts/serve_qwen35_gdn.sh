#!/usr/bin/env bash
# serve_qwen35_gdn.sh
# Launches vLLM to serve Qwen3.5-35B-A3B with the GDN op patched in.
# Uses the isolated /home/ubuntu/vllm-gdn/.venv — does NOT touch trtllm venv.
set -euo pipefail

VENV=/home/ubuntu/vllm-gdn/.venv
MODEL="${MODEL_PATH:-/var/lib/rancher/k3s/storage/pvc-75765228-fb32-4ac8-94b9-5af200bb7aac_default_openclaw-qwen3-outputs/architect-preflight-100-qwen35/final}"
SO_PATH="${GDN_OP_LIB:-/home/ubuntu/blackwell-kernel/sm_120/build/m4-vllm/src/ops/libgdn_sm120_op.so}"
PORT="${SERVE_PORT:-8001}"
REPO=/home/ubuntu/blackwell-kernel/sm_120
LOG_DIR="${REPO}/artifacts/path_b/logs"

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/serve_$(date -u +%Y%m%dT%H%M%SZ).log"

source "${VENV}/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export GDN_OP_LIB="${SO_PATH}"
export PYTHONPATH="${REPO}/src:${REPO}/src/ops:${PYTHONPATH:-}"

echo "[$(date -u +%FT%TZ)] serve_qwen35_gdn.sh" | tee "${LOG}"
echo "[$(date -u +%FT%TZ)] model=${MODEL}" | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] so=${SO_PATH}" | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] port=${PORT}" | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] vllm=$(python -c 'import vllm; print(vllm.__version__)')" | tee -a "${LOG}"

# verify SO exists
if [[ ! -f "${SO_PATH}" ]]; then
    echo "[ERROR] libgdn_sm120_op.so not found at ${SO_PATH}" | tee -a "${LOG}"
    exit 1
fi

# verify model exists
if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "[ERROR] model config.json not found at ${MODEL}" | tee -a "${LOG}"
    exit 1
fi

echo "[$(date -u +%FT%TZ)] Starting vLLM server..." | tee -a "${LOG}"

# --custom-op-lib loads the GDN .so at server startup
# PYTHONSTARTUP runs qwen35_gdn_patch.apply() before model load
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port "${PORT}" \
    --custom-op-lib "${SO_PATH}" \
    2>&1 | tee -a "${LOG}"
