#!/usr/bin/env bash
# watch_nvfp4_checkpoint.sh — Path C parallel watcher
# Polls HuggingFace for nvidia/Qwen3.5-35B-A3B-NVFP4 availability.
# Run in background: nohup bash scripts/watch_nvfp4_checkpoint.sh &
set -euo pipefail

HF_REPO="nvidia/Qwen3.5-35B-A3B-NVFP4"
CHECK_INTERVAL=3600   # check every hour
LOG=/home/ubuntu/blackwell-kernel/sm_120/artifacts/path_b/logs/nvfp4_watch.log

mkdir -p "$(dirname "${LOG}")"

echo "[$(date -u +%FT%TZ)] Watching ${HF_REPO}" | tee "${LOG}"

while true; do
    STATUS=$(curl -sf \
        "https://huggingface.co/api/models/${HF_REPO}" \
        -H "Accept: application/json" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print('EXISTS')" \
        2>/dev/null || echo "NOT_FOUND")

    echo "[$(date -u +%FT%TZ)] ${HF_REPO}: ${STATUS}" | tee -a "${LOG}"

    if [[ "${STATUS}" == "EXISTS" ]]; then
        echo "[$(date -u +%FT%TZ)] CHECKPOINT AVAILABLE — download with:" | tee -a "${LOG}"
        echo "  source /home/ubuntu/vllm-gdn/.venv/bin/activate" | tee -a "${LOG}"
        echo "  huggingface-cli download ${HF_REPO} --local-dir /home/ubuntu/${HF_REPO##*/}" | tee -a "${LOG}"
        echo "  # then switch serve_qwen35_gdn.sh MODEL_PATH to NVFP4 checkpoint" | tee -a "${LOG}"
        break
    fi

    sleep "${CHECK_INTERVAL}"
done
