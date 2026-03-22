#!/usr/bin/env bash
# smoke_qwen35_gdn.sh
# Sends a test prompt to the running Qwen3.5 GDN server and reports T/s.
# Run after serve_qwen35_gdn.sh is up.
set -euo pipefail

PORT="${SERVE_PORT:-8001}"
REPO=/home/ubuntu/blackwell-kernel/sm_120
LOG_DIR="${REPO}/artifacts/path_b/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

BASE_URL="http://localhost:${PORT}"

echo "[$(date -u +%FT%TZ)] smoke_qwen35_gdn.sh" | tee "${LOG}"

# wait for server ready
echo "Waiting for server on port ${PORT}..." | tee -a "${LOG}"
for i in $(seq 1 60); do
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "Server ready." | tee -a "${LOG}"
        break
    fi
    sleep 2
done

# benchmark: 5 requests, 512 output tokens
PROMPT="Explain the Gated Delta Network architecture and its advantages over softmax attention for long-context inference."

START=$(date +%s%3N)
TOTAL_TOKENS=0

for i in $(seq 1 5); do
    RESP=$(curl -sf "${BASE_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"qwen3.5\",
            \"prompt\": \"${PROMPT}\",
            \"max_tokens\": 512,
            \"temperature\": 0.0
        }")
    TOKENS=$(echo "${RESP}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d['usage']['completion_tokens'])
" 2>/dev/null || echo 0)
    TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
    echo "  request ${i}: ${TOKENS} tokens" | tee -a "${LOG}"
done

END=$(date +%s%3N)
ELAPSED=$(( (END - START) ))
ELAPSED_S=$(echo "scale=2; $ELAPSED / 1000" | bc)
TPS=$(echo "scale=1; $TOTAL_TOKENS * 1000 / $ELAPSED" | bc)

echo "" | tee -a "${LOG}"
echo "──────────────────────────────────────────" | tee -a "${LOG}"
echo "  Qwen3.5-35B-A3B BF16 + GDN op"           | tee -a "${LOG}"
echo "  total_tokens : ${TOTAL_TOKENS}"           | tee -a "${LOG}"
echo "  elapsed      : ${ELAPSED_S}s"             | tee -a "${LOG}"
echo "  tokens/sec   : ${TPS}"                    | tee -a "${LOG}"
echo "  baseline     : Qwen3-30B-A3B NVFP4 @ 134.84 T/s (TRT-LLM)" | tee -a "${LOG}"
echo "──────────────────────────────────────────" | tee -a "${LOG}"
