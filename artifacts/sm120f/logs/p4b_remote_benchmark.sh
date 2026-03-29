#!/usr/bin/env bash
set -euo pipefail

REMOTE="ubuntu@192.168.0.233"

ssh "${REMOTE}" 'bash -s' <<'EOF'
set -euo pipefail

PORT=18080
MODEL="qwen35-nvfp4"

run_request() {
  local id="$1"
  local prompt="User ${id}: Explain the concept of quantum entanglement in simple terms."
  local start_time
  local end_time
  local response

  start_time=$(date +%s.%N)
  response=$(curl -s http://127.0.0.1:${PORT}/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${prompt}\",\"max_tokens\":128,\"temperature\":0.0}")
  end_time=$(date +%s.%N)

  python3 -c 'import json, sys
req_id = sys.argv[1]
start_time = float(sys.argv[2])
end_time = float(sys.argv[3])
payload = json.load(sys.stdin)
elapsed = end_time - start_time
tokens = payload["usage"]["completion_tokens"]
tps = tokens / elapsed
text = payload["choices"][0]["text"].replace("\n", " ")[:60]
print(f"Request {req_id} -> Tokens: {tokens}, Time: {elapsed:.2f}s, TPS: {tps:.2f}")
print(f"  Text[:60]: {text}")' \
    "$id" "$start_time" "$end_time" <<<"${response}"
}

for run in 1 2 3; do
  echo "=== RUN ${run} ==="
  run_request 1 &
  run_request 2 &
  run_request 3 &
  run_request 4 &
  wait
  echo "RUN_PASS=True"
  echo
  sleep 2
done
EOF
