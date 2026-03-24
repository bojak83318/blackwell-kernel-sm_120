#!/usr/bin/env bash
set -euo pipefail

PORT=18080
MODEL="qwen35-nvfp4"

echo "=== P3 Batching Long-Prompt Benchmark (4 concurrent requests) ==="

run_request() {
  local id=$1
  local prompt=$(python3 -c "print('User $id: Summarize the following technical document about GDN optimization on SM_120 architecture. ' + 'Blackwell architecture provides NVFP4 precision for weights and BF16 for recurrent states. ' * 20)")
  local start_time=$(date +%s.%N)
  
  RESPONSE=$(curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",
         \"prompt\":\"${prompt}\",
         \"max_tokens\":128,\"temperature\":0.0}")
  
  local end_time=$(date +%s.%N)
  local elapsed=$(echo "$end_time - $start_time" | bc)
  
  echo "$RESPONSE" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    tokens = d['usage']['completion_tokens']
    elapsed = float('$elapsed')
    tps = tokens / elapsed
    print(f'Request {id} -> Tokens: {tokens}, Time: {elapsed:.2f}s, TPS: {tps:.2f}')
    print(f'  Text[:60]: {d[\"choices\"][0][\"text\"][:60].replace(\"\\n\", \" \")}')
except Exception as e:
    print(f'Request {id} ERROR: {e}')
"
}

# Launch 4 requests in parallel
run_request 1 &
run_request 2 &
run_request 3 &
run_request 4 &

wait

echo "=== P3 Concurrent Run Complete ==="
