#!/usr/bin/env bash
set -euo pipefail

PORT=18080
MODEL="qwen35-nvfp4"

echo "=== P1 Single-Sequence Benchmark ==="
RESPONSE=$(curl -s -w "TIME:%{time_total}" \
  http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",
       \"prompt\":\"Explain the architecture of a mixture-of-experts language model in detail.\",
       \"max_tokens\":128,\"temperature\":0.0}")

echo "$RESPONSE" | sed 's/TIME:.*//' | python3 -m json.tool || echo "$RESPONSE"
echo "Elapsed: $(echo "$RESPONSE" | grep -o 'TIME:.*' | cut -d: -f2)s"

echo -e "\n=== P1 5-run Benchmark Lock ==="
for i in 1 2 3 4 5; do
  echo -n "--- Run $i --- "
  RESPONSE=$(curl -s -w "TIME:%{time_total}" \
    http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",
         \"prompt\":\"Explain the architecture of a mixture-of-experts language model in detail.\",
         \"max_tokens\":128,\"temperature\":0.0}")
  
  ELAPSED=$(echo "$RESPONSE" | grep -o 'TIME:.*' | cut -d: -f2)
  JSON_BODY=$(echo "$RESPONSE" | sed 's/TIME:.*//')
  
  echo "$JSON_BODY" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    tokens = d['usage']['completion_tokens']
    elapsed = float('$ELAPSED')
    tps = tokens / elapsed
    print(f'Tokens: {tokens}, Time: {elapsed:.2f}s, TPS: {tps:.2f}')
    print(f'  Text[:60]: {d[\"choices\"][0][\"text\"][:60].replace(\"\\n\", \" \")}')
except Exception as e:
    print('ERROR:', e)
    print('RAW:', sys.stdin.read())
"
  sleep 2
done
