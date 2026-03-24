#!/usr/bin/env bash
set -euo pipefail

PORT=18080
MODEL="qwen35-nvfp4"

echo "=== Phase 5: Production Baseline Lock (5 x 4-concurrent) ==="

run_batch() {
  local batch_id=$1
  local start_time=$(date +%s.%N)
  
  # Launch 4 requests in parallel
  for i in 1 2 3 4; do
    (curl -s http://localhost:${PORT}/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",
           \"prompt\":\"Run $batch_id-$i: Explain the importance of high-performance kernels in AI.\",
           \"max_tokens\":128,\"temperature\":0.0}" > /dev/null) &
  done
  
  wait
  
  local end_time=$(date +%s.%N)
  local elapsed=$(echo "$end_time - $start_time" | bc)
  local total_tokens=512
  local aggregate_tps=$(echo "scale=2; $total_tokens / $elapsed" | bc)
  
  echo "Batch $batch_id -> Time: ${elapsed}s, Aggregate TPS: $aggregate_tps"
  echo $aggregate_tps >> tps_results.txt
}

rm -f tps_results.txt

for b in 1 2 3 4 5; do
  run_batch $b
  sleep 2
done

echo -e "\n=== Final Production Statistics ==="
python3 -c "
import statistics
with open('tps_results.txt') as f:
    results = [float(line.strip()) for f in [f] for line in f]
print(f'Average Aggregate TPS: {statistics.mean(results):.2f}')
print(f'Min Aggregate TPS:     {min(results):.2f}')
print(f'Max Aggregate TPS:     {max(results):.2f}')
print(f'StdDev:                {statistics.stdev(results):.2f}')
print(f'Variance %:            {(statistics.stdev(results)/statistics.mean(results))*100:.2f}%')
"
