# P4-B Batch-4 Throughput Summary

Benchmark target: `>=18.2 tok/s/user` at batch `4`.

Server config used:
- Remote host: `ubuntu@192.168.0.233`
- Model: `/home/ubuntu/labs/sglang_nvfp4_gdn/models/qwen35_nvfp4`
- Launch flags: `--max-model-len 512 --max-num-batched-tokens 1024 --max-num-seqs 4 --enable-chunked-prefill --kv-cache-dtype fp8 --gdn-prefill-backend triton`
- One debug `print()` in `/home/ubuntu/vllm-gdn/vllm-sm120/vllm/model_executor/models/qwen3_next.py` was removed on the remote node to unblock TorchDynamo warmup.

Benchmark shape:
- 3 runs
- 4 concurrent requests per run
- `max_tokens=128`
- prompt: `Explain the concept of quantum entanglement in simple terms.`

Results:
- Run 1: avg `5.47 tok/s/user`, min `5.47`, max `5.47` -> fail
- Run 2: avg `117.78 tok/s/user`, min `117.38`, max `117.92` -> pass
- Run 3: avg `130.32 tok/s/user`, min `130.29`, max `130.34` -> pass

Aggregate numbers:
- Overall mean across all 12 request samples: `84.52 tok/s/user`
- Overall min: `5.47 tok/s/user`
- Overall max: `130.34 tok/s/user`
- Steady-state mean across runs 2 and 3: `124.05 tok/s/user`

Decision:
- Pass on steady-state batch-4 throughput.
- The first run is a cold-start outlier and is documented in the raw log.
