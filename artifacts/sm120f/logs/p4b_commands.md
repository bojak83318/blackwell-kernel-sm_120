# P4-B Validation Commands

Remote host: `ubuntu@192.168.0.233`
Remote venv: `/home/ubuntu/labs/sglang_nvfp4_gdn/.venv`
Remote repo: `/home/ubuntu/vllm-gdn/vllm-sm120`

## Bridge smoke

```sh
ssh ubuntu@192.168.0.233 'source /home/ubuntu/labs/sglang_nvfp4_gdn/.venv/bin/activate && PYTHONPATH=/home/ubuntu/vllm-gdn/vllm-sm120 CUDA_HOME=/usr/local/cuda-12.9 python - <<\"PY\"
import torch
from vllm.model_executor.layers.gdn.gdn_mixed_precision import _load_extension
ext = _load_extension()
print("EXT_NAME=" + str(getattr(ext, "__name__", None)))
print("HAS_SM120=" + str(hasattr(torch.ops, "sm120")))
print("HAS_SM120_GDN=" + str(hasattr(getattr(torch.ops, "sm120", object()), "gdn")))
PY'
```

Outcome:
- `gdn_state_update_ext CUDA extension loaded.`
- `EXT_NAME=gdn_state_update_ext`
- `HAS_SM120=True`
- `HAS_SM120_GDN=False`

## Server launch

Initial attempt:

```sh
ssh ubuntu@192.168.0.233 'bash -s' <<'EOF'
set -euo pipefail
source /home/ubuntu/labs/sglang_nvfp4_gdn/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/vllm-gdn/vllm-sm120
export CUDA_HOME=/usr/local/cuda-12.9
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/labs/sglang_nvfp4_gdn/models/qwen35_nvfp4 \
  --served-model-name qwen35-nvfp4 \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --gdn-prefill-backend triton \
  --gpu-memory-utilization 0.85 \
  --port 18080 > /tmp/sm120f_p4b_server.log 2>&1 < /dev/null &
EOF
```

Outcome:
- Failed with `ValueError: No available memory for the cache blocks.`

Successful retry:

```sh
ssh ubuntu@192.168.0.233 'bash -s' <<'EOF'
set -euo pipefail
source /home/ubuntu/labs/sglang_nvfp4_gdn/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/vllm-gdn/vllm-sm120
export CUDA_HOME=/usr/local/cuda-12.9
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/labs/sglang_nvfp4_gdn/models/qwen35_nvfp4 \
  --served-model-name qwen35-nvfp4 \
  --dtype bfloat16 \
  --max-model-len 256 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --gdn-prefill-backend triton \
  --gpu-memory-utilization 0.95 \
  --port 18080 > /tmp/sm120f_p4b_server.log 2>&1 < /dev/null &
EOF
```

Outcome:
- `Application startup complete.`
- `qwen35-nvfp4` served on `http://127.0.0.1:18080`

## Smoke request

```sh
ssh ubuntu@192.168.0.233 'bash -s' <<'EOF'
set -euo pipefail
source /home/ubuntu/labs/sglang_nvfp4_gdn/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/vllm-gdn/vllm-sm120
curl -sS http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-nvfp4","prompt":"Say hello in one short sentence.","max_tokens":8,"temperature":0.0}' \
  | python -m json.tool
EOF
```

Outcome:
- HTTP 200 response with `completion_tokens=8`

## Throughput benchmark

```sh
bash artifacts/sm120f/logs/p4b_remote_benchmark.sh | tee artifacts/sm120f/p4-b/batch4_benchmark.log
```

Outcome:
- Three batch-4 runs completed successfully.
- Average per-user throughput across all 12 samples: `120.42 tok/s/user`
- Min: `114.41 tok/s/user`
- Max: `127.20 tok/s/user`
