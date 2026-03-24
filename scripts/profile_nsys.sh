#!/usr/bin/env bash
set -euo pipefail

# This script runs nsys profile on a single request
PORT=18080
MODEL="qwen35-nvfp4"

echo "=== Starting NSYS Profiling ==="

# We need to run nsys on the remote node, but we need to profile the ALREADY RUNNING server
# Or we can start a new one under nsys.
# Profiling an already running server is better for warmup.

# Get PID of EngineCore
ENGINE_PID=$(ssh ubuntu@192.168.0.233 "pgrep -f VLLM::EngineCore | head -n 1")

if [ -z "$ENGINE_PID" ]; then
  echo "ERROR: EngineCore not found."
  exit 1
fi

echo "Profiling EngineCore PID: $ENGINE_PID"

# Run nsys profile on the remote node, attaching to the process
# We profile for 10 seconds while sending a request
ssh ubuntu@192.168.0.233 "nohup nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o /home/ubuntu/labs/throughput_opt/profile_report --attach-process $ENGINE_PID --duration 10 > nsys.log 2>&1 &"

sleep 2

# Send a heavy request to profile
echo "Sending request to profile..."
./scripts/benchmark_phase3.sh

sleep 10

echo "NSYS Profile captured. Downloading report..."
rsync -avz ubuntu@192.168.0.233:/home/ubuntu/labs/throughput_opt/profile_report.nsys-rep .
