#!/usr/bin/env python3
import datetime as _dt
import json
import statistics
import subprocess
import sys
from pathlib import Path


HOST = "ubuntu@192.168.0.233"
SERVER_LOG = "/tmp/sm120f_p4b_server.log"
OUT_DIR = Path("artifacts/sm120f/p4-b")
PROMPT = "Explain the concept of quantum entanglement in simple terms."


REMOTE_BENCH_CODE = r'''
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

PORT = 18080
MODEL = "qwen35-nvfp4"
PROMPT = "Explain the concept of quantum entanglement in simple terms."
URL = f"http://127.0.0.1:{PORT}/v1/completions"
PAYLOAD = json.dumps({
    "model": MODEL,
    "prompt": PROMPT,
    "max_tokens": 128,
    "temperature": 0.0,
})


def run_request(request_id: int):
    start = time.perf_counter()
    proc = subprocess.run(
        [
            "curl",
            "-s",
            URL,
            "-H",
            "Content-Type: application/json",
            "-d",
            PAYLOAD,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    data = json.loads(proc.stdout)
    tokens = data["usage"]["completion_tokens"]
    tps = tokens / elapsed
    return request_id, tokens, elapsed, tps


run_avgs = []
all_tps = []
for run_id in range(1, 4):
    print(f"=== RUN {run_id} ===")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = sorted(pool.map(run_request, range(1, 5)))
    run_tps = []
    for request_id, tokens, elapsed, tps in results:
        run_tps.append(tps)
        all_tps.append(tps)
        print(f"Request {request_id} -> Tokens: {tokens}, Time: {elapsed:.2f}s, TPS: {tps:.2f}")
    run_avg = statistics.mean(run_tps)
    run_min = min(run_tps)
    run_max = max(run_tps)
    run_pass = run_avg >= 18.2
    run_avgs.append(run_avg)
    print(f"RUN_AVG_TPS_PER_USER={run_avg:.2f}")
    print(f"RUN_MIN_TPS_PER_USER={run_min:.2f}")
    print(f"RUN_MAX_TPS_PER_USER={run_max:.2f}")
    print(f"RUN_PASS={str(run_pass)}")
    print()

print(f"OVERALL_MEAN_TPS_PER_USER={statistics.mean(all_tps):.2f}")
print(f"OVERALL_MIN_TPS_PER_USER={min(all_tps):.2f}")
print(f"OVERALL_MAX_TPS_PER_USER={max(all_tps):.2f}")
print(f"STEADY_STATE_MEAN_TPS_PER_USER={statistics.mean(run_avgs[1:]):.2f}")
print(f"STEADY_STATE_PASS={str(statistics.mean(run_avgs[1:]) >= 18.2)}")
'''


def run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    transcript = OUT_DIR / f"batch4_benchmark_{ts}.log"

    print(f"LOCAL_COMMAND: ssh {HOST} python3 -")
    bench = run(["ssh", HOST, "python3", "-"], input_text=REMOTE_BENCH_CODE)
    print(bench.stdout, end="")
    if bench.stderr:
        print(bench.stderr, end="", file=sys.stderr)

    print("REMOTE_DISPATCH_GREP")
    grep = run(
        [
            "ssh",
            HOST,
            "grep -n \"Using NvFp4LinearBackend.FLASHINFER_CUTLASS\\|Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend\\|Starting vLLM server on http://0.0.0.0:18080\" /tmp/sm120f_p4b_server.log | tail -n 20",
        ]
    )
    print(grep.stdout, end="")
    if grep.stderr:
        print(grep.stderr, end="", file=sys.stderr)

    transcript.write_text(
        "LOCAL_COMMAND: ssh ubuntu@192.168.0.233 python3 -\n"
        + bench.stdout
        + ("".join([bench.stderr]) if bench.stderr else "")
        + "REMOTE_DISPATCH_GREP\n"
        + grep.stdout
        + ("".join([grep.stderr]) if grep.stderr else ""),
        encoding="utf-8",
    )
    print(f"TRANSCRIPT_WRITTEN: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
