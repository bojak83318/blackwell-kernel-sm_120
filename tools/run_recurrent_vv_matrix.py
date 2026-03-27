#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass


DEFAULT_HOST = "ubuntu@192.168.0.233"
DEFAULT_BASE_URL = "http://192.168.0.233:8000"
DEFAULT_LOG = "/home/ubuntu/vllm_server.log"
DEFAULT_MODEL = "/home/ubuntu/labs/sglang_nvfp4_gdn/models/qwen35_nvfp4"
DEFAULT_PROMPT = (
    "Write a detailed technical essay on the history of floating point numerical "
    "formats, covering IEEE 754, bfloat16, FP8, and FP4, including their tradeoffs "
    "in memory bandwidth, dynamic range, and use in deep learning hardware from "
    "2015 to 2026."
)


@dataclass
class SweepResult:
    seqs: int
    http_ok: bool
    sqnr_min: str
    sqnr_mean: str
    nan_inf: str
    wall_ms: str


def ssh_run(host: str, *args: str) -> str:
    proc = subprocess.run(
        ["ssh", host, *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def remote_line_count(host: str, log_path: str) -> int:
    code = (
        "from pathlib import Path\n"
        f"print(sum(1 for _ in Path({log_path!r}).open()))\n"
    )
    proc = subprocess.run(
        ["ssh", host, "python3", "-"],
        input=code,
        text=True,
        check=True,
        capture_output=True,
    )
    return int(proc.stdout.strip())


def fetch_remote_log_slice(host: str, log_path: str, start_line: int, end_line: int) -> str:
    code = (
        "from pathlib import Path\n"
        f"lines = Path({log_path!r}).read_text().splitlines()\n"
        f"start = max({start_line} - 1, 0)\n"
        f"end = min({end_line}, len(lines))\n"
        "print('\\n'.join(lines[start:end]))\n"
    )
    proc = subprocess.run(
        ["ssh", host, "python3", "-"],
        input=code,
        text=True,
        check=True,
        capture_output=True,
    )
    return proc.stdout


def http_post(url: str, payload: dict, timeout: int) -> int:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return exc.code


async def fire_batch(base_url: str, model: str, prompt: str, seqs: int, timeout: int) -> list[int]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": 0,
    }
    url = f"{base_url}/v1/completions"
    tasks = [
        asyncio.to_thread(http_post, url, payload, timeout)
        for _ in range(seqs)
    ]
    return await asyncio.gather(*tasks)


def wait_for_health(base_url: str, timeout_s: int = 10) -> bool:
    url = f"{base_url}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
            return resp.status == 200
    except Exception:
        return False


def parse_window(window_text: str) -> tuple[list[float], int, int, int, list[float]]:
    window = window_text.splitlines()
    sqnr_vals: list[float] = []
    wall_vals: list[float] = []
    nans = 0
    infs = 0
    clipped_bad = 0

    for line in window:
        if "GDN Telemetry" not in line or "SQNR=99.00" in line:
            continue

        m = re.search(r"SQNR=([^ ]+)", line)
        if m:
            try:
                sqnr_vals.append(float(m.group(1)))
            except ValueError:
                pass

        m = re.search(r"NaN=(\d+)", line)
        if m:
            nans += int(m.group(1))

        m = re.search(r"Inf=(\d+)", line)
        if m:
            infs += int(m.group(1))

        m = re.search(r"pct_clipped=([0-9.]+)%", line)
        if m and float(m.group(1)) >= 1.0:
            clipped_bad += 1

        m = re.search(r"wall=([0-9.]+) ms", line)
        if m:
            wall_vals.append(float(m.group(1)))

    return sqnr_vals, nans, infs, clipped_bad, wall_vals


async def run_tier(host: str, log_path: str, base_url: str, model: str, prompt: str, seqs: int, timeout: int) -> SweepResult:
    _token = uuid.uuid4().hex[:12]
    _ = _token
    before_warmup = remote_line_count(host, log_path)
    warmup_codes = await fire_batch(base_url, model, prompt, seqs, timeout)
    before_measure = remote_line_count(host, log_path)
    measure_codes = await fire_batch(base_url, model, prompt, seqs, timeout)
    after_measure = remote_line_count(host, log_path)

    _ = before_warmup  # retained for debugging symmetry with the warmup phase
    log_text = fetch_remote_log_slice(host, log_path, before_measure + 1, after_measure)
    sqnr_vals, nans, infs, clipped_bad, wall_vals = parse_window(log_text)

    http_ok = (
        len(warmup_codes) == seqs
        and len(measure_codes) == seqs
        and all(code == 200 for code in warmup_codes)
        and all(code == 200 for code in measure_codes)
    )

    if not sqnr_vals:
        return SweepResult(
            seqs=seqs,
            http_ok=http_ok,
            sqnr_min="NA",
            sqnr_mean="NA",
            nan_inf=f"{nans}/{infs}",
            wall_ms="NA",
        )

    sqnr_min = min(sqnr_vals)
    sqnr_mean = sum(sqnr_vals) / len(sqnr_vals)
    wall_mean = sum(wall_vals) / len(wall_vals) if wall_vals else float("nan")
    if clipped_bad:
        nan_inf = f"{nans}/{infs} clip={clipped_bad}"
    else:
        nan_inf = f"{nans}/{infs}"

    return SweepResult(
        seqs=seqs,
        http_ok=http_ok,
        sqnr_min=f"{sqnr_min:.2f}",
        sqnr_mean=f"{sqnr_mean:.2f}",
        nan_inf=nan_inf,
        wall_ms=f"{wall_mean:.3f}",
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", default=DEFAULT_HOST)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not wait_for_health(args.base_url):
        print("Remote server is not healthy.", file=sys.stderr)
        return 1

    results: list[SweepResult] = []
    for seqs in (1, 4, 8):
        started = time.time()
        result = await run_tier(
            args.ssh_host,
            args.log_path,
            args.base_url,
            args.model,
            args.prompt,
            seqs,
            args.timeout,
        )
        elapsed = time.time() - started
        print(f"completed seqs={seqs} elapsed_s={elapsed:.1f}", file=sys.stderr)
        results.append(result)

    print("| Seqs | HTTP OK | SQNR Min | SQNR Mean | NaN/Inf | Wall (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in results:
        print(
            f"| {row.seqs} | {'yes' if row.http_ok else 'no'} | {row.sqnr_min} | "
            f"{row.sqnr_mean} | {row.nan_inf} | {row.wall_ms} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
