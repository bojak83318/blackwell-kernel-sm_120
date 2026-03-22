#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${1:-${REPO_ROOT}/build/benchmarks}"
EVIDENCE_DIR="${2:-${REPO_ROOT}/artifacts/m2/benchmarks}"
BENCH_ARGS=()
if [[ "$#" -gt 2 ]]; then
  BENCH_ARGS=("${@:3}")
fi

mkdir -p "${BUILD_DIR}" "${EVIDENCE_DIR}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DSM120_BUILD_BENCHMARKS=ON -DSM120_BUILD_TESTS=OFF
cmake --build "${BUILD_DIR}" --target sm120_benchmark_harness
"${BUILD_DIR}/sm120_benchmark_main" "${BENCH_ARGS[@]:-}" | tee "${EVIDENCE_DIR}/benchmarks.log"
