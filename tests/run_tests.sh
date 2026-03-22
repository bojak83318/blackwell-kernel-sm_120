#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${1:-${REPO_ROOT}/build/tests}"
TEST_OUTPUT="${2:-${REPO_ROOT}/artifacts/m1/tests}"

mkdir -p "${BUILD_DIR}" "${TEST_OUTPUT}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DSM120_BUILD_TESTS=ON -DSM120_BUILD_BENCHMARKS=OFF
cmake --build "${BUILD_DIR}" --target sm120_test_harness
ctest --test-dir "${BUILD_DIR}" --output-on-failure | tee "${TEST_OUTPUT}/ctest.log"
