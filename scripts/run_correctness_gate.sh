#!/usr/bin/env bash
# run_correctness_gate.sh — M3 correctness gate
# Builds the test_gdn_correctness target and runs it.
# Artifacts written to artifacts/m3/logs/
# Exits 0 on PASS, 1 on FAIL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/m3"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/m3/logs"
LOG="${ARTIFACT_DIR}/correctness_gate_$(date -u +%Y%m%dT%H%M%SZ).log"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"

mkdir -p "${ARTIFACT_DIR}"

echo "[$(date -u +%FT%TZ)] run_correctness_gate.sh starting" | tee "${LOG}"
echo "[$(date -u +%FT%TZ)] BUILD_DIR=${BUILD_DIR}" | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] CUDA: $(nvcc --version | head -1)" | tee -a "${LOG}"

# ── configure ─────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Running: cmake configure" | tee -a "${LOG}"
cmake -S "${REPO_ROOT}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DSM120_BUILD_TESTS=ON \
      -DSM120_BUILD_BENCHMARKS=OFF \
      -DSM120_TARGET_ARCH=sm_120f \
      2>&1 | tee -a "${LOG}"

# ── build ─────────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Running: cmake build" | tee -a "${LOG}"
cmake --build "${BUILD_DIR}" \
      --target test_gdn_correctness \
      --parallel "$(nproc)" \
      2>&1 | tee -a "${LOG}"

# ── disassembly check (FR-7) ──────────────────────────────────────────────────
OBJ=$(find "${BUILD_DIR}" -name "test_gdn_correctness*" -name "*.o" 2>/dev/null | head -1)
if [[ -n "${OBJ}" ]]; then
    echo "[$(date -u +%FT%TZ)] cuobjdump check for cvt.rn.satfinite.e2m1x2" | tee -a "${LOG}"
    cuobjdump --dump-sass "${OBJ}" 2>/dev/null \
        | grep -iE "cvt|e2m1|satfinite" | tee -a "${LOG}" || true
else
    echo "[$(date -u +%FT%TZ)] WARNING: could not find test object for disassembly" | tee -a "${LOG}"
fi

# ── run test ──────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Running: test_gdn_correctness" | tee -a "${LOG}"
TEST_BIN=$(find "${BUILD_DIR}" -name "test_gdn_correctness" -type f | head -1)

if [[ -z "${TEST_BIN}" ]]; then
    echo "[$(date -u +%FT%TZ)] FAIL: test binary not found after build" | tee -a "${LOG}"
    exit 1
fi

"${TEST_BIN}" 2>&1 | tee -a "${LOG}"
TEST_EXIT=${PIPESTATUS[0]}

# ── node metadata ─────────────────────────────────────────────────────────────
{
    echo "---"
    echo "node: $(hostname)"
    echo "date: $(date -u +%FT%TZ)"
    echo "cuda: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "test_exit: ${TEST_EXIT}"
} | tee -a "${LOG}"

# ── PASS / FAIL banner ────────────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
if [[ ${TEST_EXIT} -eq 0 ]]; then
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
    echo "  M3 CORRECTNESS GATE: PASS"                   | tee -a "${LOG}"
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
else
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
    echo "  M3 CORRECTNESS GATE: FAIL  (exit=${TEST_EXIT})" | tee -a "${LOG}"
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
fi

echo "[$(date -u +%FT%TZ)] Artifacts: ${LOG}" | tee -a "${LOG}"
exit ${TEST_EXIT}
