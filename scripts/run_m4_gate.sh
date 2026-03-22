#!/usr/bin/env bash
# run_m4_gate.sh — M4 torch.ops registration gate
# Builds libgdn_sm120_op.so, runs test_vllm_integration.py
# Exits 0 on PASS, 1 on FAIL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/m4"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/m4/logs"
LOG="${ARTIFACT_DIR}/m4_gate_$(date -u +%Y%m%dT%H%M%SZ).log"
VENV="${VENV:-/home/ubuntu/trtllm/.venv}"
PYTHON="${VENV}/bin/python3"
PYTEST="${VENV}/bin/pytest"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"

mkdir -p "${ARTIFACT_DIR}"

echo "[$(date -u +%FT%TZ)] run_m4_gate.sh starting" | tee "${LOG}"
echo "[$(date -u +%FT%TZ)] VENV=${VENV}" | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] torch: $(${PYTHON} -c 'import torch; print(torch.__version__)')" | tee -a "${LOG}"

# ── cmake configure ───────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] cmake configure" | tee -a "${LOG}"
/usr/bin/cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSM120_BUILD_TESTS=OFF \
    -DSM120_BUILD_BENCHMARKS=OFF \
    -DSM120_BUILD_OPS=ON \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -DPython3_EXECUTABLE="${PYTHON}" \
    2>&1 | tee -a "${LOG}"

# ── build op ─────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] cmake build gdn_sm120_op" | tee -a "${LOG}"
/usr/bin/cmake --build "${BUILD_DIR}" \
    --target gdn_sm120_op \
    --parallel "$(nproc)" \
    2>&1 | tee -a "${LOG}"

SO="${BUILD_DIR}/src/ops/libgdn_sm120_op.so"
if [[ ! -f "${SO}" ]]; then
    echo "[$(date -u +%FT%TZ)] FAIL: ${SO} not found after build" | tee -a "${LOG}"
    exit 1
fi
echo "[$(date -u +%FT%TZ)] Built: ${SO}" | tee -a "${LOG}"

# ── run pytest ────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] pytest test_vllm_integration.py" | tee -a "${LOG}"
GDN_OP_LIB="${SO}" "${PYTEST}" \
    "${REPO_ROOT}/tests/test_vllm_integration.py" \
    -v --tb=short \
    2>&1 | tee -a "${LOG}"
TEST_EXIT=${PIPESTATUS[0]}

# ── node metadata ─────────────────────────────────────────────────────────────
{
    echo "---"
    echo "node: $(hostname)"
    echo "date: $(date -u +%FT%TZ)"
    echo "cuda: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "torch: $(${PYTHON} -c 'import torch; print(torch.__version__)')"
    echo "test_exit: ${TEST_EXIT}"
} | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
if [[ ${TEST_EXIT} -eq 0 ]]; then
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
    echo "  M4 GATE: PASS"                               | tee -a "${LOG}"
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
else
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
    echo "  M4 GATE: FAIL  (exit=${TEST_EXIT})"          | tee -a "${LOG}"
    echo "══════════════════════════════════════════════" | tee -a "${LOG}"
fi

echo "[$(date -u +%FT%TZ)] Artifacts: ${LOG}" | tee -a "${LOG}"
exit ${TEST_EXIT}
