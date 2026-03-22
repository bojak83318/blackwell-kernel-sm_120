#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/m1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/m1}"
LOG_DIR="$ARTIFACT_ROOT/logs"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 4)}"
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DSM120_BUILD_TESTS=ON
  -DSM120_BUILD_BENCHMARKS=ON
  -DSM120_TARGET_ARCH=sm_120
)

configure_log="$LOG_DIR/configure.log"
build_log="$LOG_DIR/build.log"
test_log="$LOG_DIR/tests.log"
bench_log="$LOG_DIR/benchmarks.log"
disasm_log="$LOG_DIR/disassembly.log"

usage() {
  cat <<USAGE
Usage: ${0##*/} [--skip-build] [--skip-tests] [--skip-bench] [--skip-disasm]

Helper that configures the M1-C build (sm_120), runs the placeholder tests/benchmark harnesses, and captures PTX/disassembly artifacts under $ARTIFACT_ROOT.

Options:
  --skip-build      skip configure/build stages
  --skip-tests      skip `ctest`
  --skip-bench      skip running the benchmark harness
  --skip-disasm     skip disassembly capture (requires cuobjdump/nvdisasm)
  --help            show this text
USAGE
}

run_and_log() {
  local log="$1"
  shift
  printf '\n[%s] Running: %s\n' "$(date --iso-8601=seconds)" "${*}" | tee -a "$log"
  if ! ("$@" 2>&1 | tee -a "$log"); then
    printf '\n[%s] ✗ Command failed: %s\n' "$(date --iso-8601=seconds)" "${*}" | tee -a "$log"
    exit 1
  fi
}

# parse args
skip_build=0
skip_tests=0
skip_bench=0
skip_disasm=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-tests)
      skip_tests=1
      shift
      ;;
    --skip-bench)
      skip_bench=1
      shift
      ;;
    --skip-disasm)
      skip_disasm=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$BUILD_DIR" "$LOG_DIR" "$ARTIFACT_ROOT/ptx"

if [[ $skip_build -eq 0 ]]; then
  run_and_log "$configure_log" cmake -S "$REPO_ROOT" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
  run_and_log "$build_log" cmake --build "$BUILD_DIR" --parallel "$PARALLEL_JOBS"
else
  printf '\nSkipping configure/build stage\n'
fi

if [[ $skip_tests -eq 0 ]]; then
  run_and_log "$test_log" ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel "$PARALLEL_JOBS"
else
  printf '\nSkipping tests\n'
fi

if [[ $skip_bench -eq 0 ]]; then
  bench_bin="$BUILD_DIR/benchmarks/sm120_benchmark_main"
  if [[ ! -x "$bench_bin" ]]; then
    echo "Benchmark binary not built: $bench_bin" | tee -a "$bench_log"
    exit 1
  fi
  run_and_log "$bench_log" "$bench_bin" --profile context_throughput --iterations 2 --verbose
else
  printf '\nSkipping benchmark run\n'
fi

if [[ $skip_disasm -eq 0 ]]; then
  ptx_files=(
    "$BUILD_DIR/src/kernel/gdn_kernel_sm120.ptx"
    "$BUILD_DIR/src/kernel/gdn_tiled_sm120.ptx"
    "$BUILD_DIR/src/kernel/tcgen05_probe_sm120.ptx"
  )
  for ptx in "${ptx_files[@]}"; do
    if [[ -f "$ptx" ]]; then
      cp -f "$ptx" "$ARTIFACT_ROOT/ptx/$(basename "$ptx")"
    fi
  done

  objfile="$(find "$BUILD_DIR" -path '*/CMakeFiles/sm120_kernel.dir/*.cu.o' -print -quit)"
  if [[ -z "$objfile" ]]; then
    objfile="$(find "$BUILD_DIR" -name 'gdn_kernel.cu.o' -print -quit)"
  fi

  if [[ -n "$objfile" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
      run_and_log "$disasm_log" cuobjdump --dump-sass "$objfile"
    elif command -v nvdisasm >/dev/null 2>&1; then
      run_and_log "$disasm_log" nvdisasm "$objfile"
    else
      run_and_log "$disasm_log" objdump -d "$objfile"
    fi
  else
    printf '\nNo kernel object found for disassembly\n' | tee -a "$disasm_log"
  fi
else
  printf '\nSkipping disassembly capture\n'
fi

printf '\nM1-C artifacts captured under %s\n' "$ARTIFACT_ROOT"
