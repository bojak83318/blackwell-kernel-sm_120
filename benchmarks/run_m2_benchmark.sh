#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_BUILD_DIR="${REPO_ROOT}/build/m2-benchmarks"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/artifacts/m2"

BUILD_DIR=""
OUTPUT_DIR=""
PROFILE="m2-c"
ITERATIONS="3"
CMAKE_EXTRA_ARGS=()
BIN_EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $0 [options] [-- extra benchmark args]

Options:
  --build-dir DIR      Configure and build under DIR (default: ${DEFAULT_BUILD_DIR})
  --output-dir DIR     Place logs, metadata, and summaries under DIR (default: ${DEFAULT_OUTPUT_DIR})
  --profile NAME       Benchmark profile name passed to the harness (default: ${PROFILE})
  --iterations N       Number of placeholder iterations (default: ${ITERATIONS})
  --cmake-arg ARG      Forward ARG to the configure command (multiple allowed)
  --help               Show this message and exit

Everything after the optional "--" is forwarded directly to the benchmark binary.
USAGE
}

require_arg() {
  if [[ $# -lt 2 ]]; then
    echo "Option '$1' requires an argument." >&2
    usage
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      require_arg "$1" "$@"
      BUILD_DIR="$2"
      shift 2
      ;;
    --output-dir)
      require_arg "$1" "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --profile)
      require_arg "$1" "$@"
      PROFILE="$2"
      shift 2
      ;;
    --iterations)
      require_arg "$1" "$@"
      ITERATIONS="$2"
      shift 2
      ;;
    --cmake-arg)
      require_arg "$1" "$@"
      CMAKE_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        BIN_EXTRA_ARGS+=("$@")
      fi
      break
      ;;
    -*?)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      BIN_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

BUILD_DIR="${BUILD_DIR:-$DEFAULT_BUILD_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

if [[ -z "$PROFILE" ]]; then
  echo "Profile name cannot be empty." >&2
  usage
  exit 1
fi

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || (( ITERATIONS < 1 )); then
  echo "Iterations must be a positive integer." >&2
  usage
  exit 1
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

CONFIG_CMD=(cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DSM120_BUILD_BENCHMARKS=ON -DSM120_BUILD_TESTS=OFF)
if [[ ${#CMAKE_EXTRA_ARGS[@]} -gt 0 ]]; then
  CONFIG_CMD+=("${CMAKE_EXTRA_ARGS[@]}")
fi

CONFIG_LOG="$OUTPUT_DIR/m2-configure.log"
{
  printf "# M2 benchmark configure log (%s)\n" "$(date --iso-8601=seconds)"
  printf "Command: %s\n\n" "${CONFIG_CMD[*]}"
} > "$CONFIG_LOG"
"${CONFIG_CMD[@]}" | tee -a "$CONFIG_LOG"

BUILD_LOG="$OUTPUT_DIR/m2-build.log"
{
  printf "# M2 benchmark build log (%s)\n" "$(date --iso-8601=seconds)"
} > "$BUILD_LOG"
cmake --build "$BUILD_DIR" --target sm120_benchmark_main 2>&1 | tee -a "$BUILD_LOG"

BENCH_BINARY="$BUILD_DIR/benchmarks/sm120_benchmark_main"
if [[ ! -x "$BENCH_BINARY" ]]; then
  echo "Benchmark binary not found or not executable: $BENCH_BINARY" >&2
  exit 1
fi

BENCH_ARGS=(--profile "$PROFILE" --iterations "$ITERATIONS")
BENCH_ARGS+=("${BIN_EXTRA_ARGS[@]}")

CMAKE_EXTRA_DISPLAY="${CMAKE_EXTRA_ARGS[*]}"
if [[ -z "$CMAKE_EXTRA_DISPLAY" ]]; then
  CMAKE_EXTRA_DISPLAY="(none)"
fi

BIN_EXTRA_DISPLAY="${BIN_EXTRA_ARGS[*]}"
if [[ -z "$BIN_EXTRA_DISPLAY" ]]; then
  BIN_EXTRA_DISPLAY="(none)"
fi

RUN_LOG="$OUTPUT_DIR/m2-benchmark.log"
{
  printf "# M2 benchmark run log (%s)\n" "$(date --iso-8601=seconds)"
  printf "Profile: %s\n" "$PROFILE"
  printf "Iterations: %s\n" "$ITERATIONS"
  printf "Binary: %s\n" "$BENCH_BINARY"
  printf "Binary args: %s\n\n" "${BENCH_ARGS[*]}"
} > "$RUN_LOG"
"$BENCH_BINARY" "${BENCH_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
GIT_STATUS="$(git status -sb 2>/dev/null || echo 'git status unavailable')"

METADATA_LOG="$OUTPUT_DIR/m2-run-metadata.txt"
{
  printf "# M2 benchmark metadata\n"
  printf "Timestamp: %s\n" "$(date --iso-8601=seconds)"
  printf "Profile: %s\n" "$PROFILE"
  printf "Iterations: %s\n" "$ITERATIONS"
  printf "Build dir: %s\n" "$BUILD_DIR"
  printf "Output dir: %s\n" "$OUTPUT_DIR"
  printf "Configure command: %s\n" "${CONFIG_CMD[*]}"
  printf "CMake extra args: %s\n" "$CMAKE_EXTRA_DISPLAY"
  printf "Build command: cmake --build %s --target sm120_benchmark_main\n" "$BUILD_DIR"
  printf "Binary: %s\n" "$BENCH_BINARY"
  printf "Binary args: %s\n" "${BENCH_ARGS[*]}"
  printf "Forwarded benchmark args: %s\n" "$BIN_EXTRA_DISPLAY"
  printf "Git commit: %s\n" "$GIT_COMMIT"
  printf "Git status:\n%s\n" "$GIT_STATUS"
  printf "System: %s\n" "$(uname -a)"
} > "$METADATA_LOG"

SUMMARY_FILE="$OUTPUT_DIR/m2-summary.md"
cat <<SUMMARY > "$SUMMARY_FILE"
# M2 Benchmark (M2-C) Summary

- **Profile:** \`$PROFILE\`
- **Iterations:** \`$ITERATIONS\`
- **Benchmark binary:** \`$BENCH_BINARY\`
- **Benchmark args:** \`${BENCH_ARGS[*]}\`
- **Forwarded benchmark args:** \`${BIN_EXTRA_DISPLAY}\`
- **Configure log:** \`$(basename "$CONFIG_LOG")\`
- **Build log:** \`$(basename "$BUILD_LOG")\`
- **Run log:** \`$(basename "$RUN_LOG")\`
- **Metadata:** \`$(basename "$METADATA_LOG")\`
SUMMARY


echo "Reproducible M2 benchmark completed. Artifacts stored under $OUTPUT_DIR"
for log in "$CONFIG_LOG" "$BUILD_LOG" "$RUN_LOG" "$METADATA_LOG" "$SUMMARY_FILE"; do
  echo "  - $log"
done
