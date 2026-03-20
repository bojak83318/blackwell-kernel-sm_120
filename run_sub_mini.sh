#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run bounded sub-mini passes against sm_120 TASK/implementation files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$SCRIPT_DIR}"
MODEL="gpt-5.1-codex-mini"
TASK_FILE="$ROOT_DIR/TASK.md"
PLAN_FILE="$ROOT_DIR/implementation.md"
SKIP_GIT_CHECK_FLAG="${SKIP_GIT_CHECK_FLAG:---skip-git-repo-check}"
K3S_NODE_SSH="${K3S_NODE_SSH:-ubuntu@192.168.0.233}"
K3S_NODE_NAME="${K3S_NODE_NAME:-kimi-k3s-node}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/home/rocm/workspace/blackwell-kernel-worktrees}"
MULTI_AGENT_TARGET="${MULTI_AGENT_TARGET:-3}"

if [[ ! -f "$TASK_FILE" || ! -f "$PLAN_FILE" ]]; then
  echo "Missing required files: $TASK_FILE and/or $PLAN_FILE" >&2
  exit 1
fi

SLICE="${1:-next}"
MODE="${2:-run}" # run | dry-run
EXTRA_ARGS=()

if [[ $# -ge 2 ]]; then
  shift 2
  EXTRA_ARGS=("$@")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 && "${EXTRA_ARGS[0]}" == "--" ]]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

if [[ "$MODE" != "run" && "$MODE" != "dry-run" ]]; then
  echo "Invalid mode: $MODE (expected: run|dry-run)" >&2
  exit 1
fi

common_prompt() {
  cat <<EOF
Use \$sub-mini workflow.
Project files:
- $PLAN_FILE
- $TASK_FILE

Execution context:
- Primary edit workspace: $ROOT_DIR
- Remote validation node: $K3S_NODE_SSH (hostname: $K3S_NODE_NAME)
- Remote node has an NVIDIA GeForce RTX 5090, Docker, and kubectl available.
- Remote base shell does not currently expose cmake or nvcc in PATH, so prefer Docker/container-based validation there for CUDA/GPU work.
- Multi-agent mode: enabled by default with target parallel units = $MULTI_AGENT_TARGET.
- Git worktree mode: enabled by default with worktree root = $WORKTREE_ROOT.
- K3s preflight is mandatory before remote build/test work: confirm DiskPressure=False.
- Current known node state: DiskPressure=False, root filesystem about 91% used with about 46G free, inode usage about 3%.
- Avoid unnecessary image pulls, stale build directories, or large retained artifacts on the node.

Core rules:
- Read implementation.md and TASK.md first.
- Execute exactly one bounded phase slice, split into 2-4 parallel units where safe.
- Use disjoint file ownership per unit; do not allow overlapping write scopes.
- Use one git worktree per unit under $WORKTREE_ROOT and one branch per unit.
- Integrate unit branches only after verification in each worktree.
- Verify locally when possible, and use the remote node when CUDA/GPU/container validation is required.
- Update TASK.md with factual progress only.
- Do not start TRT-LLM plugin work before vLLM-path validation.
- If K3s DiskPressure becomes True or node disk headroom drops materially, stop and report the blocker.
EOF
}

build_prompt() {
  local slice="$1"
  common_prompt
  printf '\nSlice instructions:\n'

  case "$slice" in
    next)
      cat <<'PROMPT'
- Select the next unfinished slice from implementation.md.
- Respect the slice boundary and do not spill into later slices.
- Split the slice into independent units and run them with multi-agent delegation using git worktrees.
- For verification, prefer remote node checks when local tool availability is insufficient.
PROMPT
      ;;
    P0-A)
      cat <<'PROMPT'
- Execute only Phase 0 Slice P0-A from implementation.md.
- Scope ownership: root build files (for example bootstrap CMake config) and base directory skeleton under `sm_120`.
- Do not do P0-B or P0-C yet.
- If possible, split P0-A into parallel units (for example root CMake unit and skeleton layout unit) with disjoint ownership.
- For verification, prefer a real configure/smoke check on the remote node if local `cmake` is unavailable.
- Update TASK.md only for items proven complete.
PROMPT
      ;;
    P0-B)
      cat <<'PROMPT'
- Execute only Phase 0 Slice P0-B from implementation.md.
- Scope ownership: test harness skeleton and benchmark harness skeleton.
- Do not do kernel logic yet.
- Run test-harness and benchmark-harness units in separate worktrees when ownership is disjoint.
- Verify locally first, then use the remote node if the slice needs container/build validation.
- Update TASK.md factually.
PROMPT
      ;;
    P0-C)
      cat <<'PROMPT'
- Execute only Phase 0 Slice P0-C from implementation.md.
- Scope ownership: docs for environment assumptions and local/remote commands and prerequisites.
- No Phase 1 code.
- Prefer splitting docs units by concern (environment, commands, node operations) when parallelizable.
- Record the remote-node workflow clearly, including disk-pressure preflight and Docker-based validation.
- Update TASK.md factually.
PROMPT
      ;;
    P1-A)
      cat <<'PROMPT'
- Execute only Phase 1 Slice P1-A.
- Scope ownership: `src/kernel` scaffold for the SM_120 GDN kernel.
- No launcher, tests, or benchmark logic in this pass.
- If parallelized, keep sub-units disjoint (for example headers vs kernel source scaffolding).
- Verify buildability of touched files if possible; use the remote node for CUDA-aware validation if local toolchain is missing.
- Update TASK.md factually.
PROMPT
      ;;
    P1-B)
      cat <<'PROMPT'
- Execute only Phase 1 Slice P1-B.
- Scope ownership: `src/runtime` host launcher and parameter validation.
- Do not expand into PyTorch or vLLM integration yet.
- If parallelized, split by disjoint runtime modules and keep one worktree per module.
- Verify locally first, then use the remote node if CUDA/container checks are needed.
- Update TASK.md factually.
PROMPT
      ;;
    P1-C)
      cat <<'PROMPT'
- Execute only Phase 1 Slice P1-C.
- Scope ownership: `tests/unit` correctness and failure-case tests.
- Do not implement Phase 2 integration.
- Parallelize test units by disjoint files or test groups when possible.
- Run tests locally when possible and use the remote node for any GPU-sensitive validation.
- Update TASK.md factually.
PROMPT
      ;;
    *)
      echo "Unknown slice: $slice" >&2
      echo "Allowed: next P0-A P0-B P0-C P1-A P1-B P1-C" >&2
      exit 1
      ;;
  esac
}

PROMPT_CONTENT="$(build_prompt "$SLICE")"

if [[ "$MODE" == "dry-run" ]]; then
  printf 'Slice: %s\nModel: %s\n\n' "$SLICE" "$MODEL"
  printf '%s\n' "$PROMPT_CONTENT"
  exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec codex exec "$SKIP_GIT_CHECK_FLAG" --full-auto --model "$MODEL" "${EXTRA_ARGS[@]}" "$PROMPT_CONTENT"
