# Implementation Plan: SM_120 GDN Kernel

## Purpose
This file is the execution plan for `blackwell-kernel/sm_120`. It is written to support narrow `sub-mini` implementation passes. `TASK.md` is the factual progress tracker; this file defines phase order, write scopes, and validation gates.

## Working Decisions From The PRD
- Use a vLLM-first integration path for the first implementation passes.
- Treat TensorRT-LLM as a later plugin phase, not the initial integration target.
- Optimize for fast correctness iteration first, then production throughput.
- Treat long-context support as a vLLM validation requirement and a TRT-LLM build-time tradeoff.

## Current Repo State
- `blackwell-kernel/sm_120/` currently contains only `PRD.md`.
- The initial implementation pass must create the project skeleton before kernel work can begin.

## Proposed Project Layout
All paths below are planned targets and do not imply the files already exist.

```text
blackwell-kernel/sm_120/
  PRD.md
  TASK.md
  implementation.md
  CMakeLists.txt
  include/
  src/
    kernel/
    runtime/
    pytorch/
    trtllm/
  python/
  tests/
    unit/
    integration/
  benchmarks/
  docker/
  scripts/
  docs/
  artifacts/
```

## Execution Rules
- Keep each delegated pass bounded to one module or write scope.
- Do not mark any task complete until code or evidence exists locally.
- Prefer verification immediately after each slice instead of large unverified batches.
- Keep fallback paths explicit so correctness can be compared at every stage.

## Phase Order

### Phase 0: Bootstrap And Shared Config
Goal: create the skeleton required for independent kernel, integration, and test work.

Deliverables:
- top-level build config for the `sm_120` project area
- initial directory layout
- shared test and benchmark entry points
- environment/version notes for SM_120 prerequisites

Suggested write scopes:
- Slice P0-A: root build files and shared config
- Slice P0-B: test harness and benchmark harness skeleton
- Slice P0-C: docs for environment assumptions and local commands

Gate to exit phase:
- project skeleton exists
- at least one no-op or placeholder build/test command runs successfully

### Phase 1: CUTLASS Kernel Baseline
Goal: stand up the custom GDN kernel and prove numerical correctness before framework integration.

Deliverables:
- kernel source targeting SM_120
- host launcher and parameter validation
- reference implementation for result comparison
- unit tests for supported shapes and failure handling
- first microbenchmark output

Suggested write scopes:
- Slice P1-A: kernel scaffold in `src/kernel/`
- Slice P1-B: launcher/runtime helpers in `src/runtime/`
- Slice P1-C: correctness tests in `tests/unit/`
- Slice P1-D: benchmark harness in `benchmarks/`

Gate to exit phase:
- kernel builds
- correctness tests pass against the reference path
- baseline benchmark artifacts are recorded

### Phase 2: PyTorch Extension And vLLM Path
Goal: expose the validated kernel through `torch.ops` and wire it into the intended vLLM execution path.

Deliverables:
- PyTorch extension build
- `torch.ops` registration
- Python wrapper that handles custom-op and fallback selection
- vLLM integration hook or patch
- smoke tests for dynamic sequence lengths

Suggested write scopes:
- Slice P2-A: C++/PyTorch binding files in `src/pytorch/`
- Slice P2-B: Python wrapper and loading logic in `python/`
- Slice P2-C: vLLM integration glue in `python/` or `scripts/`
- Slice P2-D: integration tests in `tests/integration/`

Gate to exit phase:
- `torch.ops` path loads cleanly
- fallback path still works when the extension is absent
- vLLM smoke tests pass for representative sequence lengths

### Phase 3: End-To-End Validation On The 5090/k3s Stack
Goal: prove the custom path works in the intended deployment environment and produce evidence.

Deliverables:
- reproducible environment setup or container definition
- model-level smoke test invoking the GDN path
- correctness comparisons at representative sizes
- throughput and latency measurements across context lengths
- documented limits and known issues

Suggested write scopes:
- Slice P3-A: container or environment config in `docker/`
- Slice P3-B: deployment or run scripts in `scripts/`
- Slice P3-C: end-to-end tests and evidence capture in `tests/integration/` and `artifacts/`
- Slice P3-D: operational notes in `docs/`

Gate to exit phase:
- target-stack smoke test passes
- evidence exists for correctness and performance
- known limitations are documented

### Phase 4: TensorRT-LLM Plugin Path
Goal: wrap the validated kernel in a TensorRT plugin only after the vLLM path is stable.

Deliverables:
- `IPluginV3` implementation
- plugin build integration
- capability checks for SM_120 and supported formats
- one successful engine build for a fixed `max_seq_len`
- performance comparison with the vLLM path

Suggested write scopes:
- Slice P4-A: plugin implementation in `src/trtllm/`
- Slice P4-B: build and packaging changes
- Slice P4-C: engine build scripts in `scripts/`
- Slice P4-D: plugin validation tests and benchmark capture

Gate to exit phase:
- plugin library builds
- `trtllm-build --plugin_lib` succeeds for one chosen context length
- comparison artifacts exist for vLLM vs TRT-LLM

### Phase 5: CI And Acceptance
Goal: make the work repeatable and ready for handoff.

Deliverables:
- repeatable local and CI commands
- documentation for both integration paths
- acceptance checklist tied to evidence

Suggested write scopes:
- Slice P5-A: CI workflow or scripted validation entry points
- Slice P5-B: docs cleanup and usage guides
- Slice P5-C: acceptance evidence collation

Gate to exit phase:
- required commands are documented and runnable
- acceptance checklist is complete and backed by artifacts

## Verification Strategy
- Unit verification: kernel numerics, shape validation, error handling.
- Integration verification: extension load, `torch.ops` call path, fallback behavior.
- System verification: vLLM smoke tests on representative context lengths.
- Production-path verification: one fixed-length TRT-LLM engine build after vLLM success.
- Evidence discipline: store benchmark outputs, test logs, and comparison artifacts under `artifacts/`.

## Delegation Guidance For `sub-mini`
Use one delegated worker per write scope. Good first passes are:
1. P0-A: bootstrap build files and directory skeleton.
2. P0-B: test and benchmark harness skeleton.
3. P1-A: kernel scaffold only.
4. P1-B: launcher/runtime helpers only.
5. P1-C: unit tests only.

Avoid delegating these before a local decision is made:
- supported tensor layouts and dtype contract
- fallback semantics for unsupported shapes
- the exact vLLM integration seam
- the fixed `max_seq_len` chosen for the first TRT-LLM engine build

## Stop Conditions
Stop and reassess if any of these happen:
- CUTLASS or CUDA toolchain cannot target SM_120 as assumed.
- PyTorch extension loading fails for reasons unrelated to the kernel itself.
- vLLM does not expose a clean integration seam for the GDN path.
- TRT-LLM plugin requirements force a redesign of the data layout or dtype contract.
