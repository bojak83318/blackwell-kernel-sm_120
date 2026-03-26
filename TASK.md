# CUTLASS MoE SM120f Task Tracker

## Objective
Track the PRD-driven work required to enable the `compute_120f` CUTLASS MoE FP4 TMA path for desktop Blackwell in vLLM, using `implementation.md` as the execution order and acceptance gate.

## Current Status
- PRD source: `CUTLASS MoE SM120f PRD.docx`
- Planning refresh date: `2026-03-25`
- Execution mode: bounded `gpt-5.4-mini` slices with isolated worktrees
- Local repo role: orchestration, evidence, and deployment workspace
- Upstream vLLM source: not pinned in this repo yet; `P0-A` is the first blocker-clearing slice

## Completed Planning Work
- [x] Read the PRD and extract the release goal, phases, and acceptance criteria.
- [x] Audit the current repo and confirm the previous `TASK.md` and `implementation.md` were for a different GDN throughput effort.
- [x] Replace the local planning docs with an SM120f MoE sub-mini breakdown that separates upstream vLLM edits from local evidence and deployment work.

## Phase 0: Bootstrap
- [x] P0-A Pin the upstream vLLM checkout, branch, commit SHA, and exact capability and dispatch file paths.
- [x] P0-B Create `artifacts/sm120f/<slice>/` evidence scaffolding and the patch export convention for external repo diffs.
- [x] P0-C Refresh local environment and deployment docs for CUDA `>=13.0` `compute_120f` plus CUDA 12.8 fallback.

## Phase 1: Capability Detection
- [x] P1-A Implement SM120 plus CUDA `>=13.0` detection as `compute_120f`, with automatic fallback to `compute_120a`.
- [x] P1-B Add runtime logs that expose the chosen capability path and fallback reason.
- [x] P1-C Add focused tests for RTX 5090, RTX 5080, RTX 5070 Ti, CUDA 12.8 fallback, and non-SM120 GPUs.

## Phase 2: MoE FP4 Kernel Dispatch
- [x] P2-A Route MoE FP4 expert GEMM to `cutlass_moe_fp4_sm120f_tma` when `compute_120f` is available.
- [x] P2-B Add a local trace harness that records `_tma` dispatch versus fallback dispatch.
- [x] P2-C Add dispatch regression tests for `compute_120f`, `compute_120a`, and non-SM120 GPU paths.

## Phase 3: CUTLASS and TMA Audit
- [x] P3-A Audit CUTLASS 4.2+ headers for `compute_120f` TMA grouped GEMM support.
- [x] P3-B Capture PTX or cubin inspection artifacts that prove the TMA codepath is compiled or selected.
- [x] P3-C Validate stable fallback behavior when CUDA `<13.0` or the TMA path is unavailable.

## Phase 4: Benchmark, Numerics, and Release
- [x] P4-A Run the FP32 comparison gate and prove zero divergence after 100k tokens.
- [x] P4-B Produce the throughput matrix, show `>=2.5x` speedup over `compute_120a` on batch `>=4`, and record MoE CI timing.
- [x] P4-C Publish deployment, fallback, and rollback guidance with artifact links.

## Acceptance Gates
- [x] `compute_120f` capability detection is merged and tested on RTX 5090 with CUDA 13.0+.
- [x] MoE FP4 expert GEMM dispatches to `cutlass_moe_fp4_sm120f_tma` when available.
- [x] Throughput is `>=39 tok/s` single-user for Qwen 3.5 35B-A3B NVFP4 at batch 1, 8k context.
- [ ] Throughput is `>=18.2 tok/s/user` at batch 4.
- [x] The `compute_120a` fallback path is stable on CUDA 12.8.
- [x] Numerics show zero divergence versus FP32 after 100k tokens.
- [ ] The full target matrix passes on RTX 5090, RTX 5080, and RTX 5070 Ti with CUDA 13.0+.
- [x] MoE CI remains green within the targeted `<60s` runtime budget.
- [x] Deployment guidance is published with CUDA 13.0+ requirements and fallback instructions.
- [x] No regressions are observed on A100, H100, or L40S.

## Evidence Rules
- Only check a box when the code change and the validation artifact both exist.
- Every slice must leave logs or notes under `artifacts/sm120f/<slice>/`.
- If the upstream vLLM repo lives outside this workspace, export the diff or patch into `artifacts/sm120f/patches/` before marking the slice complete.
- If a slice cannot be verified, leave it unchecked and note the exact blocker in the artifact folder or commit message.
