# Implementation Plan: CUTLASS MoE SM120f Enablement

## Source and Goal
- Source PRD: `CUTLASS MoE SM120f PRD.docx`, read on 2026-03-25.
- Goal: enable the `compute_120f` TMA-aware CUTLASS MoE FP4 path for desktop Blackwell in vLLM 0.18.0+, with the PRD release target of vLLM 0.19.0.
- This repository is the orchestration, evidence, and deployment workspace. The upstream vLLM source tree is a separate write target that must be pinned before code slices start.

## Success Gates
- Throughput: `>=39 tok/s` single-user on RTX 5090 for Qwen 3.5 35B-A3B NVFP4 at batch 1, 8k context.
- Throughput: `>=18.2 tok/s/user` at batch 4, with at least `2.5x` speedup over the `compute_120a` path on batch `>=4`.
- Numerics: zero divergence versus FP32 reference after 100k tokens.
- Compatibility: stable fallback to `compute_120a` on CUDA 12.8 and no regressions on non-SM120 GPUs.
- Release hygiene: MoE CI stays green with the targeted test budget `<60s`, and a deployment guide is published.

## Guardrails
- Do not mark any engineering slice complete without code, logs, or artifacts.
- `compute_120f` requires host CUDA `>=13.0`; CUDA 12.8 must keep the fallback path unchanged.
- No slice may claim completion if the upstream vLLM checkout path and commit are not recorded.
- Keep upstream vLLM edits isolated from this repo's existing GDN files unless a slice explicitly owns them.
- Use one git worktree per slice and keep write scopes disjoint across parallel agents.

## Execution Model
- Delegated model: `gpt-5.4-mini`
- Delegated reasoning: `medium`
- Parallelism: `2-4` agents maximum when write scopes do not overlap
- Local repo root: `/home/rocm/blackwell-kernel/sm_120`
- Recommended worktree root: `/home/rocm/workspace/blackwell-kernel-worktrees/sm120f`
- Upstream repo variable: `VLLM_SM120F_ROOT`
- Artifact root: `artifacts/sm120f/<slice>/`
- Patch export root if upstream repo is external: `artifacts/sm120f/patches/`
- Integration order: `P0 -> P1 -> P2 -> P3 -> P4`

## Repo Reality
- The current tracked `implementation.md` and `TASK.md` were for a different GDN throughput effort and are superseded by this plan.
- This repo contains local build, docs, and validation assets such as `docs/environment.md`, `docs/vllm_registration.md`, `scripts/sm120_remote_preflight.sh`, `scripts/sm120_run_runtime_smoke.sh`, and `test/test_vllm_integration.py`.
- This repo does not contain the upstream vLLM source tree, so any slice that edits vLLM core files is blocked on `P0-A`.

## Phase Plan

### Phase 0: Bootstrap the Source of Truth

#### P0-A: Pin the upstream vLLM checkout and exact target files
- Goal: establish the external repo, branch, commit SHA, and the exact file paths for capability detection and fused MoE dispatch.
- Primary write scope: repo notes plus `artifacts/sm120f/p0-a/`.
- Deliverables:
  - recorded `VLLM_SM120F_ROOT`
  - pinned upstream branch and commit SHA
  - exact path for the capability helper
  - exact path for the CUTLASS MoE selector or dispatcher
  - gap list between the current repo and the PRD assumptions
- Verification:
  - `git -C "$VLLM_SM120F_ROOT" rev-parse HEAD`
  - `python -m pip show vllm torch cutlass`
  - `nvcc --version`
- Parallelism: none; this blocks all upstream code slices.

#### P0-B: Create local evidence and patch scaffolding
- Goal: make every later slice leave reproducible artifacts.
- Primary write scope: `artifacts/sm120f/`, local helper scripts, and validation notes in this repo.
- Deliverables:
  - artifact directory convention by slice
  - patch export convention for external repo diffs
  - benchmark and trace log naming convention
  - one-command placeholders for smoke, trace, and benchmark capture
- Verification:
  - dry-run scripts create the expected artifact paths
  - artifact README or notes document the naming contract
- Parallelism: can run with `P0-C`.

#### P0-C: Refresh local environment and deployment assumptions
- Goal: remove stale CUDA 12.8 or vLLM 0.12 assumptions from the local docs for this effort.
- Primary write scope: local docs such as `docs/environment.md`, `docs/runtime-validation.md`, `docs/vllm_registration.md`, and `setup-sglang-vllm-venv.md`.
- Deliverables:
  - explicit split between `compute_120f` on CUDA `>=13.0` and fallback `compute_120a` on CUDA 12.8
  - pinned CUTLASS 4.2+ expectation
  - test matrix prerequisites for RTX 5090, 5080, and 5070 Ti
- Verification:
  - doc review against the PRD
  - no conflicting version statements remain in the refreshed local docs
- Parallelism: can run with `P0-B`.

### Phase 1: Capability Detection

#### P1-A: Implement `compute_120f` detection
- Goal: detect SM120 plus CUDA `>=13.0` and expose that as the `compute_120f` path.
- Primary write scope: the capability helper module in `VLLM_SM120F_ROOT`, resolved by `P0-A`.
- Deliverables:
  - capability helper returns `compute_120f` for SM120 on CUDA `>=13.0`
  - helper falls back to `compute_120a` for CUDA `<13.0`
- Verification:
  - focused unit test or scripted probe across mocked capability and CUDA-version inputs
- Parallelism: start serially; this unlocks `P1-B` and `P1-C`.

#### P1-B: Add runtime trace for resolved capability
- Goal: surface the chosen capability path in logs so benchmarking can prove what executed.
- Primary write scope: upstream logging or worker module plus local artifact capture notes.
- Deliverables:
  - DEBUG or INFO trace line that includes the chosen capability and fallback reason
  - sample log capture for a CUDA 13.x path and a CUDA 12.8 fallback path
- Verification:
  - logs show `compute_120f` on the target path
  - logs show `compute_120a` fallback on the older host path
- Parallelism: can run with `P1-C` after `P1-A`.

#### P1-C: Add capability tests
- Goal: lock the detection behavior down before dispatch work starts.
- Primary write scope: upstream test files only.
- Deliverables:
  - tests for RTX 5090, RTX 5080, and RTX 5070 Ti on CUDA 13.0+
  - tests for CUDA 12.8 fallback
  - test for non-SM120 GPUs to prove no regression
- Verification:
  - focused `pytest` run for the new capability tests
- Parallelism: can run with `P1-B` after `P1-A`.

### Phase 2: MoE FP4 Kernel Dispatch

#### P2-A: Route MoE FP4 expert GEMM to the TMA kernel
- Goal: select `cutlass_moe_fp4_sm120f_tma` whenever `compute_120f` is available.
- Primary write scope: upstream fused MoE selector or dispatcher files only.
- Deliverables:
  - selector or enum path for the TMA-aware grouped GEMM kernel
  - preserved fallback to the existing `sm120a` path
- Verification:
  - focused unit test or minimal runtime check shows `_tma` is selected for `compute_120f`
- Parallelism: start serially; this unlocks `P2-B` and `P2-C`.

#### P2-B: Add a local trace harness for dispatch proof
- Goal: make kernel-selection proof reproducible outside ad hoc shell history.
- Primary write scope: local serve or validation scripts and `artifacts/sm120f/p2-b/`.
- Deliverables:
  - one-command trace harness that records which MoE kernel path was selected
  - artifacts for both TMA and fallback runs
- Verification:
  - trace logs contain the `_tma` kernel on CUDA 13.x
  - trace logs contain the fallback kernel on CUDA 12.8
- Parallelism: can run with `P2-C` after `P2-A`.

#### P2-C: Add dispatch regression tests
- Goal: prove the new selector does not regress other GPU paths.
- Primary write scope: upstream MoE test files only.
- Deliverables:
  - tests for `compute_120f`
  - tests for `compute_120a`
  - tests covering non-SM120 GPUs
- Verification:
  - focused `pytest` run for the dispatch tests
- Parallelism: can run with `P2-B` after `P2-A`.

### Phase 3: CUTLASS and TMA Audit

#### P3-A: Audit CUTLASS headers for the `compute_120f` TMA path
- Goal: confirm the installed CUTLASS version actually exposes the required TMA grouped GEMM pieces.
- Primary write scope: local audit notes and `artifacts/sm120f/p3-a/`.
- Deliverables:
  - CUTLASS version record
  - header references for `compute_120f`, TMA descriptors, and grouped GEMM templates
  - explicit pass or fail note for header presence
- Verification:
  - captured grep or inspection output stored in artifacts
- Parallelism: serial start for the audit phase.

#### P3-B: Prove the TMA codepath in PTX or cubin artifacts
- Goal: show that the compiled path is real, not just selected in Python.
- Primary write scope: local inspection script(s) and `artifacts/sm120f/p3-b/`.
- Deliverables:
  - repeatable PTX or cubin inspection command
  - saved output proving the `sm120f` or TMA codepath is present
- Verification:
  - inspection artifact saved and linked from the phase notes
- Parallelism: can run after `P3-A`.

#### P3-C: Prove fallback stability when TMA prerequisites are absent
- Goal: verify graceful fallback when CUDA `<13.0` or the CUTLASS TMA path is not available.
- Primary write scope: local validation notes and `artifacts/sm120f/p3-c/`.
- Deliverables:
  - fallback trace log
  - stability note for the CUDA 12.8 path
  - failure-mode note if headers or kernels are missing
- Verification:
  - stable fallback run recorded with artifacts
- Parallelism: can run after `P3-A`; keep it separate from `P3-B`.

### Phase 4: Benchmark, Numerics, and Release Evidence

#### P4-A: Build the numerical validation gate
- Goal: enforce the PRD requirement of zero divergence versus FP32 after 100k tokens.
- Primary write scope: local validation scripts, local notes, and any upstream test harness additions required for the check.
- Deliverables:
  - reproducible FP32 comparison harness
  - saved diff report in `artifacts/sm120f/p4-a/`
- Verification:
  - numerical report shows zero divergence across the required run
- Parallelism: serial start for the release phase.

#### P4-B: Run the throughput matrix and CI timing gate
- Goal: prove the performance claims and keep test cost bounded.
- Primary write scope: local benchmark scripts, parsing helpers, CI notes, and `artifacts/sm120f/p4-b/`.
- Deliverables:
  - throughput matrix across the PRD GPU, batch, and context combinations
  - evidence of `>=2.5x` speedup over `compute_120a` on batch `>=4`
  - CI timing record for the MoE tests
- Verification:
  - saved benchmark outputs and summary table
  - CI timing output under the target budget
- Parallelism: can run in parallel with parts of `P4-C` only after the benchmark schema is fixed.

#### P4-C: Publish deployment and rollback guidance
- Goal: close the loop for operators and reviewers.
- Primary write scope: local deployment docs and final acceptance notes.
- Deliverables:
  - deployment guide with CUDA 13.0+ requirement
  - fallback instructions for CUDA 12.8
  - operator checklist for tracing the selected kernel
  - no-regression note for non-SM120 GPUs
- Verification:
  - every acceptance item links to a concrete artifact or test result
- Parallelism: starts after `P4-A`; may overlap with the report assembly portion of `P4-B`.

## Deterministic Integration Order
1. `p0-a`
2. `p0-b`
3. `p0-c`
4. `p1-a`
5. `p1-b`
6. `p1-c`
7. `p2-a`
8. `p2-b`
9. `p2-c`
10. `p3-a`
11. `p3-b`
12. `p3-c`
13. `p4-a`
14. `p4-b`
15. `p4-c`

## Definition of Done
- `compute_120f` capability detection is merged, tested, and traced on RTX 5090 with CUDA 13.0+.
- MoE FP4 expert GEMM selects `cutlass_moe_fp4_sm120f_tma` when available.
- The required throughput targets are met and backed by saved benchmark artifacts.
- The CUDA 12.8 fallback remains stable.
- Numerics show zero divergence versus FP32 after 100k tokens.
- CI remains green and within the targeted MoE runtime budget.
- Deployment, fallback, and rollback instructions are published in this repo.

## Operator Note
- Existing helper scripts in this repo were written for an older GDN milestone layout. Until they are refreshed for this plan, use the slice IDs above as the source of truth and prefer `next`-style sub-mini runs over stale hard-coded wave names.
