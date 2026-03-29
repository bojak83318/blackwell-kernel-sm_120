# PRD

**SM120 Qwen3.5 Recovery: Re-establish 8K Baseline, Preserve GDN Tracing, Prove Hot-Path Dispatch**

| Field | Value |
|---|---|
| Status | In Progress (Blocked On Matrix Hardware Availability) |
| Owner | rocm |
| Date | 2026-03-29 |
| Target System | `ubuntu@192.168.0.233` - RTX 5090 (SM120), 32 GiB VRAM |
| Runtime Family | `vllm-sm120` on CUDA 13.1 / Driver 590.48.01 |
| Current Debug Advantage | GDN tracing build can confirm extension load and supports kernel-level probe injection |
| Primary Decision Goal | Separate runtime stabilization from GDN dispatch validation and restore a reproducible path back to 8K serving |

## Background

The project had already demonstrated that Qwen3.5-35B-A3B NVFP4 can run on the
RTX 5090 at useful 8K throughput with FP8 KV cache, chunked prefill, and a
stable SM120 path. That earlier work is directionally correct and remains the
reference point for production intent.

The current build is better in one specific way: it has direct GDN tracing.
The runtime now exposes `gdn_state_update_ext` load visibility and supports a
kernel-level probe path. That is better than the earlier build for debugging,
because it gives direct evidence about whether the custom extension exists and
whether the kernel can be instrumented.

The current build is worse in a more important way: it is no longer a controlled
baseline. Runtime stabilization changes, GDN tracing changes, MoE/Triton bypass
changes, memory-shape experiments, and broad `setup.py develop` rebuilds were
all mixed into one dirty remote tree. This makes every failure ambiguous.

## Verified Facts

- The remote node working tree is dirty and includes local runtime patches plus
  untracked GDN-related files. The current runtime cannot be treated as an
  immutable benchmark reference.
- The `SparseMatrix` Triton import failure was bypassed on SM120 by patching
  `gpt_oss_triton_kernels_moe.py`.
- After that patch, the runtime still reported
  `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`, so the fix removed one startup
  blocker but did not prove that the full MoE path was running through Marlin.
- The GDN extension loads successfully. The server log contains
  `gdn_state_update_ext CUDA extension loaded.`
- A minimal eager server at `max-model-len=128`, `max-num-seqs=1`,
  `max-num-batched-tokens=128`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.98` can start and serve requests.
- A single `/v1/completions` request succeeds on that minimal server.
- The injected probe string did not appear in the server log. Current measured
  dispatch count is zero.
- An offline 8K throughput run on the patched runtime completed at
  `3098.53 total tok/s` and `442.65 output tok/s` after clearing the GPU.
- An 8K serving benchmark on the patched runtime completed with
  `3871.45 total tok/s`, `553.06 output tok/s`, and zero failed requests.

## Problem Statement

The project no longer has a clean answer to the core question:

> Is the custom GDN kernel actually on the hot path of Qwen3.5 serving on SM120?

The current build answers weaker questions only:

- Can the extension load? Yes.
- Can the runtime serve at small context? Yes.
- Can the runtime produce useful 8K throughput on the patched branch? Yes.
- Can the current probe prove kernel dispatch? No.

This ambiguity exists because the current build combined three separate tracks:

1. Runtime survival on SM120
2. Throughput benchmarking at 8K
3. GDN hot-path dispatch proof

Those tracks require different validation shapes, different acceptable flags,
and different failure criteria. Treating them as one loop created false
negatives and wasted iterations.

## Where The Current Build Went Wrong

### 1. No Immutable Source of Truth

The remote runtime was benchmarked from a dirty working tree rather than a
frozen commit or tagged patchset. This broke apples-to-apples comparison and
made stash-based A/B runs the only option.

### 2. Extension Load Was Treated As Dispatch Proof

`gdn_state_update_ext` loading is necessary but not sufficient. The active model
path can still bypass the custom update function entirely. The current probe
result proves that this distinction matters.

### 3. Runtime Survival And GDN Validation Were Coupled

The Triton/Marlin bypass, memory-shape tuning, and GDN probe loop were run
together. As a result, startup failures, CLI incompatibilities, and probe
failures all looked like one problem.

### 4. Proof-of-Life Was Attempted On Invalid Or Unsupported Paths

At different points the loop used unsupported commands or flags in this build,
including `generate` and `--disable-log-requests`. Those failures said nothing
about the GDN kernel itself.

### 5. Rebuild Scope Was Too Large

`setup.py develop` rebuilt the full extension stack, including flash-attention
and unrelated CUDA targets, when the intent was to validate one GDN probe. That
made every probe iteration expensive and noisy.

### 6. Memory Validation Started Too High

The initial proof path used shapes that still failed at KV cache allocation.
Only after reducing to a minimal eager server at context 128 was it possible to
run a real request. That should have been the first dispatch-proof step.

## Objective

Restore a reproducible development flow with two explicit branches of work:

- A clean 8K runtime baseline for production benchmarking
- A GDN-tracing debug branch for proving hot-path dispatch

The immediate goal is not to maximize throughput. The immediate goal is to prove
or disprove that the traced GDN extension is actually exercised by inference,
without losing the known-good 8K serving path.

## Goals

- Re-establish a clean, reproducible 8K baseline branch with pinned runtime
  behavior and no probe-only edits.
- Preserve the current GDN tracing additions on a separate debug branch.
- Prove hot-path dispatch with a minimal inference path before rerunning 8K
  serving benchmarks on the tracing branch.
- Separate MoE/Triton stabilization from GDN kernel validation.
- Replace ambiguous "load succeeded" evidence with direct dispatch evidence.

## Non-Goals

- Retuning all Qwen3.5 serving parameters
- Replacing FlashInfer or CUTLASS backends wholesale
- k3s rollout changes in this phase
- Full TRT-LLM migration
- Throughput optimization on the debug/probe branch until dispatch is proven

## Scope

### In Scope

- Branch separation between production-like baseline and GDN-tracing debug path
- SM120 Triton/Marlin guard handling required only for runtime survival
- Minimal eager serving shape for dispatch proof
- GDN probe injection and dispatch-path tracing
- Reproducible benchmark commands and artifact logging

### Out of Scope

- Training or requantizing model weights
- Multi-GPU inference
- Full DeerFlow integration changes
- Runtime-independent claims about world-record throughput

## Requirements

### Functional Requirements

- `FR-1`: The project must maintain one clean 8K baseline branch with no
  tracing-only edits.
- `FR-2`: The project must maintain one GDN-tracing branch that contains the
  extension-load visibility and dispatch probe instrumentation.
- `FR-3`: The tracing branch must prove hot-path dispatch by producing at least
  one `GDN KERNEL DISPATCHED` event for a real completion request.
- `FR-4`: The minimal proof server must use a memory shape that is known to
  start on the 5090 node before any larger-context probe attempt.
- `FR-5`: Every benchmark result must be tied to an explicit branch or patchset,
  not an anonymous dirty tree.
- `FR-6`: The runtime stabilization patch for SM120 Triton import failure must
  be separable from GDN dispatch logic.
- `FR-7`: The project must identify the exact Python or C++ call site that is
  expected to invoke `py_gdn_state_update_bf16`.
- `FR-8`: The project must state clearly whether the active path uses the custom
  GDN kernel, a BF16 fallback, a Triton recurrent path, or another bypass.

### Non-Functional Requirements

- `NFR-1`: Reproducibility first. Every proof or benchmark must be rerunnable
  from a repo-owned command sequence.
- `NFR-2`: Minimal proof path must be preferred over large-context serving when
  validating dispatch.
- `NFR-3`: Failures must be classified as one of: startup, memory, CLI shape,
  extension-load, or dispatch-path failure.
- `NFR-4`: The tracing build must remain available because it is strictly better
  for diagnosis than the earlier non-tracing build.

## Technical Approach

The work will proceed on two explicitly different paths.

### Path A: Clean Baseline

Freeze a production-like branch at the last known-good 8K runtime behavior.
This branch exists only to answer:

- Does the node still sustain 8K serving?
- What is the current throughput without tracing overhead?
- Are MoE/Triton startup blockers fully contained?

No probe instrumentation is allowed on this branch.

### Path B: GDN Tracing

Keep the current tracing advantages:

- extension load visibility
- custom probe injection
- runtime logging around GDN initialization

This branch exists only to answer:

- Is the custom kernel on the actual serving path?
- If not, where is the bypass?

This branch should use the smallest serving shape that can reliably start. The
current known-good proof shape is the 128-token eager server.

### Dispatch-Proof Rule

The only accepted proof that the custom kernel is active is one of:

- `GDN KERNEL DISPATCHED` from the probe
- a stronger structured counter added directly at the call site

Extension load lines alone are not proof.

## Milestones

### Milestone 1: Freeze Branches

Objective: restore reproducibility.

Deliverables:

- one clean 8K baseline branch
- one tracing branch
- exact branch notes describing which files differ and why

Exit criteria:

- both branches can be checked out and described without ambiguity

### Milestone 2: Minimal Dispatch Proof

Objective: prove or disprove hot-path invocation on the smallest viable server.

Deliverables:

- minimal eager server command
- one successful completion request
- dispatch-count artifact
- extension-load artifact

Exit criteria:

- either dispatch count is greater than zero, or the bypass point is identified

### Milestone 3: Call-Site Isolation

Objective: if dispatch count remains zero, identify the active path instead of
repeating the same probe loop.

Deliverables:

- confirmed entry point from model code to recurrent update logic
- explicit answer whether `gdn_attention_core`, mixed-precision manager, or
  another path is active
- patch plan if the traced function is not wired into active inference

Exit criteria:

- the project knows exactly why the probe does or does not fire

### Milestone 4: Rejoin 8K Serving

Objective: once dispatch is proven or the bypass is fixed, rerun 8K serving on
 the tracing branch and compare with the clean baseline.

Deliverables:

- offline throughput benchmark
- serve benchmark
- measured regression or parity statement

Exit criteria:

- tracing overhead is measured and documented

## Acceptance Criteria

- A clean baseline branch exists and can run the known 8K benchmark commands.
- A tracing branch exists and can boot the minimal eager server.
- At least one real completion request succeeds on the tracing branch.
- The project has a definitive answer to whether `py_gdn_state_update_bf16`
  executes during inference.
- If the answer is no, the exact bypass location is documented and assigned to a
  follow-up patch.
- No benchmark claim is reported without the corresponding branch or patch
  identity.

## Rollout And Exit Decision

Continue with the tracing branch if:

- dispatch is proven, or
- the bypass location is identified and can be patched cleanly

Revise the approach if:

- the tracing branch still cannot prove dispatch after the active call site is
  inspected directly

Stop investing in the current tracing integration path if:

- the active Qwen3.5 recurrent path does not call the traced function at all and
  requires a materially different integration design than currently assumed

## Open Questions

- Which exact code path is currently performing recurrent state update during the
  successful 128-token completion request?
- Is `py_gdn_state_update_bf16` supposed to be called from the mixed-precision
  manager, from `gdn_attention_core`, or from another wrapper?
- Does the SM120 Triton bypass patch need to remain as a permanent compatibility
  guard, or only as a temporary debug workaround?
- When will RTX 5080 and RTX 5070 Ti hardware become available so the remaining
  full-matrix acceptance gate can be executed and closed?
