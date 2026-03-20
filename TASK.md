# SM_120 GDN Kernel Task Tracker

## Objective
Implement and validate an SM_120-targeted GDN kernel for Qwen 3.5 style linear-attention layers, integrate it through a vLLM-first `torch.ops` path, and only then add an optional TensorRT-LLM plugin path.

## Current Status
- State: Planning complete, implementation not started.
- Execution mode: Prepare for small bounded `sub-mini` implementation passes.
- Primary path: vLLM first for correctness and fast iteration.
- Secondary path: TensorRT-LLM plugin only after the vLLM path is validated.

## Guardrails
- Do not mark items complete from documentation alone.
- Do not start the TRT-LLM plugin path until the vLLM path has passing correctness and benchmark evidence.
- Keep long-context validation dynamic in vLLM; treat TRT-LLM context lengths as build-time variants.
- Record benchmark numbers and validation evidence for every completed phase.

## T0 Bootstrap And Shared Layout
- [x] Confirm the source layout under `blackwell-kernel/sm_120/` and create the initial build/test skeleton.
- [x] Define the CUDA/CUTLASS/PyTorch/vLLM/TensorRT-LLM version matrix required for SM_120 work.
- [x] Add shared build configuration for CUDA extension builds and standalone kernel benchmarking.
- [x] Add a minimal test harness and evidence directory for benchmark and correctness outputs.
- [x] Document environment assumptions plus local and remote command prerequisites (docs/p0-c-ops.md).

## T1 CUTLASS Kernel Baseline
- [ ] Add the initial GDN kernel scaffold targeting SM_120.
- [x] Add host-side launch code and shape/config validation.
- [ ] Add a reference implementation path for correctness comparison.
- [ ] Add focused unit tests for kernel numerics, shape handling, and failure cases.
- [ ] Add a standalone microbenchmark and capture the first baseline throughput numbers.

## T2 PyTorch And vLLM Integration
- [ ] Expose the kernel through a PyTorch extension and `torch.ops` registration.
- [ ] Add Python wrappers that select the custom op or the reference fallback safely.
- [ ] Integrate the custom op into the intended vLLM GDN execution path.
- [ ] Add dynamic-context validation for representative sequence lengths without engine rebuilds.
- [ ] Add smoke tests proving the fallback path remains usable when the custom kernel is unavailable.

## T3 End-To-End Validation On Target Stack
- [ ] Add an environment or container recipe for the 5090/k3s validation path.
- [ ] Run model-level inference smoke tests that exercise the GDN path.
- [ ] Capture correctness evidence against the reference path at representative sizes.
- [ ] Capture performance evidence for short, medium, and long context settings.
- [ ] Document known limits, unsupported shapes, and operational constraints.

## T4 TensorRT-LLM Plugin Path
- [ ] Add a TensorRT `IPluginV3` wrapper around the validated kernel.
- [ ] Add plugin capability checks for SM_120 and supported dtypes/layouts.
- [ ] Add engine-build integration using `trtllm-build --plugin_lib`.
- [ ] Validate one fixed `max_seq_len` engine build end to end.
- [ ] Benchmark the plugin path against the vLLM path and record the tradeoffs.

## T5 CI, Docs, And Acceptance
- [ ] Add reproducible build and test commands for local and CI execution.
- [ ] Add docs for the vLLM-first development path and the TRT-LLM production path.
- [ ] Add an acceptance checklist tied to correctness, performance, and deployment evidence.
- [ ] Verify that all completion claims in this file are backed by code or recorded artifacts.
