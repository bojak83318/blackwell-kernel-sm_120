# Implementation Plan: GDN Throughput Optimization

## Purpose
Achieve sustained inference throughput of 133 tok/s on a single NVIDIA RTX 5090 (SM_120 Blackwell, 32 GB GDDR7) running Qwen3.5-35B-A3B with the validated Mixed-Precision GDN architecture. This document governs throughput engineering only; the architecture is frozen.

## Core Scope & Guardrails
- **Model Target:** Qwen3.5-35B-A3B on RTX 5090, CUDA 12.9 (symlinked to 12.8 for flashinfer), vLLM 0.18.0.
- **Precision Lock:** NVFP4 weights + BF16 recurrent GDN state.
- **SQNR Gate:** Must maintain SQNR ≥ 50 dB on BF16 recurrent state after every kernel change.
- **Coherence Gate:** Output must be coherent with no repetition loops or language switches at any batch size.
- **Memory Envelope:** 32 GB. OOM during CUDA Graph capture is a hard blocker.
- **Framework Lock:** Do not upgrade vLLM 0.18.0 or remove the CUDA 12.8 symlink.

## Engineering Phases

### Phase 1: CUDA Graph Capture
**Goal:** Remove `--enforce-eager` to eliminate CPU overhead and capture the static decode graph.
- **Target Throughput:** 25–45 tok/s.
- **Validation Gate:** No graph break on startup. 5-run benchmark must show variance < 10% across runs and average ≥ 25 tok/s.
- **Write Scope:** Server launch script configuration, and potentially vLLM C++ patch if GDN state resetting is needed between requests.

### Phase 2: Batching
**Goal:** Increase `max-num-seqs` to 4-8 and enable `chunked-prefill` to amortize weight load costs.
- **Target Throughput:** 50–80 tok/s.
- **Validation Gate:** No OOM during launch or concurrent requests. All 4 outputs must be coherent and distinct. Average throughput ≥ 50 tok/s.
- **Write Scope:** Launch configuration and verification scripts.

### Phase 3: Context Expansion
**Goal:** Expand `max-model-len` to 2048 to reduce KV cache fragmentation and fully utilize chunked prefill.
- **Target Throughput:** 70–95 tok/s.
- **Validation Gate:** Average throughput ≥ 65 tok/s. 256-token output remains coherent. SQNR ≥ 50 dB.
- **Write Scope:** Launch configuration, long-prompt verification scripts.

### Phase 4: Kernel Fusion
**Goal:** Profile the current state using `nsys`, then implement a fused GDN state update and flash attention kernel if HBM round-trips are the bottleneck.
- **Target Throughput:** 100–133 tok/s.
- **Validation Gate:** Bottleneck identified with >5% contribution via `nsys`. After fusion: throughput 100–133 tok/s, SQNR ≥ 50 dB, and output coherence.
- **Write Scope:** `gdn_state_update` CUDA kernel (using `__nv_bfloat162`), vLLM 0.18.0 C++ integration patch, SQNR validation script.

### Phase 5: Production Baseline Lock
**Goal:** Perform a formal 5-run production benchmark to document variance, coherence, and exact launch flags.
- **Target Throughput:** 133 tok/s sustained.
- **Validation Gate:** Average ≥ 133 tok/s AND variance < 5%.
- **Write Scope:** Final documentation and production configuration scripts.

## Sub-Mini Execution Rules
- Use git worktrees for each unit of work.
- Only mark items in `TASK.md` as complete when code and validation artifacts exist.
- Verify SQNR and coherence locally in the worktree before integration.
- Stop and evaluate if memory leaks (VRAM growth) or SQNR regressions (< 50 dB) are detected.
