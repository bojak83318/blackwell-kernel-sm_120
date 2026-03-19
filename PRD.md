Great question — TensorRT-LLM is a fundamentally different integration path for the GDN kernel vs. vLLM, with distinct tradeoffs for your 5090 k3s node.

***

## TRT-LLM vs vLLM for the GDN Kernel

The core architectural difference: vLLM runs **eager PyTorch ops** at inference time, making custom `torch.ops` registration relatively lightweight. TensorRT-LLM **compiles the full model graph into a `.engine` file ahead of time** — your GDN kernel must be registered as a TensorRT plugin *before* engine build, not at runtime  [docs.nvidia](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html).

***

## Current SM_120 State in TRT-LLM

TRT-LLM's SM_120 support has been a moving target  [zenn](https://zenn.dev/toki_mwc/articles/d2d02fe44c03f8?locale=en):

| Version | SM_120 Status |
|---|---|
| 0.19.0 | `FP4 Gemm not supported before Blackwell, nor GeForce Blackwell` — explicitly excluded  [github](https://github.com/NVIDIA/TensorRT-LLM/issues/5018) |
| 0.20.0 rc3 | SM_120 check added — NVFP4 dense GEMM enabled  [github](https://github.com/NVIDIA/TensorRT-LLM/issues/5018) |
| 1.0.0 rc2 | First officially verified SM_120 build (NGC Docker)  [zenn](https://zenn.dev/toki_mwc/articles/d2d02fe44c03f8?locale=en) |
| 1.2.0 rc8 | `TRTLLMGenFusedMoE does not support SM120` bug — MoE still broken  [forums.developer.nvidia](https://forums.developer.nvidia.com/t/bug-tensorrt-llm-1-2-0rc8-trtllmgenfusedmoe-does-not-support-sm120-error-on-dgx-spark-with-gpt-oss-120b-eagle3/357849/7) |
| Latest (1.3+) | GPT-OSS SM_120/SM_121 support added, MoE bug fixed  [github](https://github.com/NVIDIA/TensorRT-LLM/releases) |

Critically: **GDN/linear attention is not in TRT-LLM's validated model list at all**  [nvidia.github](https://nvidia.github.io/TensorRT-LLM/release-notes.html). Only transformer-standard models (Llama, GPT-OSS, Qwen dense) are validated on SM_120. Qwen 3.5's GDN layers would require a custom plugin regardless of version.

***

## How TRT-LLM Custom Plugins Work for GDN

To integrate your CUTLASS GDN kernel into TRT-LLM, you'd implement a **`IPluginV3` plugin**  [docs.nvidia](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html). The flow is:

```
Your gdn_kernel.cu (CUTLASS SM_120f)
        ↓
TensorRT IPluginV3 wrapper (C++)
  - enqueue(): calls your CUDA kernel
  - getOutputDataTypes(): returns NVFP4
  - configurePlugin(): validates SM_120 capability
        ↓
trtllm-build --checkpoint_dir /ckpt \
             --output_dir /engine \
             --plugin_lib libgdn_sm120_plugin.so   ← loaded at engine build
        ↓
Compiled .engine (GDN op fused into static graph)
```

The advantage over vLLM: once compiled, the `.engine` has **zero Python overhead** per token — op dispatch, memory planning, and kernel scheduling are all resolved at compile time  [docs.nvidia](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html). This is why TRT-LLM typically beats vLLM by 15–30% on throughput for production workloads.

***

## The Key Tradeoff for Your Use Case

| Dimension | vLLM path | TRT-LLM path |
|---|---|---|
| Integration effort | `torch.ops` registration (~50 LOC Python) | `IPluginV3` C++ wrapper + engine rebuild |
| Iteration speed | Hot-reload, no recompile | Full `trtllm-build` on every change (~10–30 min) |
| Runtime overhead | PyTorch dispatcher per op | Zero — static graph |
| GDN support today | BF16 fallback, patchable | **Not implemented at all**  [nvidia.github](https://nvidia.github.io/TensorRT-LLM/release-notes.html) |
| SM_120 stability | vLLM 0.17+ working | TRT-LLM 1.3+ — MoE still had bugs at 1.2  [forums.developer.nvidia](https://forums.developer.nvidia.com/t/bug-tensorrt-llm-1-2-0rc8-trtllmgenfusedmoe-does-not-support-sm120-error-on-dgx-spark-with-gpt-oss-120b-eagle3/357849/7) |
| k3s deployment | Any container | Requires NGC Triton server base image  [github](https://github.com/NVIDIA/TensorRT-LLM/issues/5018) |
| Long-context (262K) | PagedAttention handles it | `max_seq_len` must be set at engine build — no dynamic context |

The last row is a **hard constraint for Qwen 3.5**: TRT-LLM's `max_seq_len` is baked into the `.engine` at build time  [github](https://github.com/NVIDIA/TensorRT-LLM/issues/5018). To test 8k vs 64k vs 262k you'd need three separate engine builds. vLLM's PagedAttention handles this dynamically with a single serve command.

***

## Recommended Path: vLLM First, TRT-LLM Plugin Later

For the PRD on your 5090 k3s node, the practical sequencing is:

1. **Phase 1–3 (as per PRD):** Build and validate the CUTLASS kernel, register as `torch.ops`, test via vLLM — fast iteration, dynamic context, no recompile loop  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/155001363/04714d6b-d1c1-4ef6-a6f2-014caadab770/CUTLASS-SM_120-GDN-Kernel-Scaffolding.pdf)
2. **Phase 4 (post-validation):** Wrap the validated kernel in a TRT-LLM `IPluginV3` plugin for production deployment — you get the static-graph throughput advantage only *after* correctness is proven

```bash
# When ready for TRT-LLM, the engine build command would look like:
trtllm-build \
  --checkpoint_dir /ckpt/qwen35-14b-nvfp4 \
  --output_dir /engine \
  --plugin_lib /workspace/libgdn_sm120_plugin.so \
  --gemm_plugin disable \          # let your custom plugin own GDN layers
  --max_seq_len 65536 \            # must pick ONE context length at build time
  --use_paged_context_fmha enable
```

The standalone FP4 GEMM library that hit 129 TFLOPS on DGX Spark  [forums.developer.nvidia](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600) is also framework-independent — it can serve as a validation reference for your kernel's throughput ceiling before committing to either the vLLM or TRT-LLM integration path.
