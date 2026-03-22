// gdn_op.cu
// PyTorch custom op wrapper for the SM_120 GDN state-update kernel.
// Registers: torch.ops.gdn_sm120.state_update
//
// Signature (Python):
//   S_out_data, S_out_scales = torch.ops.gdn_sm120.state_update(
//       S_data, S_scales, k, v, alpha, beta, d)
//
// Inputs:
//   S_data   : torch.Tensor uint8, shape [(d*d)//2]   — packed E2M1
//   S_scales : torch.Tensor uint8, shape [(d*d)//16]  — UE4M3 block scales
//   k        : torch.Tensor bfloat16, shape [d]
//   v        : torch.Tensor bfloat16, shape [d]
//   alpha    : float — decay gate
//   beta     : float — update scale
//   d        : int   — state dimension
//
// Outputs:
//   S_out_data   : torch.Tensor uint8, shape [(d*d)//2]
//   S_out_scales : torch.Tensor uint8, shape [(d*d)//16]

#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

// Forward declaration — implemented in gdn_state_update.cu
extern "C" int launch_gdn_state_update(
    const uint8_t* d_S_data,
    const uint8_t* d_S_scales,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    uint8_t* d_S_out_data,
    uint8_t* d_S_out_scales,
    float alpha, float beta, int d,
    cudaStream_t stream);

// ── op implementation ─────────────────────────────────────────────────────────

std::tuple<at::Tensor, at::Tensor> gdn_state_update_op(
    const at::Tensor& S_data,
    const at::Tensor& S_scales,
    const at::Tensor& k,
    const at::Tensor& v,
    double alpha,
    double beta,
    int64_t d)
{
    // ── input validation ──────────────────────────────────────────────────────
    TORCH_CHECK(S_data.is_cuda(),   "S_data must be a CUDA tensor");
    TORCH_CHECK(S_scales.is_cuda(), "S_scales must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(),        "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(),        "v must be a CUDA tensor");

    TORCH_CHECK(S_data.dtype()   == torch::kUInt8,    "S_data must be uint8");
    TORCH_CHECK(S_scales.dtype() == torch::kUInt8,    "S_scales must be uint8");
    TORCH_CHECK(k.dtype()        == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype()        == torch::kBFloat16, "v must be bfloat16");

    TORCH_CHECK(S_data.is_contiguous(),   "S_data must be contiguous");
    TORCH_CHECK(S_scales.is_contiguous(), "S_scales must be contiguous");
    TORCH_CHECK(k.is_contiguous(),        "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(),        "v must be contiguous");

    int64_t n = d * d;
    TORCH_CHECK(S_data.numel()   == n / 2,  "S_data size mismatch: expected ", n/2, " got ", S_data.numel());
    TORCH_CHECK(S_scales.numel() == n / 16, "S_scales size mismatch");
    TORCH_CHECK(k.numel() == d, "k size mismatch");
    TORCH_CHECK(v.numel() == d, "v size mismatch");
    TORCH_CHECK(n % 16 == 0, "d*d must be divisible by 16 for NVFP4 block scaling");

    // ── allocate outputs ──────────────────────────────────────────────────────
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(S_data.device());
    at::Tensor S_out_data   = torch::empty({n / 2},  opts);
    at::Tensor S_out_scales = torch::empty({n / 16}, opts);

    // ── launch kernel ─────────────────────────────────────────────────────────
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = launch_gdn_state_update(
        reinterpret_cast<const uint8_t*>(S_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(S_scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<uint8_t*>(S_out_data.data_ptr()),
        reinterpret_cast<uint8_t*>(S_out_scales.data_ptr()),
        static_cast<float>(alpha),
        static_cast<float>(beta),
        static_cast<int>(d),
        stream);

    TORCH_CHECK(rc == 0, "launch_gdn_state_update failed with code ", rc);

    return {S_out_data, S_out_scales};
}

// ── registration ──────────────────────────────────────────────────────────────

TORCH_LIBRARY(gdn_sm120, m) {
    m.def("state_update(Tensor S_data, Tensor S_scales, Tensor k, Tensor v, "
          "float alpha, float beta, int d) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gdn_sm120, CUDA, m) {
    m.impl("state_update", &gdn_state_update_op);
}
