// =============================================================================
// gdn_state_update.cuh
// Mixed-Precision GDN Recurrent State Update Kernel for SM120 Blackwell
//
// Architecture:
//   - Static weights:    NVFP4 (E2M1)  — fed into existing TMA GEMM path
//   - Recurrent state:   BF16 (Mode A) or FP8-E5M2 (Mode B)
//   - Accumulators:      FP32 always   — downcast to state dtype on write-back
//
// Target: RTX 5090, SM_120, CUDA 13.0+, CUTLASS 4.2+
// =============================================================================

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// NVFP4 (E2M1) packed type: two FP4 values per byte
// ---------------------------------------------------------------------------
struct alignas(1) nvfp4x2_t {
    uint8_t x;  // lo nibble = first element, hi nibble = second element
};

// ---------------------------------------------------------------------------
// Telemetry record — written to device buffer, read back by host
// ---------------------------------------------------------------------------
struct GdnStateTelemetry {
    float max_abs_err;          // max |dequant(quant(x)) - x|
    float mean_rel_err;         // mean |err| / (|x| + eps)
    float pct_clipped;          // % elements hitting ±max representable
    float sqnr_db;              // 10*log10(signal_power / noise_power)
    uint32_t n_nan;             // elements that became NaN post-roundtrip
    uint32_t n_inf;             // elements that became ±Inf post-roundtrip
    uint32_t total_elements;    // for normalisation sanity-check
    uint32_t pad;
};

// ---------------------------------------------------------------------------
// Kernel interface — Mode A: BF16 state
//
//   d_W_packed   : NVFP4 weight matrix, shape [out_dim, in_dim/2], packed
//   d_W_scale    : per-block FP32 scale for NVFP4 dequantisation
//   block_size   : elements per scale block (typically 16 or 32)
//   d_S_prev     : BF16 previous recurrent state,  [batch, state_dim]
//   d_k          : BF16 key projection,            [batch, state_dim]
//   d_v          : BF16 value projection,           [batch, state_dim]
//   d_S_out      : BF16 output recurrent state,    [batch, state_dim]
//   d_telemetry  : device-side telemetry scratch (1 record per call)
//   batch        : batch size
//   state_dim    : state dimension
//   stream       : CUDA stream
// ---------------------------------------------------------------------------
void launch_gdn_state_update_bf16(
    const nvfp4x2_t*       d_W_packed,
    const float*           d_W_scale,
    int                    block_size,
    const __nv_bfloat16*   d_S_prev,
    const __nv_bfloat16*   d_k,
    const __nv_bfloat16*   d_v,
    __nv_bfloat16*         d_S_out,
    GdnStateTelemetry*     d_telemetry,
    int                    batch,
    int                    state_dim,
    cudaStream_t           stream
);

// ---------------------------------------------------------------------------
// Kernel interface — Mode B: FP8-E5M2 state
//
//   d_S_prev_fp8 / d_S_out_fp8 hold __nv_fp8_e5m2 elements.
//   k and v are still BF16 (projection outputs from attention path).
//   Scale factors are per-tensor; update them via EMA from host.
// ---------------------------------------------------------------------------
void launch_gdn_state_update_fp8e5m2(
    const nvfp4x2_t*        d_W_packed,
    const float*            d_W_scale,
    int                     block_size,
    const __nv_fp8_e5m2*    d_S_prev_fp8,
    float                   s_prev_scale,     // dequant: val * s_prev_scale
    const __nv_bfloat16*    d_k,
    const __nv_bfloat16*    d_v,
    __nv_fp8_e5m2*          d_S_out_fp8,
    float*                  d_s_out_scale,    // updated scale written here
    GdnStateTelemetry*      d_telemetry,
    int                     batch,
    int                     state_dim,
    cudaStream_t            stream
);

// ---------------------------------------------------------------------------
// Utility: host-side telemetry readback (synchronous)
// ---------------------------------------------------------------------------
GdnStateTelemetry gdn_telemetry_readback(
    const GdnStateTelemetry* d_telemetry,
    cudaStream_t             stream
);
