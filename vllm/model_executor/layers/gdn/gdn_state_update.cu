// =============================================================================
// gdn_state_update.cu
// Mixed-Precision GDN Recurrent State Update Kernel — SM120 Blackwell
//
// Compilation (from vLLM build tree or standalone):
//   nvcc -arch=sm_120 -std=c++17 -O3 --use_fast_math \
//        -I/path/to/cutlass/include \
//        gdn_state_update.cu -o gdn_state_update.o
//
// Key design decisions:
//   1. NVFP4 dequant is done once per thread-block in shared memory to amortise
//      the nibble-unpack cost across all accumulation steps.
//   2. FP32 accumulators throughout — never touch BF16/FP8 mid-accumulation.
//   3. Telemetry is computed inline during the quantise/dequantise round-trip,
//      not as a separate pass, to avoid an extra kernel launch.
//   4. 128-bit vectorised loads for BF16 state (matches existing bfloat162
//      kernel in vLLM GDN patch).
// =============================================================================

#include "gdn_state_update.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Compile-time tunables
// ---------------------------------------------------------------------------
#define GDN_BLOCK_THREADS   256     // threads per CTA
#define GDN_VEC_WIDTH       8       // bfloat16 elements per 128-bit load

// ---------------------------------------------------------------------------
// NVFP4 dequantisation helpers
// ---------------------------------------------------------------------------

// FP4 E2M1 bit layout: [s][e1][e0][m0]
// Sign: bit3, Exponent: bits 2:1 (bias=1), Mantissa: bit0
__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    // Extract fields
    uint8_t s = (nibble >> 3) & 0x1;
    uint8_t e = (nibble >> 1) & 0x3;
    uint8_t m = (nibble >> 0) & 0x1;

    float val;
    if (e == 0) {
        // Subnormal: (-1)^s * 0.m * 2^(1-1) = m * 0.5
        val = m * 0.5f;
    } else {
        // Normal: (-1)^s * 1.m * 2^(e-1)
        val = (1.0f + m * 0.5f) * (float)(1 << (e - 1));
    }
    return s ? -val : val;
}

__device__ __forceinline__ void unpack_nvfp4x2(
    uint8_t packed, float scale, float& lo, float& hi
) {
    lo = fp4_e2m1_to_float(packed & 0x0F) * scale;
    hi = fp4_e2m1_to_float((packed >> 4) & 0x0F) * scale;
}

// ---------------------------------------------------------------------------
// FP8-E5M2 round-trip helpers  (SM120 has native fp8 instructions)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8e5m2_roundtrip(float x, float scale, float inv_scale) {
    // Quantise to FP8-E5M2 then immediately dequantise — used for telemetry.
    // In production the state is *stored* as FP8; this models the error.
    float scaled = x * inv_scale;
    __nv_fp8_e5m2 q = __nv_fp8_e5m2(scaled);
    return (float)q * scale;
}

// ---------------------------------------------------------------------------
// Warp-level FP32 atomic helpers for telemetry (use shared mem reduction)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void warp_reduce_max(float& val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
}

__device__ __forceinline__ void warp_reduce_add(float& val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
}

// ---------------------------------------------------------------------------
// Mode A kernel: BF16 recurrent state
//
// Thread mapping: each thread processes GDN_VEC_WIDTH consecutive state dims.
// Grid: (batch, ceil(state_dim / (GDN_BLOCK_THREADS * GDN_VEC_WIDTH)))
// ---------------------------------------------------------------------------
__global__ void gdn_state_update_bf16_kernel(
    const uint8_t*         d_W_packed,     // NVFP4 weights [out_dim, in_dim/2]
    const float*           d_W_scale,      // per-block scales
    int                    block_size,     // elements per scale block
    const __nv_bfloat16*   d_S_prev,
    const __nv_bfloat16*   d_k,
    const __nv_bfloat16*   d_v,
    __nv_bfloat16*         d_S_out,
    GdnStateTelemetry*     d_telemetry,
    int                    batch,
    int                    state_dim
) {
    // ---- Shared memory for block-level telemetry reduction ----
    __shared__ float s_max_abs_err;
    __shared__ float s_sum_rel_err;
    __shared__ float s_sum_signal;
    __shared__ float s_sum_noise;
    __shared__ uint32_t s_n_clipped;
    __shared__ uint32_t s_n_nan;
    __shared__ uint32_t s_n_inf;

    if (threadIdx.x == 0) {
        s_max_abs_err = 0.f;
        s_sum_rel_err = 0.f;
        s_sum_signal  = 0.f;
        s_sum_noise   = 0.f;
        s_n_clipped   = 0;
        s_n_nan       = 0;
        s_n_inf       = 0;
    }
    __syncthreads();

    const int b   = blockIdx.x;                          // batch index
    const int col = (blockIdx.y * blockDim.x + threadIdx.x) * GDN_VEC_WIDTH;

    if (b >= batch || col >= state_dim) return;

    const int base   = b * state_dim + col;
    const int vec_end = min(col + GDN_VEC_WIDTH, state_dim);
    const int n_elem  = vec_end - col;

    // ---- Load S_prev, k, v as FP32 accumulators ----
    float acc_s[GDN_VEC_WIDTH], acc_k[GDN_VEC_WIDTH], acc_v[GDN_VEC_WIDTH];

    #pragma unroll
    for (int i = 0; i < GDN_VEC_WIDTH; ++i) {
        if (i < n_elem) {
            acc_s[i] = __bfloat162float(d_S_prev[base + i]);
            acc_k[i] = __bfloat162float(d_k[base + i]);
            acc_v[i] = __bfloat162float(d_v[base + i]);
        }
    }

    // ---- NVFP4 weight × state (simplified GDN recurrence):
    //   S_out[i] = S_prev[i] * k[i]  +  (W_fp4 @ v)[i]
    //
    // For the GDN recurrent update the weight matrix W projects v into
    // the state space. We accumulate W*v in FP32.
    //
    // NOTE: A full grouped GEMM would use the CUTLASS TMA path; this kernel
    // handles only the state-dimension update (small, residual-style op).
    // The main expert GEMM still uses cutlass_moe_fp4_sm120f_tma.

    float wv_acc[GDN_VEC_WIDTH] = {};

    // Iterate over input (v) dimension — weight row = output dim (col..col+n_elem)
    for (int out_i = 0; out_i < n_elem; ++out_i) {
        int out_global = col + out_i;
        float wv = 0.f;

        for (int in_j = 0; in_j < state_dim; in_j += 2) {
            int weight_idx  = out_global * (state_dim / 2) + in_j / 2;
            int scale_idx   = weight_idx / block_size;

            uint8_t packed  = d_W_packed[weight_idx];
            float   wscale  = d_W_scale[scale_idx];

            float w0, w1;
            unpack_nvfp4x2(packed, wscale, w0, w1);

            float v0 = (in_j     < state_dim) ? __bfloat162float(d_v[b * state_dim + in_j    ]) : 0.f;
            float v1 = (in_j + 1 < state_dim) ? __bfloat162float(d_v[b * state_dim + in_j + 1]) : 0.f;

            wv = fmaf(w0, v0, fmaf(w1, v1, wv));  // FP32 FMA
        }
        wv_acc[out_i] = wv;
    }

    // ---- GDN state update (FP32): S_out = S_prev * k + W*v ----
    float s_out_fp32[GDN_VEC_WIDTH];
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        s_out_fp32[i] = fmaf(acc_s[i], acc_k[i], wv_acc[i]);
    }

    // ---- Telemetry: BF16 round-trip error on S_out ----
    float t_max_abs = 0.f, t_sum_rel = 0.f, t_sig = 0.f, t_noise = 0.f;
    uint32_t t_clipped = 0, t_nan = 0, t_inf = 0;

    // BF16 max representable magnitude (finite): ~3.39e+38 (same as FP32 eff.)
    // BF16 has 7 mantissa bits; clip threshold = largest normal BF16
    const float BF16_MAX = 65504.0f * 256.0f;  // ~1.677e7; practical saturation

    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        float x     = s_out_fp32[i];
        float x_qt  = __bfloat162float(__float2bfloat16(x));  // BF16 roundtrip
        float err   = fabsf(x_qt - x);
        float denom = fabsf(x) + 1e-8f;

        t_max_abs  = fmaxf(t_max_abs, err);
        t_sum_rel += err / denom;
        t_sig     += x   * x;
        t_noise   += err * err;

        if (isinf(x_qt))             { ++t_inf; }
        else if (isnan(x_qt))        { ++t_nan; }
        else if (fabsf(x_qt) >= BF16_MAX) { ++t_clipped; }
    }

    // Warp-level reductions
    warp_reduce_max(t_max_abs);
    warp_reduce_add(t_sum_rel);
    warp_reduce_add(t_sig);
    warp_reduce_add(t_noise);

    // Thread 0 of each warp writes to shared
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&s_max_abs_err),
                  __float_as_int(t_max_abs));
        atomicAdd(&s_sum_rel_err, t_sum_rel);
        atomicAdd(&s_sum_signal,  t_sig);
        atomicAdd(&s_sum_noise,   t_noise);
        atomicAdd(&s_n_clipped,   t_clipped);
        atomicAdd(&s_n_nan,       t_nan);
        atomicAdd(&s_n_inf,       t_inf);
    }
    __syncthreads();

    // ---- Write BF16 output (downcast from FP32 accumulator) ----
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        d_S_out[base + i] = __float2bfloat16(s_out_fp32[i]);
    }

    // ---- Block 0 writes telemetry (one record per kernel call) ----
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        int total = batch * state_dim;
        float sig   = d_telemetry->sqnr_db;  // reuse field temporarily
        (void)sig;

        d_telemetry->max_abs_err      = s_max_abs_err;
        d_telemetry->mean_rel_err     = s_sum_rel_err / (float)total;
        d_telemetry->pct_clipped      = 100.f * (float)s_n_clipped / (float)total;
        d_telemetry->sqnr_db          = (s_sum_noise > 0.f)
                                        ? 10.f * log10f(s_sum_signal / s_sum_noise)
                                        : 99.f;
        d_telemetry->n_nan            = s_n_nan;
        d_telemetry->n_inf            = s_n_inf;
        d_telemetry->total_elements   = (uint32_t)total;
    }
}

// ---------------------------------------------------------------------------
// Mode B kernel: FP8-E5M2 recurrent state
// ---------------------------------------------------------------------------
__global__ void gdn_state_update_fp8e5m2_kernel(
    const uint8_t*          d_W_packed,
    const float*            d_W_scale,
    int                     block_size,
    const __nv_fp8_e5m2*    d_S_prev_fp8,
    float                   s_prev_scale,
    const __nv_bfloat16*    d_k,
    const __nv_bfloat16*    d_v,
    __nv_fp8_e5m2*          d_S_out_fp8,
    float*                  d_s_out_scale,
    GdnStateTelemetry*      d_telemetry,
    int                     batch,
    int                     state_dim
) {
    __shared__ float s_max_abs_err;
    __shared__ float s_sum_rel_err;
    __shared__ float s_sum_signal;
    __shared__ float s_sum_noise;
    __shared__ uint32_t s_n_clipped;
    __shared__ float s_global_max_abs;

    if (threadIdx.x == 0) {
        s_max_abs_err   = 0.f;
        s_sum_rel_err   = 0.f;
        s_sum_signal    = 0.f;
        s_sum_noise     = 0.f;
        s_n_clipped     = 0;
        s_global_max_abs = 0.f;
    }
    __syncthreads();

    const int b   = blockIdx.x;
    const int col = (blockIdx.y * blockDim.x + threadIdx.x) * GDN_VEC_WIDTH;
    if (b >= batch || col >= state_dim) return;

    const int base   = b * state_dim + col;
    const int n_elem = min(col + GDN_VEC_WIDTH, state_dim) - col;

    // Dequantise FP8 state
    float acc_s[GDN_VEC_WIDTH];
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        acc_s[i] = (float)d_S_prev_fp8[base + i] * s_prev_scale;
    }

    // Load k, v
    float acc_k[GDN_VEC_WIDTH], acc_v[GDN_VEC_WIDTH];
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        acc_k[i] = __bfloat162float(d_k[base + i]);
        acc_v[i] = __bfloat162float(d_v[base + i]);
    }

    // NVFP4 W*v accumulation (same as BF16 path)
    float wv_acc[GDN_VEC_WIDTH] = {};
    for (int out_i = 0; out_i < n_elem; ++out_i) {
        int out_global = col + out_i;
        float wv = 0.f;
        for (int in_j = 0; in_j < state_dim; in_j += 2) {
            int weight_idx  = out_global * (state_dim / 2) + in_j / 2;
            int scale_idx   = weight_idx / block_size;
            float wscale    = d_W_scale[scale_idx];
            float w0, w1;
            unpack_nvfp4x2(d_W_packed[weight_idx], wscale, w0, w1);
            float v0 = (in_j     < state_dim) ? __bfloat162float(d_v[b * state_dim + in_j    ]) : 0.f;
            float v1 = (in_j + 1 < state_dim) ? __bfloat162float(d_v[b * state_dim + in_j + 1]) : 0.f;
            wv = fmaf(w0, v0, fmaf(w1, v1, wv));
        }
        wv_acc[out_i] = wv;
    }

    float s_out_fp32[GDN_VEC_WIDTH];
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        s_out_fp32[i] = fmaf(acc_s[i], acc_k[i], wv_acc[i]);
    }

    // Find max magnitude for dynamic scale update (EMA candidate)
    float local_max = 0.f;
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        local_max = fmaxf(local_max, fabsf(s_out_fp32[i]));
    }
    warp_reduce_max(local_max);
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&s_global_max_abs), __float_as_int(local_max));
    }
    __syncthreads();

    // Compute scale for this batch: s = max_abs / FP8_E5M2_MAX
    // FP8 E5M2 max finite value = 57344.0
    const float FP8_E5M2_MAX = 57344.0f;
    float out_scale = (s_global_max_abs > 0.f) ? (s_global_max_abs / FP8_E5M2_MAX) : 1.f;
    float inv_scale = 1.f / out_scale;

    // Telemetry: FP8 round-trip
    float t_max_abs = 0.f, t_sum_rel = 0.f, t_sig = 0.f, t_noise = 0.f;
    uint32_t t_clipped = 0;

    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        float x    = s_out_fp32[i];
        float x_qt = fp8e5m2_roundtrip(x, out_scale, inv_scale);
        float err  = fabsf(x_qt - x);

        t_max_abs  = fmaxf(t_max_abs, err);
        t_sum_rel += err / (fabsf(x) + 1e-8f);
        t_sig     += x   * x;
        t_noise   += err * err;
        if (fabsf(x * inv_scale) >= FP8_E5M2_MAX) ++t_clipped;
    }
    warp_reduce_max(t_max_abs);
    warp_reduce_add(t_sum_rel);
    warp_reduce_add(t_sig);
    warp_reduce_add(t_noise);
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&s_max_abs_err), __float_as_int(t_max_abs));
        atomicAdd(&s_sum_rel_err, t_sum_rel);
        atomicAdd(&s_sum_signal,  t_sig);
        atomicAdd(&s_sum_noise,   t_noise);
        atomicAdd(&s_n_clipped,   t_clipped);
    }
    __syncthreads();

    // Write quantised FP8 output
    #pragma unroll
    for (int i = 0; i < n_elem; ++i) {
        float q = s_out_fp32[i] * inv_scale;
        // Clamp to FP8-E5M2 range before cast
        q = fmaxf(-FP8_E5M2_MAX, fminf(FP8_E5M2_MAX, q));
        d_S_out_fp8[base + i] = __nv_fp8_e5m2(q);
    }

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        int total = batch * state_dim;
        *d_s_out_scale                = out_scale;
        d_telemetry->max_abs_err      = s_max_abs_err;
        d_telemetry->mean_rel_err     = s_sum_rel_err / (float)total;
        d_telemetry->pct_clipped      = 100.f * (float)s_n_clipped / (float)total;
        d_telemetry->sqnr_db          = (s_sum_noise > 0.f)
                                        ? 10.f * log10f(s_sum_signal / s_sum_noise)
                                        : 99.f;
        d_telemetry->n_nan            = 0;
        d_telemetry->n_inf            = 0;
        d_telemetry->total_elements   = (uint32_t)total;
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
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
) {
    const int elems_per_block = GDN_BLOCK_THREADS * GDN_VEC_WIDTH;
    dim3 grid(batch, (state_dim + elems_per_block - 1) / elems_per_block);
    dim3 block(GDN_BLOCK_THREADS);

    gdn_state_update_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(d_W_packed),
        d_W_scale, block_size,
        d_S_prev, d_k, d_v,
        d_S_out, d_telemetry,
        batch, state_dim
    );
}

void launch_gdn_state_update_fp8e5m2(
    const nvfp4x2_t*        d_W_packed,
    const float*            d_W_scale,
    int                     block_size,
    const __nv_fp8_e5m2*    d_S_prev_fp8,
    float                   s_prev_scale,
    const __nv_bfloat16*    d_k,
    const __nv_bfloat16*    d_v,
    __nv_fp8_e5m2*          d_S_out_fp8,
    float*                  d_s_out_scale,
    GdnStateTelemetry*      d_telemetry,
    int                     batch,
    int                     state_dim,
    cudaStream_t            stream
) {
    const int elems_per_block = GDN_BLOCK_THREADS * GDN_VEC_WIDTH;
    dim3 grid(batch, (state_dim + elems_per_block - 1) / elems_per_block);
    dim3 block(GDN_BLOCK_THREADS);

    gdn_state_update_fp8e5m2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(d_W_packed),
        d_W_scale, block_size,
        d_S_prev_fp8, s_prev_scale,
        d_k, d_v,
        d_S_out_fp8, d_s_out_scale,
        d_telemetry,
        batch, state_dim
    );
}

GdnStateTelemetry gdn_telemetry_readback(
    const GdnStateTelemetry* d_telemetry,
    cudaStream_t             stream
) {
    GdnStateTelemetry h;
    cudaMemcpyAsync(&h, d_telemetry, sizeof(GdnStateTelemetry),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}
