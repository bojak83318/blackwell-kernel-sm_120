// gdn_state_update_ref.cu
// BF16 reference implementation of the GDN state update.
// Used as the correctness oracle in test_gdn_correctness.cu.
// Formula: S_t[i] = alpha * S_prev[i] + beta * (k[row] * v[col])
// All arithmetic in BF16 accumulated to FP32, output as FP32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

#include "gdn_fp4_intrinsics.cuh"

// ── NVFP4 helpers (shared with fused kernel tests) ───────────────────────────

// E2M1 4-bit float: sign(1) | exp(2) | mantissa(1)
// Packed two-per-byte, low nibble = element 0, high nibble = element 1.
// UE4M3 scale: unsigned, no bias, 8-bit, one per 16 elements.

__host__ __device__ inline float decode_e2m1_nibble(uint8_t nibble) {
    // nibble is 4 bits: [s e1 e0 m]
    uint8_t s = (nibble >> 3) & 1u;
    uint8_t e = (nibble >> 1) & 3u;
    uint8_t m = nibble & 1u;
    float mag;
    if (e == 0) {
        // subnormal: value = m * 2^{-1}  (implicit leading 0, exp bias = 1)
        mag = (float)m * 0.5f;
    } else {
        // normal: value = (1 + m*0.5) * 2^{e-1}
        mag = (1.0f + (float)m * 0.5f) * (float)(1u << (e - 1u));
    }
    return s ? -mag : mag;
}

__host__ __device__ inline float decode_ue4m3(uint8_t scale_byte) {
    return sm120::intrinsics::decode_ue4m3_to_f32(scale_byte);
}

__host__ __device__ inline uint8_t encode_f32_to_ue4m3(float v) {
    return sm120::intrinsics::encode_f32_to_ue4m3(v);
}

// ── Reference kernel ─────────────────────────────────────────────────────────

// S_prev_fp32: input state [d x d] in FP32 (already dequantised by host)
// k, v: [d] in BF16
// S_out_fp32: output state [d x d] in FP32
// alpha, beta: scalar gating factors
// d: state dimension
__global__ void gdn_state_update_ref_kernel(
    const float*   __restrict__ S_prev_fp32,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float*         __restrict__ S_out_fp32,
    float alpha, float beta, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // k dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // v dimension
    if (row >= d || col >= d) return;

    float k_fp32 = __bfloat162float(k[row]);
    float v_fp32 = __bfloat162float(v[col]);
    float s_prev = S_prev_fp32[row * d + col];

    S_out_fp32[row * d + col] = alpha * s_prev + beta * k_fp32 * v_fp32;
}

// ── Host-side helpers ─────────────────────────────────────────────────────────

// Dequantise a packed NVFP4 block into FP32.
// data_packed: (d*d)/2 bytes — two E2M1 nibbles per byte
// scales:      (d*d)/16 bytes — one UE4M3 per 16 elements
// out_fp32:    d*d floats
void nvfp4_dequantise_host(
    const uint8_t* data_packed,
    const uint8_t* scales,
    float*         out_fp32,
    int            num_elements)
{
    for (int i = 0; i < num_elements; i++) {
        int byte_idx  = i / 2;
        int nibble    = (i % 2 == 0) ? (data_packed[byte_idx] & 0x0Fu)
                                      : ((data_packed[byte_idx] >> 4) & 0x0Fu);
        float raw     = decode_e2m1_nibble((uint8_t)nibble);
        float scale   = decode_ue4m3(scales[i / 16]);
        out_fp32[i]   = raw * scale;
    }
}

// Quantise FP32 block into NVFP4 (modifies data_packed and scales in place).
// Block size = 16.
void nvfp4_quantise_host(
    const float*   in_fp32,
    uint8_t*       data_packed,   // (num_elements / 2) bytes
    uint8_t*       scales,        // (num_elements / 16) bytes
    int            num_elements)
{
    int num_blocks = num_elements / 16;
    for (int b = 0; b < num_blocks; b++) {
        const float* block = in_fp32 + b * 16;
        // find amax
        float amax = 0.0f;
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(block[i]));
        // encode scale
        float target_max = 6.0f;  // max representable E2M1 magnitude
        float local_scale = (amax > 0.0f) ? (amax / target_max) : 1.0f;
        uint8_t enc_scale = encode_f32_to_ue4m3(local_scale);
        scales[b] = enc_scale;
        float dec_scale = decode_ue4m3(enc_scale);
        float inv_scale = (dec_scale > 0.0f) ? (1.0f / dec_scale) : 0.0f;
        // encode elements in pairs
        for (int i = 0; i < 16; i += 2) {
            float v0 = block[i]     * inv_scale;
            float v1 = block[i + 1] * inv_scale;
            // clamp to [-6, 6] and quantise to E2M1
            auto clamp_e2m1 = [](float x) -> uint8_t {
                float ax = fabsf(x);
                uint8_t s = x < 0.0f ? 1u : 0u;
                // find closest E2M1 code
                // E2M1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
                static const float mags[8] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
                static const uint8_t codes[8] = {0,1,2,3,4,5,6,7};
                uint8_t best = 0;
                float best_err = fabsf(ax - mags[0]);
                for (int k = 1; k < 8; k++) {
                    float err = fabsf(ax - mags[k]);
                    if (err < best_err) { best_err = err; best = codes[k]; }
                }
                // remap: code 0-7 → nibble with sign
                // nibble = [s e1 e0 m]
                // code 0 → 0b0000=0, code 1 → 0b0001=1, etc. for positive
                uint8_t nibble = best | (s << 3);
                return nibble;
            };
            uint8_t n0 = clamp_e2m1(v0);
            uint8_t n1 = clamp_e2m1(v1);
            data_packed[(b * 16 + i) / 2] = (uint8_t)(n0 | (n1 << 4));
        }
    }
}

// Launch the BF16 reference kernel.
// S_prev_nvfp4_data, S_prev_nvfp4_scales: device pointers to packed NVFP4 state
// k_bf16, v_bf16: device pointers to BF16 vectors [d]
// S_out_fp32: device pointer to FP32 output [d*d]
extern "C" void launch_gdn_ref(
    const uint8_t* d_S_prev_data,    // device, (d*d)/2 bytes
    const uint8_t* d_S_prev_scales,  // device, (d*d)/16 bytes
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    float*         d_S_out,
    float alpha, float beta, int d,
    cudaStream_t   stream)
{
    // Step 1: dequantise S_prev on device via a simple element-wise kernel
    // (inline for simplicity — no separate kernel needed for correctness path)
    int n = d * d;

    // alloc temp FP32 S_prev on device
    float* d_S_prev_fp32 = nullptr;
    cudaMalloc(&d_S_prev_fp32, (size_t)n * sizeof(float));

    // dequantise kernel — one thread per element
    // launch dequant kernel (defined as static __global__ below)
    extern void launch_dequant_kernel(const uint8_t*, const uint8_t*, float*, int, cudaStream_t);
    launch_dequant_kernel(d_S_prev_data, d_S_prev_scales, d_S_prev_fp32, n, stream);

    // Step 2: run BF16 reference update
    dim3 block(16, 16);
    dim3 grid((d + 15) / 16, (d + 15) / 16);
    gdn_state_update_ref_kernel<<<grid, block, 0, stream>>>(
        d_S_prev_fp32, d_k, d_v, d_S_out, alpha, beta, d);

    cudaFree(d_S_prev_fp32);
}

static __global__ void dequant_kernel_impl(
    const uint8_t* data, const uint8_t* scales, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int byte_idx = i / 2;
    uint8_t byte = data[byte_idx];
    uint8_t nibble = (i % 2 == 0) ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
    float raw   = decode_e2m1_nibble(nibble);
    float scale = decode_ue4m3(scales[i / 16]);
    out[i] = raw * scale;
}

void launch_dequant_kernel(
    const uint8_t* data, const uint8_t* scales, float* out, int n, cudaStream_t s)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    dequant_kernel_impl<<<blocks, threads, 0, s>>>(data, scales, out, n);
}


// Compatibility wrapper used by test_gdn_correctness.cu.
// Produces NVFP4 output buffers from the FP32 BF16 reference path.
extern "C" int launch_gdn_state_update(
    const uint8_t* d_S_prev_data,
    const uint8_t* d_S_prev_scales,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    uint8_t* d_S_out_data,
    uint8_t* d_S_out_scales,
    float alpha,
    float beta,
    int d,
    cudaStream_t stream)
{
    if (!d_S_prev_data || !d_S_prev_scales || !d_k || !d_v ||
        !d_S_out_data || !d_S_out_scales || d <= 0) {
        return 1;
    }

    const int n = d * d;
    const size_t fp32_bytes  = static_cast<size_t>(n) / 1u * sizeof(float);
    const size_t data_bytes  = static_cast<size_t>(n) / 2u;
    const size_t scale_bytes = static_cast<size_t>(n) / 16u;

    float* d_ref_out = nullptr;
    if (cudaMalloc(&d_ref_out, fp32_bytes) != cudaSuccess) {
        return 2;
    }

    launch_gdn_ref(d_S_prev_data, d_S_prev_scales, d_k, d_v,
                   d_ref_out, alpha, beta, d, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_ref_out);
        return 3;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaFree(d_ref_out);
        return 4;
    }

    std::vector<float> h_ref_out(static_cast<size_t>(n));
    std::vector<uint8_t> h_out_data(data_bytes);
    std::vector<uint8_t> h_out_scales(scale_bytes);

    if (cudaMemcpy(h_ref_out.data(), d_ref_out, fp32_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_ref_out);
        return 5;
    }

    nvfp4_quantise_host(h_ref_out.data(), h_out_data.data(), h_out_scales.data(), n);

    if (cudaMemcpyAsync(d_S_out_data, h_out_data.data(), data_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_S_out_scales, h_out_scales.data(), scale_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFree(d_ref_out);
        return 6;
    }

    cudaFree(d_ref_out);
    return 0;
}
