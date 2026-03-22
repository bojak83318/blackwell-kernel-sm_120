// test_gdn_correctness.cu
// M3 correctness gate: GDN NVFP4 fused kernel vs BF16 reference.
// Pass criterion (FR-6): max element-wise relative error < 5% for d=2048.
// Also tests d=512 and d=1024.
//
// Exit codes: 0 = ALL PASS, 1 = any FAIL

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// ── host-side helpers (compiled from ref file via shared header) ──────────────

// Forward declarations from gdn_state_update_ref.cu
void nvfp4_quantise_host(const float*, uint8_t*, uint8_t*, int);
void nvfp4_dequantise_host(const uint8_t*, const uint8_t*, float*, int);

// Forward declaration of ref launcher
extern "C" void launch_gdn_ref(
    const uint8_t*, const uint8_t*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    float*, float, float, int, cudaStream_t);

// Forward declaration of fused kernel launcher (from gdn_state_update.cu)
// Signature must match what gdn_state_update.cu exports.
extern "C" int launch_gdn_state_update(
    const uint8_t*, const uint8_t*,
    const __nv_bfloat16*, const __nv_bfloat16*,
    uint8_t*, uint8_t*,
    float, float, int, cudaStream_t);

// ── utilities ─────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void fill_random_fp32(std::vector<float>& v, float lo, float hi) {
    for (auto& x : v) {
        x = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
    }
}

static void fp32_to_bf16(const std::vector<float>& src,
                          std::vector<__nv_bfloat16>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = __float2bfloat16(src[i]);
}

// Compare fused kernel FP32 output (after dequant) vs ref FP32 output.
// Returns max relative error and prints mismatches.
static float compare_outputs(
    const float* ref,  // [d*d] FP32
    const float* fused_dequant, // [d*d] FP32 (dequantised from NVFP4 output)
    int d, const char* tag,
    float rel_tol)
{
    int n = d * d;
    float max_rel_err = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < n; i++) {
        float denom = fabsf(ref[i]);
        if (denom < 1e-6f) denom = 1e-6f;
        float rel = fabsf(ref[i] - fused_dequant[i]) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > rel_tol && mismatch_count < 5) {
            fprintf(stderr, "  [%s] MISMATCH i=%d ref=%.6f fused=%.6f rel=%.4f\n",
                    tag, i, ref[i], fused_dequant[i], rel);
            mismatch_count++;
        }
    }
    return max_rel_err;
}

// ── single test run ───────────────────────────────────────────────────────────

static bool run_test(int d, float alpha, float beta, float rel_tol) {
    printf("  d=%-5d alpha=%.2f beta=%.2f  ", d, alpha, beta);
    fflush(stdout);

    int n = d * d;
    size_t data_bytes  = (size_t)n / 2;      // packed E2M1
    size_t scale_bytes = (size_t)n / 16;     // UE4M3 scales
    size_t fp32_bytes  = (size_t)n * sizeof(float);
    size_t bf16_bytes  = (size_t)d * sizeof(__nv_bfloat16);

    // ── host buffers ──────────────────────────────────────────────────────────
    std::vector<float> h_S_prev_fp32(n), h_k_fp32(d), h_v_fp32(d);
    fill_random_fp32(h_S_prev_fp32, -4.0f, 4.0f);
    fill_random_fp32(h_k_fp32, -1.0f, 1.0f);
    fill_random_fp32(h_v_fp32, -1.0f, 1.0f);

    std::vector<__nv_bfloat16> h_k_bf16, h_v_bf16;
    fp32_to_bf16(h_k_fp32, h_k_bf16);
    fp32_to_bf16(h_v_fp32, h_v_bf16);

    // Quantise S_prev to NVFP4
    std::vector<uint8_t> h_S_data(data_bytes), h_S_scales(scale_bytes);
    nvfp4_quantise_host(h_S_prev_fp32.data(),
                        h_S_data.data(), h_S_scales.data(), n);

    // ── device buffers ────────────────────────────────────────────────────────
    uint8_t *d_S_data = nullptr, *d_S_scales = nullptr;
    uint8_t *d_S_out_data = nullptr, *d_S_out_scales = nullptr;
    __nv_bfloat16 *d_k = nullptr, *d_v = nullptr;
    float *d_ref_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_S_data,       data_bytes));
    CUDA_CHECK(cudaMalloc(&d_S_scales,     scale_bytes));
    CUDA_CHECK(cudaMalloc(&d_S_out_data,   data_bytes));
    CUDA_CHECK(cudaMalloc(&d_S_out_scales, scale_bytes));
    CUDA_CHECK(cudaMalloc(&d_k,            bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_v,            bf16_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref_out,      fp32_bytes));

    CUDA_CHECK(cudaMemcpy(d_S_data,   h_S_data.data(),   data_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S_scales, h_S_scales.data(), scale_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── run BF16 reference ────────────────────────────────────────────────────
    launch_gdn_ref(d_S_data, d_S_scales, d_k, d_v,
                   d_ref_out, alpha, beta, d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_ref_out(n);
    CUDA_CHECK(cudaMemcpy(h_ref_out.data(), d_ref_out, fp32_bytes, cudaMemcpyDeviceToHost));

    // ── run fused NVFP4 kernel ────────────────────────────────────────────────
    int fused_rc = launch_gdn_state_update(
        d_S_data, d_S_scales, d_k, d_v,
        d_S_out_data, d_S_out_scales,
        alpha, beta, d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (fused_rc != 0) {
        printf("FAIL (fused kernel returned %d)\n", fused_rc);
        goto cleanup;
    }

    {
        // dequantise fused output on host
        std::vector<uint8_t> h_out_data(data_bytes), h_out_scales(scale_bytes);
        CUDA_CHECK(cudaMemcpy(h_out_data.data(),   d_S_out_data,   data_bytes,  cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_scales.data(), d_S_out_scales, scale_bytes, cudaMemcpyDeviceToHost));

        std::vector<float> h_fused_dequant(n);
        nvfp4_dequantise_host(h_out_data.data(), h_out_scales.data(),
                               h_fused_dequant.data(), n);

        char tag[32];
        snprintf(tag, sizeof(tag), "d=%d", d);
        float max_rel = compare_outputs(h_ref_out.data(), h_fused_dequant.data(),
                                        d, tag, rel_tol);

        bool pass = (max_rel <= rel_tol);
        printf("%s  max_rel_err=%.4f  (tol=%.4f)\n",
               pass ? "PASS" : "FAIL", max_rel, rel_tol);

        // cleanup and return
        CUDA_CHECK(cudaStreamDestroy(stream));
        cudaFree(d_S_data); cudaFree(d_S_scales);
        cudaFree(d_S_out_data); cudaFree(d_S_out_scales);
        cudaFree(d_k); cudaFree(d_v); cudaFree(d_ref_out);
        return pass;
    }

cleanup:
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_S_data); cudaFree(d_S_scales);
    cudaFree(d_S_out_data); cudaFree(d_S_out_scales);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_ref_out);
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    srand(42);

    printf("GDN Correctness Gate (M3)\n");
    printf("Tolerance: 5%% max relative error (FR-6)\n");
    printf("─────────────────────────────────────────────\n");

    // Confirm device
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM %d.%d\n\n", prop.name, prop.major, prop.minor);

    const float REL_TOL = 0.05f;  // 5% — FR-6 gate
    bool all_pass = true;

    // Three problem sizes per PRD M3 exit criteria
    printf("Test suite:\n");
    all_pass &= run_test( 512, 0.9f, 0.1f, REL_TOL);
    all_pass &= run_test(1024, 0.8f, 0.2f, REL_TOL);
    all_pass &= run_test(2048, 0.95f, 0.05f, REL_TOL);  // primary gate

    printf("─────────────────────────────────────────────\n");
    printf("M3 correctness gate: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
