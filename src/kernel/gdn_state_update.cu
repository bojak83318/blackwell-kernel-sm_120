#include <cstdint>
#include <cstddef>
#include <cmath>

#include <cuda_runtime.h>

#include "gdn_fp4_intrinsics.cuh"

namespace sm120 {
namespace kernel {

namespace detail {

constexpr int kNvfp4BlockElements = 16;
constexpr int kNvfp4MantissaBytes = kNvfp4BlockElements / 2;
constexpr int kNvfp4BlockBytes = 1 + kNvfp4MantissaBytes;
constexpr float kMinBlockScale = 1e-6f;

__device__ __forceinline__ std::uint8_t read_e2m1(const std::uint8_t* block_mantissa, int lane) noexcept {
  const int byte_index = lane >> 1;
  const bool high_nibble = (lane & 1) != 0;
  const std::uint8_t value = block_mantissa[byte_index];
  return high_nibble ? (value >> 4) : (value & 0xFu);
}

__device__ __forceinline__ void write_e2m1(std::uint8_t* block_mantissa, int lane, std::uint8_t code) noexcept {
  const int byte_index = lane >> 1;
  const bool high_nibble = (lane & 1) != 0;
  std::uint8_t current = block_mantissa[byte_index];
  if (high_nibble) {
    current = static_cast<std::uint8_t>((current & 0x0Fu) | (static_cast<std::uint8_t>(code & 0xFu) << 4));
  } else {
    current = static_cast<std::uint8_t>((current & 0xF0u) | (code & 0xFu));
  }
  block_mantissa[byte_index] = current;
}

__device__ __forceinline__ std::uint8_t encode_e2m1(float normalized) noexcept {
  const bool sign = normalized < 0.0f;
  const float magnitude = sign ? -normalized : normalized;
  std::uint8_t exponent = 0;
  std::uint8_t mantissa = 0;

  if (magnitude >= 4.0f) {
    exponent = 3;
    mantissa = magnitude >= 6.0f ? 1 : 0;
  } else if (magnitude >= 2.0f) {
    exponent = 2;
    mantissa = magnitude >= 3.0f ? 1 : 0;
  } else if (magnitude >= 1.0f) {
    exponent = 1;
    mantissa = magnitude >= 1.5f ? 1 : 0;
  } else if (magnitude >= 0.5f) {
    exponent = 0;
    mantissa = 1;
  }

  return static_cast<std::uint8_t>((static_cast<std::uint8_t>(sign) << 3) |
                                   (exponent << 1) |
                                   mantissa);
}

}  // namespace detail

struct GdnStateUpdateParams {
  const std::uint8_t* state_src = nullptr;
  std::uint8_t* state_dst = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  const float* alpha = nullptr;
  const float* beta = nullptr;
  int32_t state_dim = 0;
  int32_t batch_size = 0;
  int32_t head_count = 0;
};

struct GdnStateUpdateLaunchConfig {
  dim3 grid;
  dim3 block;
};

extern "C" __global__ void gdn_state_update_kernel(GdnStateUpdateParams params);

namespace {

[[nodiscard]] bool validate_params(GdnStateUpdateParams const& params) noexcept {
  return params.state_src != nullptr && params.state_dst != nullptr &&
         params.k != nullptr && params.v != nullptr && params.state_dim > 0 &&
         params.batch_size > 0 && params.head_count > 0;
}

}  // namespace

[[nodiscard]] GdnStateUpdateLaunchConfig compute_launch_config(GdnStateUpdateParams const& params) noexcept {
  const int blocks_per_row = (params.state_dim + detail::kNvfp4BlockElements - 1) / detail::kNvfp4BlockElements;
  const int instance_count = params.batch_size * params.head_count;
  return GdnStateUpdateLaunchConfig{
      dim3(static_cast<uint32_t>(blocks_per_row), static_cast<uint32_t>(params.state_dim), static_cast<uint32_t>(instance_count)),
      dim3(detail::kNvfp4BlockElements, 1, 1)};
}

cudaError_t launch_state_update(GdnStateUpdateParams const& params, cudaStream_t stream) noexcept {
  if (!validate_params(params)) {
    return cudaErrorInvalidValue;
  }

  const auto config = compute_launch_config(params);
  gdn_state_update_kernel<<<config.grid, config.block, 0, stream>>>(params);
  return cudaGetLastError();
}

extern "C" __global__ void gdn_state_update_kernel(GdnStateUpdateParams params) {
  const int blocks_per_row = (params.state_dim + detail::kNvfp4BlockElements - 1) / detail::kNvfp4BlockElements;
  const int instance_count = params.batch_size * params.head_count;

  const int block_col = blockIdx.x;
  const int row = blockIdx.y;
  const int instance = blockIdx.z;

  if (block_col >= blocks_per_row || row >= params.state_dim || instance >= instance_count) {
    return;
  }

  const int lane = threadIdx.x;
  const int col = block_col * detail::kNvfp4BlockElements + lane;
  const bool active = col < params.state_dim;

  const std::size_t row_stride = static_cast<std::size_t>(blocks_per_row) * detail::kNvfp4BlockBytes;
  const std::size_t matrix_stride = static_cast<std::size_t>(params.state_dim) * row_stride;
  const std::size_t instance_offset = static_cast<std::size_t>(instance) * matrix_stride;
  const std::size_t block_offset = instance_offset + static_cast<std::size_t>(row) * row_stride +
                                   static_cast<std::size_t>(block_col) * detail::kNvfp4BlockBytes;

  const std::uint8_t* src_block = params.state_src + block_offset;
  std::uint8_t* dst_block = params.state_dst + block_offset;

  const float* k_ptr = params.k + static_cast<std::size_t>(instance) * params.state_dim;
  const float* v_ptr = params.v + static_cast<std::size_t>(instance) * params.state_dim;
  const float alpha = params.alpha ? params.alpha[instance] : 1.0f;
  const float beta = params.beta ? params.beta[instance] : 1.0f;

  float prev_value = 0.0f;
  if (active) {
    const float scale = sm120::intrinsics::decode_ue4m3_to_f32(src_block[0]);
    const auto encoded = detail::read_e2m1(src_block + 1, lane);
    prev_value = sm120::intrinsics::decode_e2m1_to_f32(encoded) * scale;
  }

  const float k_val = active ? k_ptr[row] : 0.0f;
  const float v_val = active ? v_ptr[col] : 0.0f;
  const float updated = alpha * prev_value + beta * (k_val * v_val);

  __shared__ float shared_updates[detail::kNvfp4BlockElements];
  __shared__ float shared_magnitudes[detail::kNvfp4BlockElements];
  __shared__ float shared_block_scale;
  __shared__ std::uint8_t shared_scale_code;

  shared_updates[lane] = active ? updated : 0.0f;
  shared_magnitudes[lane] = active ? fabsf(updated) : 0.0f;
  __syncthreads();

  if (lane == 0) {
    float block_max = detail::kMinBlockScale;
    for (int i = 0; i < detail::kNvfp4BlockElements; ++i) {
      block_max = fmaxf(block_max, shared_magnitudes[i]);
    }
    shared_scale_code = sm120::intrinsics::encode_f32_to_ue4m3(block_max);
    shared_block_scale = sm120::intrinsics::decode_ue4m3_to_f32(shared_scale_code);
    dst_block[0] = shared_scale_code;
  }

  __syncthreads();

  if (!active) {
    return;
  }

  const float quant_scale = shared_block_scale;
  const float normalized = quant_scale > 0.0f ? shared_updates[lane] / quant_scale : 0.0f;
  const std::uint8_t encoded = detail::encode_e2m1(normalized);
  detail::write_e2m1(dst_block + 1, lane, encoded);
}

}  // namespace kernel
}  // namespace sm120
