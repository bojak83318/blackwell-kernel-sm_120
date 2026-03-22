#include "gdn_kernel.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <limits>

// This path currently targets NVIDIA Blackwell's compute_120f pipeline and emits the standalone
// K=1 outer-product update that drives Gated DeltaNet's fwd_h accumulation.
namespace sm_120 {
namespace kernel {

namespace {

constexpr int32_t kDefaultThreads = 256;
constexpr int32_t kSupportedSequenceLength = 1;

template <typename T>
__device__ float load_scalar(const T* data, std::int64_t index) {
  return static_cast<float>(data[index]);
}

template <>
__device__ float load_scalar<__half>(const __half* data, std::int64_t index) {
  return __half2float(data[index]);
}

template <typename T>
__device__ void store_scalar(T* data, std::int64_t index, float value) {
  data[index] = static_cast<T>(value);
}

template <>
__device__ void store_scalar<__half>(__half* data, std::int64_t index, float value) {
  data[index] = __float2half_rn(value);
}

template <typename T>
__global__ void gdn_k1_outer_product_kernel(
    const T* __restrict__ queries,
    const T* __restrict__ values,
    T* __restrict__ output,
    int32_t batch,
    int32_t head_count,
    int32_t hidden_size,
    int32_t state_size,
    int32_t sequence_length) {
  const auto total_threads = static_cast<std::int64_t>(batch) * head_count * hidden_size * state_size;
  const auto linear_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= total_threads) {
    return;
  }

  const auto output_index = linear_index / state_size;
  const auto state_index = static_cast<int32_t>(linear_index - output_index * state_size);
  const auto hidden_index = static_cast<int32_t>(output_index % hidden_size);
  const auto batch_head_index = output_index / hidden_size;
  const auto head_index = static_cast<int32_t>(batch_head_index % head_count);
  const auto batch_index = static_cast<int32_t>(batch_head_index / head_count);

  const auto query_batch_stride = static_cast<std::int64_t>(sequence_length) * head_count * hidden_size;
  const auto value_batch_stride = static_cast<std::int64_t>(sequence_length) * head_count * state_size;

  const auto query_offset = batch_index * query_batch_stride +
      static_cast<std::int64_t>(head_index) * hidden_size + hidden_index;
  const auto value_offset = batch_index * value_batch_stride +
      static_cast<std::int64_t>(head_index) * state_size + state_index;

  const float query_value = load_scalar(queries, query_offset);
  const float value_value = load_scalar(values, value_offset);
  const float accumulated = load_scalar(output, linear_index);

  const float product = query_value * value_value + accumulated;
  store_scalar(output, linear_index, product);
}

}  // namespace

[[nodiscard]] GdnLaunchConfig compute_launch_config(const GdnLaunchParams& params) noexcept {
  const auto total_outputs = static_cast<std::int64_t>(params.batch) * params.head_count *
      params.hidden_size * params.state_size;
  const auto block_count = (total_outputs + kDefaultThreads - 1) / kDefaultThreads;
  const auto safe_blocks = static_cast<uint32_t>(std::min(
      block_count, static_cast<std::int64_t>(std::numeric_limits<uint32_t>::max())));
  const auto launch_blocks = std::max<uint32_t>(1u, safe_blocks);

  return GdnLaunchConfig{{launch_blocks, 1, 1}, {static_cast<uint32_t>(kDefaultThreads), 1, 1}, 0};
}

[[nodiscard]] bool validate_launch_params(const GdnLaunchParams& params) noexcept {
  if (params.queries == nullptr || params.values == nullptr || params.output == nullptr) {
    return false;
  }
  if (params.batch <= 0 || params.head_count <= 0 || params.hidden_size <= 0 || params.state_size <= 0) {
    return false;
  }
  // Shape guard: the current compute_120f kernel only handles K=1 outer-product tiles.
  if (params.sequence_length != kSupportedSequenceLength) {
    return false;
  }
  if (params.hidden_size % sm120::runtime::kHiddenAlign != 0) {
    return false;
  }
  // The GDN outer-product path assumes the state dimension is a multiple of the per-head hidden size.
  if (params.state_size % params.hidden_size != 0) {
    return false;
  }
  if (params.dtype != sm120::runtime::DataType::Float16 &&
      params.dtype != sm120::runtime::DataType::Float32) {
    return false;
  }
  return true;
}

cudaError_t launch_kernel(const GdnLaunchParams& params, cuda_stream_t stream) noexcept {
  if (!validate_launch_params(params)) {
    return cudaErrorInvalidValue;
  }

  const auto config = compute_launch_config(params);
  switch (params.dtype) {
    case sm120::runtime::DataType::Float32:
      gdn_k1_outer_product_kernel<float><<<config.grid, config.block, config.shared_mem_bytes, stream>>>(
          reinterpret_cast<const float*>(params.queries),
          reinterpret_cast<const float*>(params.values),
          reinterpret_cast<float*>(params.output),
          params.batch,
          params.head_count,
          params.hidden_size,
          params.state_size,
          params.sequence_length);
      break;

    case sm120::runtime::DataType::Float16:
      gdn_k1_outer_product_kernel<__half><<<config.grid, config.block, config.shared_mem_bytes, stream>>>(
          reinterpret_cast<const __half*>(params.queries),
          reinterpret_cast<const __half*>(params.values),
          reinterpret_cast<__half*>(params.output),
          params.batch,
          params.head_count,
          params.hidden_size,
          params.state_size,
          params.sequence_length);
      break;

    default:
      return cudaErrorInvalidValue;
  }

  return cudaGetLastError();
}

}  // namespace kernel
}  // namespace sm_120
