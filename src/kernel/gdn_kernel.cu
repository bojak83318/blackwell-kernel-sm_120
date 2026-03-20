#include "gdn_kernel.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdint>

namespace sm_120 {
namespace kernel {

namespace {

constexpr int32_t kDefaultThreads = 128;

__global__ void gdn_kernel_stub(const float* source, float* destination, std::int64_t elements) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }

  destination[idx] = source[idx];
}

}  // namespace

[[nodiscard]] GdnLaunchConfig compute_launch_config(const GdnLaunchParams& params) noexcept {
  const auto elements = static_cast<std::int64_t>(params.batch) * params.seq_len * params.hidden_size;
  const auto blocks = static_cast<int32_t>(std::max<std::int64_t>(
      1, (elements + kDefaultThreads - 1) / kDefaultThreads));

  return GdnLaunchConfig{
      dim3(static_cast<uint32_t>(blocks), 1, 1),
      dim3(kDefaultThreads, 1, 1),
      0};
}

[[nodiscard]] bool validate_launch_params(const GdnLaunchParams& params) noexcept {
  return params.input != nullptr && params.output != nullptr && params.batch > 0 && params.seq_len > 0 && params.hidden_size > 0;
}

cudaError_t launch_kernel(const GdnLaunchParams& params, cuda_stream_t stream) noexcept {
  if (!validate_launch_params(params)) {
    return cudaErrorInvalidValue;
  }

  const auto config = compute_launch_config(params);
  const auto elements = static_cast<std::int64_t>(params.batch) * params.seq_len * params.hidden_size;
  if (elements <= 0) {
    return cudaErrorInvalidValue;
  }

  gdn_kernel_stub<<<config.grid, config.block, config.shared_mem_bytes, stream>>>(
      reinterpret_cast<const float*>(params.input),
      reinterpret_cast<float*>(params.output),
      elements);

  return cudaGetLastError();
}

}  // namespace kernel
}  // namespace sm_120
