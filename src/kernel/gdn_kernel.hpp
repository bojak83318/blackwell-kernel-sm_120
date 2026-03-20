#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sm_120 {
namespace kernel {

using cuda_stream_t = cudaStream_t;

struct GdnLaunchParams {
  const void* input;
  void* output;
  int32_t batch;
  int32_t seq_len;
  int32_t hidden_size;
};

struct GdnLaunchConfig {
  dim3 grid;
  dim3 block;
  int32_t shared_mem_bytes;
};

[[nodiscard]] GdnLaunchConfig compute_launch_config(const GdnLaunchParams& params) noexcept;
[[nodiscard]] bool validate_launch_params(const GdnLaunchParams& params) noexcept;
cudaError_t launch_kernel(const GdnLaunchParams& params, cuda_stream_t stream) noexcept;

}  // namespace kernel
}  // namespace sm_120
