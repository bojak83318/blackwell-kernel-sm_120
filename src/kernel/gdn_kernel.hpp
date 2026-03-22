#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "sm120/runtime/parameter_validation.h"

namespace sm_120 {
namespace kernel {

using cuda_stream_t = cudaStream_t;

struct GdnLaunchParams {
  const void* queries = nullptr;  // layout: batch, sequence_length, head_count, hidden_size
  const void* values = nullptr;   // layout: batch, sequence_length, head_count, state_size
  void* output = nullptr;         // layout: batch, head_count, hidden_size, state_size
  int32_t batch = 0;
  int32_t sequence_length = 0;  // currently restricted to 1 (K=1)
  int32_t head_count = 0;
  int32_t hidden_size = 0;  // per-head dimension
  int32_t state_size = 0;   // usually expand_v * hidden_size
  sm120::runtime::DataType dtype = sm120::runtime::DataType::Float16;
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
