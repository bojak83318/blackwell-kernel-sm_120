#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "sm120/kernel/gdn_kernel.hpp"
#include "sm120/runtime/parameter_validation.h"

namespace {

constexpr int64_t kExpectedSequenceLength = 1;

int32_t to_int32_checked(int64_t value, const char* label) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
              value <= std::numeric_limits<int32_t>::max(),
              "SM_120 GDN: ", label, " must fit in int32");
  return static_cast<int32_t>(value);
}

sm120::runtime::DataType infer_data_type(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Half:
      return sm120::runtime::DataType::Float16;
    case at::ScalarType::Float:
      return sm120::runtime::DataType::Float32;
    default:
      TORCH_CHECK(false, "SM_120 GDN: unsupported dtype");
  }
}

void validate_tensors(const at::Tensor& queries,
                      const at::Tensor& values,
                      const at::Tensor& output) {
  TORCH_CHECK(queries.is_cuda(), "SM_120 GDN: queries must be CUDA tensors");
  TORCH_CHECK(values.is_cuda(), "SM_120 GDN: values must be CUDA tensors");
  TORCH_CHECK(output.is_cuda(), "SM_120 GDN: output must be a CUDA tensor");

  TORCH_CHECK(queries.dim() == 4, "SM_120 GDN: queries must be 4D");
  TORCH_CHECK(values.dim() == 4, "SM_120 GDN: values must be 4D");
  TORCH_CHECK(output.dim() == 4, "SM_120 GDN: output must be 4D");

  TORCH_CHECK(queries.device() == values.device(), "SM_120 GDN: queries and values must share the same device");
  TORCH_CHECK(queries.device() == output.device(), "SM_120 GDN: queries and output must share the same device");

  TORCH_CHECK(queries.scalar_type() == values.scalar_type(), "SM_120 GDN: queries and values must share the same dtype");
  TORCH_CHECK(queries.scalar_type() == output.scalar_type(), "SM_120 GDN: queries and output must share the same dtype");

  TORCH_CHECK(queries.is_contiguous(), "SM_120 GDN: queries must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "SM_120 GDN: values must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "SM_120 GDN: output must be contiguous");

  const int64_t batch = queries.size(0);
  const int64_t sequence_length = queries.size(1);
  const int64_t head_count = queries.size(2);
  const int64_t hidden_size = queries.size(3);

  TORCH_CHECK(batch > 0, "SM_120 GDN: batch must be > 0");
  TORCH_CHECK(sequence_length == kExpectedSequenceLength,
              "SM_120 GDN: only sequence_length=1 is supported");
  TORCH_CHECK(head_count > 0, "SM_120 GDN: head_count must be > 0");
  TORCH_CHECK(hidden_size > 0, "SM_120 GDN: hidden_size must be > 0");
  TORCH_CHECK(hidden_size % sm120::runtime::kHiddenAlign == 0,
              "SM_120 GDN: hidden_size must be a multiple of ", sm120::runtime::kHiddenAlign);

  TORCH_CHECK(values.size(0) == batch, "SM_120 GDN: values batch dimension mismatch");
  TORCH_CHECK(values.size(1) == sequence_length, "SM_120 GDN: values sequence_length mismatch");
  TORCH_CHECK(values.size(2) == head_count, "SM_120 GDN: values head_count mismatch");

  const int64_t state_size = values.size(3);
  TORCH_CHECK(state_size > 0, "SM_120 GDN: state_size must be > 0");
  TORCH_CHECK(state_size % hidden_size == 0,
              "SM_120 GDN: state_size must be divisible by hidden_size");

  TORCH_CHECK(output.size(0) == batch, "SM_120 GDN: output batch dimension mismatch");
  TORCH_CHECK(output.size(1) == head_count, "SM_120 GDN: output head_count mismatch");
  TORCH_CHECK(output.size(2) == hidden_size, "SM_120 GDN: output hidden_size mismatch");
  TORCH_CHECK(output.size(3) == state_size, "SM_120 GDN: output state_size mismatch");
}

} // namespace

at::Tensor gdn_cuda_impl(const at::Tensor& queries,
                         const at::Tensor& values,
                         at::Tensor& output) {
  validate_tensors(queries, values, output);

  const int64_t batch = queries.size(0);
  const int64_t sequence_length = queries.size(1);
  const int64_t head_count = queries.size(2);
  const int64_t hidden_size = queries.size(3);
  const int64_t state_size = values.size(3);

  sm120::kernel::GdnLaunchParams params;
  params.queries = queries.data_ptr();
  params.values = values.data_ptr();
  params.output = output.data_ptr();
  params.batch = to_int32_checked(batch, "batch");
  params.sequence_length = to_int32_checked(sequence_length, "sequence_length");
  params.head_count = to_int32_checked(head_count, "head_count");
  params.hidden_size = to_int32_checked(hidden_size, "hidden_size");
  params.state_size = to_int32_checked(state_size, "state_size");
  params.dtype = infer_data_type(queries.scalar_type());

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto err = sm120::kernel::launch_kernel(params, stream);
  TORCH_CHECK(err == cudaSuccess,
              "SM_120 GDN: kernel launch failed: ", cudaGetErrorString(err));
  return output;
}

TORCH_LIBRARY(sm120, m) {
  m.def("gdn(Tensor queries, Tensor values, Tensor output) -> Tensor");
}

TORCH_LIBRARY_IMPL(sm120, CUDA, m) {
  m.impl("gdn", gdn_cuda_impl);
}
