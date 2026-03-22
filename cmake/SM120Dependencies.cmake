set(SM120_CUDA_VERSION "12.8" CACHE STRING "Minimum CUDA toolkit version required for SM_120")
set(SM120_CUTLASS_VERSION "4.2.1" CACHE STRING "Pinned CUTLASS version for the SM_120 kernel")
set(SM120_TORCH_VERSION "2.9.1+cu128" CACHE STRING "Target PyTorch minor version for the extension")
set(SM120_VLLM_VERSION "0.12" CACHE STRING "Minimum vLLM version expected for integration")
set(SM120_TENSORRT_VERSION "9.2" CACHE STRING "Expected TensorRT version for TRT-LLM plugin work")
set(SM120_TARGET_ARCH "sm_120f" CACHE STRING "CUDA GPU code target for the custom kernel (e.g., sm_120f)")
set(SM120_COMPUTE_TARGET "compute_120f" CACHE STRING "CUDA compute architecture used during compilation (nvcc --gpu-architecture)")

set(SM120_DEPENDENCY_MATRIX
  CUDA ${SM120_CUDA_VERSION}
  CUTLASS ${SM120_CUTLASS_VERSION}
  PyTorch ${SM120_TORCH_VERSION}
  vLLM ${SM120_VLLM_VERSION}
  TensorRT ${SM120_TENSORRT_VERSION}
)

function(sm120_print_dependency_matrix)
  message(STATUS "SM_120 dependency matrix:")
  list(LENGTH SM120_DEPENDENCY_MATRIX matrix_length)
  if(NOT matrix_length)
    return()
  endif()
  math(EXPR pair_count "${matrix_length} / 2")
  math(EXPR last_pair_index "${pair_count} - 1")
  foreach(pair_index RANGE 0 ${last_pair_index})
    math(EXPR name_index "${pair_index} * 2")
    math(EXPR version_index "${name_index} + 1")
    list(GET SM120_DEPENDENCY_MATRIX ${name_index} dep_name)
    list(GET SM120_DEPENDENCY_MATRIX ${version_index} dep_version)
    message(STATUS "  ${dep_name}: ${dep_version}")
  endforeach()
endfunction()
