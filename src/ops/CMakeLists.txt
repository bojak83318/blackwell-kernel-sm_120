enable_language(CUDA)

find_package(Torch REQUIRED)

add_library(sm120_ops SHARED gdn_op.cu)
target_include_directories(sm120_ops PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${SM120_BINARY_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/src/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sm120_ops PRIVATE
  Torch::Torch
  sm120_kernel
  sm120_runtime
)
set_target_properties(sm120_ops PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES OFF
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED YES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_compile_features(sm120_ops PRIVATE cxx_std_20)
target_compile_options(sm120_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${SM120_NVCC_GENCODE_FLAGS}
          --expt-relaxed-constexpr
          --extended-lambda>
)
