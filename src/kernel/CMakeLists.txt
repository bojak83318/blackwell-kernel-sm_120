enable_language(CUDA)

add_library(sm120_kernel STATIC gdn_kernel.cu)
target_include_directories(sm120_kernel PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${SM120_BINARY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sm120_kernel PUBLIC sm120_config)
set_target_properties(sm120_kernel PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

string(REGEX REPLACE "sm_" "" SM120_COMPUTE ${SM120_TARGET_ARCH})
if(SM120_COMPUTE MATCHES "^[0-9]+$")
  set_target_properties(sm120_kernel PROPERTIES CUDA_ARCHITECTURES ${SM120_COMPUTE})
endif()
