enable_language(CUDA)

add_library(sm120_kernel STATIC gdn_kernel.cu)
target_include_directories(sm120_kernel PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${SM120_BINARY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sm120_kernel PUBLIC sm120_config)
set_target_properties(sm120_kernel PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)

string(REGEX REPLACE "sm_" "" SM120_COMPUTE ${SM120_TARGET_ARCH})
if(SM120_COMPUTE MATCHES "^[0-9]+$")
  set_target_properties(sm120_kernel PROPERTIES CUDA_ARCHITECTURES ${SM120_COMPUTE})
endif()

# PTX-only target for early SM120 validation without linking.
add_custom_target(gdn_kernel_ptx
  COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx -arch=sm_120 -std=c++17 --use_fast_math -O3
    -Xptxas -O3 --maxrregcount=64
    -DUSE_NVFP4=1 -DSTRUCTURED_SPARSITY=24
    ${CMAKE_CUDA_FLAGS}
    -o ${CMAKE_CURRENT_BINARY_DIR}/gdn_kernel_sm120.ptx
    gdn_kernel.cu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS gdn_kernel.cu
  COMMENT "Emit SM120 NVFP4 PTX"
)
