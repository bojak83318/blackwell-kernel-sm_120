set(SM120_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "SM_120 source directory")
if(NOT DEFINED SM120_BINARY_INCLUDE_DIR)
  set(SM120_BINARY_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
endif()
add_library(sm120_config INTERFACE)
target_include_directories(sm120_config INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${SM120_BINARY_INCLUDE_DIR}
)

add_subdirectory(kernel)
add_subdirectory(runtime)
add_subdirectory(pytorch)
add_subdirectory(trtllm)
