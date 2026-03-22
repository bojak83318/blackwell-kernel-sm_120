add_library(sm120_intrinsics INTERFACE)
target_include_directories(sm120_intrinsics INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${SM120_BINARY_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/src/intrinsics)
target_compile_features(sm120_intrinsics INTERFACE cxx_std_20)
add_library(sm120_intrinsics_alias ALIAS sm120_intrinsics)
