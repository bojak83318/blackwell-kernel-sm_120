set(SM120_RUNTIME_SOURCES
  parameter_validation.cpp
  host_launcher.cpp
)

add_library(sm120_runtime STATIC ${SM120_RUNTIME_SOURCES})
target_include_directories(sm120_runtime PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(sm120_runtime PUBLIC cxx_std_20)
add_library(sm120_runtime_alias ALIAS sm120_runtime)
