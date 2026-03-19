add_library(sm120_runtime INTERFACE)
# Placeholder for runtime/launcher helpers before implementation.
target_include_directories(sm120_runtime INTERFACE ${CMAKE_SOURCE_DIR}/include)
add_library(sm120_runtime_alias ALIAS sm120_runtime)
