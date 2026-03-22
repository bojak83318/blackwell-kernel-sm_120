add_library(sm120_pytorch INTERFACE)
target_link_libraries(sm120_pytorch INTERFACE sm120_ops)
# Placeholder for PyTorch binding helper targets.
target_include_directories(sm120_pytorch INTERFACE ${CMAKE_SOURCE_DIR}/include)
