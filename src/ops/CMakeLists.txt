# src/ops/CMakeLists.txt — builds libgdn_sm120_op.so (PyTorch custom op)

enable_language(CUDA)

# ── find torch ────────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)

if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(WARNING "Could not find torch cmake prefix — skipping gdn_sm120_op target")
    return()
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# ── shared library target ─────────────────────────────────────────────────────
add_library(gdn_sm120_op SHARED gdn_op.cu)

set_target_properties(gdn_sm120_op PROPERTIES
    CUDA_ARCHITECTURES   OFF
    CUDA_STANDARD        20
    CUDA_STANDARD_REQUIRED YES
    PREFIX               "lib"
    OUTPUT_NAME          "gdn_sm120_op"
)

target_compile_options(gdn_sm120_op PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_120f,code=sm_120
        --expt-relaxed-constexpr
        --extended-lambda
        -O3
    >
)

target_link_libraries(gdn_sm120_op PRIVATE
    sm120_kernel
    ${TORCH_LIBRARIES}
)

target_include_directories(gdn_sm120_op PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/intrinsics
    ${PROJECT_SOURCE_DIR}/src/kernel
    ${TORCH_INCLUDE_DIRS}
)

# suppress torch ABI warnings
target_compile_definitions(gdn_sm120_op PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS}
)
