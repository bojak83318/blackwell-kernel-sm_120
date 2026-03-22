if(NOT TARGET sm120_tests)
  add_custom_target(sm120_tests)
endif()

add_subdirectory(unit)
add_subdirectory(integration)

add_custom_target(sm120_test_harness
  DEPENDS sm120_tests
  COMMENT "Aggregate target for SM_120 test harness"
)

# ── M3 correctness gate ───────────────────────────────────────────────────────
if(SM120_BUILD_TESTS)
  add_executable(test_gdn_correctness
      ${CMAKE_CURRENT_SOURCE_DIR}/test_gdn_correctness.cu
  )
  set_target_properties(test_gdn_correctness PROPERTIES
      CUDA_ARCHITECTURES "120f"
  )
  target_compile_options(test_gdn_correctness PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
          -gencode arch=compute_120f,code=sm_120f
          --expt-relaxed-constexpr
          -O3
      >
  )
  target_link_libraries(test_gdn_correctness PRIVATE
      sm120_kernel
  )
  target_include_directories(test_gdn_correctness PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src/intrinsics
      ${PROJECT_SOURCE_DIR}/src/kernel
  )
endif()
