if(NOT TARGET sm120_tests)
  add_custom_target(sm120_tests)
endif()

add_subdirectory(unit)
add_subdirectory(integration)

add_custom_target(sm120_test_harness
  DEPENDS sm120_tests
  COMMENT "Aggregate target for SM_120 test harness"
)
