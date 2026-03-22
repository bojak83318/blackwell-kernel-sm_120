add_executable(sm120_benchmark_main benchmark_main.cpp)

set(SM120_BENCHMARK_EVIDENCE_DIR "${CMAKE_SOURCE_DIR}/artifacts/m2/benchmarks" CACHE PATH "Directory for benchmark logs")
file(MAKE_DIRECTORY ${SM120_BENCHMARK_EVIDENCE_DIR})

add_custom_target(sm120_benchmarks
  DEPENDS sm120_benchmark_main
  COMMENT "Placeholder target for SM_120 benchmark harness"
)

add_custom_target(sm120_benchmark_harness
  DEPENDS sm120_benchmarks
  COMMENT "Aggregate target for SM_120 benchmark harness"
)
