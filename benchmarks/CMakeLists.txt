add_executable(sm120_benchmark_main benchmark_main.cpp)

add_custom_target(sm120_benchmarks
  DEPENDS sm120_benchmark_main
  COMMENT "Placeholder target for SM_120 benchmark harness"
)
