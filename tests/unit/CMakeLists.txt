add_executable(sm120_unit_tests main.cpp)

add_test(NAME sm120_unit_tests COMMAND sm120_unit_tests)
add_dependencies(sm120_tests sm120_unit_tests)
