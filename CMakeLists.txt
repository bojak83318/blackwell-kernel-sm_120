cmake_minimum_required(VERSION 3.25)
project(sm_120_gdn VERSION 0.1 LANGUAGES C CXX)

include(GNUInstallDirs)
include(cmake/SM120Options.cmake)
include(cmake/SM120Dependencies.cmake)

sm120_print_build_options()
sm120_print_dependency_matrix()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile commands" FORCE)

set(SM120_BINARY_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include" CACHE PATH "Directory for configured SM_120 headers")
file(MAKE_DIRECTORY ${SM120_BINARY_INCLUDE_DIR}/sm120)
configure_file(include/sm120/config.h.in ${SM120_BINARY_INCLUDE_DIR}/sm120/config.h @ONLY)

add_subdirectory(src)
