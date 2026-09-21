cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(work1 LANGUAGES CXX CUDA)

add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(benchmarks)