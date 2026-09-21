cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(hsys LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

find_package(CUDAToolkit REQUIRED)

include(cmake/utils/cuda_auto_arch.cmake)
include(cmake/utils/cuda_setup_targets.cmake)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    get_cuda_gpu_arch(GPU_ARCH)
    if(GPU_ARCH GREATER "0")
        if(GPU_ARCH GREATER_EQUAL 75)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

include(cmake/deps/deps_ops.cmake)
include(cmake/deps/eigen.cmake)
include(cmake/deps/gtest.cmake)
include(cmake/deps/gbench.cmake)

add_subdirectory(work1)
