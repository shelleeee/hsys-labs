add_library(core_lib STATIC
    src/kernel.cu
)

target_include_directories(core_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(core_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(core_lib PUBLIC
    CUDA::cudart
)

cuda_setup_targets(
    TARGETS core_lib
    CUDA_MIN_ARCH 75
)