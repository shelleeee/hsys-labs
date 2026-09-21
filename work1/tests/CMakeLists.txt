add_executable(run_tests
    src/test_vector.cu
)

target_include_directories(run_tests PRIVATE
    include
)

target_link_libraries(run_tests PRIVATE
    core_lib
    GTest::gtest
    GTest::gtest_main
    Eigen3::Eigen
    CUDA::cudart
)

set_target_properties(run_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cuda_setup_targets(
    TARGETS run_tests
    CUDA_MIN_ARCH 75
)

include(GoogleTest)
gtest_discover_tests(run_tests)