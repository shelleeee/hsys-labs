add_executable(run_benchmarks
    src/bench_vector.cu
)

target_include_directories(run_benchmarks PRIVATE
    include
)

target_link_libraries(run_benchmarks PRIVATE
    core_lib
    benchmark::benchmark
    benchmark::benchmark_main
    Eigen3::Eigen
    CUDA::cudart
)

set_target_properties(run_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cuda_setup_targets(
    TARGETS run_benchmarks
    CUDA_MIN_ARCH 75
)