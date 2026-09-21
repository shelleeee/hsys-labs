#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "vector.cuh"
#include "cuda_check.cuh"

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int p = 1; p <= 9; ++p) {
        std::size_t n = 1ULL << (3 * p);
        b->Args({static_cast<int64_t>(n)});
    }
}

static void BM_VectorAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Vector<float> a(n, 1.0f);
    Vector<float> b(n, 2.0f);
    Vector<float> c(n);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    add(a, b, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        add(a, b, c);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(static_cast<double>(ms) * 1e-3);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(float) * 3);
    state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void BM_EigenAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Eigen::VectorXf a = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 1.0f);
    Eigen::VectorXf b = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 2.0f);
    Eigen::VectorXf c(static_cast<Eigen::Index>(n));

    for (auto _ : state) {
        c.noalias() = a + b;
        benchmark::DoNotOptimize(c.data());
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(float) * 3);
    state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(n));
}

BENCHMARK(BM_VectorAdd)
    ->Name("CUDA Vector Addition (GPU)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_EigenAdd)
    ->Name("Eigen Vector Addition (CPU)")
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMillisecond);
