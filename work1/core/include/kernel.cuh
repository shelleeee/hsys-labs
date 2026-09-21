#pragma once

#include "vector_view.cuh"
#include <cuda_runtime.h>
#include <cstddef>

template <typename T>
__global__ void kernel_vecadd(VectorView<const T> a, VectorView<const T> b, VectorView<T> c) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < c.size()) {
        c[idx] = a[idx] + b[idx];
    }
}
