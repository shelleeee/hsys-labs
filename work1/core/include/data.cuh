#pragma once

#include "cuda_check.cuh"
#include "vector_view.cuh"

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cuda_runtime.h>

template <typename T>
requires std::is_arithmetic_v<T>
class Data {
public:
    using value_type = T;
    using size_type = std::size_t;

    Data() noexcept : data_(nullptr), size_(0) {}

    explicit Data(size_type size) : data_(nullptr), size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
        }
    }

    Data(const T* host_data, size_type size) : data_(nullptr), size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
            if (host_data != nullptr) {
                CUDA_CHECK(cudaMemcpy(data_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
            }
        }
    }

    ~Data() noexcept {
        reset();
    }

    Data(const Data& other) : data_(nullptr), size_(other.size_) {
        if (size_ > 0 && other.data_ != nullptr) {
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            Data temp(other);
            swap(temp);
        }
        return *this;
    }

    Data(Data&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void swap(Data& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type size_bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] VectorView<T> view() noexcept {
        return VectorView<T>(data_, size_);
    }

    [[nodiscard]] VectorView<const T> view() const noexcept {
        return VectorView<const T>(data_, size_);
    }

    [[nodiscard]] VectorView<const T> cview() const noexcept {
        return VectorView<const T>(data_, size_);
    }

    void copy_to_host(T* host_dst) const {
        if (size_ > 0 && data_ != nullptr) {
            if (host_dst == nullptr) {
                throw std::invalid_argument("Destination host pointer is null");
            }
            CUDA_CHECK(cudaMemcpy(host_dst, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void copy_from_host(const T* host_src, size_type count) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds Data capacity");
        }
        if (count > 0 && data_ != nullptr) {
            if (host_src == nullptr) {
                throw std::invalid_argument("Source host pointer is null");
            }
            CUDA_CHECK(cudaMemcpy(data_, host_src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

private:
    void reset() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    T* data_ = nullptr;
    size_type size_ = 0;
};

template <typename T>
void swap(Data<T>& lhs, Data<T>& rhs) noexcept {
    lhs.swap(rhs);
}
