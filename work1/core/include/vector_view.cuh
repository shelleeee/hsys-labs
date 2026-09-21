#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>

template <typename T>
requires std::is_arithmetic_v<std::remove_cv_t<T>>
class VectorView {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    constexpr VectorView() noexcept = default;

    __host__ __device__ constexpr VectorView(pointer data, size_type size) noexcept
        : data_(data), size_(size) {}

    template <typename U>
    requires std::is_convertible_v<U*, T*>
    __host__ __device__ constexpr VectorView(const VectorView<U>& other) noexcept
        : data_(other.data()), size_(other.size()) {}

    VectorView(const VectorView&) noexcept = default;
    VectorView& operator=(const VectorView&) noexcept = default;
    VectorView(VectorView&&) noexcept = default;
    VectorView& operator=(VectorView&&) noexcept = default;
    ~VectorView() noexcept = default;

    __host__ __device__ constexpr reference operator[](size_type index) noexcept {
        return data_[index];
    }

    __host__ __device__ constexpr const_reference operator[](size_type index) const noexcept {
        return data_[index];
    }

    __host__ __device__ constexpr pointer data() noexcept {
        return data_;
    }

    __host__ __device__ constexpr const_pointer data() const noexcept {
        return data_;
    }

    __host__ __device__ constexpr size_type size() const noexcept {
        return size_;
    }

    __host__ __device__ constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(T);
    }

    __host__ __device__ constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    __host__ __device__ constexpr VectorView subview(size_type offset, size_type count) const noexcept {
        if (offset >= size_) {
            return VectorView(nullptr, 0);
        }
        size_type actual_count = (offset + count > size_) ? (size_ - offset) : count;
        return VectorView(data_ + offset, actual_count);
    }

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>);
static_assert(std::is_trivially_copyable_v<VectorView<double>>);
static_assert(std::is_trivially_copyable_v<VectorView<int>>);
static_assert(std::is_trivially_copyable_v<VectorView<const float>>);
