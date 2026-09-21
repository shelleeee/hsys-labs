#pragma once

#include "data.cuh"
#include "vector_view.cuh"
#include "kernel.cuh"
#include "cuda_check.cuh"

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
requires std::is_arithmetic_v<T>
class Vector {
public:
    using value_type = T;
    using size_type = std::size_t;

    Vector() noexcept : data_(nullptr), view_() {}

    explicit Vector(size_type size)
        : data_(std::make_shared<Data<T>>(size)),
          view_(data_->view()) {}

    Vector(size_type size, T init_val)
        : Vector(size) {
        if (size > 0) {
            std::vector<T> host_init(size, init_val);
            data_->copy_from_host(host_init.data(), size);
        }
    }

    Vector(const T* host_data, size_type size)
        : data_(std::make_shared<Data<T>>(host_data, size)),
          view_(data_->view()) {}

    explicit Vector(const std::vector<T>& host_vec)
        : Vector(host_vec.data(), host_vec.size()) {}

    Vector(std::initializer_list<T> init)
        : Vector(std::vector<T>(init)) {}

    Vector(std::shared_ptr<Data<T>> data, VectorView<T> view) noexcept
        : data_(std::move(data)), view_(view) {}

    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector() = default;

    [[nodiscard]] Vector clone() const {
        if (!data_ || view_.empty()) {
            return Vector();
        }
        auto new_data = std::make_shared<Data<T>>(view_.size());
        CUDA_CHECK(cudaMemcpy(new_data->data(), view_.data(), view_.size_bytes(), cudaMemcpyDeviceToDevice));
        return Vector(new_data, new_data->view());
    }

    [[nodiscard]] Vector subvector(size_type offset, size_type count) const {
        return Vector(data_, view_.subview(offset, count));
    }

    [[nodiscard]] size_type size() const noexcept { return view_.size(); }
    [[nodiscard]] size_type size_bytes() const noexcept { return view_.size_bytes(); }
    [[nodiscard]] bool empty() const noexcept { return view_.empty(); }

    [[nodiscard]] T* data() noexcept { return view_.data(); }
    [[nodiscard]] const T* data() const noexcept { return view_.data(); }

    [[nodiscard]] VectorView<T> view() noexcept { return view_; }
    [[nodiscard]] VectorView<const T> view() const noexcept { return view_; }
    [[nodiscard]] VectorView<const T> cview() const noexcept { return view_; }

    operator VectorView<T>() noexcept { return view_; }
    operator VectorView<const T>() const noexcept { return view_; }

    [[nodiscard]] long use_count() const noexcept { return data_ ? data_.use_count() : 0; }
    [[nodiscard]] const std::shared_ptr<Data<T>>& data_ptr() const noexcept { return data_; }

    void copy_to_host(T* host_dst) const {
        if (!empty() && data()) {
            if (host_dst == nullptr) {
                throw std::invalid_argument("Destination pointer is null");
            }
            CUDA_CHECK(cudaMemcpy(host_dst, data(), size_bytes(), cudaMemcpyDeviceToHost));
        }
    }

    [[nodiscard]] std::vector<T> to_host() const {
        std::vector<T> res(size());
        if (!empty()) {
            copy_to_host(res.data());
        }
        return res;
    }

    void copy_from_host(const T* host_src, size_type count) {
        if (count > size()) {
            throw std::out_of_range("Count exceeds Vector size");
        }
        if (count > 0 && data()) {
            if (host_src == nullptr) {
                throw std::invalid_argument("Source pointer is null");
            }
            CUDA_CHECK(cudaMemcpy(data(), host_src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

private:
    std::shared_ptr<Data<T>> data_;
    VectorView<T> view_;
};

template <typename T>
void add(const Vector<T>& a, const Vector<T>& b, Vector<T>& result, cudaStream_t stream = 0) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Vector sizes must match for add");
    }
    if (result.empty()) {
        return;
    }

    constexpr unsigned int block_size = 256;
    unsigned int num_blocks = static_cast<unsigned int>((result.size() + block_size - 1) / block_size);

    kernel_vecadd<<<num_blocks, block_size, 0, stream>>>(a.cview(), b.cview(), result.view());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
Vector<T> operator+(const Vector<T>& a, const Vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for operator+");
    }

    Vector<T> result(a.size());
    add(a, b, result);
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}
