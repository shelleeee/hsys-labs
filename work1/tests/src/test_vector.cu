#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <random>
#include "data.cuh"
#include "vector.cuh"

TEST(DataTest, DefaultConstructor) {
    Data<float> d;
    EXPECT_EQ(d.size(), 0);
    EXPECT_EQ(d.data(), nullptr);
    EXPECT_TRUE(d.empty());
    EXPECT_EQ(d.size_bytes(), 0);
}

TEST(DataTest, AllocateAndSize) {
    constexpr std::size_t n = 128;
    Data<float> d(n);
    EXPECT_EQ(d.size(), n);
    EXPECT_NE(d.data(), nullptr);
    EXPECT_FALSE(d.empty());
    EXPECT_EQ(d.size_bytes(), n * sizeof(float));
}

TEST(DataTest, HostToDeviceAndBack) {
    constexpr std::size_t n = 64;
    std::vector<float> host_in(n);
    for (std::size_t i = 0; i < n; ++i) {
        host_in[i] = static_cast<float>(i * 1.5f);
    }

    Data<float> d(host_in.data(), n);
    EXPECT_EQ(d.size(), n);

    std::vector<float> host_out(n, 0.0f);
    d.copy_to_host(host_out.data());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host_in[i], host_out[i]);
    }
}

TEST(DataTest, CopyConstructorIsDeep) {
    constexpr std::size_t n = 32;
    std::vector<int> host_in(n, 42);
    Data<int> d1(host_in.data(), n);

    Data<int> d2 = d1;
    EXPECT_EQ(d2.size(), d1.size());
    EXPECT_NE(d2.data(), d1.data());

    std::vector<int> host_out(n, 0);
    d2.copy_to_host(host_out.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(host_out[i], 42);
    }
}

TEST(DataTest, CopyAssignmentIsDeep) {
    constexpr std::size_t n = 16;
    std::vector<double> host_in(n, 3.14159);
    Data<double> d1(host_in.data(), n);

    Data<double> d2;
    d2 = d1;

    EXPECT_EQ(d2.size(), d1.size());
    EXPECT_NE(d2.data(), d1.data());

    std::vector<double> host_out(n, 0.0);
    d2.copy_to_host(host_out.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(host_out[i], 3.14159);
    }
}

TEST(DataTest, MoveConstructor) {
    constexpr std::size_t n = 50;
    std::vector<float> host_in(n, 10.0f);
    Data<float> d1(host_in.data(), n);
    float* original_ptr = d1.data();

    Data<float> d2(std::move(d1));
    EXPECT_EQ(d2.size(), n);
    EXPECT_EQ(d2.data(), original_ptr);
    EXPECT_EQ(d1.size(), 0);
    EXPECT_EQ(d1.data(), nullptr);

    std::vector<float> host_out(n, 0.0f);
    d2.copy_to_host(host_out.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host_out[i], 10.0f);
    }
}

TEST(DataTest, MoveAssignment) {
    constexpr std::size_t n = 50;
    std::vector<float> host_in(n, 20.0f);
    Data<float> d1(host_in.data(), n);
    float* original_ptr = d1.data();

    Data<float> d2;
    d2 = std::move(d1);
    EXPECT_EQ(d2.size(), n);
    EXPECT_EQ(d2.data(), original_ptr);
    EXPECT_EQ(d1.size(), 0);
    EXPECT_EQ(d1.data(), nullptr);

    std::vector<float> host_out(n, 0.0f);
    d2.copy_to_host(host_out.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host_out[i], 20.0f);
    }
}

template <typename T>
__global__ void fill_view_kernel(VectorView<T> v, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < v.size()) {
        v[idx] = val + static_cast<T>(idx);
    }
}

template <typename T>
__global__ void double_view_kernel(VectorView<const T> in, VectorView<T> out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in.size()) {
        out[idx] = in[idx] * static_cast<T>(2);
    }
}

TEST(VectorViewTest, HostAccess) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    VectorView<int> view(data.data(), data.size());

    EXPECT_EQ(view.size(), 5);
    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[4], 5);

    view[2] = 42;
    EXPECT_EQ(data[2], 42);
}

TEST(VectorViewTest, FromDataView) {
    Data<float> d(10);
    VectorView<float> view = d.view();

    EXPECT_EQ(view.data(), d.data());
    EXPECT_EQ(view.size(), d.size());
}

TEST(VectorViewTest, Subview) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60};
    VectorView<int> view(data.data(), data.size());

    auto sub = view.subview(2, 3);
    EXPECT_EQ(sub.size(), 3);
    EXPECT_EQ(sub[0], 30);
    EXPECT_EQ(sub[1], 40);
    EXPECT_EQ(sub[2], 50);

    auto out_of_bounds = view.subview(10, 5);
    EXPECT_EQ(out_of_bounds.size(), 0);
    EXPECT_TRUE(out_of_bounds.empty());
}

TEST(VectorViewTest, DeviceKernelWrite) {
    constexpr std::size_t n = 64;
    Data<float> d(n);

    fill_view_kernel<<<2, 32>>>(d.view(), 100.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_res(n);
    d.copy_to_host(host_res.data());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host_res[i], 100.0f + static_cast<float>(i));
    }
}

TEST(VectorViewTest, DeviceKernelReadWriteConst) {
    constexpr std::size_t n = 128;
    std::vector<float> host_in(n);
    for (std::size_t i = 0; i < n; ++i) {
        host_in[i] = static_cast<float>(i);
    }

    Data<float> in_data(host_in.data(), n);
    Data<float> out_data(n);

    double_view_kernel<<<4, 32>>>(in_data.cview(), out_data.view());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_res(n);
    out_data.copy_to_host(host_res.data());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host_res[i], static_cast<float>(i * 2));
    }
}

TEST(VectorTest, ConstructorsAndAccess) {
    Vector<float> v_empty;
    EXPECT_EQ(v_empty.size(), 0);
    EXPECT_TRUE(v_empty.empty());

    Vector<float> v_size(10);
    EXPECT_EQ(v_size.size(), 10);
    EXPECT_FALSE(v_size.empty());
    EXPECT_NE(v_size.data(), nullptr);

    Vector<int> v_init(5, 7);
    auto host_v_init = v_init.to_host();
    for (int x : host_v_init) {
        EXPECT_EQ(x, 7);
    }

    Vector<float> v_list = {1.0f, 2.0f, 3.0f};
    EXPECT_EQ(v_list.size(), 3);
    auto host_list = v_list.to_host();
    EXPECT_FLOAT_EQ(host_list[0], 1.0f);
    EXPECT_FLOAT_EQ(host_list[1], 2.0f);
    EXPECT_FLOAT_EQ(host_list[2], 3.0f);
}

TEST(VectorTest, DataSharingViaSharedPtr) {
    Vector<float> v1(10, 1.0f);
    EXPECT_EQ(v1.use_count(), 1);

    Vector<float> v2 = v1;
    EXPECT_EQ(v1.use_count(), 2);
    EXPECT_EQ(v2.use_count(), 2);
    EXPECT_EQ(v1.data(), v2.data());
}

TEST(VectorTest, CloneCreatesIndependentCopy) {
    Vector<float> v1(10, 5.0f);
    Vector<float> v2 = v1.clone();

    EXPECT_EQ(v1.use_count(), 1);
    EXPECT_EQ(v2.use_count(), 1);
    EXPECT_NE(v1.data(), v2.data());

    auto host_v2 = v2.to_host();
    for (float x : host_v2) {
        EXPECT_FLOAT_EQ(x, 5.0f);
    }
}

TEST(VectorTest, SubvectorSharing) {
    Vector<int> v1 = {10, 20, 30, 40, 50};
    Vector<int> v_sub = v1.subvector(1, 3);

    EXPECT_EQ(v_sub.size(), 3);
    EXPECT_EQ(v1.use_count(), 2);

    auto host_sub = v_sub.to_host();
    EXPECT_EQ(host_sub[0], 20);
    EXPECT_EQ(host_sub[1], 30);
    EXPECT_EQ(host_sub[2], 40);
}

TEST(VectorTest, SizeMismatchThrows) {
    Vector<float> v1(10, 1.0f);
    Vector<float> v2(20, 2.0f);
    EXPECT_THROW(auto v3 = v1 + v2, std::invalid_argument);
}

class VectorAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddTest, MatchesEigen) {
    const std::size_t n = GetParam();

    Eigen::VectorXf eigen_a(n);
    Eigen::VectorXf eigen_b(n);
    std::mt19937 gen(static_cast<unsigned int>(n * 42 + 1));
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (std::size_t i = 0; i < n; ++i) {
        eigen_a[static_cast<Eigen::Index>(i)] = dis(gen);
        eigen_b[static_cast<Eigen::Index>(i)] = dis(gen);
    }

    Eigen::VectorXf eigen_expected = eigen_a + eigen_b;

    Vector<float> vec_a(eigen_a.data(), n);
    Vector<float> vec_b(eigen_b.data(), n);
    Vector<float> vec_res = vec_a + vec_b;

    std::vector<float> host_res = vec_res.to_host();
    Eigen::Map<const Eigen::VectorXf> eigen_actual(host_res.data(), n);

    EXPECT_TRUE(eigen_actual.isApprox(eigen_expected, 1e-6f));
}

INSTANTIATE_TEST_SUITE_P(
    MandatorySizes,
    VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029)
);
