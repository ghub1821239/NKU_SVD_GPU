// bidiagonalization.cpp
// 将 m x n 矩阵（本框架保证 m >= n）通过 Householder 变换化为上双对角形。

#include "matrix.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(USE_HIPBLAS_BACKEND) && defined(USE_HIP_KERNEL_BACKEND)
#error "USE_HIPBLAS_BACKEND and USE_HIP_KERNEL_BACKEND are mutually exclusive"
#endif

#if defined(USE_HIPBLAS_BACKEND) || defined(USE_HIP_KERNEL_BACKEND)
#include <hip/hip_runtime.h>
#endif

#ifdef USE_HIPBLAS_BACKEND
#include <hipblas/hipblas.h>
#endif

namespace
{

struct HouseholderData
{
    std::vector<double> v;
    double beta = 0.0;
    bool active = false;
};

static HouseholderData make_householder(const std::vector<double> &x);

#if defined(USE_HIPBLAS_BACKEND) || defined(USE_HIP_KERNEL_BACKEND)
static void throw_backend_error(const char *where, const char *what)
{
    std::ostringstream oss;
    oss << where << " failed: " << what;
    throw std::runtime_error(oss.str());
}

static void check_hip(hipError_t status, const char *where)
{
    if (status != hipSuccess)
    {
        throw_backend_error(where, hipGetErrorString(status));
    }
}

#ifdef USE_HIPBLAS_BACKEND
static void check_hipblas(hipblasStatus_t status, const char *where)
{
    if (status != HIPBLAS_STATUS_SUCCESS)
    {
        throw_backend_error(where, "hipBLAS status != SUCCESS");
    }
}
#endif

struct DeviceBuffer
{
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count)
    {
        reset(count);
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr)
        {
            (void)hipFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr(other.ptr), count(other.count)
    {
        other.ptr = nullptr;
        other.count = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            if (ptr != nullptr)
            {
                (void)hipFree(ptr);
            }
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    void reset(size_t new_count)
    {
        if (ptr != nullptr)
        {
            (void)hipFree(ptr);
            ptr = nullptr;
        }
        count = new_count;
        if (count > 0)
        {
            check_hip(hipMalloc(&ptr, count * sizeof(double)), "hipMalloc");
        }
    }

    double *get() { return ptr; }
    const double *get() const { return ptr; }

    double *ptr = nullptr;
    size_t count = 0;
};

static double *matrix_data(Matrix &M)
{
    return M.rows() == 0 || M.cols() == 0 ? nullptr : &M.at(0, 0);
}

static const double *matrix_data(const Matrix &M)
{
    return M.rows() == 0 || M.cols() == 0 ? nullptr : &M.at(0, 0);
}

static void zero_column_below_diag_host(Matrix &B, int k)
{
    for (int i = k + 1; i < B.rows(); ++i)
    {
        B.at(i, k) = 0.0;
    }
}

static void zero_row_right_of_superdiag_host(Matrix &B, int k)
{
    for (int j = k + 2; j < B.cols(); ++j)
    {
        B.at(k, j) = 0.0;
    }
}

__global__ static void gemv_transposed_row_major_kernel(const double *A,
                                                        int lda,
                                                        int rows,
                                                        int cols,
                                                        const double *v,
                                                        double *w)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols)
    {
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < rows; ++i)
    {
        sum += v[i] * A[i * lda + col];
    }
    w[col] = sum;
}

__global__ static void gemv_row_major_kernel(const double *A,
                                             int lda,
                                             int rows,
                                             int cols,
                                             const double *v,
                                             double *w)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    double sum = 0.0;
    const double *row_ptr = A + row * lda;
    for (int j = 0; j < cols; ++j)
    {
        sum += row_ptr[j] * v[j];
    }
    w[row] = sum;
}

__global__ static void ger_row_major_kernel(double *A,
                                            int lda,
                                            int rows,
                                            int cols,
                                            const double *lhs,
                                            const double *rhs,
                                            double alpha)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols)
    {
        return;
    }

    A[row * lda + col] += alpha * lhs[row] * rhs[col];
}

__global__ static void zero_column_tail_kernel(double *A, int lda, int row0, int row1, int col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = row0 + idx;
    if (row < row1)
    {
        A[row * lda + col] = 0.0;
    }
}

__global__ static void zero_row_tail_kernel(double *A, int lda, int row, int col0, int col1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col0 + idx;
    if (col < col1)
    {
        A[row * lda + col] = 0.0;
    }
}

struct HipBidiagBackend
{
    HipBidiagBackend(const Matrix &A, Matrix &U_ref, Matrix &V_ref)
        : m(A.rows()),
          n(A.cols()),
          dB(static_cast<size_t>(m) * static_cast<size_t>(n)),
          dU(static_cast<size_t>(m) * static_cast<size_t>(m)),
          dV(static_cast<size_t>(n) * static_cast<size_t>(n)),
          dv(static_cast<size_t>(std::max(m, n))),
          dw(static_cast<size_t>(std::max(m, n)))
    {
#ifdef USE_HIPBLAS_BACKEND
        check_hipblas(hipblasCreate(&handle), "hipblasCreate");
#endif
        upload_matrix(A, dB.get(), m, n);
        upload_matrix(U_ref, dU.get(), m, m);
        upload_matrix(V_ref, dV.get(), n, n);
    }

    ~HipBidiagBackend()
    {
#ifdef USE_HIPBLAS_BACKEND
        if (handle != nullptr)
        {
            hipblasDestroy(handle);
        }
#endif
    }

    void download(Matrix &B, Matrix &U, Matrix &V)
    {
        download_matrix(dB.get(), B, m, n);
        download_matrix(dU.get(), U, m, m);
        download_matrix(dV.get(), V, n, n);
    }

    void fetch_column_tail_from_b(int row0, int col, std::vector<double> &out)
    {
        out.resize(m - row0);
        check_hip(hipMemcpy2D(out.data(),
                              sizeof(double),
                              dB.get() + static_cast<size_t>(row0) * n + col,
                              static_cast<size_t>(n) * sizeof(double),
                              sizeof(double),
                              out.size(),
                              hipMemcpyDeviceToHost),
                  "hipMemcpy2D column tail");
    }

    void fetch_row_tail_from_b(int row, int col0, std::vector<double> &out)
    {
        out.resize(n - col0);
        check_hip(hipMemcpy(out.data(),
                            dB.get() + static_cast<size_t>(row) * n + col0,
                            out.size() * sizeof(double),
                            hipMemcpyDeviceToHost),
                  "hipMemcpy row tail");
    }

    void apply_left_householder_to_b(int k, const HouseholderData &hh)
    {
        const int rows = static_cast<int>(hh.v.size());
        const int cols = n - k;
        copy_vector_to_device(hh.v);

#ifdef USE_HIPBLAS_BACKEND
        const double one = 1.0;
        const double zero = 0.0;
        const double alpha = -hh.beta;
        double *sub = dB.get() + static_cast<size_t>(k) * n + k;

        check_hipblas(hipblasDgemv(handle, HIPBLAS_OP_N, cols, rows, &one, sub, n, dv.get(), 1, &zero, dw.get(), 1),
                      "hipblasDgemv left B");
        check_hipblas(hipblasDger(handle, cols, rows, &alpha, dw.get(), 1, dv.get(), 1, sub, n),
                      "hipblasDger left B");
#else
        launch_gemv_transposed(dB.get() + static_cast<size_t>(k) * n + k, n, rows, cols);
        launch_ger(dB.get() + static_cast<size_t>(k) * n + k, n, rows, cols, dv.get(), dw.get(), -hh.beta);
#endif
    }

    void accumulate_left_householder_to_u(int k, const HouseholderData &hh)
    {
        const int cols = static_cast<int>(hh.v.size());
        copy_vector_to_device(hh.v);

#ifdef USE_HIPBLAS_BACKEND
        const double one = 1.0;
        const double zero = 0.0;
        const double alpha = -hh.beta;
        double *sub = dU.get() + k;

        check_hipblas(hipblasDgemv(handle, HIPBLAS_OP_T, cols, m, &one, sub, m, dv.get(), 1, &zero, dw.get(), 1),
                      "hipblasDgemv left U");
        check_hipblas(hipblasDger(handle, cols, m, &alpha, dv.get(), 1, dw.get(), 1, sub, m),
                      "hipblasDger left U");
#else
        launch_gemv(dU.get() + k, m, m, cols);
        launch_ger(dU.get() + k, m, m, cols, dw.get(), dv.get(), -hh.beta);
#endif
    }

    void apply_right_householder_to_b(int k, const HouseholderData &hh)
    {
        const int rows = m - k;
        const int cols = static_cast<int>(hh.v.size());
        copy_vector_to_device(hh.v);

#ifdef USE_HIPBLAS_BACKEND
        const double one = 1.0;
        const double zero = 0.0;
        const double alpha = -hh.beta;
        double *sub = dB.get() + static_cast<size_t>(k) * n + (k + 1);

        check_hipblas(hipblasDgemv(handle, HIPBLAS_OP_T, cols, rows, &one, sub, n, dv.get(), 1, &zero, dw.get(), 1),
                      "hipblasDgemv right B");
        check_hipblas(hipblasDger(handle, cols, rows, &alpha, dv.get(), 1, dw.get(), 1, sub, n),
                      "hipblasDger right B");
#else
        launch_gemv(dB.get() + static_cast<size_t>(k) * n + (k + 1), n, rows, cols);
        launch_ger(dB.get() + static_cast<size_t>(k) * n + (k + 1), n, rows, cols, dw.get(), dv.get(), -hh.beta);
#endif
    }

    void accumulate_right_householder_to_v(int k, const HouseholderData &hh)
    {
        const int cols = static_cast<int>(hh.v.size());
        copy_vector_to_device(hh.v);

#ifdef USE_HIPBLAS_BACKEND
        const double one = 1.0;
        const double zero = 0.0;
        const double alpha = -hh.beta;
        double *sub = dV.get() + (k + 1);

        check_hipblas(hipblasDgemv(handle, HIPBLAS_OP_T, cols, n, &one, sub, n, dv.get(), 1, &zero, dw.get(), 1),
                      "hipblasDgemv right V");
        check_hipblas(hipblasDger(handle, cols, n, &alpha, dv.get(), 1, dw.get(), 1, sub, n),
                      "hipblasDger right V");
#else
        launch_gemv(dV.get() + (k + 1), n, n, cols);
        launch_ger(dV.get() + (k + 1), n, n, cols, dw.get(), dv.get(), -hh.beta);
#endif
    }

    void zero_column_below_diag(int k)
    {
        const int count = m - (k + 1);
        if (count <= 0)
        {
            return;
        }
        const int block = 256;
        const int grid = (count + block - 1) / block;
        hipLaunchKernelGGL(zero_column_tail_kernel, dim3(grid), dim3(block), 0, 0, dB.get(), n, k + 1, m, k);
        check_hip(hipGetLastError(), "zero_column_tail_kernel");
    }

    void zero_row_right_of_superdiag(int k)
    {
        const int count = n - (k + 2);
        if (count <= 0)
        {
            return;
        }
        const int block = 256;
        const int grid = (count + block - 1) / block;
        hipLaunchKernelGGL(zero_row_tail_kernel, dim3(grid), dim3(block), 0, 0, dB.get(), n, k, k + 2, n);
        check_hip(hipGetLastError(), "zero_row_tail_kernel");
    }

    void sync()
    {
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

    int m;
    int n;
    DeviceBuffer dB;
    DeviceBuffer dU;
    DeviceBuffer dV;
    DeviceBuffer dv;
    DeviceBuffer dw;

#ifdef USE_HIPBLAS_BACKEND
    hipblasHandle_t handle = nullptr;
#endif

private:
    void upload_matrix(const Matrix &src, double *dst, int rows, int cols)
    {
        check_hip(hipMemcpy(dst, matrix_data(src), static_cast<size_t>(rows) * cols * sizeof(double), hipMemcpyHostToDevice),
                  "hipMemcpy HostToDevice");
    }

    void download_matrix(const double *src, Matrix &dst, int rows, int cols)
    {
        check_hip(hipMemcpy(matrix_data(dst), src, static_cast<size_t>(rows) * cols * sizeof(double), hipMemcpyDeviceToHost),
                  "hipMemcpy DeviceToHost");
    }

    void copy_vector_to_device(const std::vector<double> &v)
    {
        check_hip(hipMemcpy(dv.get(), v.data(), v.size() * sizeof(double), hipMemcpyHostToDevice),
                  "hipMemcpy vector to device");
    }

#ifdef USE_HIP_KERNEL_BACKEND
    void launch_gemv_transposed(double *sub, int lda, int rows, int cols)
    {
        const int block = 256;
        const int grid = (cols + block - 1) / block;
        hipLaunchKernelGGL(gemv_transposed_row_major_kernel, dim3(grid), dim3(block), 0, 0, sub, lda, rows, cols, dv.get(), dw.get());
        check_hip(hipGetLastError(), "gemv_transposed_row_major_kernel");
    }

    void launch_gemv(double *sub, int lda, int rows, int cols)
    {
        const int block = 256;
        const int grid = (rows + block - 1) / block;
        hipLaunchKernelGGL(gemv_row_major_kernel, dim3(grid), dim3(block), 0, 0, sub, lda, rows, cols, dv.get(), dw.get());
        check_hip(hipGetLastError(), "gemv_row_major_kernel");
    }

    void launch_ger(double *sub,
                    int lda,
                    int rows,
                    int cols,
                    const double *lhs,
                    const double *rhs,
                    double alpha)
    {
        const dim3 block(16, 16);
        const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        hipLaunchKernelGGL(ger_row_major_kernel, grid, block, 0, 0, sub, lda, rows, cols, lhs, rhs, alpha);
        check_hip(hipGetLastError(), "ger_row_major_kernel");
    }
#endif
};

static Matrix to_bidiagonal_gpu(const Matrix &A, Matrix &U, Matrix &V)
{
    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
    {
        U.at(i, i) = 1.0;
    }

    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        V.at(i, i) = 1.0;
    }

    HipBidiagBackend backend(B, U, V);

    for (int k = 0; k < n; ++k)
    {
        if (k < m - 1)
        {
            std::vector<double> x;
            backend.fetch_column_tail_from_b(k, k, x);
            HouseholderData left_hh = make_householder(x);
            if (left_hh.active)
            {
                backend.apply_left_householder_to_b(k, left_hh);
                backend.accumulate_left_householder_to_u(k, left_hh);
            }
            backend.zero_column_below_diag(k);
            zero_column_below_diag_host(B, k);
        }

        if (k < n - 2)
        {
            std::vector<double> y;
            backend.fetch_row_tail_from_b(k, k + 1, y);
            HouseholderData right_hh = make_householder(y);
            if (right_hh.active)
            {
                backend.apply_right_householder_to_b(k, right_hh);
                backend.accumulate_right_householder_to_v(k, right_hh);
            }
            backend.zero_row_right_of_superdiag(k);
            zero_row_right_of_superdiag_host(B, k);
        }
    }

    backend.sync();
    backend.download(B, U, V);

    for (int k = 0; k < n; ++k)
    {
        zero_column_below_diag_host(B, k);
        if (k < n - 2)
        {
            zero_row_right_of_superdiag_host(B, k);
        }
    }

    return B;
}
#endif

#ifdef __ARM_NEON
static double horizontal_sum(float64x2_t x)
{
    double tmp[2];
    vst1q_f64(tmp, x);
    return tmp[0] + tmp[1];
}
#endif

static double dot_product(const double *lhs, const double *rhs, int len)
{
    double sum = 0.0;

#ifdef __ARM_NEON
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 7 < len; i += 8)
    {
        float64x2_t l0 = vld1q_f64(lhs + i);
        float64x2_t l1 = vld1q_f64(lhs + i + 2);
        float64x2_t l2 = vld1q_f64(lhs + i + 4);
        float64x2_t l3 = vld1q_f64(lhs + i + 6);
        float64x2_t r0 = vld1q_f64(rhs + i);
        float64x2_t r1 = vld1q_f64(rhs + i + 2);
        float64x2_t r2 = vld1q_f64(rhs + i + 4);
        float64x2_t r3 = vld1q_f64(rhs + i + 6);
        acc0 = vaddq_f64(acc0, vmulq_f64(l0, r0));
        acc1 = vaddq_f64(acc1, vmulq_f64(l1, r1));
        acc2 = vaddq_f64(acc2, vmulq_f64(l2, r2));
        acc3 = vaddq_f64(acc3, vmulq_f64(l3, r3));
    }
    for (; i + 3 < len; i += 4)
    {
        float64x2_t l0 = vld1q_f64(lhs + i);
        float64x2_t l1 = vld1q_f64(lhs + i + 2);
        float64x2_t r0 = vld1q_f64(rhs + i);
        float64x2_t r1 = vld1q_f64(rhs + i + 2);
        acc0 = vaddq_f64(acc0, vmulq_f64(l0, r0));
        acc1 = vaddq_f64(acc1, vmulq_f64(l1, r1));
    }
    for (; i + 1 < len; i += 2)
    {
        float64x2_t l0 = vld1q_f64(lhs + i);
        float64x2_t r0 = vld1q_f64(rhs + i);
        acc0 = vaddq_f64(acc0, vmulq_f64(l0, r0));
    }

    sum += horizontal_sum(acc0);
    sum += horizontal_sum(acc1);
    sum += horizontal_sum(acc2);
    sum += horizontal_sum(acc3);

    for (; i < len; ++i)
    {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#else
    for (int i = 0; i < len; ++i)
    {
        sum += lhs[i] * rhs[i];
    }
    return sum;
#endif
}

static void axpy_inplace(double alpha, const double *x, double *y, int len)
{
#ifdef __ARM_NEON
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    int i = 0;
    for (; i + 7 < len; i += 8)
    {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t x2 = vld1q_f64(x + i + 4);
        float64x2_t x3 = vld1q_f64(x + i + 6);
        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        float64x2_t y2 = vld1q_f64(y + i + 4);
        float64x2_t y3 = vld1q_f64(y + i + 6);
        y0 = vaddq_f64(y0, vmulq_f64(alpha_vec, x0));
        y1 = vaddq_f64(y1, vmulq_f64(alpha_vec, x1));
        y2 = vaddq_f64(y2, vmulq_f64(alpha_vec, x2));
        y3 = vaddq_f64(y3, vmulq_f64(alpha_vec, x3));
        vst1q_f64(y + i, y0);
        vst1q_f64(y + i + 2, y1);
        vst1q_f64(y + i + 4, y2);
        vst1q_f64(y + i + 6, y3);
    }
    for (; i + 3 < len; i += 4)
    {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t x1 = vld1q_f64(x + i + 2);
        float64x2_t y0 = vld1q_f64(y + i);
        float64x2_t y1 = vld1q_f64(y + i + 2);
        y0 = vaddq_f64(y0, vmulq_f64(alpha_vec, x0));
        y1 = vaddq_f64(y1, vmulq_f64(alpha_vec, x1));
        vst1q_f64(y + i, y0);
        vst1q_f64(y + i + 2, y1);
    }
    for (; i + 1 < len; i += 2)
    {
        float64x2_t x0 = vld1q_f64(x + i);
        float64x2_t y0 = vld1q_f64(y + i);
        y0 = vaddq_f64(y0, vmulq_f64(alpha_vec, x0));
        vst1q_f64(y + i, y0);
    }
    for (; i < len; ++i)
    {
        y[i] += alpha * x[i];
    }
#else
    for (int i = 0; i < len; ++i)
    {
        y[i] += alpha * x[i];
    }
#endif
}

static double squared_norm(const std::vector<double> &v)
{
    if (v.empty())
    {
        return 0.0;
    }
    return dot_product(v.data(), v.data(), static_cast<int>(v.size()));
}

static double vector_norm(const std::vector<double> &v)
{
    return std::sqrt(squared_norm(v));
}

static std::vector<double> extract_column_tail(const Matrix &A, int row0, int col)
{
    std::vector<double> out(A.rows() - row0);
    for (int i = 0; i < static_cast<int>(out.size()); ++i)
    {
        out[i] = A.at(row0 + i, col);
    }
    return out;
}

static std::vector<double> extract_row_tail(const Matrix &A, int row, int col0)
{
    std::vector<double> out(A.cols() - col0);
    for (int j = 0; j < static_cast<int>(out.size()); ++j)
    {
        out[j] = A.at(row, col0 + j);
    }
    return out;
}

static HouseholderData make_householder(const std::vector<double> &x)
{
    HouseholderData hh;
    const double norm_x = vector_norm(x);
    if (norm_x <= 1e-14)
    {
        return hh;
    }

    hh.v = x;
    const double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
    hh.v[0] += sigma;

    const double vtv = squared_norm(hh.v);
    if (vtv <= 1e-28)
    {
        hh.v.clear();
        return hh;
    }

    hh.beta = 2.0 / vtv;
    hh.active = true;
    return hh;
}

static void gemv_transposed_submatrix(const Matrix &A,
                                      int row0,
                                      int col0,
                                      const std::vector<double> &v,
                                      std::vector<double> &w)
{
    const int rows = static_cast<int>(v.size());
    const int cols = static_cast<int>(w.size());
    std::fill(w.begin(), w.end(), 0.0);

    for (int i = 0; i < rows; ++i)
    {
        const double *row_ptr = &A.at(row0 + i, col0);
        axpy_inplace(v[i], row_ptr, w.data(), cols);
    }
}

static void gemv_submatrix(const Matrix &A,
                           int row0,
                           int col0,
                           int rows,
                           const std::vector<double> &v,
                           std::vector<double> &w)
{
    const int cols = static_cast<int>(v.size());
    for (int i = 0; i < rows; ++i)
    {
        const double *row_ptr = &A.at(row0 + i, col0);
        w[i] = dot_product(row_ptr, v.data(), cols);
    }
}

static void ger_submatrix(Matrix &A,
                          int row0,
                          int col0,
                          const std::vector<double> &lhs,
                          const std::vector<double> &rhs,
                          double alpha)
{
    const int rows = static_cast<int>(lhs.size());
    const int cols = static_cast<int>(rhs.size());
    for (int i = 0; i < rows; ++i)
    {
        double *row_ptr = &A.at(row0 + i, col0);
        axpy_inplace(alpha * lhs[i], rhs.data(), row_ptr, cols);
    }
}

static void apply_left_householder_to_b(Matrix &B, int k, const HouseholderData &hh)
{
    std::vector<double> w(B.cols() - k, 0.0);
    gemv_transposed_submatrix(B, k, k, hh.v, w);
    ger_submatrix(B, k, k, hh.v, w, -hh.beta);
}

static void accumulate_left_householder_to_u(Matrix &U, int k, const HouseholderData &hh)
{
    std::vector<double> w(U.rows(), 0.0);
    gemv_submatrix(U, 0, k, U.rows(), hh.v, w);
    ger_submatrix(U, 0, k, w, hh.v, -hh.beta);
}

static void apply_right_householder_to_b(Matrix &B, int k, const HouseholderData &hh)
{
    std::vector<double> w(B.rows() - k, 0.0);
    gemv_submatrix(B, k, k + 1, B.rows() - k, hh.v, w);
    ger_submatrix(B, k, k + 1, w, hh.v, -hh.beta);
}

static void accumulate_right_householder_to_v(Matrix &V, int k, const HouseholderData &hh)
{
    std::vector<double> w(V.rows(), 0.0);
    gemv_submatrix(V, 0, k + 1, V.rows(), hh.v, w);
    ger_submatrix(V, 0, k + 1, w, hh.v, -hh.beta);
}

} // namespace

static Matrix to_bidiagonal_cpu(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
    {
        U.at(i, i) = 1.0;
    }

    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        V.at(i, i) = 1.0;
    }

    for (int k = 0; k < n; ++k)
    {
        if (k < m - 1)
        {
            HouseholderData left_hh = make_householder(extract_column_tail(B, k, k));
            if (left_hh.active)
            {
                apply_left_householder_to_b(B, k, left_hh);
                accumulate_left_householder_to_u(U, k, left_hh);
            }
        }

        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        if (k < n - 2)
        {
            HouseholderData right_hh = make_householder(extract_row_tail(B, k, k + 1));
            if (right_hh.active)
            {
                apply_right_householder_to_b(B, k, right_hh);
                accumulate_right_householder_to_v(V, k, right_hh);
            }

            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}

Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

#if defined(USE_HIPBLAS_BACKEND) || defined(USE_HIP_KERNEL_BACKEND)
    return to_bidiagonal_gpu(A, U, V);
#else
    return to_bidiagonal_cpu(A, U, V);
#endif
}
