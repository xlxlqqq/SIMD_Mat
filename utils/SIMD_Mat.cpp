#include <cstddef>
#include <immintrin.h>
#include <vector>
#include <iostream>

using std::size_t;
using std::cout;
using std::endl;

namespace SIMD {
    void add_avx(float* dst, const float* a, const float* b, size_t n) {
        size_t i = 0;

        size_t simd_end = n & ~size_t(7);  // n 向下取整到8的倍数

        for (; i < simd_end; i += 8) {  // 256位的SIMD指令，每次处理8个float类型的数据
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 b_vec = _mm256_loadu_ps(b + i);
            __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(dst + i, sum_vec);
        }

        for (; i < n; i++) {
            dst[i] = a[i] + b[i];
        }

        return;
    }

    // 计算向量点积
    float dot_avx(const float* a, const float* b, size_t n) {
        size_t i = 0;
        size_t simd_end = n & ~size_t(7);  // n 向下取整到8的倍数
        __m256 acc = _mm256_setzero_ps();
        float dot = 0.0f;
        
        for (; i < simd_end; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 b_vec = _mm256_loadu_ps(b + i);
            __m256 dot_vec = _mm256_mul_ps(a_vec, b_vec);
            acc = _mm256_add_ps(acc, dot_vec);  // 暂时保留256位
        }

        alignas(32) float acc_arr[8];
        _mm256_storeu_ps(acc_arr, acc);

        dot += acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] +
                      acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];

        for (; i < n; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    // 矩阵向量乘法
    // A: M * N
    // x: N
    // out: M
    void matvec_avx(const float* Mat_A, const float* vec_x, float* out, int M, int N) {
        size_t i = 0;
        for (; i < M; i++) {
            out[i] = dot_avx(Mat_A + i * N, vec_x, N);
        }

        return;
    }

    // 矩阵乘法
    // A: M * K
    // B: K * N
    // C: M * N
    void matmul_avx(const float* Mat_A, const float* Mat_B, float* out_C, int M, int K, int N) {
        size_t i = 0;

        std::vector<float> B_T(static_cast<size_t>(N) * K);
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                B_T[j * K + k] = Mat_B[N * k + j];
            }
        }
        
        for (int i = 0; i < M; ++i) {
            matvec_avx(
                B_T.data(),                    // N x K
                Mat_A + static_cast<size_t>(i) * K,// K
                out_C + static_cast<size_t>(i) * N,// N
                N,                             // matvec 的 M
                K                              // matvec 的 N
            );
        }
        return;
    }

    void axpy_avx(float* y, float a, const float* x, size_t n) {
        size_t SIMD_size = n & ~size_t(7);
        
        size_t i = 0;
        __m256 a_vec = _mm256_set1_ps(a);
        for (; i < SIMD_size; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 ax_vec = _mm256_mul_ps(a_vec, x_vec);
            __m256 y_vec = _mm256_loadu_ps(y + i);
            __m256 yax_vec = _mm256_add_ps(y_vec, ax_vec);
            _mm256_storeu_ps(y + i, yax_vec);
        }
        for (; i < n; i++) {
            y[i] = a * x[i] + y[i];
        }
        return;
    }

    void scal_avx(float* x, float a, size_t n) {
        size_t SIMD_size = n & ~size_t(7);
        size_t i = 0;
        __m256 a_vec = _mm256_set1_ps(a);

        for (; i < SIMD_size; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 ax_vec = _mm256_mul_ps(a_vec, x_vec);
            _mm256_storeu_ps(x + i, ax_vec);
        }

        for (; i < n; i++) {
            x[i] = a * x[i];
        }
        return;
    }

    void add_inplace_avx(float* dst, const float* src, size_t n) {
        size_t SIMD_size = n & ~size_t(7);
        size_t i = 0;

        for (; i < SIMD_size; i += 8) {
            __m256 src_vec = _mm256_loadu_ps(src + i);
            __m256 dst_vec = _mm256_loadu_ps(dst + i);
            __m256 sum_vec = _mm256_add_ps(dst_vec, src_vec);
            _mm256_storeu_ps(dst + i, sum_vec);
        }

        for (; i < n; i++) {
            dst[i] += src[i];
        }
        return;
    }

}  // namespace SIMD
