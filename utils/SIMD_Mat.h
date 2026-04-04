#pragma once

#include <cstddef>
#include <iostream>

using std::size_t;

namespace SIMD {
   
    void add_avx(float* dst, const float* a, const float* b, size_t n);

    // 计算向量点积
    float dot_avx(const float* a, const float* b, size_t n);

    // 矩阵向量乘法
    // A: M * N
    // x: N
    // out: M
    void matvec_avx(const float* Mat_A, const float* vec_x, float* out, int M, int N);

    // 矩阵乘法
    // A: M * K
    // B: K * N
    // C: M * N
    void matmul_avx(const float* Mat_A, const float* Mat_B, float* out_C, int M, int K, int N);

    // y: M
    // a: scalar
    // x: M
    void axpy_avx(float* y, float a, const float* x, size_t n);

    // x: M
    // a: scalar
    void scal_avx(float* x, float a, size_t n);

    // dst: M
    // src: M
    void add_inplace_avx(float* dst, const float* src, size_t n);

}  // namespace SIMD
