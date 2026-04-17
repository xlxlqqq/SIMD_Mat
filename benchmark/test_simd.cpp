#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include "../utils/SIMD_Mat.h"

// 测试矩阵乘法的性能, 使用AVX指令集, 并使用SIMD指令进行优化

// 生成随机矩阵
void init_matrix(std::vector<float>& mat, int rows, int cols) {
    std::mt19937 gen(42);  // 固定种子,保证可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

int main() {
    std::cout << "===== SIMD MatMul Benchmark (AVX) =====" << std::endl;

    // 使用较大矩阵才能体现 SIMD 优势
    constexpr int M = 512;
    constexpr int K = 512;
    constexpr int N = 512;
    constexpr int iterations = 10;  // 大矩阵减少迭代次数

    std::cout << "Matrix size: " << M << " x " << K << " * " << K << " x " << N << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N);

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    float checksum = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        SIMD::matmul_avx(A.data(), B.data(), C.data(), M, K, N);
        checksum += C[0] + C[static_cast<size_t>(M) * N - 1];
        asm volatile("" ::: "memory");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double avg_ms = static_cast<double>(total_ms.count()) / iterations;

    // 计算 GFLOPS: 2*M*K*N FLOPs per matmul (multiply + add)
    double flops = 2.0 * M * K * N;
    double gflops = (flops / 1e9) / (avg_ms / 1000.0);

    std::cout << "Total time: " << total_ms.count() << " ms (" << iterations << " iterations)" << std::endl;
    std::cout << "Avg time:  " << avg_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Checksum: " << checksum << std::endl;

    return 0;
}
