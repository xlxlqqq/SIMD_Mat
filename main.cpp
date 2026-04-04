#include <iostream>
#include <vector>

#include "utils/SIMD_Mat.h"

int main() {
    std::cout << "Test SIMD MatMul" << std::endl;
    int m = 2;
    int k = 8;
    int n = 3;
    std::vector<float> A = {1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,9};
    std::vector<float> B = {1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
    };

    std::vector<float> x(n);

    SIMD::matmul_avx(
        A.data(),
        B.data(),
        x.data(),
        m,
        k,
        n
    );

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << x[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
