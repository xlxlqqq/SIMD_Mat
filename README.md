# HighPerformanceComputing

## 项目介绍
本项目是一个基于 SIMD 指令的矩阵乘法实现，用于加速矩阵乘法运算。

## 项目结构

## TODO

高优先级
- axpy_avx(y, a, x, n) ： y = a*x + y ，线代核心基元，很多算法都会用。
- scal_avx(x, a, n) ：向量缩放。
- add_inplace_avx(dst, src, n) ：就地加法，减少额外内存。
- fma_avx(...) ：FMA 版本点积（支持时更快更准）。
- matmul_blocked_avx(...) ：分块矩阵乘，比当前 naive 版更接近高性能。

中优先级
- transpose_avx(src, dst, rows, cols) ：转置单独做成接口，方便复用。
- gemv_avx(A, x, y, M, N, alpha, beta) ：比 matvec 更通用，兼容 BLAS 语义。
- gemm_avx(A, B, C, M, K, N, alpha, beta) ：同上，支持 C = alphaAB + betaC 。
- reduce_sum_avx(x, n) / reduce_max_avx(x, n) ：常见归约操作。

工程化接口（非常推荐）
- bool has_avx2() / bool has_fma() ：运行时能力检测。
- *_auto(...) ：自动分发（AVX2/FMA/标量 fallback）。
- *_aligned(...) 与 *_unaligned(...) ：显式区分对齐路径。
- batched 接口： matmul_batched_avx(...) ，深度学习/批量任务很实用。


# g++ 编译
### main.exe
- 主要文件： main.cpp
- 编译命令： g++ -O2 -mavx2 main.cpp utils/SIMD_Mat.cpp -o main.exe

### test_simd.exe
- 主要文件： test_simd.cpp
- 编译命令： g++ -O2 -mavx2 test_simd.cpp utils/SIMD_Mat.cpp -o test_simd.exe

### SIMD_Mat.dll
- 主要文件： utils/SIMD_Mat.cpp
- 编译命令： g++ -O2 -mavx2 -shared utils/SIMD_Mat.cpp -o SIMD_Mat.dll