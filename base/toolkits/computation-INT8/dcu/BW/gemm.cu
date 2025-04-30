// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>


constexpr int M = 8192;
constexpr int N = 13312;
constexpr int K = 17792;

struct PrecisionConfig {
    hipblasDatatype_t cudaType;
    hipblasDatatype_t cublasType;
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS ;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    int8_t  *d_A, *d_B;
    int32_t *d_C;

    hipMalloc(&d_A, M * K * config.bytesPerElement);
    hipMalloc(&d_B, K * N * config.bytesPerElement);
    if (config.cudaType == HIPBLAS_R_8I) {
        hipMalloc(&d_C, M * N * sizeof(float));
    } else {
        hipMalloc(&d_C, M * N * config.bytesPerElement);
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    int alpha = 1;
    int beta = 0;

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
        if (config.cudaType == HIPBLAS_R_8I) {
            hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, HIPBLAS_R_32I, M,
                         config.cublasType, HIPBLAS_GEMM_DEFAULT);
        } else {
            hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, HIPBLAS_GEMM_DEFAULT);
        }
    }

    hipError_t syncError = hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != hipSuccess) {
        std::cout << "CUDA error: " << hipGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
        if (config.cudaType == HIPBLAS_R_8I) {
            hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, HIPBLAS_R_32I, M,
                         config.cublasType, HIPBLAS_GEMM_DEFAULT);
        } else {
            hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, HIPBLAS_GEMM_DEFAULT);
        }
    }
    syncError = hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (syncError != hipSuccess) {
        std::cout << "CUDA error: " << hipGetErrorString(syncError) << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << config.name << " Single Op Duration: " << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double ops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double OPS = ops / time_second;
    double TOPS = OPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-INT8=" << TOPS << "TOPS" << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    hipblasDestroy(handle);
}

int main() {
    PrecisionConfig int8 = {
        HIPBLAS_R_8I,
        HIPBLAS_R_32I,
        1,
        "INT8",
        100,
        10
    };

    test(int8);

    return 0;
}

