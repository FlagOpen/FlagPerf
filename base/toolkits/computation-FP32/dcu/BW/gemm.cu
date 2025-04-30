// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>

constexpr int M = 2560;
constexpr int N = 8192;
constexpr int K = 65536;

struct PrecisionConfig {
    hipblasDatatype_t cudaType;
    hipblasDatatype_t cublasType;
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, M * K * config.bytesPerElement);
    hipMalloc(&d_B, K * N * config.bytesPerElement);
    if (config.cudaType == HIPBLAS_R_8I) {
        hipMalloc(&d_C, M * N * sizeof(float));
    } else {
        hipMalloc(&d_C, M * N * config.bytesPerElement);
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

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
    double flops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double FLOPS = flops / time_second;
    double TFLOPS = FLOPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-FP32=" << TFLOPS << "TFLOPS" << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    hipblasDestroy(handle);
}

int main() {
    PrecisionConfig fp32 = {
        HIPBLAS_R_32F,
        HIPBLAS_R_32F,
        4,
        "FP32",
        100,
        10
    };

    test(fp32);

    return 0;
}

