// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>


constexpr int M = 8192;
constexpr int N = 8192;
constexpr int K = 8192;

struct PrecisionConfig {
    cudaDataType_t cudaType;
    cublasComputeType_t cublasType;
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS ;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    int8_t  *d_A, *d_B;
    int32_t *d_C;

    cudaMallocManaged(&d_A, M * K * config.bytesPerElement);
    cudaMallocManaged(&d_B, K * N * config.bytesPerElement);
    if (config.cudaType == CUDA_R_8I) {
        cudaMallocManaged(&d_C, M * N * sizeof(float));
    } else {
        cudaMallocManaged(&d_C, M * N * config.bytesPerElement);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    int alpha = 1;
    int beta = 0;

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
        if (config.cudaType == CUDA_R_8I) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, CUDA_R_32I, M,
                         config.cublasType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    cudaError_t syncError = cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
        if (config.cudaType == CUDA_R_8I) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, CUDA_R_32I, M,
                         config.cublasType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    syncError = cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (syncError != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(syncError) << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << config.name << " Single Op Duration: " << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double ops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double OPS = ops / time_second;
    double TOPS = OPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-INT8=" << TOPS << "TOPS" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
}

int main() {
    PrecisionConfig int8 = {
        CUDA_R_8I,
        CUBLAS_COMPUTE_32I,
        1,
        "INT8",
        50000,
        10
    };

    test(int8);

    return 0;
}

