// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <mcblas/mcblas.h>
#include <mc_runtime.h>
#include <chrono>
#include <iostream>


// constexpr int M = 8192;
// constexpr int N = 8192;
// constexpr int K = 8192;
constexpr int M = 3328;
constexpr int N = 2048;
constexpr int K = 49152;

struct PrecisionConfig {
    macaDataType_t cudaType;
    mcblasComputeType_t cublasType;
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS ;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    int8_t  *d_A, *d_B;
    int32_t *d_C;

    mcMallocManaged(&d_A, M * K * config.bytesPerElement);
    mcMallocManaged(&d_B, K * N * config.bytesPerElement);
    if (config.cudaType == MACA_R_8I) {
        mcMallocManaged(&d_C, M * N * sizeof(float));
    } else {
        mcMallocManaged(&d_C, M * N * config.bytesPerElement);
    }

    mcblasHandle_t handle;
    mcblasCreate(&handle);

    int alpha = 1;
    int beta = 0;

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
        if (config.cudaType == MACA_R_8I) {
            mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, MACA_R_32I, M,
                         config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, K,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    mcError_t syncError = mcDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != mcSuccess) {
        std::cout << "CUDA error: " << mcGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
        if (config.cudaType == MACA_R_8I) {
            mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, MACA_R_32I, M,
                         config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
            // mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_N,
            //              M, N, K, &alpha,
            //              d_A, config.cudaType, M,
            //              d_B, config.cudaType, K,
            //              &beta,
            //              d_C, MACA_R_32I, M,
            //              config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_T,
                         M, N, K, &alpha,
                         d_A, config.cudaType, M,
                         d_B, config.cudaType, N,
                         &beta,
                         d_C, config.cudaType, M,
                         config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
            // mcblasGemmEx(handle, MCBLAS_OP_N, MCBLAS_OP_N,
            //              M, N, K, &alpha,
            //              d_A, config.cudaType, M,
            //              d_B, config.cudaType, K,
            //              &beta,
            //              d_C, config.cudaType, M,
            //              config.cublasType, MCBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    syncError = mcDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (syncError != mcSuccess) {
        std::cout << "CUDA error: " << mcGetErrorString(syncError) << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << config.name << " Single Op Duration: " << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double ops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double OPS = ops / time_second;
    double TOPS = OPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-INT8=" << TOPS << "TOPS" << std::endl;

    mcFree(d_A);
    mcFree(d_B);
    mcFree(d_C);

    mcblasDestroy(handle);
}

int main() {
    PrecisionConfig int8 = {
        MACA_R_8I,
        MCBLAS_COMPUTE_32I,
        1,
        "INT8",
        100,
        10
    };

    test(int8);

    return 0;
}

