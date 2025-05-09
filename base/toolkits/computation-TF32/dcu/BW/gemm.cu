// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <iostream>
#include <chrono>

#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>

constexpr int M = 8192;
constexpr int N = 8896;
constexpr int K = 13312;

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

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_set_math_mode(handle, rocblas_xf32_xdl_math_op);

    // hipblasHandle_t handle;
    // hipblasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {

        rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
                M, N, K,
                &alpha,
                d_A, M,
                d_B, N,
                &beta,
                d_C, M);
    }

    hipError_t syncError = hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != hipSuccess) {
        std::cout << "CUDA error: " << hipGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
          rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
                M, N, K,
                &alpha,
                d_A, M,
                d_B, N,
                &beta,
                d_C, M);
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

    std::cout << "[FlagPerf Result]" << "computation-TF32=" << TFLOPS << "TFLOPS" << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    rocblas_destroy_handle(handle);
}

int main() {
    PrecisionConfig tf32 = {
        HIPBLAS_R_32F,
        HIPBLAS_R_32F,
        4,
        "TF32",
        100,
        10
    };

    test(tf32);

    return 0;
}