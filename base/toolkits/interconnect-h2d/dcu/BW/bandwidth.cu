// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
#include <stdio.h>
#include <hip/hip_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (16ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 1000

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_src, *d_dst;
    hipEvent_t start, end;
    float elapsed_time;

    checkCudaError(hipHostMalloc(&d_src, SIZE), "hipHostMalloc");
    checkCudaError(hipMalloc(&d_dst, SIZE), "hipMalloc");

    checkCudaError(hipEventCreate(&start), "hipEventCreate");
    checkCudaError(hipEventCreate(&end), "hipEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkCudaError(hipMemcpy(d_dst, d_src, SIZE, hipMemcpyHostToDevice), "hipMemcpy");
    }

    checkCudaError(hipEventRecord(start), "hipEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkCudaError(hipMemcpy(d_dst, d_src, SIZE, hipMemcpyHostToDevice), "hipMemcpy");
    }

    checkCudaError(hipEventRecord(end), "hipEventRecord");
    checkCudaError(hipEventSynchronize(end), "hipEventSynchronize");

    checkCudaError(hipEventElapsedTime(&elapsed_time, start, end), "hipEventElapsedTime");

    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("# Avg Unidirectional  bandwidth :=%.2f GB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));
    printf("# Avg Bidirectional  bandwidth :=%.2f GB/s\n", 2 * bandwidth / (1000.0 * 1000.0 * 1000.0));

    printf("[FlagPerf Result]transfer-bandwidth=%.2fGiB/s\n", 2 * bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]transfer-bandwidth=%.2fGB/s\n", 2 * bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(hipHostFree(d_src), "hipHostFree");
    checkCudaError(hipFree(d_dst), "hipFree");
    checkCudaError(hipEventDestroy(start), "hipEventDestroy");
    checkCudaError(hipEventDestroy(end), "hipEventDestroy");

    return 0;
}