// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <stdio.h>
#include <hip/hip_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (16ULL * GB)
#define WARMUP_ITERATIONS 10
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

    checkCudaError(hipMalloc(&d_src, SIZE), "hipMalloc");
    checkCudaError(hipMalloc(&d_dst, SIZE), "hipMalloc");

    checkCudaError(hipEventCreate(&start), "hipEventCreate");
    checkCudaError(hipEventCreate(&end), "hipEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkCudaError(hipMemcpy(d_dst, d_src, SIZE, hipMemcpyDeviceToDevice), "hipMemcpy");
    }

    checkCudaError(hipEventRecord(start), "hipEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkCudaError(hipMemcpy(d_dst, d_src, SIZE, hipMemcpyDeviceToDevice), "hipMemcpy");
    }

    checkCudaError(hipEventRecord(end), "hipEventRecord");
    checkCudaError(hipEventSynchronize(end), "hipEventSynchronize");

    checkCudaError(hipEventElapsedTime(&elapsed_time, start, end), "hipEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(hipFree(d_src), "hipFree");
    checkCudaError(hipFree(d_dst), "hipFree");
    checkCudaError(hipEventDestroy(start), "hipEventDestroy");
    checkCudaError(hipEventDestroy(end), "hipEventDestroy");

    return 0;
}
