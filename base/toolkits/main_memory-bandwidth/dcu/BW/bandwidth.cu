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

__global__ void copyKernel(void* d_dst, const void* d_src, size_t size) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < size) {
        ((double*)d_dst)[offset] = ((const double*)d_src)[offset];
    }
}

int main() {

    double *d_src, *d_dst;
    hipEvent_t start, end;
    float elapsed_time;

    checkCudaError(hipMalloc(&d_src, SIZE), "hipMalloc");
    checkCudaError(hipMalloc(&d_dst, SIZE), "hipMalloc");

    checkCudaError(hipEventCreate(&start), "hipEventCreate");
    checkCudaError(hipEventCreate(&end), "hipEventCreate");

    int threadsPerBlock = 1024;
    size_t numElem = SIZE/sizeof(double);
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dst, d_src, SIZE);
    }
    hipDeviceSynchronize();
    checkCudaError(hipEventRecord(start), "hipEventRecord");
    for (int i = 0; i < ITERATIONS; ++i) {
        copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dst, d_src, SIZE);
    }
    hipDeviceSynchronize();
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
