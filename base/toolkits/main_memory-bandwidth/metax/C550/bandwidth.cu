// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <stdio.h>
#include <cuda_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (16ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 1000

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
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
    cudaEvent_t start, end;
    float elapsed_time;

    checkCudaError(cudaMalloc(&d_src, SIZE), "cudaMalloc");
    checkCudaError(cudaMalloc(&d_dst, SIZE), "cudaMalloc");

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    int threadsPerBlock = 1024;
    size_t numElem = SIZE/sizeof(double);
    int blocksPerGrid = (numElem + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
	    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dst, d_src, SIZE);
    }
    cudaDeviceSynchronize();
    checkCudaError(cudaEventRecord(start), "cudaEventRecord");
    for (int i = 0; i < ITERATIONS; ++i) {
	    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dst, d_src, SIZE);
    }
   cudaDeviceSynchronize();
    checkCudaError(cudaEventRecord(end), "cudaEventRecord");
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");

    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(cudaFree(d_src), "cudaFree");
    checkCudaError(cudaFree(d_dst), "cudaFree");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");

    return 0;
}
