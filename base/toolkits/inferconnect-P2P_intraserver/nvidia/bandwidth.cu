// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE (16ULL * 1024ULL * 1024ULL * sizeof(float))
#define WARMUP_ITERATIONS 50
#define ITERATIONS 2000

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_src, *d_dst;
    cudaEvent_t start, end;
    float elapsed_time;
    int gpu_n;
    checkCudaError(cudaGetDeviceCount(&gpu_n), "cudaGetDeviceCount");
    printf("[FlagPerf Info]CUDA-capable device count: %i\n", gpu_n);
    if (gpu_n < 2) {
        fprintf(stderr, "Two or more GPUs with Peer-to-Peer access capability are required for inferconnect-P2P_intraserver-bandwidth test\n");
        exit(EXIT_FAILURE);
    }
    int can_access_peer;
    int p2pCapableGPUs[2];  // We take only 1 pair of P2P capable GPUs
    p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

    // Show all the combinations of supported P2P GPUs
    for (int i = 0; i < gpu_n; i++) {
        for (int j = 0; j < gpu_n; j++) {
            if (i == j) {
                continue;
            }
            checkCudaError(cudaDeviceCanAccessPeer(&can_access_peer, i, j), "cudaDeviceCanAccessPeer");
            printf("[FlagPerf Info]> Peer access from (GPU%d) -> (GPU%d) : %s\n",
                    i, j, can_access_peer ? "Yes" : "No");
            if (can_access_peer && p2pCapableGPUs[0] == -1) {
                p2pCapableGPUs[0] = i;
                p2pCapableGPUs[1] = j;
            }
        }
    }
    if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1) {
        printf(
            "[FlagPerf Info]Two or more GPUs with Peer-to-Peer access capability are required for inferconnect-P2P_intraserver-bandwidth test\n");
        printf(
            "[FlagPerf Info]Peer to Peer access is not available amongst GPUs in the system, "
            "waiving test.\n");
        return 0;
    }
    int gpuid[2];
    gpuid[0] = p2pCapableGPUs[0];
    gpuid[1] = p2pCapableGPUs[1];
    printf("[FlagPerf Info]Enabling peer access between GPU%d and GPU%d...\n", gpuid[0],
            gpuid[1]);
    checkCudaError(cudaSetDevice(gpuid[0]), "cudaSetDevice");
    checkCudaError(cudaDeviceEnablePeerAccess(gpuid[1], 0), "cudaDeviceEnablePeerAccess");
    checkCudaError(cudaMalloc(&d_src, SIZE), "cudaMalloc");
    
    checkCudaError(cudaSetDevice(gpuid[1]), "cudaSetDevice");
    checkCudaError(cudaDeviceEnablePeerAccess(gpuid[0], 0), "cudaDeviceEnablePeerAccess");
    checkCudaError(cudaMalloc(&d_dst, SIZE), "cudaMalloc");
    
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (i % 2 == 0) {
            checkCudaError(cudaMemcpy(d_dst, d_src, SIZE, cudaMemcpyDefault), "cudaMemcpy");
        } else {
            checkCudaError(cudaMemcpy(d_src, d_dst, SIZE, cudaMemcpyDefault), "cudaMemcpy");
        }
    }


    checkCudaError(cudaEventRecord(start), "cudaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        if (i % 2 == 0) {
            checkCudaError(cudaMemcpy(d_dst, d_src, SIZE, cudaMemcpyDefault), "cudaMemcpy");
        } else {
            checkCudaError(cudaMemcpy(d_src, d_dst, SIZE, cudaMemcpyDefault), "cudaMemcpy");
        } 
    }

    checkCudaError(cudaEventRecord(end), "cudaEventRecord");
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");

    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]inferconnect-P2P_intraserver-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]inferconnect-P2P_intraserver-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(cudaFree(d_src), "cudaFree");
    checkCudaError(cudaFree(d_dst), "cudaFree");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");

    return 0;
}
