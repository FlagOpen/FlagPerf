// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <iostream>


#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (1ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 1000

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkNcclError(ncclResult_t result, const char *msg) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL Error: %s: %s\n", msg, ncclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int num_gpus = 8;
    int devs[num_gpus] = {0, 1, 2, 3, 4, 5, 6, 7};

    cudaEvent_t start, end;
    float elapsed_time;
    std::vector<float*> d_src(num_gpus);
    std::vector<float*> d_dst(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    for (int i = 0; i < num_gpus; ++i) {
        checkCudaError(cudaSetDevice(devs[i]), "cudaSetDevice");
        checkCudaError(cudaMalloc(&d_src[i], SIZE), "cudaMalloc");
        checkCudaError(cudaMalloc(&d_dst[i], SIZE), "cudaMalloc");
        checkCudaError(cudaMemset(d_src[i], 1.0f, SIZE), "cudaMemset");
        checkCudaError(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
    }

    checkNcclError(ncclCommInitAll(comms.data(), num_gpus, devs), "ncclCommInitAll");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        for (int j = 0; j < num_gpus; ++j) {
            checkNcclError(ncclAllReduce(d_src[j], d_dst[j], SIZE, ncclFloat, ncclSum, comms[j], streams[j]), "ncclAllReduce");
        }
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        for (int j = 0; j < num_gpus; ++j){
            checkCudaError(cudaStreamSynchronize(streams[j]), "cudaStreamSynchronize");
        } 
    }

    checkCudaError(cudaEventRecord(start), "cudaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        for (int j = 0; j < num_gpus; ++j) {
            checkNcclError(ncclAllReduce(d_src[j], d_dst[j], SIZE, ncclFloat, ncclSum, comms[j], streams[j]), "ncclAllReduce");
        }
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        for (int j = 0; j < num_gpus; ++j){
            checkCudaError(cudaStreamSynchronize(streams[j]), "cudaStreamSynchronize");
        }
    } 

    checkCudaError(cudaEventRecord(end), "cudaEventRecord");
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");

    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");

    /*
        algbw = S/t
    Considering that each rank has a bandwidth to the outside world of B, the time to perform an allReduce operation of S elements is at best :
        t = (S*2*(n-1)) / (n*B)
    Indeed, we have S elements, 2*(n-1) operations per element, and n links of bandwidth B to perform them. Reordering the equation, we find that
        t = (S/B) * (2*(n-1)/n)
    Therefore, to get an AllReduce bandwidth measurement which we can compare to the hardware peak bandwidth, we compute :
        B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
    More details can be found in https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    The final calculation is the two-way bandwidth, so we multiply by 2.
    */
    double algbw = SIZE * ITERATIONS / (elapsed_time / 1000.0);
    double bandwidth = algbw * (2 * (num_gpus-1) / num_gpus);
    bandwidth = bandwidth * 2;

    printf("[FlagPerf Result]transfer-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]transfer-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    for (int i = 0; i < num_gpus; ++i) {
        checkCudaError(cudaFree(d_src[i]), "cudaFree");
        checkCudaError(cudaFree(d_dst[i]), "cudaFree");
        checkNcclError(ncclCommDestroy(comms[i]), "ncclCommDestroy");
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");
    return 0;
}