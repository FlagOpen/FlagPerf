// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define SIZE (1024ULL * 1024ULL * 1024ULL * sizeof(float))
#define WARMUP_ITERATIONS 100
#define ITERATIONS 2000

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

void checkMPIError(int result, const char *msg) {
    if (result != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(result, error_string, &length);
        fprintf(stderr, "MPI Error: %s: %s\n", msg, error_string);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    float *d_tensor;
    cudaEvent_t start, end;
    float elapsed_time;

    checkMPIError(MPI_Init(&argc, &argv), "MPI_Init");
    int rank, nranks;
    checkMPIError(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &nranks), "MPI_Comm_size");
    checkCudaError(cudaSetDevice(0), "cudaSetDevice");

    ncclComm_t comm;
    cudaStream_t stream;

    ncclUniqueId id;
    if (rank == 0) {
        checkNcclError(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    checkNcclError(ncclCommInitRank(&comm, nranks, id, rank), "ncclCommInitRank");
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    checkCudaError(cudaMalloc(&d_tensor, SIZE), "cudaMalloc");

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    printf("Rank %d: Running...\n", rank);
    checkNcclError(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (rank == 0) {
            checkNcclError(ncclSend(d_tensor, SIZE / sizeof(float), ncclFloat, 1, comm, stream), "ncclSend");
        }
        else if (rank == 1){
            checkNcclError(ncclRecv(d_tensor, SIZE / sizeof(float), ncclFloat, 0, comm, stream), "ncclRecv");
        }
        printf("Rank %d: Warmup iteration %d\n", rank, i);
    }
    checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    printf("Rank %d: Warmup done\n", rank);
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

    printf("Rank %d: Running...\n", rank);
    checkCudaError(cudaEventRecord(start), "cudaEventRecord");
    checkNcclError(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < ITERATIONS; ++i) {
        if (rank == 0) {
            checkNcclError(ncclSend(d_tensor, SIZE / sizeof(float), ncclFloat, 1, comm, stream), "ncclSend");
        }
        else if (rank == 1){
            checkNcclError(ncclRecv(d_tensor, SIZE / sizeof(float), ncclFloat, 0, comm, stream), "ncclRecv");
        }
        printf("Rank %d: Iteration %d\n", rank, i);
    }
    checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    printf("Rank %d: Done\n", rank);
    checkCudaError(cudaEventRecord(end), "cudaEventRecord");
    printf("Rank %d: Recording done\n", rank); 
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    printf("Rank %d: Synchronization done\n", rank);
    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");
    printf("Rank %d: Elapsed time: %.2fms\n", rank, elapsed_time);

    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0);
    printf("[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");
    checkCudaError(cudaFree(d_tensor), "cudaFree");
    checkNcclError(ncclCommDestroy(comm), "ncclCommDestroy");
    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkMPIError(MPI_Finalize(), "MPI_Finalize");
    return 0;
}
