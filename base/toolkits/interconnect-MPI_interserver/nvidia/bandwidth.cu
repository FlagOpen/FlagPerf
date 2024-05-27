#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <vector>
#include <iostream>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (4ULL * GB)
#define WARMUP_ITERATIONS 200
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
        fprintf(stderr, "MPI Error: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    checkMPIError(MPI_Init(&argc, &argv), "MPI_Init");

    int rank, size;
    checkMPIError(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    int num_gpus_per_node = 8;
    int total_gpus = size;
    int gpu_id = rank % num_gpus_per_node;

    cudaEvent_t start, end;
    float elapsed_time;
    float* d_src;
    float* d_dst;
    ncclComm_t comm;
    cudaStream_t stream;

    checkCudaError(cudaSetDevice(gpu_id), "cudaSetDevice");
    checkCudaError(cudaMalloc(&d_src, SIZE), "cudaMalloc");
    checkCudaError(cudaMalloc(&d_dst, SIZE), "cudaMalloc");
    checkCudaError(cudaMemset(d_src, 1.0f, SIZE), "cudaMemset");
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

    ncclUniqueId id;
    if (rank == 0) checkNcclError(ncclGetUniqueId(&id), "ncclGetUniqueId");
    checkMPIError(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast");
    checkNcclError(ncclCommInitRank(&comm, total_gpus, id, rank), "ncclCommInitRank");
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkNcclError(ncclAllReduce((const void*)d_src, (void*)d_dst, SIZE / sizeof(float), ncclFloat, ncclSum, comm, stream), "ncclAllReduce");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    checkCudaError(cudaEventRecord(start), "cudaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkNcclError(ncclAllReduce((const void*)d_src, (void*)d_dst, SIZE / sizeof(float), ncclFloat, ncclSum, comm, stream), "ncclAllReduce");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    checkCudaError(cudaEventRecord(end), "cudaEventRecord"); 
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");
    /*
    The following are the three performance metrics commonly used
        1. samples/s (algbw): This metric measures the number of samples processed per second, indicating the algorithmic bandwidth. It reflects the computational efficiency of the algorithm.
        2. busbw: This metric represents the bus bandwidth, which measures the data transfer rate across the system's bus. It is crucial for understanding the communication efficiency between different parts of the system.
        3. busbw * 2: This metric is an extension of busbw, accounting for bidirectional data transfer. It doubles the bus bandwidth to reflect the full duplex capability of the system.
    The second metric, busbw, is chosen for the following reasons:
        1. This number is obtained applying a formula to the algorithm bandwidth to reflect the speed of the inter-GPU communication. Using this bus bandwidth, we can compare it with the hardware peak bandwidth, independently of the number of ranks used.
    The following is the derivation:
        algbw = S/t
    Considering that each rank has a bandwidth to the outside world of B, the time to perform an allReduce operation of S elements is at best :
        t = (S*2*(n-1)) / (n*B)
    Indeed, we have S elements, 2*(n-1) operations per element, and n links of bandwidth B to perform them. Reordering the equation, we find that
        t = (S/B) * (2*(n-1)/n)
    Therefore, to get an AllReduce bandwidth measurement which we can compare to the hardware peak bandwidth, we compute :
        B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
    More details can be found in https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    The final calculation is the unidirectional bandwidth.
    */
    double algbw = SIZE * ITERATIONS / (elapsed_time / 1000.0);
    double bandwidth = algbw * (2.0 * (total_gpus - 1) / total_gpus);
    if (rank == 0) {
        printf("[FlagPerf Result]interconnect-MPI_interserver-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
        printf("[FlagPerf Result]interconnect-MPI_interserver-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));
    }
    checkCudaError(cudaFree(d_src), "cudaFree");
    checkCudaError(cudaFree(d_dst), "cudaFree");
    checkNcclError(ncclCommDestroy(comm), "ncclCommDestroy");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");
    checkMPIError(MPI_Finalize(), "MPI_Finalize");
    return 0;
}
