#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <vector>
#include <iostream>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (4ULL * GB)
#define WARMUP_ITERATIONS 50
#define ITERATIONS 100

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

    printf("Rank: %d, Size: %d\n", rank, size);

    int num_gpus_per_node = 8; // 每个节点的GPU数量
    int total_gpus = num_gpus_per_node * size; // 总的GPU数量
    int gpu_id = rank % num_gpus_per_node; // 当前进程使用的GPU ID

    cudaEvent_t start, end;
    float elapsed_time;
    float* d_src;
    float* d_dst;
    ncclComm_t comm;
    cudaStream_t stream;

    // 设置当前进程使用的GPU
    checkCudaError(cudaSetDevice(gpu_id), "cudaSetDevice");
    checkCudaError(cudaMalloc(&d_src, SIZE), "cudaMalloc");
    checkCudaError(cudaMalloc(&d_dst, SIZE), "cudaMalloc");
    checkCudaError(cudaMemset(d_src, 1.0f, SIZE), "cudaMemset");
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

    ncclUniqueId id;
    if (rank == 0){
        printf("Generated NCCL unique id on rank 0 st\n");
        checkNcclError(ncclGetUniqueId(&id), "ncclGetUniqueId");
        printf("Generated NCCL unique id on rank 0 ed\n");
    }
    checkMPIError(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast");
    printf("Before init nccl comm rank %d\n", rank);


    // 初始化NCCL通信器
    checkNcclError(ncclCommInitRank(&comm, total_gpus, id, rank), "ncclCommInitRank");
    printf("init nccl comm rank %d\n", rank);

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    printf("start bandwidth test\n");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        checkNcclError(ncclAllReduce((const void*)d_src, (void*)d_dst, SIZE / sizeof(float), ncclFloat, ncclSum, comm, stream), "ncclAllReduce");
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    printf("warmup finished\n");
    checkCudaError(cudaEventRecord(start), "cudaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        checkNcclError(ncclAllReduce((const void*)d_src, (void*)d_dst, SIZE / sizeof(float), ncclFloat, ncclSum, comm, stream), "ncclAllReduce");
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    printf("iterations finished\n");

    checkCudaError(cudaEventRecord(end), "cudaEventRecord"); 
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");

    double algbw = SIZE * ITERATIONS / (elapsed_time / 1000.0);
    double bandwidth = algbw * (2.0 * (total_gpus - 1) / total_gpus);

    double global_bandwidth;
    printf("MPI Reduce\n");
    checkMPIError(MPI_Reduce(&bandwidth, &global_bandwidth, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "MPI_Reduce");
    printf("MPI Reduce finished\n");

    if (rank == 0) {
        printf("[FlagPerf Result]transfer-bandwidth=%.2fGiB/s\n", global_bandwidth / (1024.0 * 1024.0 * 1024.0));
        printf("[FlagPerf Result]transfer-bandwidth=%.2fGB/s\n", global_bandwidth / (1000.0 * 1000.0 * 1000.0));
    }

    checkCudaError(cudaFree(d_src), "cudaFree");
    checkCudaError(cudaFree(d_dst), "cudaFree");
    checkNcclError(ncclCommDestroy(comm), "ncclCommDestroy");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");

    checkMPIError(MPI_Finalize(), "MPI_Finalize");

    return 0;
}
