#include <mccl.h>
#include <mpi.h>
#include <musa_runtime.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>

#define SIZE (1024ULL * 1024ULL * 1024ULL * sizeof(float))
#define WARMUP_ITERATIONS 100
#define ITERATIONS 1000

void checkMusaError(musaError_t err, const char* msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "MUSA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkMcclError(mcclResult_t result, const char* msg) {
    if (result != mcclSuccess) {
        fprintf(stderr, "MCCL Error: %s: %s\n", msg, mcclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void checkMPIError(int result, const char* msg) {
    if (result != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(result, error_string, &length);
        fprintf(stderr, "MPI Error: %s: %s\n", msg, error_string);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    float* d_tensor;
    musaEvent_t start, end;
    float elapsed_time;

    checkMPIError(MPI_Init(&argc, &argv), "MPI_Init");
    int rank, nranks;
    checkMPIError(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &nranks), "MPI_Comm_size");
    checkMusaError(musaSetDevice(0), "musaSetDevice");

    mcclComm_t comm;
    musaStream_t stream;

    mcclUniqueId id;
    if (rank == 0) {
        checkMcclError(mcclGetUniqueId(&id), "mcclGetUniqueId");
    }
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    checkMcclError(mcclCommInitRank(&comm, nranks, id, rank), "mcclCommInitRank");
    checkMusaError(musaStreamCreate(&stream), "musaStreamCreate");

    checkMusaError(musaMalloc(&d_tensor, SIZE), "musaMalloc");

    checkMusaError(musaEventCreate(&start), "musaEventCreate");
    checkMusaError(musaEventCreate(&end), "musaEventCreate");

    checkMcclError(mcclGroupStart(), "mcclGroupStart");
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (rank == 0) {
            checkMcclError(
                mcclSend(d_tensor, SIZE / sizeof(float), mcclFloat, 1, comm, stream),
                "mcclSend");
        }
        else if (rank == 1) {
            checkMcclError(
                mcclRecv(d_tensor, SIZE / sizeof(float), mcclFloat, 0, comm, stream),
                "mcclRecv");
        }
    }
    checkMcclError(mcclGroupEnd(), "mcclGroupEnd");
    checkMusaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

    checkMusaError(musaEventRecord(start), "musaEventRecord");
    checkMcclError(mcclGroupStart(), "mcclGroupStart");
    for (int i = 0; i < ITERATIONS; ++i) {
        if (rank == 0) {
            checkMcclError(
                mcclSend(d_tensor, SIZE / sizeof(float), mcclFloat, 1, comm, stream),
                "mcclSend");
        }
        else if (rank == 1) {
            checkMcclError(
                mcclRecv(d_tensor, SIZE / sizeof(float), mcclFloat, 0, comm, stream),
                "mcclRecv");
        }
    }
    checkMcclError(mcclGroupEnd(), "mcclGroupEnd");
    checkMusaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    checkMusaError(musaEventRecord(end), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");
    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end),
        "musaEventElapsedTime");

    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0) +
        SIZE * ITERATIONS / (elapsed_time / 1000.0);
    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth="
        << std::fixed << std::setprecision(2)
        << bandwidth / (1024.0 * 1024.0 * 1024.0) << "GiB/s" << std::endl;

    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth="
        << std::fixed << std::setprecision(2)
        << bandwidth / (1000.0 * 1000.0 * 1000.0) << "GB/s" << std::endl;
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");
    checkMusaError(musaFree(d_tensor), "musaFree");
    checkMcclError(mcclCommDestroy(comm), "mcclCommDestroy");
    checkMusaError(musaStreamDestroy(stream), "musaStreamDestroy");
    checkMPIError(MPI_Finalize(), "MPI_Finalize");
    return 0;
}