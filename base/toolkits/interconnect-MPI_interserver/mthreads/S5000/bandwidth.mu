#include <iomanip>
#include <iostream>
#include <mccl.h>
#include <mpi.h>
#include <musa_runtime.h>
#include <stdio.h>
#include <vector>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (4ULL * GB)
#define WARMUP_ITERATIONS 10
#define ITERATIONS 400

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
    fprintf(stderr, "MPI Error: %s\n", msg);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  checkMPIError(MPI_Init(&argc, &argv), "MPI_Init");

  int rank, size;
  checkMPIError(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
  checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

  int num_gpus_per_node = 8;
  int total_gpus = size;
  int gpu_id = rank % num_gpus_per_node;

  musaEvent_t start, end;
  float elapsed_time;
  float* d_src;
  float* d_dst;
  mcclComm_t comm;
  musaStream_t stream;

  checkMusaError(musaSetDevice(gpu_id), "musaSetDevice");
  checkMusaError(musaMalloc(&d_src, SIZE), "musaMalloc");
  checkMusaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

  std::vector<float> host_data(SIZE / sizeof(float), 1.0f);
  checkMusaError(musaMemcpy(d_src, host_data.data(), SIZE, musaMemcpyHostToDevice), "musaMemcpy");

  // checkMusaError(musaMemset(d_src, 1.0f, SIZE), "musaMemset");
  checkMusaError(musaStreamCreate(&stream), "musaStreamCreate");

  mcclUniqueId id;
  if (rank == 0)
    checkMcclError(mcclGetUniqueId(&id), "mcclGetUniqueId");
  checkMPIError(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
    "MPI_Bcast");
  checkMcclError(mcclCommInitRank(&comm, total_gpus, id, rank),
    "mcclCommInitRank");
  checkMusaError(musaEventCreate(&start), "musaEventCreate");
  checkMusaError(musaEventCreate(&end), "musaEventCreate");

  for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
    checkMcclError(mcclAllReduce((const void*)d_src, (void*)d_dst,
      SIZE / sizeof(float), mcclFloat, mcclSum, comm,
      stream),
      "mcclAllReduce");
    checkMusaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
  }
  checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
  checkMusaError(musaEventRecord(start), "musaEventRecord");

  for (int i = 0; i < ITERATIONS; ++i) {
    checkMcclError(mcclAllReduce((const void*)d_src, (void*)d_dst,
      SIZE / sizeof(float), mcclFloat, mcclSum, comm,
      stream),
      "mcclAllReduce");
    checkMusaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
  }
  checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
  checkMusaError(musaEventRecord(end), "musaEventRecord");
  checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");
  checkMusaError(musaEventElapsedTime(&elapsed_time, start, end),
    "musaEventElapsedTime");
  
  double algbw = SIZE * ITERATIONS / (elapsed_time / 1000.0);
  double bandwidth = algbw * (2.0 * (total_gpus - 1) / total_gpus);
  bandwidth = bandwidth + bandwidth;
  if (rank == 0) {
    std::cout << "[FlagPerf Result]interconnect-MPI_interserver-algbw="
      << std::fixed << std::setprecision(2)
      << algbw / (1024.0 * 1024.0 * 1024.0) << "GiB/s" << std::endl;
    std::cout << "[FlagPerf Result]interconnect-MPI_interserver-algbw="
      << std::fixed << std::setprecision(2)
      << algbw / (1000.0 * 1000.0 * 1000.0) << "GB/s" << std::endl;
    std::cout << "[FlagPerf Result]interconnect-MPI_interserver-bandwidth="
      << std::fixed << std::setprecision(2)
      << bandwidth / (1024.0 * 1024.0 * 1024.0) << "GiB/s" << std::endl;
    std::cout << "[FlagPerf Result]interconnect-MPI_interserver-bandwidth="
      << std::fixed << std::setprecision(2)
      << bandwidth / (1000.0 * 1000.0 * 1000.0) << "GB/s" << std::endl;
  }
  checkMusaError(musaFree(d_src), "musaFree");
  checkMusaError(musaFree(d_dst), "musaFree");
  checkMcclError(mcclCommDestroy(comm), "mcclCommDestroy");
  checkMusaError(musaEventDestroy(start), "musaEventDestroy");
  checkMusaError(musaEventDestroy(end), "musaEventDestroy");
  checkMPIError(MPI_Finalize(), "MPI_Finalize");
  return 0;
}
