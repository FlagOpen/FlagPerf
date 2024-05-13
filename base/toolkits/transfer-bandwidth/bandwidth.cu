#include <stdio.h>
#include <cuda_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (1ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 100000

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

    checkCudaError(cudaMallocHost(&d_src, SIZE), "cudaMallocHost");
    checkCudaError(cudaMalloc(&d_dst, SIZE), "cudaMalloc");

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&end), "cudaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkCudaError(cudaMemcpy(d_dst, d_src, SIZE, cudaMemcpyHostToDevice), "cudaMemcpy");
    }

    checkCudaError(cudaEventRecord(start), "cudaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkCudaError(cudaMemcpy(d_dst, d_src, SIZE, cudaMemcpyHostToDevice), "cudaMemcpy");
    }

    checkCudaError(cudaEventRecord(end), "cudaEventRecord");
    checkCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");

    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end), "cudaEventElapsedTime");

    double bandwidth = 2.0 * (SIZE) * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]transfer-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]transfer-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(cudaFreeHost(d_src), "cudaFreeHost");
    checkCudaError(cudaFree(d_dst), "cudaFree");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy");
    checkCudaError(cudaEventDestroy(end), "cudaEventDestroy");

    return 0;
}