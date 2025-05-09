#include <stdio.h>
#include <musa_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (1ULL * GB)
#define WARMUP_ITERATIONS 10
#define ITERATIONS 250000

void checkMusaError(musaError_t err, const char* msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "MUSA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* d2d */
#define LOOP_NUM 1
#define UNROLL_NUM 1

__global__ void global_bandwidth(float4 *dst, float4 *src) {
  int id, dist;
  id = blockIdx.x * blockDim.x * LOOP_NUM * UNROLL_NUM + threadIdx.x;
  dist = blockDim.x;
  // id = blockIdx.x * blockDim.x + threadIdx.x;
  // dist = gridDim.x * blockDim.x;
#pragma unroll 1
  for (int i = 0; i < LOOP_NUM; i++) {
#pragma unroll
    for (int j = 0; j < UNROLL_NUM; j++) {
      dst[id] = src[id];
      id += dist;
    }
  }
}

void runMemcpy(void *dst, void *src, size_t total_size) {
  dim3 block_size(256),
      block_num(total_size / sizeof(float4) / LOOP_NUM / UNROLL_NUM / 256);
  global_bandwidth<<<block_num, block_size>>>((float4 *)dst, (float4 *)src);
}

int main() {
    float* d_src, * d_dst;
    musaEvent_t start, end;
    float elapsed_time;

    checkMusaError(musaMalloc(&d_src, SIZE), "musaMalloc");
    checkMusaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

    checkMusaError(musaEventCreate(&start), "musaEventCreate");
    checkMusaError(musaEventCreate(&end), "musaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        runMemcpy(d_dst, d_src, SIZE);
    }

    checkMusaError(musaEventRecord(start), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        runMemcpy(d_dst, d_src, SIZE);
    }

    checkMusaError(musaEventRecord(end), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");

    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);
    
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkMusaError(musaFree(d_src), "musaFree");
    checkMusaError(musaFree(d_dst), "musaFree");
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}