#include <stdio.h>
#include <musa_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (1ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 210000

void checkMusaError(musaError_t err, const char* msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "MUSA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
        checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDeviceToDevice), "musaMemcpy");
    }

    checkMusaError(musaEventRecord(start), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDeviceToDevice), "musaMemcpy");
    }

    checkMusaError(musaEventRecord(end), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");

    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);
    printf("[FlagPerf Result]elapsed_time=%.2fs\n",elapsed_time / 1000.0);
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkMusaError(musaFree(d_src), "musaFree");
    checkMusaError(musaFree(d_dst), "musaFree");
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}