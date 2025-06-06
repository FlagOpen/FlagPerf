#include <stdio.h>
#include <musa_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (16ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 1200

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

    checkMusaError(musaMallocHost(&d_src, SIZE), "musaMallocHost");
    checkMusaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

    checkMusaError(musaEventCreate(&start), "musaEventCreate");
    checkMusaError(musaEventCreate(&end), "musaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyHostToDevice), "musaMemcpy");
    }

    checkMusaError(musaEventRecord(start), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyHostToDevice), "musaMemcpy");
    }

    checkMusaError(musaEventRecord(end), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");

    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]transfer-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]transfer-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkMusaError(musaFreeHost(d_src), "musaFreeHost");
    checkMusaError(musaFree(d_dst), "musaFree");
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}