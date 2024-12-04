#include <stdio.h>
#include <musa_runtime.h>
#include <iostream>
#include <iomanip>

#define SIZE (1024ULL * 1024ULL * 1024ULL * sizeof(float))
#define WARMUP_ITERATIONS 100
#define ITERATIONS 2000

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
    int gpu_n;
    checkMusaError(musaGetDeviceCount(&gpu_n), "musaGetDeviceCount");
    printf("[FlagPerf Info]MUSA-capable device count: %i\n", gpu_n);
    if (gpu_n < 2) {
        fprintf(stderr, "Two or more GPUs with Peer-to-Peer access capability are required for inferconnect-P2P_intraserver-bandwidth test\n");
        exit(EXIT_FAILURE);
    }
    int can_access_peer;
    int p2pCapableGPUs[2];  // We take only 1 pair of P2P capable GPUs
    p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

    // Show all the combinations of supported P2P GPUs
    for (int i = 0; i < gpu_n; i++) {
        for (int j = 0; j < gpu_n; j++) {
            if (i == j) {
                continue;
            }
            checkMusaError(musaDeviceCanAccessPeer(&can_access_peer, i, j), "musaDeviceCanAccessPeer");
            printf("[FlagPerf Info]> Peer access from (GPU%d) -> (GPU%d) : %s\n",
                i, j, can_access_peer ? "Yes" : "No");
            if (can_access_peer && p2pCapableGPUs[0] == -1) {
                p2pCapableGPUs[0] = i;
                p2pCapableGPUs[1] = j;
            }
        }
    }
    if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1) {
        printf(
            "[FlagPerf Info]Two or more GPUs with Peer-to-Peer access capability are required for inferconnect-P2P_intraserver-bandwidth test\n");
        printf(
            "[FlagPerf Info]Peer to Peer access is not available amongst GPUs in the system, "
            "waiving test.\n");
        return 0;
    }
    int gpuid[2];
    gpuid[0] = p2pCapableGPUs[0];
    gpuid[1] = p2pCapableGPUs[1];
    printf("[FlagPerf Info]Enabling peer access between GPU%d and GPU%d...\n", gpuid[0],
        gpuid[1]);
    printf("Allocating buffers (%iGB on GPU%d, GPU%d and CPU Host)...\n",
        int(SIZE / 1024 / 1024 / 1024), gpuid[0], gpuid[1]);

    checkMusaError(musaSetDevice(gpuid[0]), "musaSetDevice");
    checkMusaError(musaDeviceEnablePeerAccess(gpuid[1], 0), "musaDeviceEnablePeerAccess");
    checkMusaError(musaSetDevice(gpuid[1]), "musaSetDevice");
    checkMusaError(musaDeviceEnablePeerAccess(gpuid[0], 0), "musaDeviceEnablePeerAccess");

    checkMusaError(musaSetDevice(gpuid[0]), "musaSetDevice");
    checkMusaError(musaMalloc(&d_src, SIZE), "musaMalloc");
    checkMusaError(musaSetDevice(gpuid[1]), "musaSetDevice");
    checkMusaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

    checkMusaError(musaEventCreate(&start), "musaEventCreate");
    checkMusaError(musaEventCreate(&end), "musaEventCreate");


    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (i % 2 == 0) {
            checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDefault), "musaMemcpy");
        }
        else {
            checkMusaError(musaMemcpy(d_src, d_dst, SIZE, musaMemcpyDefault), "musaMemcpy");
        }
    }


    checkMusaError(musaEventRecord(start, 0), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        if (i % 2 == 0) {
            checkMusaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDefault), "musaMemcpy");
        }
        else {
            checkMusaError(musaMemcpy(d_src, d_dst, SIZE, musaMemcpyDefault), "musaMemcpy");
        }
    }
    checkMusaError(musaEventRecord(end, 0), "musaEventRecord");
    checkMusaError(musaEventSynchronize(end), "musaEventSynchronize");
    checkMusaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");
    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0) + SIZE * ITERATIONS / (elapsed_time / 1000.0);
    std::cout << "[FlagPerf Result]inferconnect-P2P_intraserver-bandwidth="
        << std::fixed << std::setprecision(2) << bandwidth / (1024.0 * 1024.0 * 1024.0)
        << "GiB/s" << std::endl;

    std::cout << "[FlagPerf Result]inferconnect-P2P_intraserver-bandwidth="
        << std::fixed << std::setprecision(2) << bandwidth / (1000.0 * 1000.0 * 1000.0)
        << "GB/s" << std::endl;
    checkMusaError(musaSetDevice(gpuid[0]), "musaSetDevice");
    checkMusaError(musaDeviceDisablePeerAccess(gpuid[1]), "musaDeviceDisablePeerAccess");
    checkMusaError(musaSetDevice(gpuid[1]), "musaSetDevice");
    checkMusaError(musaDeviceDisablePeerAccess(gpuid[0]), "musaDeviceDisablePeerAccess");

    checkMusaError(musaFree(d_src), "musaFree");
    checkMusaError(musaFree(d_dst), "musaFree");
    checkMusaError(musaEventDestroy(start), "musaEventDestroy");
    checkMusaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}