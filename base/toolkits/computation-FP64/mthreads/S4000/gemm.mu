#include <chrono>
#include <iostream>
#include <mublas.h>
#include <musa_runtime.h>
#include <vector>

constexpr int M = 8192;
constexpr int N = 8192;
constexpr int K = 8192;

struct PrecisionConfig {
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    double* d_A, * d_B, * d_C;
    std::vector<double> h_A(M * K, double(1.0f));
    std::vector<double> h_B(K * N, double(1.0f));
    std::vector<double> h_C(M * N);

    musaMalloc(&d_A, M * K * config.bytesPerElement);
    musaMalloc(&d_B, K * N * config.bytesPerElement);
    musaMalloc(&d_C, M * N * config.bytesPerElement);

    musaMemcpy(d_A, h_A.data(), M * K * config.bytesPerElement, musaMemcpyHostToDevice);
    musaMemcpy(d_B, h_B.data(), K * N * config.bytesPerElement, musaMemcpyHostToDevice);

    mublasHandle_t handle;
    mublasCreate(&handle);

    double alpha = 1.0f;
    double beta = 0.0f;

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
            M, N, K, &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M);

    }

    musaError_t syncError = musaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != musaSuccess) {
        std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
            M, N, K, &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M);
    }
    syncError = musaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (syncError != musaSuccess) {
        std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
    }
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << config.name << " Single Op Duration: "
        << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double flops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double FLOPS = flops / time_second;
    double TFLOPS = FLOPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-FP32=" << TFLOPS << "TFLOPS"
        << std::endl;

    musaMemcpy(h_C.data(), d_C, M * N * config.bytesPerElement, musaMemcpyDeviceToHost);

    musaFree(d_A);
    musaFree(d_B);
    musaFree(d_C);

    mublasDestroy(handle);
}

int main() {
    PrecisionConfig fp64 = { sizeof(double), "FP64", 70, 10 };

    test(fp64);

    return 0;
}