#pragma once

#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif
#include "musa_runtime.h"
#include <string>
#include <chrono>

#define TAB "  "
#define NEWLINE "\n"
#ifndef __FreeBSD__
#define uint unsigned int
#endif
#define ulong unsigned long

#if defined(__APPLE__) || defined(__MACOSX)
#define OS_NAME "Macintosh"
#elif defined(__ANDROID__)
#define OS_NAME "Android"
#elif defined(_WIN32)
#if defined(_WIN64)
#define OS_NAME "Win64"
#else
#define OS_NAME "Win32"
#endif
#elif defined(__linux__)
#if defined(__x86_64__)
#define OS_NAME "Linux x64"
#elif defined(__i386__)
#define OS_NAME "Linux x86"
#elif defined(__arm__)
#define OS_NAME "Linux ARM"
#elif defined(__aarch64__)
#define OS_NAME "Linux ARM64"
#else
#define OS_NAME "Linux unknown"
#endif
#elif defined(__FreeBSD__)
#define OS_NAME "FreeBSD"
#else
#define OS_NAME "Unknown"
#endif

int check_musa_error(musaError _err, int line, const char* func_name);

#define _PERF_CHECK_MUSA_ERROR_INNER(cond, func, line) \
  do {                                                 \
    if (check_musa_error(cond, line, func))            \
      exit(1);                                         \
  } while (0)

#define CHECK_MUSA_ERROR(cond) \
  _PERF_CHECK_MUSA_ERROR_INNER(cond, __PRETTY_FUNCTION__, __LINE__)

typedef struct {
    std::string device_name;
    std::string driver_version;
    int device_arch;

    uint num_compute_units;
    uint max_work_group_size;
    uint64_t max_alloc_size;
    uint64_t max_global_size;
    uint max_clock_freq;

    bool half_supported;
    bool double_supported;
    bool imma_supported;

    // Test specific options
    uint bw_global_iters;
    uint bw_shmem_iters;
    uint64_t bw_global_max_size;
    uint64_t bw_shmem_max_size;
    uint compute_work_groups_per_cu;
    uint compute_dp_work_groups_per_cu;
    uint shmem_work_groups_per_cu;
    uint compute_iters;
    uint bw_transfer_iters;
    uint kernel_latency_iters;
    uint64_t bw_transfer_max_size;
    std::string extension;
} device_info_t;

class Timer {
public:
    explicit Timer(float* dur);
    ~Timer();

private:
    float* duration_us;
    std::chrono::high_resolution_clock::time_point tick;
    std::chrono::high_resolution_clock::time_point tock;
};

class MUSAEvent {
public:
    explicit MUSAEvent(float* dur);
    ~MUSAEvent();

private:
    float* duration_us;
    musaEvent_t startTime;
    musaEvent_t stopTime;
};

int GetDeviceInfo(int dev, device_info_t* dev_info);

// Round down to next multiple of the given base with an optional maximum value
uint64_t RoundToMultipleOf(uint64_t number, uint64_t base,
    uint64_t maxValue = UINT64_MAX);

void Populate(float* ptr, uint64_t N);
void Populate(double* ptr, uint64_t N);

void TrimString(std::string& str);