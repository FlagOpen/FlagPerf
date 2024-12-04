#pragma once

#include <common.h>
#include <logger.h>

#define DEFAULT_BANDWIDTH_MEM_SIZE 2048
#define DEFAULT_BANDWIDTH_ITERS 30

typedef enum class BANDWIDTH_MODE {
    ALL,
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
} BW_MODE_T;

typedef enum class OFFSET_MODE { LOCAL, GLOBAL } OFFSET_MODE_T;

typedef enum GPU_ARCH {
    MP_10 = 10,
    MP_21 = 21,
    MP_22 = 22,
    MP_31 = 31
} GPU_ARCH_T;

class Benchmark {
public:
    // devices
    int specified_device;
    const char* specified_device_name;
    const char* specified_type_name;
    bool force_device;
    bool force_device_name;
    bool force_type;
    // option
    bool all_cases;
    bool use_event_timer;
    BW_MODE_T bandwidth_mode;
    int bandwidth_mem_size;
    int bandwidth_iters;

    logger* log;

    Benchmark();
    ~Benchmark();

    template <class T, typename... Args>
    float RunKernel(T func, dim3 block_num, dim3 block_size, uint iters,
        Args... args);

    int RunComputeMMABF16(device_info_t& dev_info);

    int RunBF16Test();
};