#include "benchmark_bf16.h"
#include <cstring>
#include <string>

Benchmark::Benchmark()
    : specified_device(0),
    specified_device_name(0),
    specified_type_name(0),
    force_device(false),
    force_device_name(false),
    force_type(false),
    all_cases(false),
    use_event_timer(false),
    bandwidth_mode(BANDWIDTH_MODE::ALL),
    bandwidth_mem_size(DEFAULT_BANDWIDTH_MEM_SIZE),
    bandwidth_iters(DEFAULT_BANDWIDTH_ITERS) {
    log = new logger();
}

Benchmark::~Benchmark() {
    if (log) {
        delete log;
    }
}

int Benchmark::RunBF16Test() {
    musaSetDevice(0);
    device_info_t dev_info;
    if (GetDeviceInfo(0, &dev_info)) {
        log->print(TAB "Can not get informations for Device " +
            std::to_string(0) + NEWLINE);
    }

    RunComputeMMABF16(dev_info);
    return 0;
}
