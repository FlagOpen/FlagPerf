#include "common.h"

#define MUSA_ERROR_CASE(ERR)                                \
  case ERR: {                                               \
    printf("" #ERR " in %s on line %i\n", func_name, line); \
    return 1;                                               \
  }

int check_musa_error(musaError _err, int line, const char* func_name) {
    switch (_err) {
        case musaSuccess:
            return 0;
            MUSA_ERROR_CASE(musaErrorInvalidValue)
                MUSA_ERROR_CASE(musaErrorMemoryAllocation)
                MUSA_ERROR_CASE(musaErrorInitializationError)
                MUSA_ERROR_CASE(musaErrorInvalidMemcpyDirection)
                MUSA_ERROR_CASE(musaErrorAddressOfConstant)
                MUSA_ERROR_CASE(musaErrorSynchronizationError)
                MUSA_ERROR_CASE(musaErrorNotYetImplemented)
                MUSA_ERROR_CASE(musaErrorMemoryValueTooLarge)
                MUSA_ERROR_CASE(musaErrorNoDevice)
                MUSA_ERROR_CASE(musaErrorInvalidDevice)
                MUSA_ERROR_CASE(musaErrorHostMemoryAlreadyRegistered)
                MUSA_ERROR_CASE(musaErrorHostMemoryNotRegistered)
                MUSA_ERROR_CASE(musaErrorIllegalInstruction)
                MUSA_ERROR_CASE(musaErrorInvalidAddressSpace)
                MUSA_ERROR_CASE(musaErrorLaunchFailure)
                MUSA_ERROR_CASE(musaErrorNotSupported)
                MUSA_ERROR_CASE(musaErrorTimeout)
                MUSA_ERROR_CASE(musaErrorUnknown)
                MUSA_ERROR_CASE(musaErrorApiFailureBase)
        default:
            printf("Unknown MUSA error %i in %s on line %i\n", _err, func_name, line);
            return 1;
    }
}

int GetDeviceInfo(int dev, device_info_t* dev_info) {
    musaDeviceProp deviceProp;
    if (musaSuccess != musaGetDeviceProperties(&deviceProp, dev)) {
        return -1;
    }
    dev_info->device_name = deviceProp.name;
    dev_info->device_arch = 10 * deviceProp.major + deviceProp.minor;
    int driverVersion = 0;
    musaDriverGetVersion(&driverVersion);
    dev_info->driver_version = std::to_string(driverVersion);
    TrimString(dev_info->device_name);
    TrimString(dev_info->driver_version);

    dev_info->num_compute_units = deviceProp.multiProcessorCount;
    dev_info->max_work_group_size = deviceProp.maxThreadsPerBlock;

    // Limiting max work-group size to 512
#define MAX_WG_SIZE 256
    dev_info->max_work_group_size =
        std::min(dev_info->max_work_group_size, (uint)MAX_WG_SIZE);
#undef MAX_WG_SIZE

    /*  Size of global device memory in bytes.  */
    dev_info->max_global_size = static_cast<uint64_t>(deviceProp.totalGlobalMem);
    /*  Max size of memory object allocation in bytes.*/
    dev_info->max_alloc_size = dev_info->max_global_size / 3;
    dev_info->max_clock_freq = static_cast<uint>(deviceProp.clockRate / 1000);
    dev_info->double_supported = true;
    dev_info->half_supported = true;

    dev_info->bw_global_max_size = 1 << 31;
    dev_info->bw_shmem_max_size = 1 << 28;
    dev_info->bw_transfer_max_size = 1 << 28;
    dev_info->compute_work_groups_per_cu = 2048;
    dev_info->compute_dp_work_groups_per_cu = 512;
    dev_info->shmem_work_groups_per_cu = 128;
    dev_info->compute_iters = 20;
    dev_info->bw_global_iters = 30;
    dev_info->bw_shmem_iters = 20;
    dev_info->bw_transfer_iters = 20;
    dev_info->kernel_latency_iters = 20000;

    return 0;
}

MUSAEvent::MUSAEvent(float* dur) : duration_us(dur) {
    CHECK_MUSA_ERROR(musaEventCreate(&startTime));
    CHECK_MUSA_ERROR(musaEventCreate(&stopTime));
    CHECK_MUSA_ERROR(musaEventRecord(startTime));
}

MUSAEvent::~MUSAEvent() {
    CHECK_MUSA_ERROR(musaEventRecord(stopTime));
    CHECK_MUSA_ERROR(musaEventSynchronize(stopTime));
    CHECK_MUSA_ERROR(musaEventElapsedTime(duration_us, startTime, stopTime));
    *duration_us *= 1e3f;
    CHECK_MUSA_ERROR(musaEventDestroy(startTime));
    CHECK_MUSA_ERROR(musaEventDestroy(stopTime));
}

Timer::Timer(float* dur) : duration_us(dur) {
    tick = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    tock = std::chrono::high_resolution_clock::now();
    *duration_us =
        (float)(std::chrono::duration_cast<std::chrono::microseconds>(tock - tick)
            .count());
}

void Populate(float* ptr, uint64_t num) {
    srand((unsigned int)time(NULL));

    for (uint64_t i = 0; i < num; i++) {
        // ptr[i] = (float)rand();
        // to ensure the sum of arr is a positive number
        // avoid the STORE in ReadOnly
        ptr[i] = (float)i;
    }
}

void Populate(double* ptr, uint64_t num) {
    srand((unsigned int)time(NULL));
    for (uint64_t i = 0; i < num; i++) {
        // ptr[i] = (double)rand();
        ptr[i] = (double)i;
    }
}

uint64_t RoundToMultipleOf(uint64_t number, uint64_t base, uint64_t max_value) {
    uint64_t n = (number > max_value) ? max_value : number;
    return (n / base) * base;
}

void TrimString(std::string& str) {
    size_t pos = str.find('\0');
    if (pos != std::string::npos) {
        str.erase(pos);
    }
}