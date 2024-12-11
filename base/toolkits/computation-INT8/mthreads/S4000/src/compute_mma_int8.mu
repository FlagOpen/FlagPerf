#include "benchmark.muh"
#include "compute_mma_int8.muh"

#define WARPSIZE 128
#define TEST_DETAIL(show_, kernel_, tag_, M_, N_, K_, ARCH_)                  \
  work_per_warp = M_ * N_ * K_ * 2 * ITERS * UNROLL_NUM;                      \
  timed = RunKernel(kernel_<M_, N_, K_, ARCH_>, block_num, block_size, iters, \
                    d_x);                                                     \
  gops = (static_cast<float>(total_num) / WARPSIZE) * work_per_warp / timed / \
      1e3f;                                                                   \
  gops_max = std::max(gops_max, gops);                                        \
  if (show_) {                                                                \
    log->print(TAB TAB TAB #tag_ "  : ");                                     \
    log->print(gops);                                                         \
    log->print(NEWLINE);                                                      \
  }

#define TEST_END(show_, type_)             \
  if (!show_) {                            \
    log->print("[FlagPerf Result]computation-" #type_ "=");\
    log->print(gops_max/1e3);\
    log->print("TFLOPS");\
    log->print(NEWLINE); \
  }

int Benchmark::RunComputeMMAINT8(device_info_t& dev_info) {

    // mtgpu imma only

    float timed, gops, gops_max;
    int work_per_warp;
    dim3 block_size(1024);
    int grid_size = std::min((dev_info.num_compute_units) *
        (dev_info.compute_work_groups_per_cu) *
        (block_size.x) * sizeof(int),
        dev_info.max_alloc_size) /
        ((block_size.x) * sizeof(int));
    grid_size = std::min(grid_size, 2048);
    dim3 block_num(grid_size);
    size_t total_num = block_size.x * block_num.x;
    uint iters = dev_info.compute_iters;
    {
        if (dev_info.device_arch == MP_21) {
            void* d_x;
            CHECK_MUSA_ERROR(musaMalloc(&d_x, total_num * sizeof(int)));
            gops_max = 0.0f;
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_16_8_16, 16, 16, 16,
                MP_21);
            TEST_END(all_cases, UINT8);
            CHECK_MUSA_ERROR(musaFree(d_x));
        }
        else if (dev_info.device_arch == MP_22) {
            void* d_x;
            CHECK_MUSA_ERROR(musaMalloc(&d_x, total_num * sizeof(int)));
            gops_max = 0.0f;
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_16_16_16, 16, 16, 16,
                MP_22);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_32_8_16, 32, 8, 16,
                MP_22);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_8_32_16, 8, 32, 16,
                MP_22);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_32_32_32, 32, 32, 32,
                MP_22);
            TEST_END(all_cases, INT8);
            CHECK_MUSA_ERROR(musaFree(d_x));
        }
        else if (dev_info.device_arch == MP_31) {
            void* d_x;
            CHECK_MUSA_ERROR(musaMalloc(&d_x, total_num * sizeof(int)));
            gops_max = 0.0f;
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_16_16_16, 16, 16, 16,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_32_8_16, 32, 8, 16,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_8_32_16, 8, 32, 16,
                MP_31);
            // TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_32_32_32, 32, 32, 32,
            //             MP_31);
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_16_16_32, 16, 16, 32,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_uint8, UINT8_16_16_64, 16, 16, 64,
                MP_31);
            TEST_END(all_cases, UINT8);
            gops_max = 0.0f;
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_16_16_16, 16, 16, 16,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_32_8_16, 32, 8, 16,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_8_32_16, 8, 32, 16,
                MP_31);
            // TEST_DETAIL(all_cases, compute_mma_int8, INT8_32_32_32, 32, 32, 32,
            //             MP_31);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_16_16_32, 16, 16, 32,
                MP_31);
            TEST_DETAIL(all_cases, compute_mma_int8, INT8_16_16_64, 16, 16, 64,
                MP_31);
            TEST_END(all_cases, INT8);
            CHECK_MUSA_ERROR(musaFree(d_x));
        }
        else {
            log->print(TAB TAB TAB "NOT SUPPORT mp_" +
                std::to_string(dev_info.device_arch) + NEWLINE);
        }
    }

    return 0;
}