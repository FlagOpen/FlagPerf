#include <musa_runtime.h>
#include <stdio.h>

bool CHECK(musaError_t call){
    const musaError_t error = call;
    return (error == musaSuccess);
}


void test_gpu_memory_capacity() {
    size_t initial_byte_size = 65536;
    size_t current_byte_size = initial_byte_size;
    size_t min_byte_size = 1;
    size_t total_allocated = 0;

    printf("Init tensor size:  %zu MiB...\n", initial_byte_size);
 
    while (current_byte_size >= min_byte_size) {
        void* ptr = NULL;
        bool allocation_failed = false;

        while (!allocation_failed) {
            if (CHECK(musaMalloc(&ptr, current_byte_size * 1024 * 1024))){
                total_allocated += current_byte_size;
                printf("Allocated: %zu MiB\n", total_allocated);
            }
            else{
                printf("MUSA OOM at tensor size %zu MiB. Allocated:%zu MiB\n", current_byte_size, total_allocated);
                allocation_failed = true;
            }
        }

        current_byte_size /= 2;
        printf("Reduce tensor size to %zu MiB\n", current_byte_size);
    }
	    
    
    printf("[FlagPerf Result]main_memory-capacity=%.2fGiB\n", total_allocated / (1024.0));
    printf("[FlagPerf Result]main_memory-capacity=%.2fGB\n", total_allocated * 1024.0 * 1024.0 / (1000.0 * 1000.0 * 1000.0));  
}

int main() {
    test_gpu_memory_capacity();
    musaDeviceReset();
    return 0;
}