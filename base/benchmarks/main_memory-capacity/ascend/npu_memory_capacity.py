import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu


def test_gpu_memory_capacity():
    # Initial tensor size in MiB
    initial_byte_size = 10240
    current_byte_size = initial_byte_size
    min_byte_size = 1
    total_allocated = 0

    tensor_list = []

    print(f"Init tensor size: {initial_byte_size} MiB...")

    # Loop to reduce tensor size until it reaches the minimum size
    while current_byte_size >= min_byte_size:
        allocation_failed = False

        # Attempt to allocate memory until failure
        while not allocation_failed:
            try:
                # Allocate tensor of size `current_byte_size` MiB on the GPU
                tensor = torch.cuda.FloatTensor(int(current_byte_size * 1024 * 1024 / 4))
                tensor_list.append(tensor)
                total_allocated += current_byte_size
                print(f"Allocated: {total_allocated} MiB")
            except RuntimeError as e:
                # Handle out-of-memory error
                print(f"CUDA OOM at tensor size {current_byte_size} MiB. Allocated: {total_allocated} MiB")
                allocation_failed = True

        # Halve the tensor size for the next iteration
        current_byte_size /= 2
        print(f"Reduce tensor size to {current_byte_size} MiB")

    # Print the total allocated memory in GiB
    print(f"[FlagPerf Result]main_memory-capacity={total_allocated / 1024.0:.2f}GiB")
    # Print the total allocated memory in GB (decimal)
    print(
        f"[FlagPerf Result]main_memory-capacity={total_allocated * 1024.0 * 1024.0 / (1000.0 * 1000.0 * 1000.0):.2f}GB")


if __name__ == "__main__":
    test_gpu_memory_capacity()
