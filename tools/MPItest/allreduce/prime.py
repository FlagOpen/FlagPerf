import torch
import torch.distributed as dist


def sieve_of_eratosthenes(n):
    if n < 2:
        return []

    primes = [True] * (n + 1)
    primes[0], primes[1] = False, False

    for current in range(2, int(n**0.5) + 1):
        if primes[current]:
            for multiple in range(current * current, n + 1, current):
                primes[multiple] = False

    return [number for number, is_prime in enumerate(primes) if is_prime]


if __name__ == "__main__":

    iters = 1000

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    prime_numbers = sieve_of_eratosthenes(world_size)
    print("all_prime_ranks")

    group = dist.new_group(prime_numbers)

    a = torch.zeros(1000000000, dtype=torch.float).to(rank % 8)
    dist.barrier()
    torch.cuda.synchronize()

    import time

    start = time.time()

    for _ in range(iters):
        dist.all_reduce(a, group=group, op=dist.ReduceOp.SUM)
    dist.barrier()
    torch.cuda.synchronize()
    print(a)
    end = time.time()
    inter = end - start

    gbps = 4 * iters / inter
    print(gbps)

    dist.destroy_process_group()
