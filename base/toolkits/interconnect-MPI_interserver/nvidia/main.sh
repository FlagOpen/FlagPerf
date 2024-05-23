export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0;
export NCCL_IB_CUDA_SUPPORT=1;
export NCCL_IB_GID_INDEX=0;
export NCCL_IB_HCA=mlx5_2,mlx5_8;
nvcc -c -o bandwidth.o bandwidth.cu -I/usr/local/cuda/include -I/usr/local/nccl/include -I/usr/local/mpi/include
mpic++ -o bdtest bandwidth.o -L/usr/local/cuda/lib64 -L/usr/local/nccl/lib -L/usr/local/mpi/lib -lcudart -lnccl -lcuda -lmpi
mpirun --allow-run-as-root -np 16 ./bdtest