hipcc -O3 gemm.cu -lhipblas -o gemm -std=c++17 --offload-arch=gfx936
./gemm
