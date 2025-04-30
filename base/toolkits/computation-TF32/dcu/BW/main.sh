hipcc -O3 gemm.cu -lrocblas -o gemm -std=c++17 --offload-arch=gfx936
./gemm
