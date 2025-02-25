mcc gemm.mu -std=c++17 -fopenmp -lmudnn -lmusart -o gemm -O2 --offload-arch=mp_31
./gemm