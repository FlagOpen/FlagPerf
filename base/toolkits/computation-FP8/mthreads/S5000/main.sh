g++ gemm-new.cpp -std=c++17 -I/usr/local/musa/include -L /usr/local/musa/lib/ -fopenmp -lmudnn -lmusart -o gemm -O2
./gemm