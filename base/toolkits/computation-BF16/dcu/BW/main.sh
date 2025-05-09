hipcc  -O3 gemm.cu -lhipblas -o gemm -std=c++17 --offload-arch=gfx936
./gemm
# /opt/dtk/rocblas/lib/rocblas/benchmark_tool/rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 3840 -n 3840 -k 3840 --lda 3840 --ldb 3840 --ldc 3840 --alpha 1 --beta 0 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --algo 0 

