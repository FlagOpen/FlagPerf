export MACA_PATH=/opt/maca
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export LD_LIBRARY_PATH=./:$MACA_PATH/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$MACA_CLANG_PATH:$PATH
export MACA_VISIBLE_DEVICES=2
cucc gemm.cu -lcublas -o gemm
./gemm
