hipcc bandwidth.cu -o bdtest -std=c++17 --offload-arch=gfx936 --gpu-max-threads-per-block=1024
./bdtest
