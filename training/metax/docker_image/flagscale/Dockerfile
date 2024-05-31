FROM maca-c500-pytorch-2.19.0.9-ubuntu18.04-amd64:FlagPerf_aquila2_7b
ENV PATH="/opt/conda/bin:${PATH}"
ENV MACA_SMALL_PAGESIZE_ENABLE=1
ENV MHA_USE_BLAS=ON
ENV MHA_BWD_NO_ATOMIC_F64=1
ENV CUCC_PATH=${MACA_PATH}/tools/cu-bridge
ENV CUDA_PATH=${CUCC_PATH}
RUN /bin/bash -c "uname -a"
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  pybind11 regex
