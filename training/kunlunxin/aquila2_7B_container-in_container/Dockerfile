FROM 10.1.15.74:5000/zhiyuan_flagperf_ai_platform:v2.0

ENV CUDART_DUMMY_REGISTER 1
ENV XPU_DUMMY_EVENT 1

ENV XPU_ZEBU_MODE 1
ENV BKCL_XLINK_C2C 1
ENV XBLAS_FC_HBM_VERSION 40
ENV XPU_FORCE_CODE_PARAM_LOCATE_IN_L3 1
ENV XPU_FORCE_USERMODE_LAUNCH 1

ENV LD_LIBRARY_PATH /usr/local/xpu/xre/so:/usr/local/xpu/xccl/so:$LD_LIBRARY_PATH
ENV XMLIR_FA_GEMM_TYPE float16
ENV FAST_SWIGLU_ENABLE 1
ENV XDNN_FAST_DIV_SCALAR true
ENV FAST_SUB_MN_M true

RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagScale -b release/v0.2"
RUN /bin/bash -c "git clone https://github.com/FlagOpen/FlagPerf.git -b main"
