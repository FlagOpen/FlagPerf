# =================================================
# Export variables
# =================================================

export BKCL_PCIE_RING=1
export BKCL_TIMEOUT=1800
# when using tree allreduce, the number of nodes must be a multiple of 2
export BKCL_SOCKET_FORCE_TREE=1

export XMLIR_D_XPU_L3_SIZE=32505856

export BKCL_CCIX_RING=1
export BKCL_FORCE_SYNC=1

export ALLREDUCE_ASYNC=false
export ALLREDUCE_FUSION=0

export XMLIR_F_XPU_FC_GEMM_MODE=float
export XMLIR_F_FAST_INDEX_PUT=true


#export https_proxy=http://10.1.0.34:7890
#export http_proxy=http://10.1.0.34:7890
#export all_proxy=socks5h://10.1.0.34:7891

KLX_WEB_SERVER_URL=https://bd.bcebos.com/klx-pytorch-ipipe-bd

wget --tries=1 --read-timeout=1 10.1.2.158:8111/hello.txt || true
if [ -e hello.txt ]; then
    if [ "$(cat hello.txt)" = "hello from 158!" ]; then
        KLX_WEB_SERVER_URL=http://10.1.2.158:8111
    fi
fi

echo "KLX_WEB_SERVER_URL is set to $KLX_WEB_SERVER_URL"
rm -fv hello.txt
#https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/archives/111e7d45/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
#https://bd.bcebos.com/klx-pytorch-ipipe-bd/flagperf/archives/111e7d45/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip install ${KLX_WEB_SERVER_URL}/flagperf/archives/111e7d45/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
pip install ${KLX_WEB_SERVER_URL}/flagperf/archives/111e7d45/xacc-0.1.0-cp38-cp38-linux_x86_64.whl

export XACC_ENABLE=1
python -m xacc.install
