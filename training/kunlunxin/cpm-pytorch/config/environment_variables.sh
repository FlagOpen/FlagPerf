# =================================================
# Export variables
# =================================================
set -x

export XMLIR_F_XPU_ENABLED_BOOL=true
export XMLIR_TORCH_XCCL_ENABLED=true

##===----------------------------------------------------------------------===##
## R480 config
##===----------------------------------------------------------------------===##

# BKCL
topo_file=${WORKSPACE-"."}/topo.txt
touch topo_file
export XPUSIM_TOPOLOGY_FILE=$(readlink -f $topo_file)

## workaround due to ccix bug
export BKCL_CCIX_RING="1"
export ALLREDUCE_ASYNC="0"
export ALLREDUCE_FUSION="0"

KLX_WEB_SERVER_URL=https://bd.bcebos.com/klx-pytorch-ipipe-bd

wget --tries=1 --read-timeout=1 10.1.2.158:8111/hello.txt || true
if [ -e hello.txt ]; then
    if [ "$(cat hello.txt)" = "hello from 158!" ]; then
        KLX_WEB_SERVER_URL=http://10.1.2.158:8111
    fi
fi

echo "KLX_WEB_SERVER_URL is set to $KLX_WEB_SERVER_URL"
rm -f hello.txt

pip install ${KLX_WEB_SERVER_URL}/flagperf/202307/cpm/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip install ${KLX_WEB_SERVER_URL}/flagperf/202307/cpm/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
