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