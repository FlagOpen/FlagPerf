# =================================================
# Export variables
# =================================================
set -x

export XMLIR_F_XPU_ENABLED_BOOL=true
export XMLIR_TORCH_XCCL_ENABLED=true

### whl before f5cfc3b need fallback aten::bernoulli_.float
# export XMLIR_D_FORCE_FALLBACK_STR="aten::bernoulli_.float"

### whl after f5cfc3b need fallback aten::bernoulli_.float,aten::_index_put_impl_
## export XMLIR_D_FORCE_FALLBACK_STR="aten::bernoulli_.float,aten::_index_put_impl_"

##===----------------------------------------------------------------------===##
## R200 config
##===----------------------------------------------------------------------===##

# export BKCL_RING_ALL_REDUCE=1 

##===----------------------------------------------------------------------===##
## R480 config
##===----------------------------------------------------------------------===##

# BKCL
topo_file=${WORKSPACE-"."}/topo.txt
touch topo_file
export XPUSIM_TOPOLOGY_FILE=$(readlink -f $topo_file)

# export BKCL_CCIX_RING=1

## workaround due to ccix bug
export BKCL_CCIX_RING="1"
export ALLREDUCE_ASYNC="0"
export ALLREDUCE_FUSION="0"

# To Be modified after the accuracy verification
export _XMLIR_D_GRAPH_COMPARE_ACCURACY_FLOAT=0.001