import torch

# 'segmented' or 'full_iteration' options for CUDA graph capture.
# 'segmented' option: Pytorch Autograd orchestrates execution of backward ops every iteration.
# 'full_iteration' option: CUDA graph orchestrates execution of bwd ops every iteration without Autograd involvement (has composability limitations but could be more performant allowing optimizer                              and collectives capture).
cuda_graph_mode: str = "segmented"

# Maximum number of iterations to capture in a single graph.
# Requires 'full_iteration' option for '--cuda_graph_mode'.
max_iterations_per_graph: int = 4

# Whether to do allreduces during gradient accumulation steps.
allreduce_post_accumulation: bool = False

# Whether to do fp16 allreduce post accumulation.
allreduce_post_accumulation_fp16: bool = False

# Whether to run with unpadding.
unpad: bool = False

# Whether to run with unpadding.
unpad_fmha: bool = False

# Whether to pad tokens.
pad: bool = False

# Whether to disable fusion of the scaling to BMM1.
disable_fuse_scale: bool = False

# Whether to disable fusion of the QKV GEMMs.
disable_fuse_qkv: bool = False

# Whether to disable apex softmax.
disable_apex_softmax: bool = False

# Enable use of streams for pad case.
enable_stream: bool = False

# Whether to run with optimizations.
fused_mha: bool = False

# Enable CUDA graph execution.
use_cuda_graph: bool = False

# DDP type: 'apex' or 'native'.
ddp_type: str = "apex"

# Bypass AMP unscaling and inf/nan checks for SOL measurements.
bypass_amp: bool = False

# Whether to use distributed lamb.
distributed_lamb: bool = False

# distributed weight update group size. If arg is 0, defaults to one node
dwu_group_size: int = 0

# number of blocks in dwu scheme
dwu_num_blocks: int = 4

# number of chunks in dwu scheme
dwu_num_chunks: int = 1

# number of reduction-scatter streams in dwu scheme
dwu_num_rs_pg: int = 2

# number of all-reduce streams in dwu scheme
dwu_num_ar_pg: int = 4

# number of all-gather streams in dwu scheme
dwu_num_ag_pg: int = 2

# whether to overlap reductions with backprop
dwu_overlap_reductions: bool = False

# do allgather with e5m2 floats
dwu_e5m2_allgather: bool = False

# the apex optimization level, value: [O1, O2]
opt_level: str = "O2"


def get_gpu_mem():
    return torch.cuda.get_device_properties("cuda:0").total_memory / 1e+9
