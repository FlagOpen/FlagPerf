import math
import torch
import importlib
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier
import torch.distributed.distributed_c10d as c10d

import inspect

no_copy_args = dict()
if 'no_copy' in inspect.getfullargspec(torch.distributed.reduce_scatter).args:
    no_copy_args["no_copy"] = True

## Update to APEX (https://github.com/NVIDIA/apex.git)
## changes incorporated in apex, compared to container APEX version 082f999 generated 4/13/2021
## 1) function to support gradient clipping before all reduce (late rule change to MLPerf, now grad clipping before and after all reduce are both allowed)


## Excerpted from PR # 1099 in apex library (https://github.com/NVIDIA/apex.git)
## for supporting gradient clipping before allreduce
## PR # 1099 adds the option to do either clip-before-allreduce or clip-after-allreduce
def _pipeline_block_reductions_patched(self, block_id):
    # Copy model grads to flat grads buffer
    self._flatten_grad_mt(1.0)

    # Compute L2 grad norm
    self._l2_grad_norm_st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self._l2_grad_norm_st):
        self._L2_grad_norm = self._flat_grads.norm(dtype=torch.float16,
                                                   p=2).float()
    torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)

    # Apply clipping & pre-reduction scaling on grads
    loss_scale = self.global_scale
    max_grad_norm = loss_scale * self.defaults['max_grad_norm']
    coeff = max_grad_norm / (1e-6 + self.L2_grad_norm)
    coeff = (coeff > 1) * self._one + (coeff <= 1) * coeff
    tmp = torch.cat(((self._one), (coeff)))
    index = (coeff + 1 > coeff).int()
    scale = tmp.index_select(0, index).half() / self._world_size
    self._flat_grads.mul_(scale)

    # Reduction within each node
    # Changes gradient format from [block * chunk * shard] to [shard * block * chunk]
    # The output format is the same as the fp32 master parameters
    works = [None] * self._num_chunks
    for chunk_id in range(self._num_chunks):
        glob_chunk_id = block_id * self._num_chunks + chunk_id
        rs_stream = self._rs_st[glob_chunk_id % self._num_rs_pg]
        rs_stream.wait_stream(torch.cuda.current_stream())
        rs_stream.wait_stream(self._l2_grad_norm_st)
        with torch.cuda.stream(rs_stream):
            works[chunk_id] = torch.distributed.reduce_scatter(
                self._fp16_g_chunks[block_id][chunk_id],
                self._flat_grads_shards[block_id][chunk_id],
                group=self._rs_pg[glob_chunk_id % self._num_rs_pg],
                async_op=True,
                **no_copy_args)

    # Reduction across nodes for each rank
    if self._num_groups > 1:
        for chunk_id in range(self._num_chunks):
            glob_chunk_id = block_id * self._num_chunks + chunk_id
            ar_stream = self._ar_st[glob_chunk_id % self._num_ar_pg]
            with torch.cuda.stream(ar_stream):
                works[chunk_id].wait()
                works[chunk_id] = torch.distributed.all_reduce(
                    self._fp16_g_chunks[block_id][chunk_id],
                    group=self._ar_pg[glob_chunk_id % self._num_ar_pg],
                    async_op=True)
    self._reductions_works[block_id] = works

    if block_id == 0:
        for block_id in range(self._num_blocks):
            for chunk_id in range(self._num_chunks):
                self._reductions_works[block_id][chunk_id].wait()


## Excerpted from PR # 1099 in apex library (https://github.com/NVIDIA/apex.git)
## for supporting gradient clipping before allreduce
## PR # 1099 adds the option to do either clip-before-allreduce or clip-after-allreduce
def _pipeline_step_patched(self):
    global_scale = self.global_scale
    # if clip before ar, set max_grad_norm to 0
    max_grad_norm = 0.0
    self._completion_st.wait_stream(self._l2_grad_norm_st)
    global_grad_norm = self.L2_grad_norm

    # check global_grad_norm and fill overflow_buf
    is_finite = (global_grad_norm + 1 > global_grad_norm).int()
    self._overflow_buf = self._one * (is_finite ^ self._one
                                      )  # toggle between 0 and 1
    torch.distributed.all_reduce(is_finite,
                                 op=torch.distributed.ReduceOp.MIN,
                                 group=self._current_process_group)
    torch.distributed.all_reduce(self._overflow_buf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=self._current_process_group)

    # increment step counter if no overflow
    self._step += is_finite
    self._completion_st.wait_stream(torch.cuda.current_stream())
    self._completion_st.wait_stream(self._l2_grad_norm_st)

    # Call step kernel once per step
    # Call all-gather once per step
    with torch.cuda.stream(self._completion_st):
        for block_id in range(self._num_blocks):
            for chunk_id in range(self._num_chunks):
                self._reductions_works[block_id][chunk_id].wait()
        #param_norm = self.__compute_contrib_param_norm()
        param_norm = self._DistributedFusedLAMB__compute_contrib_param_norm()
        multi_tensor_applier(
            self.multi_tensor_lamb_compute_update_term,
            self._overflow_buf,
            self._contrib_compute_update_term_tensor_list,  # g, p, m, v, u
            self._contrib_beta1,
            self._contrib_beta2,
            self._contrib_beta3,
            self._contrib_bias_correction,
            self._step,
            self._contrib_epsilon,
            self._adam_w_mode,
            self._contrib_weight_decay,
            global_scale,
            global_grad_norm,
            max_grad_norm)
        upd_norm = self._DistributedFusedLAMB__compute_contrib_update_norm()
        multi_tensor_applier(
            self.multi_tensor_lamb_update_weights,
            self._overflow_buf,
            self._contrib_update_weights_tensor_list,  # u, p, p_copy
            param_norm,
            upd_norm,
            self._offsets,
            self._lr,
            self._contrib_weight_decay,
            global_grad_norm,
            self._use_nvlamb)
        torch.distributed.all_gather(self._new_params_mega_shards,
                                     self._fp16_p,
                                     group=self._ag_pg[0],
                                     **no_copy_args)
