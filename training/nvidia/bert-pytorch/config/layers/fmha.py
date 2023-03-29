# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from collections import OrderedDict

try:
    from apex import fused_dense
except:
    fused_dense = None
import numpy as np
import fmhalib as mha


class FMHAFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training):
        b = cu_seqlens.numel() - 1

        if b < 4:
            max_s = 512
            context, S_dmask = mha.fwd_nl(qkv, cu_seqlens, p_dropout, max_s,
                                          is_training, None)
        else:
            context, S_dmask = mha.fwd(qkv, cu_seqlens, p_dropout, max_s,
                                       is_training, None)
        ctx.save_for_backward(qkv, S_dmask)
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        return context

    @staticmethod
    def backward(ctx, dout):
        b = ctx.cu_seqlens.numel() - 1
        qkv, S_dmask = ctx.saved_tensors
        if b < 4:
            dqkv, dp, dkv = mha.bwd_nl(dout, qkv, S_dmask, ctx.cu_seqlens,
                                       ctx.p_dropout, ctx.max_s)
        else:
            dqkv, dp = mha.bwd(dout, qkv, S_dmask, ctx.cu_seqlens,
                               ctx.p_dropout, ctx.max_s)

        return dqkv, None, None, None, None


class TestParam(torch.nn.Parameter):

    def __init__(self, data, requires_grad=True):
        super(TestParam, self).__init__()
        self.data = data
        self.requires_grad = requires_grad
        self.tag = 'qkv'
        self.counter = 0


class NoopCat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Wq, Wk, Wv, Bq, Bk, Bv, Wqkv, Bqkv, hidden_size):
        assert not Wqkv.requires_grad and not Bqkv.requires_grad, "hye!"
        Wtmp = Wqkv.view(3, hidden_size, hidden_size)
        Btmp = Bqkv.view(3, hidden_size)
        Wq.data = Wtmp[0, :, :]
        Wk.data = Wtmp[1, :, :]
        Wv.data = Wtmp[2, :, :]

        Bq.data = Btmp[0, :]
        Bk.data = Btmp[1, :]
        Bv.data = Btmp[2, :]

        Wtmp = Wqkv.new()
        Wtmp.set_(Wqkv.storage(), Wqkv.storage_offset(), Wqkv.size(),
                  Wqkv.stride())
        Wtmp.requires_grad = True

        Btmp = Bqkv.new()
        Btmp.set_(Bqkv.storage(), Bqkv.storage_offset(), Bqkv.size(),
                  Bqkv.stride())
        Btmp.requires_grad = True
        ctx.save_for_backward(Wqkv, Bqkv, Wq, Wk, Wv, Bq, Bk, Bv)
        ctx.hidden_size = hidden_size
        return Wtmp, Btmp

    @staticmethod
    def backward(ctx, dWqkv, dBqkv):
        Wqkv, Bqkv, Wq, Wk, Wv, Bq, Bk, Bv = ctx.saved_tensors
        Wtmp = Wqkv.view(3, ctx.hidden_size, ctx.hidden_size)
        Btmp = Bqkv.view(3, ctx.hidden_size)
        Wq.data = Wtmp[0, :, :]
        Wk.data = Wtmp[1, :, :]
        Wv.data = Wtmp[2, :, :]

        Bq.data = Btmp[0, :]
        Bk.data = Btmp[1, :]
        Bv.data = Btmp[2, :]

        dWtmp = dWqkv.view(3, ctx.hidden_size, ctx.hidden_size)
        dBtmp = dBqkv.view(3, ctx.hidden_size)
        return dWtmp[0, :, :], dWtmp[1, :, :], dWtmp[2, :, :], dBtmp[
            0, :], dBtmp[1, :], dBtmp[2, :], None, None, None


class FMHA(torch.nn.Module):

    def __init__(self, config):

        super(FMHA, self).__init__()

        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        # self.fuse_bias = config.fused_bias_mha
        self.fuse_bias = fused_dense is not None
        assert self.d * self.h == self.hidden_size, "Invalid hidden size/num_heads"

        self.register_buffer(
            "Wqkv", torch.zeros(3 * config.hidden_size, config.hidden_size))
        self.register_buffer("Bqkv", torch.zeros(3 * config.hidden_size))
        self.Wqkv.requires_grad = False
        self.Bqkv.requires_grad = False
        self.Wqkv.detach()
        self.Bqkv.detach()
        with torch.no_grad():
            params = []
            Wtmp = self.Wqkv.view(3, self.hidden_size, self.hidden_size)
            Btmp = self.Bqkv.view(3, self.hidden_size)
            for tag, idx in zip('qkv', range(3)):
                params.append(('W' + tag, torch.nn.Parameter(Wtmp[idx, :, :])))
            for tag, idx in zip('qkv', range(3)):
                params.append(('B' + tag, torch.nn.Parameter(Btmp[idx, :])))

            self.param_views = OrderedDict(params)

            self._reset_param_views()

        def prep_weights(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
            Wq = state_dict.pop(prefix + 'query.weight')
            bq = state_dict.pop(prefix + 'query.bias')

            Wk = state_dict.pop(prefix + 'key.weight')
            bk = state_dict.pop(prefix + 'key.bias')

            Wv = state_dict.pop(prefix + 'value.weight')
            bv = state_dict.pop(prefix + 'value.bias')

            weight = torch.cat([
                Wq.view(self.h, self.d, self.hidden_size),
                Wk.view(self.h, self.d, self.hidden_size),
                Wv.view(self.h, self.d, self.hidden_size)
            ],
                               dim=0).reshape(config.hidden_size * 3,
                                              config.hidden_size).contiguous()

            bias = torch.cat([
                bq.view(self.h, self.d),
                bk.view(self.h, self.d),
                bv.view(self.h, self.d)
            ],
                             dim=0).reshape(3 *
                                            config.hidden_size).contiguous()

            state_dict[prefix + 'Wqkv'] = weight
            state_dict[prefix + 'Bqkv'] = bias
            state_dict[prefix + 'Wq'] = Wq
            state_dict[prefix + 'Wk'] = Wk
            state_dict[prefix + 'Wv'] = Wv
            state_dict[prefix + 'Bq'] = bq
            state_dict[prefix + 'Bk'] = bk
            state_dict[prefix + 'Bv'] = bv

        self._register_load_state_dict_pre_hook(prep_weights)

    def _reset_param_views(self):
        with torch.no_grad():
            Wtmp = self.Wqkv.view(3, self.hidden_size, self.hidden_size)
            Btmp = self.Bqkv.view(3, self.hidden_size)

            for tag, idx in zip('qkv', range(3)):
                self.param_views['W' + tag].data = Wtmp[idx, :, :]
            for tag, idx in zip('qkv', range(3)):
                self.param_views['B' + tag].data = Btmp[idx, :]

    def _apply(self, fn):

        with torch.no_grad():
            self.Wqkv = fn(self.Wqkv)

            if self.Wqkv.grad is not None:
                self.Wqkv.grad = fn(self.Wqkv.grad)

            self.Bqkv = fn(self.Bqkv)

            if self.Bqkv.grad is not None:
                self.Bqkv.grad = fn(self.Bqkv.grad)

            self._reset_param_views()

    @property
    def _parameters(self):
        self._reset_param_views()
        return self.param_views

    @_parameters.setter
    def _parameters(self, _):
        if 'Wqkv' in self.__dict__ and self.Wqkv is not None and self.Wqkv.device == torch.device(
                'cuda:0'):
            import traceback
            traceback.print_stack()
        pass

    def forward(self,
                hidden_states,
                cu_seqlens,
                seqlens,
                max_s,
                is_training=True):

        Wqkv, Bqkv = NoopCat.apply(
            *[self.param_views[x + y] for x in 'WB' for y in 'qkv'], self.Wqkv,
            self.Bqkv, self.hidden_size)
        if not self.fuse_bias:
            qkv = F.linear(hidden_states, Wqkv, Bqkv)
        else:
            qkv = fused_dense.fused_dense_function(hidden_states, Wqkv, Bqkv)
        p_dropout = self.p_dropout

        ctx = FMHAFun.apply(qkv.view(-1, 3, self.h, self.d), cu_seqlens,
                            p_dropout, max_s, is_training)

        return ctx.view(-1, self.hidden_size)
